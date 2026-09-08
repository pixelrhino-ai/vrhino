#include "vrhino/product/run.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <signal.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/input.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/product/safetensors.h"
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"
#include "vrhino/tokenizer.h"

namespace vrhino::product {
namespace {

using Clock = std::chrono::steady_clock;

std::atomic<uint64_t> output_probe_counter = 0;

[[noreturn]] void product_fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) product_fail(ModelPackageErrorCode::ArtifactMissing,
                                     "cannot open artifact: " + path.string());
    return std::string(std::istreambuf_iterator<char>(stream), {});
}

const ResolvedArtifact& artifact(const ResolvedRunnableModel& model,
                                 const std::string& id) {
    const auto found = model.artifacts.find(id);
    if (found == model.artifacts.end())
        product_fail(ModelPackageErrorCode::ArtifactMissing,
                     "run profile references unresolved artifact: " + id);
    return found->second;
}

const Json& selected_preset(const Json& manifest, const std::string& name) {
    const Json& presets = manifest.at("defaults").at("presets");
    const Json* selected = presets.find(name);
    if (selected == nullptr)
        product_fail(ModelPackageErrorCode::InvalidInput, "unknown preset: " + name);
    return *selected;
}

std::string selected_preset_name(const ResolvedRunnableModel& model,
                                 const std::string& requested) {
    return requested.empty() ? model.manifest.default_preset : requested;
}

std::string gibibytes(const uint64_t bytes) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << static_cast<long double>(bytes) /
                  static_cast<long double>(1024ULL * 1024ULL * 1024ULL)
           << " GiB";
    return output.str();
}

std::filesystem::path absolute_output_path(const std::filesystem::path& output) {
    if (output.empty() || output.string().find('\0') != std::string::npos)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output path is empty or contains an invalid character");
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(output, error);
    if (error)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot resolve output path " + output.string() + ": " +
                         error.message());
    return absolute.lexically_normal();
}

std::string output_probe_name(const char* suffix) {
    return ".vrhino-output-preflight-" + std::to_string(getpid()) + "-" +
           std::to_string(output_probe_counter.fetch_add(1)) + suffix;
}

void remove_probe(const std::filesystem::path& path) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

void check_output_destination(const std::filesystem::path& requested,
                              const bool overwrite) {
    const std::filesystem::path output = absolute_output_path(requested);
    std::error_code error;
    const std::filesystem::file_status output_status =
        std::filesystem::symlink_status(output, error);
    if (error && error != std::errc::no_such_file_or_directory)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot inspect output path " + output.string() + ": " +
                         error.message());
    if (!error && std::filesystem::exists(output_status)) {
        if (!std::filesystem::is_regular_file(output_status) &&
            !std::filesystem::is_symlink(output_status))
            product_fail(ModelPackageErrorCode::OutputInvalid,
                         "output path is not a regular file: " + output.string());
        if (!overwrite)
            product_fail(ModelPackageErrorCode::OutputExists,
                         "output already exists; pass --overwrite: " + output.string());
    }

    const std::filesystem::path parent = output.parent_path();
    error.clear();
    std::filesystem::create_directories(parent, error);
    if (error)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot create output parent " + parent.string() +
                         " for " + output.string() + ": " + error.message());
    const std::filesystem::file_status parent_status =
        std::filesystem::status(parent, error);
    if (error || !std::filesystem::is_directory(parent_status))
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output parent is not a directory: " + parent.string());

    std::filesystem::path partial = output;
    partial += ".partial";
    const std::filesystem::file_status partial_status =
        std::filesystem::symlink_status(partial, error);
    if (error && error != std::errc::no_such_file_or_directory)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot inspect temporary output path " + partial.string() +
                         ": " + error.message());
    if (!error && std::filesystem::exists(partial_status) &&
        !std::filesystem::is_regular_file(partial_status))
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "temporary output path is not a regular file: " +
                         partial.string());

    const std::filesystem::path probe = parent / output_probe_name(".partial");
    const int descriptor = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                  S_IRUSR | S_IWUSR);
    if (descriptor < 0)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output parent is not writable for " + output.string() +
                         " (parent " + parent.string() + "): " +
                         std::strerror(errno));
    if (::close(descriptor) != 0) {
        const std::string reason = std::strerror(errno);
        remove_probe(probe);
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot close output preflight probe in " + parent.string() +
                         ": " + reason);
    }
    if (::unlink(probe.c_str()) != 0) {
        const std::string reason = std::strerror(errno);
        remove_probe(probe);
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot remove output preflight probe from " + parent.string() +
                         ": " + reason);
    }
}

void check_media_encoder(const std::filesystem::path& requested) {
    if (requested.empty())
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder path is empty; installation is "
                     "incomplete or damaged; inference did not start");
    std::error_code error;
    const std::filesystem::path encoder = std::filesystem::absolute(requested, error);
    if (error || !std::filesystem::is_regular_file(encoder, error) || error)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder is missing: " + requested.string() +
                         "; installation is incomplete or damaged; inference did not start");
    if (::access(encoder.c_str(), X_OK) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder is not executable: " + encoder.string() +
                         "; installation is incomplete or damaged; inference did not start");

    const pid_t child = ::fork();
    if (child < 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot launch bundled VRhino media encoder preflight: " +
                         encoder.string() + "; inference did not start");
    if (child == 0) {
        const int null_fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_fd >= 0) {
            ::dup2(null_fd, STDOUT_FILENO);
            ::dup2(null_fd, STDERR_FILENO);
            ::close(null_fd);
        }
        ::execl(encoder.c_str(), encoder.c_str(), "-version",
                static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder failed its readiness check: " +
                         encoder.string() +
                         "; installation is incomplete or damaged; inference did not start");
}

Json load_run_profile(const ResolvedRunnableModel& model, const std::string& preset) {
    const Json manifest = Json::parse(model.manifest.raw_json);
    const Json& declaration = selected_preset(manifest, preset);
    const Json* profile_id = declaration.find("profile_artifact");
    if (profile_id == nullptr || !profile_id->is_string())
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "preset does not declare profile_artifact: " + preset);
    const Json profile = Json::parse(read_text(artifact(model, profile_id->string()).path));
    if (!profile.find("run") || !profile.at("run").is_object())
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "execution profile is not prompt-runnable; publish a new immutable "
                     "package version with a generic run declaration");
    return profile;
}

std::optional<uint64_t> optional_u64(const Json* value) {
    if (value == nullptr || value->is_null()) return std::nullopt;
    const int64_t integer = value->integer();
    if (integer < 0) product_fail(ModelPackageErrorCode::PackageInvalid,
                                  "hardware byte declaration is negative");
    return static_cast<uint64_t>(integer);
}

struct PresetHardware {
    std::optional<uint64_t> minimum;
    std::optional<uint64_t> recommended;
    std::optional<int> minimum_compute_major;
    std::optional<int> minimum_compute_minor;
};

PresetHardware preset_hardware(const ResolvedRunnableModel& model,
                               const std::string& preset) {
    PresetHardware result;
    const Json manifest = Json::parse(model.manifest.raw_json);
    const Json* hardware = manifest.find("hardware");
    if (hardware == nullptr) return result;
    const Json* presets = hardware->find("presets");
    if (presets == nullptr) return result;
    const Json* selected = presets->find(preset);
    if (selected == nullptr) return result;
    result.minimum = optional_u64(selected->find("minimum_vram_bytes"));
    result.recommended = optional_u64(selected->find("recommended_vram_bytes"));
    if (const Json* capability = selected->find("minimum_compute_capability");
        capability != nullptr && !capability->is_null()) {
        if (capability->is_array() && capability->array().size() == 2) {
            result.minimum_compute_major = static_cast<int>(capability->array()[0].integer());
            result.minimum_compute_minor = static_cast<int>(capability->array()[1].integer());
        } else {
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "minimum_compute_capability must be null or [major,minor]");
        }
    }
    return result;
}

HardwareSnapshot hardware_snapshot() {
    HardwareSnapshot result;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
        product_fail(ModelPackageErrorCode::UnsupportedGpu,
                     "no CUDA-capable NVIDIA GPU is available");
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp properties {};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        product_fail(ModelPackageErrorCode::UnsupportedGpu,
                     "CUDA device properties are unavailable");
    result.gpu_name = properties.name;
    result.compute_major = properties.major;
    result.compute_minor = properties.minor;
    result.total_vram_bytes = properties.totalGlobalMem;
    size_t available = 0, total = 0;
    const cudaError_t memory_status = cudaMemGetInfo(&available, &total);
    if (memory_status != cudaSuccess)
        product_fail(ModelPackageErrorCode::UnsupportedGpu,
                     "cannot query currently available VRAM for selected GPU: " +
                         std::string(cudaGetErrorString(memory_status)));
    result.available_vram_bytes = available;
    result.total_vram_bytes = total;
    cudaDriverGetVersion(&result.driver_version);
    cudaRuntimeGetVersion(&result.runtime_version);
    return result;
}

std::vector<int32_t> i32_array(const Json* value) {
    std::vector<int32_t> result;
    if (value == nullptr) return result;
    for (const Json& item : value->array())
        result.push_back(static_cast<int32_t>(item.integer()));
    return result;
}

TokenizerSpec tokenizer_spec(const Json& value) {
    TokenizerSpec spec;
    const std::string format = value.at("format").string();
    if (format == "sentencepiece") spec.format = TokenizerAssetFormat::SentencePiece;
    else if (format == "huggingface_json")
        spec.format = TokenizerAssetFormat::HuggingFaceJson;
    else product_fail(ModelPackageErrorCode::PackageInvalid,
                      "unsupported tokenizer format: " + format);
    spec.max_length = value.at("max_length").integer();
    spec.pad_id = static_cast<int32_t>(value.at("pad_id").integer());
    spec.prefix_ids = i32_array(value.find("prefix_ids"));
    spec.suffix_ids = i32_array(value.find("suffix_ids"));
    if (const Json* side = value.find("padding_side")) {
        if (side->string() == "left") spec.padding_side = SequenceSide::Left;
        else if (side->string() != "right")
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "unsupported tokenizer padding_side");
    }
    if (const Json* side = value.find("truncation_side")) {
        if (side->string() == "left") spec.truncation_side = SequenceSide::Left;
        else if (side->string() != "right")
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "unsupported tokenizer truncation_side");
    }
    if (const Json* empty = value.find("empty_input")) {
        if (empty->string() == "all_padding") spec.empty_input = EmptyInputPolicy::AllPadding;
        else if (empty->string() != "encode_normally")
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "unsupported tokenizer empty-input policy");
    }
    if (const Json* suppress = value.find("suppress_metaspace_after_added_token"))
        spec.suppress_metaspace_after_added_token = suppress->boolean();
    if (const Json* added = value.find("added_tokens")) {
        for (const Json& item : added->array()) {
            spec.added_tokens.push_back({item.at("content").string(),
                static_cast<int32_t>(item.at("id").integer()),
                item.at("lstrip").boolean(), item.at("rstrip").boolean()});
        }
    }
    return spec;
}

Tensor ids_tensor(const TokenizedInput& input) {
    std::vector<int64_t> values(input.input_ids.begin(), input.input_ids.end());
    return host_i64({1, static_cast<int64_t>(values.size())}, values);
}

Tensor mask_tensor(const TokenizedInput& input) {
    return host_bool({1, static_cast<int64_t>(input.attention_mask.size())},
                     input.attention_mask);
}

Tensor host_mask_slice(const Tensor& input, int64_t start, int64_t stop) {
    require(input.device().is_host() && input.dtype() == DType::Bool && input.ndim() == 2 &&
                start >= 0 && stop >= start && stop <= input.dim(1),
            "Invalid host mask slice");
    const int64_t batch = input.dim(0), width = stop - start;
    std::vector<uint8_t> values(static_cast<size_t>(batch * width));
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t column = 0; column < width; ++column)
            values[static_cast<size_t>(row * width + column)] =
                input.data_as<uint8_t>()[row * input.dim(1) + start + column];
    return host_bool({batch, width}, values);
}

Tensor binding_value(const Json& binding, const ConditioningComponentResult& result,
                     const Tensor& ids, const Tensor& mask) {
    const std::string source = binding.at("source").string();
    if (source == "hidden_states") return result.hidden_states;
    if (source == "pooled_output") return result.pooled_output;
    if (source == "input_ids") return ids;
    if (source == "attention_mask") {
        const int64_t start = binding.find("slice_start")
            ? binding.at("slice_start").integer() : 0;
        const int64_t stop = binding.find("slice_stop")
            ? binding.at("slice_stop").integer() : mask.dim(1);
        return host_mask_slice(mask, start, stop);
    }
    product_fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported conditioning binding source: " + source);
}

float load_float(const Tensor& tensor, int64_t index) {
    if (tensor.dtype() == DType::F32) return tensor.data_as<float>()[index];
    if (tensor.dtype() == DType::BF16) {
        const uint32_t bits = static_cast<uint32_t>(tensor.data_as<uint16_t>()[index]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    product_fail(ModelPackageErrorCode::RuntimeError,
                 "finite check requires FP32 or BF16 tensor");
}

void require_finite(Backend& backend, const Tensor& tensor, const std::string& name) {
    if (!tensor.defined()) return;
    Tensor host = tensor.device().is_host() ? tensor : backend.copy_to_host(tensor);
    if (host.dtype() != DType::F32 && host.dtype() != DType::BF16) return;
    for (int64_t index = 0; index < host.numel(); ++index) {
        if (!std::isfinite(load_float(host, index)))
            product_fail(ModelPackageErrorCode::RuntimeError,
                         name + " contains NaN or Inf");
    }
}

std::string source_text(const RunOptions& options, const Json& run,
                        const Json& component) {
    const std::string source = component.at("text_source").string();
    std::string text;
    if (source == "prompt") text = options.prompt;
    else if (source == "negative_prompt") {
        const Json* negative = run.find("negative_prompt");
        if (negative == nullptr)
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "negative_prompt source has no declared default");
        text = negative->string();
    } else {
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "unsupported component text source: " + source);
    }
    if (const Json* trim = component.find("trim"); trim && trim->boolean()) {
        const size_t begin = text.find_first_not_of(" \t\r\n");
        const size_t end = text.find_last_not_of(" \t\r\n");
        text = begin == std::string::npos ? "" : text.substr(begin, end - begin + 1);
    }
    const std::string prefix = component.find("prefix") ? component.at("prefix").string() : "";
    const std::string suffix = component.find("suffix") ? component.at("suffix").string() : "";
    return prefix + text + suffix;
}

int64_t input_i64(const Json& inputs, const std::string& name) {
    return inputs.at(name).integer();
}

double input_number(const Json& inputs, const std::string& name) {
    return inputs.at(name).number();
}

void add_runtime_inputs(TensorBundle& input, const Json& run,
                        const Json& inputs, uint64_t seed) {
    input.emplace("seed", scalar_i64(static_cast<int64_t>(seed)));
    for (const Json& item : run.at("runtime_inputs").array()) {
        const std::string target = item.at("target").string();
        const std::string dtype = item.at("dtype").string();
        const std::string source = item.at("source").string();
        if (source == "seed") {
            const int64_t offset = item.find("offset") ? item.at("offset").integer() : 0;
            input.emplace(target, scalar_i64(static_cast<int64_t>(seed) + offset));
        } else if (source.rfind("inputs.", 0) == 0) {
            const std::string name = source.substr(7);
            if (dtype == "i64") input.emplace(target, scalar_i64(input_i64(inputs, name)));
            else if (dtype == "f32")
                input.emplace(target, scalar_f32(static_cast<float>(input_number(inputs, name))));
            else product_fail(ModelPackageErrorCode::PackageInvalid,
                              "unsupported runtime input dtype: " + dtype);
        } else if (source == "literal") {
            std::vector<int64_t> shape;
            for (const Json& dimension : item.at("shape").array())
                shape.push_back(dimension.integer());
            if (dtype == "i64") {
                std::vector<int64_t> values;
                for (const Json& value : item.at("values").array())
                    values.push_back(value.integer());
                input.emplace(target, host_i64(shape, values));
            } else if (dtype == "f32") {
                std::vector<float> values;
                for (const Json& value : item.at("values").array())
                    values.push_back(static_cast<float>(value.number()));
                input.emplace(target, host_f32(shape, values));
            } else product_fail(ModelPackageErrorCode::PackageInvalid,
                                "unsupported literal input dtype: " + dtype);
        } else product_fail(ModelPackageErrorCode::PackageInvalid,
                            "unsupported runtime input source: " + source);
    }
    if (const Json* derived = run.find("derived_inputs")) {
        for (const Json& item : derived->array()) {
            const std::string kind = item.at("kind").string();
            if (kind != "grid_coordinates_thw")
                product_fail(ModelPackageErrorCode::PackageInvalid,
                             "unsupported derived input kind: " + kind);
            const std::string grid_source = item.at("grid_source").string();
            const Tensor& grid = input.at(grid_source);
            require(grid.dtype() == DType::I64 && grid.shape() == std::vector<int64_t>({3}),
                    "grid coordinate source must be i64[3]");
            const int64_t t = grid.data_as<int64_t>()[0];
            const int64_t h = grid.data_as<int64_t>()[1];
            const int64_t w = grid.data_as<int64_t>()[2];
            require(t > 0 && h > 0 && w > 0, "grid dimensions must be positive");
            std::vector<float> coordinates;
            coordinates.reserve(static_cast<size_t>(3 * t * h * w));
            for (int axis = 0; axis < 3; ++axis) {
                for (int64_t ti = 0; ti < t; ++ti)
                    for (int64_t hi = 0; hi < h; ++hi)
                        for (int64_t wi = 0; wi < w; ++wi)
                            coordinates.push_back(static_cast<float>(
                                axis == 0 ? ti : axis == 1 ? hi : wi));
            }
            input.emplace(item.at("target").string(),
                          host_f32({1, 3, t * h * w}, coordinates));
        }
    }
}

std::filesystem::path product_encoder_path() {
    std::array<char, 4096> executable {};
    const ssize_t length = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length <= 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot locate product executable");
    executable[static_cast<size_t>(length)] = '\0';
    return std::filesystem::path(executable.data()).parent_path() / "vrhino-ffmpeg";
}

uint8_t video_u8(const Tensor& video, int64_t index, float& minimum, float& maximum,
                 int64_t& nan_count, int64_t& inf_count,
                 const float declared_minimum, const float declared_maximum) {
    const float value = load_float(video, index);
    if (std::isnan(value)) ++nan_count;
    if (std::isinf(value)) ++inf_count;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    if (!std::isfinite(value)) return 0;
    const float normalized = (value - declared_minimum) /
                             (declared_maximum - declared_minimum);
    return static_cast<uint8_t>(std::lround(std::clamp(normalized, 0.0f, 1.0f) * 255.0f));
}

void write_all(int fd, const uint8_t* data, size_t bytes) {
    while (bytes > 0) {
        const ssize_t written = write(fd, data, bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0)
            product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                         "video encoder input pipe failed");
        data += written;
        bytes -= static_cast<size_t>(written);
    }
}

struct EncodeResult {
    double seconds = 0.0;
    uint64_t bytes = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    int64_t nan_count = 0;
    int64_t inf_count = 0;
};

EncodeResult encode_mp4(const Tensor& video, int64_t fps,
                        const std::filesystem::path& encoder,
                        const std::filesystem::path& output, bool overwrite,
                        const std::function<bool()>& cancelled,
                        const float declared_minimum, const float declared_maximum) {
    if (!video.device().is_host() || (video.dtype() != DType::F32 && video.dtype() != DType::BF16) ||
        video.shape().size() != 5 || video.dim(0) != 1 || video.dim(1) != 3)
        product_fail(ModelPackageErrorCode::RuntimeError,
                     "video output must be host FP32/BF16 BCTHW with batch=1 and RGB channels");
    if (fps <= 0) product_fail(ModelPackageErrorCode::PackageInvalid, "fps must be positive");
    if (!std::isfinite(declared_minimum) || !std::isfinite(declared_maximum) ||
        declared_minimum >= declared_maximum)
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "video output range must contain two increasing finite values");
    if (std::filesystem::exists(output) && !overwrite)
        product_fail(ModelPackageErrorCode::OutputExists,
                     "output already exists; pass --overwrite: " + output.string());
    if (!std::filesystem::exists(encoder))
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled encoder is missing: " + encoder.string());
    std::filesystem::path partial = output;
    partial += ".partial";
    std::error_code remove_error;
    std::filesystem::remove(partial, remove_error);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());

    int pipe_fds[2];
    if (pipe(pipe_fds) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot create video encoder pipe");
    const std::string size = std::to_string(video.dim(4)) + "x" +
                             std::to_string(video.dim(3));
    const std::string rate = std::to_string(fps);
    const pid_t child = fork();
    if (child < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot start bundled video encoder");
    }
    if (child == 0) {
        dup2(pipe_fds[0], STDIN_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        execl(encoder.c_str(), encoder.c_str(), "-hide_banner", "-loglevel", "error",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", size.c_str(),
              "-r", rate.c_str(), "-i", "pipe:0", "-an", "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
              partial.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(pipe_fds[0]);
    const auto started = Clock::now();
    EncodeResult result;
    const int64_t frames = video.dim(2), height = video.dim(3), width = video.dim(4);
    std::vector<uint8_t> frame(static_cast<size_t>(height * width * 3));
    bool interrupted = false;
    signal(SIGPIPE, SIG_IGN);
    for (int64_t t = 0; t < frames; ++t) {
        if (cancelled && cancelled()) { interrupted = true; break; }
        size_t destination = 0;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t source = c * frames * height * width +
                                           t * height * width + y * width + x;
                    frame[destination++] = video_u8(video, source, result.minimum,
                        result.maximum, result.nan_count, result.inf_count,
                        declared_minimum, declared_maximum);
                }
            }
        }
        if (!interrupted) write_all(pipe_fds[1], frame.data(), frame.size());
    }
    close(pipe_fds[1]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    if (interrupted) {
        std::filesystem::remove(partial, remove_error);
        product_fail(ModelPackageErrorCode::Cancelled, "run cancelled during video encoding");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        result.nan_count != 0 || result.inf_count != 0 ||
        result.minimum < declared_minimum - 1.0e-6f ||
        result.maximum > declared_maximum + 1.0e-6f) {
        std::filesystem::remove(partial, remove_error);
        if (result.nan_count != 0 || result.inf_count != 0)
            product_fail(ModelPackageErrorCode::RuntimeError,
                         "video output contains NaN or Inf");
        if (result.minimum < declared_minimum - 1.0e-6f ||
            result.maximum > declared_maximum + 1.0e-6f)
            product_fail(ModelPackageErrorCode::RuntimeError,
                         "video output violates its declared range");
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled FFmpeg failed to encode MP4");
    }
    // POSIX rename replaces an existing regular file atomically. Do not unlink
    // first: --overwrite must never create a window in which the old completed
    // output is absent before the new completed output is published.
    std::filesystem::rename(partial, output);
    result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    result.bytes = std::filesystem::file_size(output);
    return result;
}

const ComponentDeclaration& package_component(const ResolvedRunnableModel& model,
                                              const std::string& id) {
    const auto found = std::find_if(model.manifest.components.begin(),
        model.manifest.components.end(), [&](const ComponentDeclaration& item) {
            return item.id == id;
        });
    if (found == model.manifest.components.end())
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "run profile references undeclared package component: " + id);
    return *found;
}

void require_component_artifact(const ComponentDeclaration& component,
                                const std::string& artifact_id) {
    if (std::find(component.artifact_ids.begin(), component.artifact_ids.end(), artifact_id) ==
        component.artifact_ids.end())
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "component " + component.id + " does not declare artifact " + artifact_id);
}

}  // namespace

HardwareSnapshot query_hardware() { return hardware_snapshot(); }

PreflightResult preflight_runnable_model(const ResolvedRunnableModel& model,
                                         const std::string& requested_preset) {
    return preflight_runnable_model(model, requested_preset, hardware_snapshot());
}

PreflightResult preflight_runnable_model(const ResolvedRunnableModel& model,
                                         const std::string& requested_preset,
                                         const HardwareSnapshot& hardware) {
    const std::string preset = selected_preset_name(model, requested_preset);
    const PresetHardware declared = preset_hardware(model, preset);
    PreflightResult result;
    result.hardware = hardware;
    result.minimum_vram_bytes = declared.minimum;
    result.recommended_vram_bytes = declared.recommended;
    if (declared.minimum_compute_major &&
        (result.hardware.compute_major < *declared.minimum_compute_major ||
         (result.hardware.compute_major == *declared.minimum_compute_major &&
          result.hardware.compute_minor < *declared.minimum_compute_minor))) {
        result.status = PreflightStatus::UnsupportedGpu;
        result.message = "GPU compute capability is below the preset minimum";
    } else if (declared.minimum &&
               result.hardware.available_vram_bytes < *declared.minimum) {
        result.status = PreflightStatus::InsufficientVram;
        result.message = "model " + model.manifest.identity.reference() + " preset " +
            preset + " requires " + gibibytes(*declared.minimum) +
            " available VRAM; selected GPU has " +
            gibibytes(result.hardware.available_vram_bytes) + " available of " +
            gibibytes(result.hardware.total_vram_bytes) + " total";
    } else if (!declared.minimum ||
               (declared.recommended && result.hardware.total_vram_bytes < *declared.recommended)) {
        result.status = PreflightStatus::SupportedWithWarning;
        result.message = !declared.minimum
            ? "minimum VRAM is unknown; admission uses qualification metadata"
            : "GPU VRAM is below the recommended amount";
    } else {
        result.status = PreflightStatus::Supported;
        result.message = "hardware satisfies declared preset admission";
    }
    return result;
}

void preflight_output_destination(const std::filesystem::path& output,
                                  const bool overwrite) {
    check_output_destination(output, overwrite);
}

void preflight_media_encoder(const std::filesystem::path& encoder) {
    check_media_encoder(encoder);
}

std::filesystem::path default_media_encoder_path() {
    return product_encoder_path();
}

RunResult run_runnable_model(const ResolvedRunnableModel& model,
                             const RunOptions& options, RunEventSink progress) {
    if (model.manifest.product.family == "lip_sync" &&
        model.manifest.product.workflow_identity ==
            "lip_sync_diffusion_workflow_v1")
        return run_lip_sync_diffusion_product(model, options, std::move(progress));
    if (model.manifest.product.family == "lip_sync")
        return run_lip_sync_product(model, options, std::move(progress));
    if (model.manifest.product.family != "text_to_video")
        product_fail(ModelPackageErrorCode::PackageVersionUnsupported,
                     "unsupported product family: " + model.manifest.product.family);
    const auto process_started = Clock::now();
    const ProductInputSchema* product_schema =
        model.manifest.product.input_schema
            ? &*model.manifest.product.input_schema : nullptr;
    const std::string output = resolve_product_output(
        product_schema, options.output.string());
    preflight_output_destination(output, options.overwrite);
    std::filesystem::path encoder = options.encoder_path;
    if (encoder.empty()) {
        if (const char* development_encoder = std::getenv("VRHINO_FFMPEG"))
            encoder = development_encoder;
        else encoder = default_media_encoder_path();
    }
    preflight_media_encoder(encoder);
    if (options.prompt.empty())
        product_fail(ModelPackageErrorCode::InvalidInput,
                     "text-to-video product requires --prompt");
    const std::string preset = selected_preset_name(model, options.preset);
    const Json profile = load_run_profile(model, preset);
    const Json& run = profile.at("run");
    const Json& inputs = profile.at("inputs");
    if (product_schema != nullptr)
        validate_product_execution_consistency(
            model.manifest.product.family,
            model.manifest.product.workflow_identity,
            *product_schema,
            *model.manifest.product.frozen_profile,
            profile);
    const uint64_t seed = resolve_product_seed(
        product_schema, options.seed,
        static_cast<uint64_t>(run.at("default_seed").integer()));
    PreflightResult admission = preflight_runnable_model(model, preset);
    if (admission.status == PreflightStatus::InsufficientVram)
        product_fail(ModelPackageErrorCode::InsufficientVram, admission.message);
    if (admission.status == PreflightStatus::UnsupportedGpu)
        product_fail(ModelPackageErrorCode::UnsupportedGpu, admission.message);
    if (admission.status == PreflightStatus::DriverIncompatible)
        product_fail(ModelPackageErrorCode::DriverIncompatible, admission.message);
    if (progress && admission.status == PreflightStatus::SupportedWithWarning)
        progress(RunEvent::diagnostic(
            "Hardware warning: " + admission.message, RunStage::Admission));
    if (progress)
        progress(RunEvent::stage_changed(RunStage::Loading, "Loading model"));

    try {
        VrmModel runtime_model(model.runtime_model_path);
        const std::string execution_dtype = run.at("execution_dtype").string();
        const DType dtype = execution_dtype == "bfloat16" ? DType::BF16 :
                            execution_dtype == "float32" ? DType::F32 : DType::F16;
        if (dtype == DType::F16)
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "run execution_dtype must be bfloat16 or float32");
        const Json& memory = run.at("memory_runtime");
        const uint64_t device_budget = static_cast<uint64_t>(memory.at("device_budget_bytes").integer());
        MemoryBudget budget{device_budget,
            static_cast<size_t>(memory.at("activation_reserve_bytes").integer()),
            static_cast<size_t>(memory.at("host_budget_bytes").integer()),
            static_cast<size_t>(memory.at("host_staging_bytes").integer()),
            static_cast<size_t>(memory.at("safety_margin_bytes").integer())};
        MemoryRuntimeOptions memory_options;
        memory_options.enabled = true;
        memory_options.host_staging = false;
        memory_options.prefetch = false;
        auto configure_backend = [&](CudaBackend& target) {
            target.set_execution_dtype(dtype);
            target.configure_memory_runtime(budget, memory_options);
        };

        TensorBundle runtime_input;
        add_runtime_inputs(runtime_input, run, inputs, seed);
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Conditioning, "Encoding prompt"));
        const auto conditioning_started = Clock::now();
        size_t conditioning_peak_device_bytes = 0;
        {
            // Conditioning is a completed package component phase. Its large
            // text-encoder cache must not remain resident during denoiser/VAE.
            CudaBackend conditioning_backend;
            configure_backend(conditioning_backend);
            std::vector<SafeTensorAsset> weight_assets;
            std::map<std::string, size_t> weight_asset_indices;
            size_t conditioning_mapped_bytes = 0;
            for (const Json& component : run.at("components").array()) {
                const std::string component_id = component.at("component_id").string();
                const ComponentDeclaration& declaration = package_component(model, component_id);
                const std::string graph_id = component.at("graph_artifact").string();
                const std::string tokenizer_id = component.at("tokenizer_artifact").string();
                const std::string index_id = component.at("weights_index_artifact").string();
                require_component_artifact(declaration, graph_id);
                require_component_artifact(declaration, tokenizer_id);
                require_component_artifact(declaration, index_id);
                size_t asset_index;
                const auto existing = weight_asset_indices.find(index_id);
                if (existing == weight_asset_indices.end()) {
                    std::map<std::string, std::filesystem::path> shards;
                    for (const auto& [logical_name, value] : component.at("weight_shards").object()) {
                        const std::string shard_id = value.string();
                        require_component_artifact(declaration, shard_id);
                        shards.emplace(logical_name, artifact(model, shard_id).path);
                    }
                    asset_index = weight_assets.size();
                    weight_assets.push_back(SafeTensorAsset::indexed(
                        artifact(model, index_id).path, shards));
                    conditioning_mapped_bytes += weight_assets.back().mapped_bytes();
                    weight_asset_indices.emplace(index_id, asset_index);
                } else asset_index = existing->second;
                NativeTokenizer tokenizer(tokenizer_spec(component.at("tokenizer_spec")),
                                          read_text(artifact(model, tokenizer_id).path));
                const TokenizedInput tokenized = tokenizer.encode(
                    source_text(options, run, component));
                const Tensor ids = ids_tensor(tokenized);
                const Tensor mask = mask_tensor(tokenized);
                ConditioningComponentExecutor executor(
                    conditioning_backend, weight_assets[asset_index].weights());
                const ConditioningComponentResult result = executor.execute(
                    Json::parse(read_text(artifact(model, graph_id).path)), ids, mask);
                require_finite(conditioning_backend, result.hidden_states,
                               component_id + ".hidden_states");
                require_finite(conditioning_backend, result.pooled_output,
                               component_id + ".pooled_output");
                for (const Json& binding : component.at("bindings").array()) {
                    const std::string target = binding.at("target").string();
                    Tensor value = binding_value(binding, result, ids, mask);
                    require(value.defined(), "Undefined conditioning binding: " + target);
                    if (!value.device().is_host())
                        value = conditioning_backend.copy_to_host(value);
                    require(runtime_input.emplace(target, value).second,
                            "Duplicate conditioning target: " + target);
                }
            }
            conditioning_backend.synchronize();
            conditioning_backend.set_vrm_mapped_bytes(conditioning_mapped_bytes);
            conditioning_peak_device_bytes = conditioning_backend.peak_device_bytes();
        }
        const double conditioning_seconds = std::chrono::duration<double>(
            Clock::now() - conditioning_started).count();

        CudaBackend backend;
        configure_backend(backend);
        backend.set_vrm_mapped_bytes(runtime_model.file_size());

        const PrecisionPolicy policy = PrecisionPolicy::from_json(
            Json::parse(read_text(artifact(model, run.at("precision_policy_artifact").string()).path)));
        require(policy.requested_dtype() == dtype,
                "Precision policy requested dtype does not match execution dtype");
        ComponentExecutionConfig component_execution;
        if (const Json* descriptor = run.find("component_execution"))
            component_execution = component_execution_config_from_json(*descriptor);
        auto architecture = create_architecture(runtime_model);
        NativeRuntime runtime(backend, policy, std::move(component_execution));
        runtime.set_sampling_step_observer([&](int step, int total) {
            if (options.cancellation_requested && options.cancellation_requested())
                product_fail(ModelPackageErrorCode::Cancelled,
                             "run cancelled after sampling step " + std::to_string(step));
            if (progress)
                progress(RunEvent::progress(
                    RunStage::Sampling, static_cast<uint64_t>(step),
                    static_cast<uint64_t>(total), RunProgressUnit::Step,
                    "Sampling " + std::to_string(step) + "/" +
                        std::to_string(total)));
            if (progress && step == total)
                progress(RunEvent::stage_changed(
                    RunStage::Decoding, "Decoding video"));
        });
        RuntimeResult execution = runtime.execute(*architecture, runtime_input);
        if (options.cancellation_requested && options.cancellation_requested())
            product_fail(ModelPackageErrorCode::Cancelled,
                         "run cancelled after Native execution");
        const Tensor& video = execution.outputs.at("video");
        const Json& output_contract = run.at("output");
        if (output_contract.at("layout").string() != "BCTHW" ||
            output_contract.at("range").array().size() != 2)
            product_fail(ModelPackageErrorCode::PackageInvalid,
                         "product MP4 path requires declared BCTHW output with a range");
        const float output_minimum = static_cast<float>(
            output_contract.at("range").array()[0].number());
        const float output_maximum = static_cast<float>(
            output_contract.at("range").array()[1].number());
        if (progress)
            progress(RunEvent::stage_changed(
                RunStage::Encoding, "Encoding MP4"));
        const EncodeResult encoded = encode_mp4(video, input_i64(inputs, "fps"), encoder,
            output, options.overwrite, options.cancellation_requested,
            output_minimum, output_maximum);
        if (progress)
            progress(RunEvent::stage_changed(RunStage::Finalizing, "Done"));
        RunResult result;
        result.identity = model.manifest.identity;
        result.preset = preset;
        result.seed = seed;
        result.width = input_i64(inputs, "width");
        result.height = input_i64(inputs, "height");
        result.frames = video.dim(2);
        result.fps = input_i64(inputs, "fps");
        result.conditioning_seconds = conditioning_seconds;
        result.sampling_seconds = execution.sampling_seconds;
        result.decode_seconds = execution.decode_seconds;
        result.encoding_seconds = encoded.seconds;
        result.total_seconds = std::chrono::duration<double>(Clock::now() - process_started).count();
        result.peak_device_bytes = std::max<uint64_t>(
            conditioning_peak_device_bytes, execution.peak_device_bytes);
        result.output_bytes = encoded.bytes;
        result.video_minimum = encoded.minimum;
        result.video_maximum = encoded.maximum;
        result.nan_count = encoded.nan_count;
        result.inf_count = encoded.inf_count;
        result.preflight = admission;
        return result;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        if (options.cancellation_requested && options.cancellation_requested())
            product_fail(ModelPackageErrorCode::Cancelled, "run cancelled");
        const std::string message = error.what();
        const bool oom = message.find("out of memory") != std::string::npos ||
                         message.find("Out of memory") != std::string::npos ||
                         message.find("cudaErrorMemoryAllocation") != std::string::npos;
        product_fail(oom ? ModelPackageErrorCode::OutOfMemory
                         : ModelPackageErrorCode::RuntimeError,
                     message);
    }
}

}  // namespace vrhino::product
