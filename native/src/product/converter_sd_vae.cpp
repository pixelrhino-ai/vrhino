#include "vrhino/product/converter.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr const char* kRevision =
    "31f26fdeee1355a5c34592e401dd41e45d25a493";
constexpr uint64_t kCheckpointBytes = 334707217ULL;
constexpr uint64_t kRawTensorBytes = 334615452ULL;
constexpr const char* kCheckpointSha256 =
    "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc";

[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> output;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(separator, begin);
        output.push_back(value.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin));
        if (end == std::string::npos) return output;
        begin = end + 1;
    }
}

std::vector<int64_t> parse_shape(const std::string& value) {
    std::vector<int64_t> shape;
    for (const std::string& field : split(value, ',')) {
        size_t consumed = 0;
        int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid AutoencoderKL tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid AutoencoderKL tensor dimension");
        shape.push_back(dimension);
    }
    return shape;
}

uint64_t tensor_bytes(const std::vector<int64_t>& shape) {
    uint64_t elements = 1;
    for (int64_t dimension : shape) {
        if (dimension != 0 && elements >
            std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "AutoencoderKL tensor shape overflow");
        elements *= static_cast<uint64_t>(dimension);
    }
    return elements * sizeof(float);
}

struct MappingSet {
    std::map<std::string, SourceTensorDescriptor> tensors;
    std::vector<TensorMapping> mappings;
};

MappingSet load_map(const fs::path& path) {
    std::ifstream input(path);
    if (!input) fail(ModelPackageErrorCode::ArtifactMissing,
                     "missing AutoencoderKL tensor map");
    std::string line;
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    if (!std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid AutoencoderKL tensor-map header");
    MappingSet result;
    uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "diffusion_pytorch_model.bin" ||
            fields[2] != "F32" || fields[5] != "identity_bytes" ||
            fields[7] != "F32" || fields[9] != "autoencoder_kl" ||
            fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported AutoencoderKL tensor-map row");
        const auto source_shape = parse_shape(fields[3]);
        const auto destination_shape = parse_shape(fields[8]);
        if (source_shape != destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "AutoencoderKL tensor-map shape transformation is forbidden");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid AutoencoderKL tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid AutoencoderKL tensor offset");
        const uint64_t bytes = tensor_bytes(source_shape);
        raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[1], DType::F32, "F32",
            source_shape, 0, offset, bytes};
        if (!result.tensors.emplace(fields[1], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate AutoencoderKL source tensor");
        result.mappings.push_back(TensorMapping{fields[1], DType::F32,
            source_shape, fields[5], fields[6], DType::F32,
            destination_shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != 248 || raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "AutoencoderKL frozen tensor inventory drift");
    return result;
}

Json string_value(const std::string& value) { return Json(Json::Value(value)); }
Json integer_value(int64_t value) { return Json(Json::Value(value)); }
Json number_value(double value) { return Json(Json::Value(value)); }
Json object_value(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) result.emplace(value.first, value.second);
    return Json(Json::Value(std::move(result)));
}

Json graph(const std::vector<TensorMapping>& mappings) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.source_name, string_value(mapping.destination_name));
    Json::Array channels;
    for (int value : {128, 256, 512, 512})
        channels.emplace_back(Json::Value(static_cast<int64_t>(value)));
    Json::Array entries{string_value("encode"), string_value("decode")};
    Json::Array primitives;
    for (const char* value : {"conv2d", "group_norm", "activation", "pad",
             "interpolate_nearest", "attention", "linear", "reshape",
             "permute", "split", "add", "mul", "div", "clamp", "exp",
             "rng_normal"})
        primitives.push_back(string_value(value));
    return object_value({
        {"schema_version", integer_value(1)},
        {"kind", string_value("autoencoder_kl")},
        {"entry_points", Json(Json::Value(std::move(entries)))},
        {"config", object_value({
            {"dtype", string_value("float32")},
            {"input_channels", integer_value(3)},
            {"latent_channels", integer_value(4)},
            {"sample_size", integer_value(256)},
            {"block_out_channels", Json(Json::Value(std::move(channels)))},
            {"layers_per_block", integer_value(2)},
            {"decoder_layers_per_block", integer_value(3)},
            {"norm_num_groups", integer_value(32)},
            {"norm_epsilon", number_value(1.0e-6)},
            {"mid_block_attention", Json(Json::Value(true))},
            {"downsample", string_value("constant_pad_right_bottom_then_conv2d")},
            {"upsample", string_value("nearest_then_conv2d")},
            {"posterior_logvar_clamp_min", number_value(-30.0)},
            {"posterior_logvar_clamp_max", number_value(20.0)},
            {"scaling_factor", number_value(0.18215)},
        })},
        {"semantic_outputs", Json(Json::Value(Json::Array{
            string_value("posterior_mean"), string_value("posterior_logvar"),
            string_value("sampled_latent"), string_value("scaled_latent"),
            string_value("decoded"), string_value("rgb_0_1")}))},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))},
    });
}

Json metadata() {
    return object_value({
        {"architecture", string_value("autoencoder-kl")},
        {"profile", string_value("component")},
        {"default_dtype", string_value("float32")},
        {"component", object_value({
            {"kind", string_value("autoencoder_kl")},
            {"implementation", string_value("generic.autoencoder_kl_2d.v1")},
            {"source_repository", string_value("stabilityai/sd-vae-ft-mse")},
            {"source_revision", string_value(kRevision)},
            {"license", string_value("creativeml-openrail-m")},
            {"checkpoint_bytes", integer_value(kCheckpointBytes)},
            {"checkpoint_sha256", string_value(kCheckpointSha256)},
            {"tensor_count", integer_value(248)},
            {"raw_tensor_bytes", integer_value(kRawTensorBytes)},
        })},
    });
}

fs::path checked_source(const fs::path& root, const char* name,
                        uint64_t bytes, const char* sha,
                        const WorkProgressCallback& progress) {
    const fs::path path = root / name;
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen AutoencoderKL source: " + std::string(name));
    if (fs::file_size(path, error) != bytes || error ||
        sha256_file(path, progress) != sha)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen AutoencoderKL source identity mismatch: " +
                 std::string(name));
    return path;
}

}  // namespace

FrozenComponentConversionResult convert_sd_vae_ft_mse_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "AutoencoderKL conversion interrupted; source retained");
    (void)checked_source(source_directory, "config.json", 547ULL,
        "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
        progress);
    const fs::path checkpoint = checked_source(source_directory,
        "diffusion_pytorch_model.bin", kCheckpointBytes, kCheckpointSha256,
        progress);
    MappingSet mapping = load_map(tensor_map);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));
    std::error_code error;
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "AutoencoderKL component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create AutoencoderKL output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "autoencoder-kl", metadata(), graph(mapping.mappings), source,
            mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish AutoencoderKL component");
        return {output, kCheckpointBytes, kRawTensorBytes, 248,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
