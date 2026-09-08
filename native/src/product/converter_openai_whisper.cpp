#include "vrhino/product/converter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unistd.h>

#include "vrhino/product/pytorch_zip.h"

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr const char* kCodeRevision =
    "a229c3948406bc2cf6eaf4873e662e70c6a04746";
constexpr const char* kModelRevision =
    "c42c7e6c8e9c213626389fa7d9a3c444b8536353";
constexpr uint64_t kCheckpointBytes = 75572083ULL;
constexpr uint64_t kCheckpointTensorBytes = 75521280ULL;
constexpr uint64_t kRetainedSourceBytes = 16416768ULL;
constexpr uint64_t kRetainedOutputBytes = 32833536ULL;
constexpr uint64_t kTensorCount = 67;
constexpr uint64_t kStorageCount = 167;
constexpr uint64_t kArchiveEntryCount = 169;
constexpr const char* kCheckpointSha256 =
    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9";
constexpr const char* kTensorMapSha256 =
    "c0c872e35d42d6c085483121940f6609fd8e372133a69fffe3a1d1db4b983b6d";

[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    for (;;) {
        const size_t end = value.find(separator, begin);
        result.push_back(value.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin));
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

std::vector<int64_t> parse_shape(const std::string& text) {
    std::vector<int64_t> shape;
    for (const std::string& field : split(text, ',')) {
        size_t consumed = 0;
        int64_t value = 0;
        try { value = std::stoll(field, &consumed); }
        catch (...) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid OpenAI Whisper tensor shape");
        }
        if (consumed != field.size() || value <= 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid OpenAI Whisper tensor dimension");
        shape.push_back(value);
    }
    if (shape.empty())
        fail(ModelPackageErrorCode::PackageInvalid,
             "empty OpenAI Whisper tensor shape");
    return shape;
}

uint64_t elements(const std::vector<int64_t>& shape) {
    uint64_t result = 1;
    for (int64_t dimension : shape) {
        if (result > std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "OpenAI Whisper tensor shape overflow");
        result *= static_cast<uint64_t>(dimension);
    }
    return result;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fU;
    const uint32_t fraction = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            uint32_t normalized = fraction;
            int shift = 0;
            while ((normalized & 0x0400U) == 0) {
                normalized <<= 1;
                ++shift;
            }
            normalized &= 0x03ffU;
            bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 |
                   normalized << 13;
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | fraction << 13;
    } else {
        bits = sign | (exponent + (127 - 15)) << 23 | fraction << 13;
    }
    float output = 0.0f;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

struct AuditMapping {
    std::string source_name;
    std::string storage_name;
    std::vector<int64_t> shape;
    std::string destination_name;
    std::string logical_name;
    bool transpose_2d = false;
};

std::string logical_name(const std::string& destination) {
    constexpr const char* prefix = "audio_encoder.";
    if (!destination.starts_with(prefix))
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper destination prefix drift");
    return "encoder." + destination.substr(
        std::char_traits<char>::length(prefix));
}

std::vector<AuditMapping> load_map(const fs::path& path,
                                   const PytorchZipArchive& archive,
                                   const RestrictedPickleInventory& pickle) {
    std::ifstream input(path);
    if (!input)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing OpenAI Whisper tensor map");
    const std::string header =
        "source_name\tstorage_name\tsource_dtype\tsource_shape\tstorage_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid OpenAI Whisper tensor-map header");
    std::vector<AuditMapping> result;
    std::set<std::string> storages;
    std::set<std::string> sources;
    std::set<std::string> destinations;
    uint64_t input_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        const bool transpose_2d = fields.size() == 11 &&
            fields[5] == "fp16_transpose_2d_to_fp32";
        if (fields.size() != 11 || fields[2] != "F16" || fields[4] != "0" ||
            (fields[5] != "fp16_to_fp32" && !transpose_2d) ||
            fields[7] != "F32" ||
            fields[8] != fields[3] ||
            fields[9] != "audio_encoder_transformer" || fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported OpenAI Whisper tensor-map row");
        const std::vector<int64_t> shape = parse_shape(fields[3]);
        const uint64_t bytes = elements(shape) * sizeof(uint16_t);
        const PytorchZipEntry& entry = archive.at("archive/data/" + fields[1]);
        if (entry.byte_length != bytes || !storages.insert(fields[1]).second ||
            !sources.insert(fields[0]).second ||
            !destinations.insert(fields[6]).second ||
            pickle.unicode_strings.count(fields[0]) != 1 ||
            pickle.unicode_strings.count(fields[1]) != 1)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "OpenAI Whisper tensor/storage inventory mismatch");
        input_bytes += bytes;
        if (transpose_2d && shape.size() != 2)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "OpenAI Whisper transposed storage must be rank two");
        result.push_back({fields[0], fields[1], shape, fields[6],
                          logical_name(fields[6]), transpose_2d});
    }
    if (result.size() != kTensorCount || input_bytes != kRetainedSourceBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper retained tensor inventory drift");
    return result;
}

class ConvertedHalfSource final : public TensorSource {
public:
    ConvertedHalfSource(const PytorchZipArchive& archive,
                        const std::vector<AuditMapping>& mappings) {
        for (const AuditMapping& mapping : mappings) {
            const std::vector<uint8_t> bytes = archive.read(
                "archive/data/" + mapping.storage_name,
                elements(mapping.shape) * sizeof(uint16_t));
            std::vector<float> values(bytes.size() / sizeof(uint16_t));
            const auto read_half = [&](size_t index) {
                const uint16_t half = static_cast<uint16_t>(bytes[2 * index]) |
                    static_cast<uint16_t>(bytes[2 * index + 1]) << 8;
                return half_to_float(half);
            };
            if (mapping.transpose_2d) {
                const size_t rows = static_cast<size_t>(mapping.shape[0]);
                const size_t columns = static_cast<size_t>(mapping.shape[1]);
                for (size_t row = 0; row < rows; ++row)
                    for (size_t column = 0; column < columns; ++column)
                        values[row * columns + column] =
                            read_half(column * rows + row);
            } else {
                for (size_t index = 0; index < values.size(); ++index)
                    values[index] = read_half(index);
            }
            const uint64_t output_bytes = values.size() * sizeof(float);
            descriptors_.emplace(mapping.source_name, SourceTensorDescriptor{
                mapping.source_name, DType::F32, "F32", mapping.shape, 0, 0,
                output_bytes});
            values_.emplace(mapping.source_name, std::move(values));
        }
    }

    const std::map<std::string, SourceTensorDescriptor>& tensors()
            const noexcept override {
        return descriptors_;
    }

    void read_tensor(const SourceTensorDescriptor& tensor,
                     uint64_t relative_offset, void* destination,
                     size_t bytes) const override {
        const auto found = values_.find(tensor.name);
        if (found == values_.end() || relative_offset > tensor.byte_length ||
            bytes > tensor.byte_length - relative_offset) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "OpenAI Whisper converted tensor range is invalid");
        }
        std::memcpy(destination,
                    reinterpret_cast<const uint8_t*>(found->second.data()) +
                        relative_offset,
                    bytes);
    }

private:
    std::map<std::string, SourceTensorDescriptor> descriptors_;
    std::map<std::string, std::vector<float>> values_;
};

Json string_value(const std::string& value) { return Json(Json::Value(value)); }
Json integer_value(int64_t value) { return Json(Json::Value(value)); }
Json number_value(double value) { return Json(Json::Value(value)); }
Json array_value(std::initializer_list<Json> values) {
    return Json(Json::Value(Json::Array(values)));
}
Json object_value(
        std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) result.emplace(value.first, value.second);
    return Json(Json::Value(std::move(result)));
}

Json linear_descriptor(const std::string& prefix, bool bias = true) {
    Json::Object result;
    result.emplace("weight", string_value(prefix + ".weight"));
    if (bias) result.emplace("bias", string_value(prefix + ".bias"));
    else result.emplace("bias", Json(Json::Value(nullptr)));
    return Json(Json::Value(std::move(result)));
}

Json norm_descriptor(const std::string& prefix) {
    return object_value({
        {"kind", string_value("layer_norm")},
        {"eps", number_value(1.0e-5)},
        {"weight", string_value(prefix + ".weight")},
        {"bias", string_value(prefix + ".bias")},
    });
}

Json encoder_block(int index) {
    const std::string prefix = "encoder.layers." + std::to_string(index);
    return object_value({
        {"input_norm", norm_descriptor(prefix + ".self_attn_layer_norm")},
        {"attention", object_value({
            {"heads", integer_value(6)},
            {"scale", number_value(0.125)},
            {"query", linear_descriptor(prefix + ".self_attn.q_proj")},
            {"key", linear_descriptor(prefix + ".self_attn.k_proj", false)},
            {"value", linear_descriptor(prefix + ".self_attn.v_proj")},
            {"output", linear_descriptor(prefix + ".self_attn.out_proj")},
        })},
        {"post_attention_norm", norm_descriptor(prefix + ".final_layer_norm")},
        {"mlp", object_value({
            {"activation", string_value("gelu")},
            {"width", integer_value(1536)},
            {"input", linear_descriptor(prefix + ".fc1")},
            {"output", linear_descriptor(prefix + ".fc2")},
        })},
    });
}

Json graph(const std::vector<AuditMapping>& mappings) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.logical_name,
                         string_value(mapping.destination_name));
    Json::Array blocks;
    for (int index = 0; index < 4; ++index)
        blocks.push_back(encoder_block(index));
    Json::Array outputs;
    for (const auto& [name, source] :
         std::array<std::pair<const char*, const char*>, 5>{{
             {"frontend_hidden", "frontend"},
             {"encoder_block_1", "block.0"},
             {"encoder_block_2", "block.1"},
             {"encoder_block_3", "block.2"},
             {"encoder_block_4", "block.3"},
         }})
        outputs.push_back(object_value({{"name", string_value(name)},
                                        {"source", string_value(source)}}));
    return object_value({
        {"schema_version", integer_value(1)},
        {"kind", string_value("audio_encoder_transformer")},
        {"config", object_value({
            {"input_layout", string_value("NCL")},
            {"mel_bins", integer_value(80)},
            {"input_frames", integer_value(3000)},
            {"hidden_width", integer_value(384)},
            {"output_frames", integer_value(1500)},
            {"encoder_blocks", integer_value(4)},
            {"attention_heads", integer_value(6)},
            {"ffn_width", integer_value(1536)},
            {"dtype", string_value("float32")},
        })},
        {"frontend", object_value({
            {"conv1", object_value({
                {"weight", string_value("encoder.conv1.weight")},
                {"bias", string_value("encoder.conv1.bias")},
                {"kernel", integer_value(3)},
                {"stride", integer_value(1)},
                {"padding", integer_value(1)},
            })},
            {"activation1", string_value("gelu")},
            {"conv2", object_value({
                {"weight", string_value("encoder.conv2.weight")},
                {"bias", string_value("encoder.conv2.bias")},
                {"kernel", integer_value(3)},
                {"stride", integer_value(2)},
                {"padding", integer_value(1)},
            })},
            {"activation2", string_value("gelu")},
        })},
        {"absolute_position_embedding", object_value({
            {"weight", string_value("encoder.embed_positions.weight")},
        })},
        {"blocks", Json(Json::Value(std::move(blocks)))},
        {"final_norm", norm_descriptor("encoder.layer_norm")},
        {"outputs", Json(Json::Value(std::move(outputs)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))},
        {"required_primitives", array_value({
            string_value("conv1d"), string_value("conv2d"),
            string_value("activation"), string_value("permute"),
            string_value("add"), string_value("layer_norm"),
            string_value("linear"), string_value("attention"),
            string_value("reshape")})},
    });
}

Json metadata() {
    return object_value({
        {"architecture", string_value("audio-encoder")},
        {"profile", string_value("component")},
        {"default_dtype", string_value("float32")},
        {"component", object_value({
            {"kind", string_value("audio_encoder")},
            {"implementation", string_value("generic.conv_transformer_encoder.v1")},
            {"source_repository", string_value("ByteDance/LatentSync-1.6")},
            {"source_revision", string_value(kModelRevision)},
            {"source_path", string_value("whisper/tiny.pt")},
            {"source_sha256", string_value(kCheckpointSha256)},
            {"code_repository", string_value("bytedance/LatentSync")},
            {"code_revision", string_value(kCodeRevision)},
            {"tensor_map_sha256", string_value(kTensorMapSha256)},
            {"retained_tensor_count", integer_value(kTensorCount)},
            {"retained_source_bytes", integer_value(kRetainedSourceBytes)},
            {"decoder_included", Json(Json::Value(false))},
            {"tokenizer_included", Json(Json::Value(false))},
            {"pickle_execution", Json(Json::Value(false))},
            {"license", string_value("MIT")},
        })},
    });
}

}  // namespace

FrozenComponentConversionResult convert_openai_whisper_tiny_encoder_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "OpenAI Whisper conversion interrupted; source retained");
    const fs::path checkpoint = source_directory / "tiny.pt";
    std::error_code error;
    if (!fs::is_regular_file(checkpoint, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen OpenAI Whisper tiny.pt");
    if (fs::file_size(checkpoint, error) != kCheckpointBytes || error ||
        sha256_file(checkpoint, progress) != kCheckpointSha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen OpenAI Whisper source identity mismatch");
    if (!fs::is_regular_file(tensor_map) ||
        sha256_file(tensor_map) != kTensorMapSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper frozen tensor-map identity drift");

    PytorchZipArchive archive(checkpoint, 256);
    if (archive.entries().size() != kArchiveEntryCount ||
        archive.file_size() != kCheckpointBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper ZIP inventory drift");
    const auto version = archive.read("archive/version", 16);
    if (std::string(version.begin(), version.end()) != "3\n")
        fail(ModelPackageErrorCode::PackageInvalid,
             "unsupported OpenAI Whisper serialization version");
    const auto pickle_bytes = archive.read("archive/data.pkl", 1ULL << 20);
    if (pickle_bytes.size() != 19363)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper pickle size drift");
    const RestrictedPickleInventory pickle = inspect_restricted_tensor_pickle(
        pickle_bytes, {"torch._utils._rebuild_tensor_v2", "torch.HalfStorage",
                       "collections.OrderedDict"});
    if (pickle.globals.size() != 3 || pickle.opcode_count != 5205 ||
        pickle.unicode_strings.size() != 348 ||
        pickle.unicode_strings.count("model_state_dict") != 1 ||
        pickle.unicode_strings.count("storage") != 1 ||
        pickle.unicode_strings.count("cpu") != 1)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper restricted pickle inventory drift");

    uint64_t data_members = 0;
    uint64_t checkpoint_tensor_bytes = 0;
    for (const auto& [name, entry] : archive.entries()) {
        if (!name.starts_with("archive/data/")) continue;
        ++data_members;
        checkpoint_tensor_bytes += entry.byte_length;
    }
    if (data_members != kStorageCount ||
        checkpoint_tensor_bytes != kCheckpointTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "OpenAI Whisper full storage inventory drift");

    const std::vector<AuditMapping> audit = load_map(tensor_map, archive, pickle);
    ConvertedHalfSource source(archive, audit);
    std::vector<TensorMapping> mappings;
    mappings.reserve(audit.size());
    for (const AuditMapping& mapping : audit)
        mappings.push_back({mapping.source_name, DType::F32, mapping.shape,
            "identity_bytes", mapping.destination_name, DType::F32,
            mapping.shape, "audio_encoder_transformer", "weight"});

    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "OpenAI Whisper component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot create OpenAI Whisper output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        const VrmWriteResult result = write_vrm_streaming(
            temporary, "component", "audio-encoder", metadata(), graph(audit),
            source, mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error)
            fail(ModelPackageErrorCode::InstallFailed,
                 "cannot atomically publish OpenAI Whisper component");
        return {output, kCheckpointBytes, kRetainedOutputBytes, kTensorCount,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
