#include "vrhino/product/converter.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <unistd.h>

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr const char* kRevision =
    "3ef28bc5cff08c90ad8178a25f1b570cd800170f";
constexpr uint64_t kCheckpointBytes = 3400074924ULL;
constexpr uint64_t kRawTensorBytes = 3399791376ULL;
constexpr const char* kCheckpointSha256 =
    "7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007";
constexpr const char* kTensorMapSha256 =
    "8bcdd2f083d2a9b79db392b73b0c919202ab0e1dccb87acd28587bf27358f514";

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
                           "invalid conditional UNet tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid conditional UNet tensor dimension");
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
                 "conditional UNet tensor shape overflow");
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
                     "missing conditional UNet tensor map");
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid conditional UNet tensor-map header");
    MappingSet result;
    uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "unet.pth" ||
            fields[2] != "F32" || fields[5] != "identity_bytes" ||
            fields[7] != "F32" || fields[9] != "conditional_unet_2d" ||
            fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported conditional UNet tensor-map row");
        const auto source_shape = parse_shape(fields[3]);
        const auto destination_shape = parse_shape(fields[8]);
        if (source_shape != destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "conditional UNet shape transformation is forbidden");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid conditional UNet tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid conditional UNet tensor offset");
        const uint64_t bytes = tensor_bytes(source_shape);
        raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[1], DType::F32, "F32",
            source_shape, 0, offset, bytes};
        if (!result.tensors.emplace(fields[1], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate conditional UNet source tensor");
        result.mappings.push_back(TensorMapping{fields[1], DType::F32,
            source_shape, fields[5], fields[6], DType::F32,
            destination_shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != 686 || raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "conditional UNet frozen tensor inventory drift");
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
    for (int value : {320, 640, 1280, 1280})
        channels.emplace_back(Json::Value(static_cast<int64_t>(value)));
    Json::Array primitives;
    for (const char* value : {"conv2d", "group_norm", "layer_norm", "silu",
             "gelu", "linear", "reshape", "permute", "concat", "split",
             "add", "mul", "attention", "sinusoidal_embedding",
             "interpolate_nearest"})
        primitives.push_back(string_value(value));
    return object_value({
        {"schema_version", integer_value(1)},
        {"kind", string_value("conditional_unet_2d")},
        {"entry_point", string_value("execute")},
        {"config", object_value({
            {"dtype", string_value("float32")},
            {"input_channels", integer_value(8)},
            {"output_channels", integer_value(4)},
            {"sample_size", integer_value(32)},
            {"conditioning_width", integer_value(384)},
            {"block_out_channels", Json(Json::Value(std::move(channels)))},
            {"layers_per_block", integer_value(2)},
            {"up_layers_per_block", integer_value(3)},
            {"attention_heads", integer_value(8)},
            {"norm_num_groups", integer_value(32)},
            {"norm_epsilon", number_value(1.0e-5)},
            {"transformer_group_norm_epsilon", number_value(1.0e-6)},
            {"layer_norm_epsilon", number_value(1.0e-5)},
            {"timestep_width", integer_value(320)},
            {"timestep_embedding_width", integer_value(1280)},
            {"feed_forward", string_value("geglu")},
            {"downsample", string_value("conv2d_stride_2_padding_1")},
            {"upsample", string_value("nearest_then_conv2d")}
        })},
        {"inputs", object_value({
            {"latent", string_value("B,8,H,W")},
            {"timestep", string_value("B")},
            {"conditioning", string_value("B,S,384")}
        })},
        {"outputs", object_value({
            {"predicted_latent", string_value("B,4,H,W")}
        })},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))}
    });
}

Json metadata() {
    return object_value({
        {"architecture", string_value("cond-unet-2d")},
        {"profile", string_value("component")},
        {"component", object_value({
            {"kind", string_value("conditional_unet_2d")},
            {"graph_identity", string_value("conditional_unet_2d.v1")},
            {"source_repository", string_value("TMElyralab/MuseTalk")},
            {"source_revision", string_value(kRevision)},
            {"source_path", string_value("musetalkV15/unet.pth")},
            {"source_sha256", string_value(kCheckpointSha256)},
            {"tensor_map_sha256", string_value(kTensorMapSha256)},
            {"tensor_count", integer_value(686)},
            {"source_raw_tensor_bytes", integer_value(
                static_cast<int64_t>(kRawTensorBytes))},
            {"license", string_value("UNKNOWN")},
            {"production_python_dependency", Json(Json::Value(false))}
        })}
    });
}

fs::path checked_source(const fs::path& root, const char* name,
                        uint64_t bytes, const char* sha,
                        const WorkProgressCallback& progress) {
    const fs::path path = root / name;
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen conditional UNet source: " + std::string(name));
    if (fs::file_size(path, error) != bytes || error ||
        sha256_file(path, progress) != sha)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen conditional UNet source identity mismatch: " +
                 std::string(name));
    return path;
}

}  // namespace

FrozenComponentConversionResult convert_musetalk_v15_unet_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "conditional UNet conversion interrupted; source retained");
    (void)checked_source(source_directory, "musetalk.json", 748ULL,
        "5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47",
        progress);
    const fs::path checkpoint = checked_source(source_directory, "unet.pth",
        kCheckpointBytes, kCheckpointSha256, progress);
    if (!fs::is_regular_file(tensor_map) ||
        sha256_file(tensor_map) != kTensorMapSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "conditional UNet frozen tensor-map identity drift");
    MappingSet mapping = load_map(tensor_map);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));
    std::error_code error;
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "conditional UNet component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create conditional UNet output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "cond-unet-2d", metadata(), graph(mapping.mappings), source,
            mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish conditional UNet component");
        return {output, kCheckpointBytes, kRawTensorBytes, 686,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
