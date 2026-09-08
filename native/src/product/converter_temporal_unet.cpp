#include "vrhino/product/converter.h"

#include <algorithm>
#include <atomic>
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
constexpr uint64_t kCheckpointBytes = 5072222488ULL;
constexpr uint64_t kRawTensorBytes = 5071778576ULL;
constexpr const char* kCheckpointSha256 =
    "0a478e89eb660f82da4c35dbdde8a5adfb27f99d1b4e50edd03729e1e98316d3";
constexpr const char* kConfigSha256 =
    "652bbf469d3baf68f1b364fd47409901cd8d1bf8bb7754133aa69a28132312e6";
constexpr const char* kSchedulerSha256 =
    "14561e18160bf30bbcf7c93259af07ae82bbf611defeb93e995883fce1461315";
constexpr const char* kTensorMapSha256 =
    "932a1bf00c97142071e47bcb749e9273fb6efcb08d86226e5be534b7832d1b56";
constexpr uint64_t kTensorCount = 1246;
constexpr uint64_t kArchiveEntryCount = 1250;

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

std::vector<int64_t> parse_shape(const std::string& text) {
    std::vector<int64_t> shape;
    for (const std::string& field : split(text, ',')) {
        size_t consumed = 0;
        int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid temporal UNet tensor shape"); }
        if (consumed != field.size() || dimension <= 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid temporal UNet tensor dimension");
        shape.push_back(dimension);
    }
    if (shape.empty()) fail(ModelPackageErrorCode::PackageInvalid,
                            "empty temporal UNet tensor shape");
    return shape;
}

uint64_t tensor_bytes(const std::vector<int64_t>& shape) {
    uint64_t elements = 1;
    for (int64_t dimension : shape) {
        if (elements > std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "temporal UNet tensor shape overflow");
        elements *= static_cast<uint64_t>(dimension);
    }
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(float))
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet tensor byte size overflow");
    return elements * sizeof(float);
}

struct MappingSet {
    std::map<std::string, SourceTensorDescriptor> tensors;
    std::vector<TensorMapping> mappings;
    std::map<std::string, std::string> storage_by_tensor;
};

MappingSet load_map(const fs::path& path, const PytorchZipArchive& archive,
                    const RestrictedPickleInventory& pickle) {
    std::ifstream input(path);
    if (!input) fail(ModelPackageErrorCode::ArtifactMissing,
                     "missing temporal UNet tensor map");
    const std::string header =
        "source_name\tstorage_name\tsource_dtype\tsource_shape\tstorage_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid temporal UNet tensor-map header");
    MappingSet result;
    std::set<std::string> storages;
    uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[2] != "F32" || fields[4] != "0" ||
            fields[5] != "identity_bytes" || fields[7] != "F32" ||
            fields[9] != "temporal_conditional_unet_2d" ||
            fields[10] != "weight" || fields[0] != fields[6] ||
            fields[3] != fields[8])
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported temporal UNet tensor-map row");
        const std::vector<int64_t> shape = parse_shape(fields[3]);
        const uint64_t bytes = tensor_bytes(shape);
        const std::string member = "latentsync_unet/data/" + fields[1];
        const PytorchZipEntry& entry = archive.at(member);
        if (entry.byte_length != bytes)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "temporal UNet storage size/shape mismatch: " + fields[0]);
        if (!storages.insert(fields[1]).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "temporal UNet aliased/duplicate storage");
        if (pickle.unicode_strings.count(fields[0]) != 1 ||
            pickle.unicode_strings.count(fields[1]) != 1)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "temporal UNet pickle tensor/storage inventory mismatch");
        if (raw_bytes > std::numeric_limits<uint64_t>::max() - bytes)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "temporal UNet raw tensor byte count overflow");
        raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[0], DType::F32, "F32", shape,
            0, entry.data_offset, bytes};
        if (!result.tensors.emplace(fields[0], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate temporal UNet tensor name");
        result.storage_by_tensor.emplace(fields[0], fields[1]);
        result.mappings.push_back(TensorMapping{fields[0], DType::F32, shape,
            fields[5], fields[6], DType::F32, shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != kTensorCount || storages.size() != kTensorCount ||
        raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet frozen tensor inventory drift");

    size_t data_members = 0;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& [name, entry] : archive.entries()) {
        if (!name.starts_with("latentsync_unet/data/")) continue;
        ++data_members;
        const std::string storage = name.substr(std::string(
            "latentsync_unet/data/").size());
        if (!storages.contains(storage))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "extra temporal UNet tensor storage");
        ranges.emplace_back(entry.data_offset, entry.data_offset + entry.byte_length);
    }
    if (data_members != kTensorCount)
        fail(ModelPackageErrorCode::PackageInvalid,
             "missing temporal UNet tensor storage");
    std::sort(ranges.begin(), ranges.end());
    for (size_t index = 1; index < ranges.size(); ++index)
        if (ranges[index - 1].second > ranges[index].first)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "overlapping temporal UNet tensor storage ranges");
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
        {"kind", string_value("temporal_conditional_unet_2d")},
        {"entry_point", string_value("execute")},
        {"config", object_value({
            {"dtype", string_value("float32")},
            {"input_channels", integer_value(13)},
            {"output_channels", integer_value(4)},
            {"sample_size", integer_value(64)},
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
            {"motion_module_count", integer_value(20)},
            {"motion_transformer_blocks", integer_value(1)},
            {"motion_attention_sublayers", integer_value(2)},
            {"temporal_position_encoding", string_value("sinusoidal_interleaved")},
            {"temporal_position_maximum", integer_value(24)},
            {"feed_forward", string_value("geglu")},
            {"downsample", string_value("spatial_conv2d_stride_2_padding_1")},
            {"upsample", string_value("spatial_nearest_then_conv2d")}
        })},
        {"inputs", object_value({
            {"latent", string_value("B,13,F,H,W")},
            {"timestep", string_value("B_or_scalar")},
            {"conditioning", string_value("B,F,50,384")}
        })},
        {"outputs", object_value({
            {"epsilon", string_value("B,4,F,H,W")}
        })},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))}
    });
}

Json metadata() {
    return object_value({
        {"architecture", string_value("temporal-unet")},
        {"profile", string_value("component")},
        {"component", object_value({
            {"kind", string_value("temporal_conditional_unet_2d")},
            {"graph_identity", string_value("temporal_conditional_unet_2d.v1")},
            {"source_repository", string_value("ByteDance/LatentSync-1.6")},
            {"source_revision", string_value(kModelRevision)},
            {"source_path", string_value("latentsync_unet.pt")},
            {"source_sha256", string_value(kCheckpointSha256)},
            {"code_repository", string_value("bytedance/LatentSync")},
            {"code_revision", string_value(kCodeRevision)},
            {"tensor_map_sha256", string_value(kTensorMapSha256)},
            {"tensor_count", integer_value(kTensorCount)},
            {"source_raw_tensor_bytes", integer_value(kRawTensorBytes)},
            {"license", string_value("OpenRAIL++")},
            {"production_python_dependency", Json(Json::Value(false))},
            {"pickle_execution", Json(Json::Value(false))}
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
             "missing frozen temporal UNet source: " + std::string(name));
    if (fs::file_size(path, error) != bytes || error ||
        sha256_file(path, progress) != sha)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen temporal UNet source identity mismatch: " +
                 std::string(name));
    return path;
}

}  // namespace

FrozenComponentConversionResult convert_temporal_conditional_unet_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "temporal UNet conversion interrupted; source retained");
    const fs::path checkpoint = checked_source(source_directory,
        "latentsync_unet.pt", kCheckpointBytes, kCheckpointSha256, progress);
    (void)checked_source(source_directory, "stage2_512.yaml", 2542ULL,
                         kConfigSha256, progress);
    (void)checked_source(source_directory, "scheduler_config.json", 275ULL,
                         kSchedulerSha256, progress);
    if (!fs::is_regular_file(tensor_map) ||
        sha256_file(tensor_map) != kTensorMapSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet frozen tensor-map identity drift");

    PytorchZipArchive archive(checkpoint, 2048);
    if (archive.file_size() != kCheckpointBytes ||
        archive.entries().size() != kArchiveEntryCount)
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet ZIP inventory drift");
    const auto byteorder = archive.read("latentsync_unet/byteorder", 16);
    const auto version = archive.read("latentsync_unet/version", 16);
    const auto serialization = archive.read(
        "latentsync_unet/.data/serialization_id", 128);
    if (std::string(byteorder.begin(), byteorder.end()) != "little" ||
        std::string(version.begin(), version.end()) != "3\n" ||
        serialization.size() != 40)
        fail(ModelPackageErrorCode::PackageInvalid,
             "unsupported temporal UNet PyTorch serialization metadata");
    const auto pickle_bytes = archive.read("latentsync_unet/data.pkl", 1ULL << 20);
    if (pickle_bytes.size() != 192468)
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet pickle size drift");
    const RestrictedPickleInventory pickle = inspect_restricted_tensor_pickle(
        pickle_bytes, {"torch._utils._rebuild_tensor_v2", "torch.FloatStorage",
                       "collections.OrderedDict"});
    if (pickle.globals.size() != 3 || pickle.unicode_strings.size() != 2495 ||
        pickle.unicode_strings.count("state_dict") != 1 ||
        pickle.unicode_strings.count("storage") != 1 ||
        pickle.unicode_strings.count("cpu") != 1)
        fail(ModelPackageErrorCode::PackageInvalid,
             "temporal UNet restricted pickle inventory drift");
    MappingSet mapping = load_map(tensor_map, archive, pickle);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));

    std::error_code error;
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "temporal UNet component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create temporal UNet output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "temporal-unet", metadata(), graph(mapping.mappings), source,
            mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish temporal UNet component");
        return {output, kCheckpointBytes, kRawTensorBytes, kTensorCount,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
