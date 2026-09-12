#include "vrhino/product/converter.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr const char* kRepository = "yzd-v/DWPose";
constexpr const char* kRevision =
    "1a7144101628d69ee7a3768d1ee3a094070dc388";
constexpr uint64_t kCheckpointBytes = 406878486ULL;
constexpr uint64_t kConfigBytes = 7364ULL;
constexpr uint64_t kRawTensorBytes = 134599836ULL;
constexpr const char* kCheckpointSha256 =
    "0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07";
constexpr const char* kConfigSha256 =
    "7c0da61ee9517ee6c97ec10dd6be592eb51501434a93fa0a8e7f3f3739b5e477";
constexpr const char* kTensorMapSha256 =
    "5e593a715f7a0c6676ea5709c8d9ab14d34a6fbadfcbcf8ce34249bca965bd1f";

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
    if (value.empty()) return shape;
    for (const std::string& field : split(value, ',')) {
        size_t consumed = 0;
        int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid pose tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid pose tensor dimension");
        shape.push_back(dimension);
    }
    return shape;
}

uint64_t tensor_bytes(const std::vector<int64_t>& shape) {
    uint64_t elements = 1;
    for (int64_t dimension : shape) {
        if (dimension != 0 && elements >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "pose tensor shape overflow");
        elements *= static_cast<uint64_t>(dimension);
    }
    return elements * sizeof(float);
}

struct MappingSet {
    std::map<std::string, SourceTensorDescriptor> tensors;
    std::vector<TensorMapping> mappings;
};

MappingSet load_map(const fs::path& path) {
    if (sha256_file(path) != kTensorMapSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pose estimator frozen tensor-map identity drift");
    std::ifstream input(path);
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!input || !std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid pose estimator tensor-map header");
    MappingSet result;
    uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "dw-ll_ucoco_384.pth" ||
            fields[2] != "F32" || fields[5] != "identity_bytes" ||
            fields[7] != "F32" || fields[9] != "pose_estimator_2d" ||
            fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported pose estimator tensor-map row");
        const auto source_shape = parse_shape(fields[3]);
        const auto destination_shape = parse_shape(fields[8]);
        if (source_shape != destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "pose estimator shape transformation is forbidden");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid pose estimator tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid pose estimator tensor offset");
        const uint64_t bytes = tensor_bytes(source_shape);
        raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[1], DType::F32, "F32",
            source_shape, 0, offset, bytes};
        if (!result.tensors.emplace(fields[1], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate pose estimator source tensor");
        result.mappings.push_back(TensorMapping{fields[1], DType::F32,
            source_shape, fields[5], fields[6], DType::F32,
            destination_shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != 395 || raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pose estimator frozen tensor inventory drift");
    return result;
}

Json text(const std::string& value) { return Json(Json::Value(value)); }
Json integer(int64_t value) { return Json(Json::Value(value)); }
Json number(double value) { return Json(Json::Value(value)); }
Json boolean(bool value) { return Json(Json::Value(value)); }
Json object(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) result.emplace(value.first, value.second);
    return Json(Json::Value(std::move(result)));
}

Json graph(const std::vector<TensorMapping>& mappings) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.source_name, text(mapping.destination_name));
    Json::Array primitives;
    for (const char* value : {"conv2d", "batch_norm_composition", "silu",
             "max_pool2d", "reduce_sum", "sqrt", "clamp", "concat",
             "linear", "reshape", "permute", "split", "add", "mul",
             "div", "relu", "batched_matmul"})
        primitives.push_back(text(value));
    return object({
        {"schema_version", integer(1)},
        {"kind", text("pose_estimator_2d")},
        {"entry_point", text("execute")},
        {"config", object({
            {"dtype", text("float32")},
            {"input_height", integer(384)},
            {"input_width", integer(288)},
            {"keypoint_count", integer(133)},
            {"backbone", text("cspnext_l_p5")},
            {"backbone_output_channels", integer(1024)},
            {"head", text("rtmcc_simcc")},
            {"head_hidden_width", integer(256)},
            {"gau_expanded_width", integer(512)},
            {"gau_attention_width", integer(128)},
            {"simcc_x_bins", integer(576)},
            {"simcc_y_bins", integer(768)},
            {"simcc_split_ratio", number(2.0)},
            {"flip_test", boolean(true)},
            {"batch_norm_epsilon", number(1.0e-5)},
            {"scale_norm_epsilon", number(1.0e-5)}})},
        {"inputs", object({{"image", text("B,3,384,288")}})},
        {"outputs", object({{"simcc_x", text("B,133,576")},
                              {"simcc_y", text("B,133,768")}})},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))}
    });
}

Json metadata() {
    return object({
        {"architecture", text("pose-estimator")},
        {"profile", text("component")},
        {"default_dtype", text("float32")},
        {"component", object({
            {"kind", text("pose_estimator_2d")},
            {"graph_identity", text("pose_estimator_2d.v1")},
            {"implementation", text("generic.cspnext_rtmcc.v1")},
            {"source_repository", text(kRepository)},
            {"source_revision", text(kRevision)},
            {"source_path", text("dw-ll_ucoco_384.pth")},
            {"source_sha256", text(kCheckpointSha256)},
            {"reference_config_sha256", text(kConfigSha256)},
            {"tensor_map_sha256", text(kTensorMapSha256)},
            {"source_tensor_count", integer(470)},
            {"retained_tensor_count", integer(395)},
            {"source_raw_tensor_bytes", integer(134600436)},
            {"retained_raw_tensor_bytes", integer(kRawTensorBytes)},
            {"license", text("Apache-2.0")},
            {"production_python_dependency", boolean(false)}})}
    });
}

fs::path checked_source(const fs::path& root, const char* name,
                        uint64_t bytes, const char* sha,
                        const WorkProgressCallback& progress) {
    const fs::path path = root / name;
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen pose source: " + std::string(name));
    if (fs::file_size(path, error) != bytes || error ||
        sha256_file(path, progress) != sha)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen pose source identity mismatch: " + std::string(name));
    return path;
}

}  // namespace

FrozenComponentConversionResult convert_dwpose_keypoint_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "pose estimator conversion interrupted; source retained");
    const fs::path checkpoint = checked_source(source_directory,
        "dw-ll_ucoco_384.pth", kCheckpointBytes, kCheckpointSha256, progress);
    (void)checked_source(source_directory,
        "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",
        kConfigBytes, kConfigSha256, progress);
    MappingSet mapping = load_map(tensor_map);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));
    std::error_code error;
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "pose estimator component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create pose estimator output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "pose-estimator", metadata(), graph(mapping.mappings), source,
            mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish pose estimator component");
        return {output, kCheckpointBytes, kRawTensorBytes, 395,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
