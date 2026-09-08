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

constexpr uint64_t kCheckpointBytes = 53289463ULL;
constexpr uint64_t kRawTensorBytes = 52650752ULL;
constexpr const char* kCheckpointSha256 =
    "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567";
constexpr const char* kTensorMapSha256 =
    "4915ff72ee673a412508ee3c6656bbfc99e927f6de25e7225a7a16106fb0e814";

[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> output; size_t begin = 0;
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
        size_t consumed = 0; int64_t dimension = 0;
        try { dimension = std::stoll(field, &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid semantic segmenter tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid semantic segmenter tensor dimension");
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
                 "semantic segmenter tensor shape overflow");
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
             "semantic segmenter frozen tensor-map identity drift");
    std::ifstream input(path);
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!input || !std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid semantic segmenter tensor-map header");
    MappingSet result; uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "79999_iter.pth" ||
            fields[2] != "F32" || fields[5] != "identity_bytes" ||
            fields[7] != "F32" || fields[9] != "semantic_segmenter_2d" ||
            fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported semantic segmenter tensor-map row");
        const auto source_shape = parse_shape(fields[3]);
        const auto destination_shape = parse_shape(fields[8]);
        if (source_shape != destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "semantic segmenter shape transformation is forbidden");
        size_t consumed = 0; uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid semantic segmenter tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid semantic segmenter tensor offset");
        const uint64_t bytes = tensor_bytes(source_shape); raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[1], DType::F32, "F32",
            source_shape, 0, offset, bytes};
        if (!result.tensors.emplace(fields[1], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate semantic segmenter source tensor");
        result.mappings.push_back(TensorMapping{fields[1], DType::F32,
            source_shape, fields[5], fields[6], DType::F32,
            destination_shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != 148 || raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter frozen tensor inventory drift");
    return result;
}

Json text(const std::string& value) { return Json(Json::Value(value)); }
Json integer(int64_t value) { return Json(Json::Value(value)); }
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
    for (const char* value : {"conv2d", "batch_norm_composition", "relu",
             "max_pool2d", "reduce_sum", "add", "mul", "div", "exp",
             "concat", "interpolate_nearest", "interpolate_bilinear_2d"})
        primitives.push_back(text(value));
    return object({
        {"schema_version", integer(1)},
        {"kind", text("semantic_segmenter_2d")},
        {"entry_point", text("execute")},
        {"config", object({
            {"dtype", text("float32")},
            {"input_height", integer(512)},
            {"input_width", integer(512)},
            {"class_count", integer(19)},
            {"backbone", text("resnet18_context_path")},
            {"bilinear_align_corners", boolean(true)},
            {"batch_norm_epsilon", Json(Json::Value(1.0e-5))}})},
        {"inputs", object({{"image", text("B,3,512,512")}})},
        {"outputs", object({{"semantic_logits", text("B,19,512,512")}})},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))}
    });
}

Json metadata() {
    return object({
        {"architecture", text("segmenter-2d")},
        {"profile", text("component")},
        {"default_dtype", text("float32")},
        {"component", object({
            {"kind", text("semantic_segmenter_2d")},
            {"graph_identity", text("semantic_segmenter_2d.v1")},
            {"implementation", text("generic.resnet_context_bisenet.v1")},
            {"reference_revision", text("9deb9bea0d46b085d171d9a417671f6ef1af2f69")},
            {"source_path", text("79999_iter.pth")},
            {"source_sha256", text(kCheckpointSha256)},
            {"tensor_map_sha256", text(kTensorMapSha256)},
            {"source_tensor_count", integer(191)},
            {"retained_tensor_count", integer(148)},
            {"retained_raw_tensor_bytes", integer(kRawTensorBytes)},
            {"resnet18_source_required", boolean(false)},
            {"license", text("UNKNOWN")},
            {"production_python_dependency", boolean(false)}})}
    });
}

}  // namespace

FrozenComponentConversionResult convert_bisenet_face_parser_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "semantic segmenter conversion interrupted; source retained");
    const fs::path checkpoint = source_directory / "79999_iter.pth";
    std::error_code error;
    if (!fs::is_regular_file(checkpoint, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen semantic segmenter source");
    if (fs::file_size(checkpoint, error) != kCheckpointBytes || error ||
        sha256_file(checkpoint, progress) != kCheckpointSha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen semantic segmenter source identity mismatch");
    MappingSet mapping = load_map(tensor_map);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "semantic segmenter component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create semantic segmenter output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "segmenter-2d", metadata(), graph(mapping.mappings), source,
            mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish semantic segmenter component");
        return {output, kCheckpointBytes, kRawTensorBytes, 148,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error); throw;
    }
}

}  // namespace vrhino::product
