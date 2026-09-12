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

constexpr uint64_t kCheckpointBytes = 89843225ULL;
constexpr uint64_t kRawTensorBytes = 89836440ULL;
constexpr const char* kCheckpointSha256 =
    "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543";
constexpr const char* kTensorMapSha256 =
    "53ffcf678205fbcaf23766c5f9ef7641de00db5720ad30bbed193d773b2a66d0";

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
                           "invalid vision detector tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid vision detector tensor dimension");
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
                 "vision detector tensor shape overflow");
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
             "vision detector frozen tensor-map identity drift");
    std::ifstream input(path);
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!input || !std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid vision detector tensor-map header");
    MappingSet result;
    uint64_t raw_bytes = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "s3fd-619a316812.pth" ||
            fields[2] != "F32" || fields[5] != "identity_bytes" ||
            fields[7] != "F32" || fields[9] != "vision_detector_multiscale" ||
            fields[10] != "weight")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported vision detector tensor-map row");
        auto source_shape = parse_shape(fields[3]);
        auto destination_shape = parse_shape(fields[8]);
        if (source_shape != destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "vision detector shape transformation is forbidden");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid vision detector tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid vision detector tensor offset");
        const uint64_t bytes = tensor_bytes(source_shape);
        raw_bytes += bytes;
        SourceTensorDescriptor descriptor{fields[1], DType::F32, "F32",
            source_shape, 0, offset, bytes};
        if (!result.tensors.emplace(fields[1], descriptor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate vision detector source tensor");
        result.mappings.push_back(TensorMapping{fields[1], DType::F32,
            source_shape, fields[5], fields[6], DType::F32,
            destination_shape, fields[9], fields[10]});
    }
    if (result.mappings.size() != 65 || raw_bytes != kRawTensorBytes)
        fail(ModelPackageErrorCode::PackageInvalid,
             "vision detector frozen tensor inventory drift");
    return result;
}

Json text(const std::string& value) { return Json(Json::Value(value)); }
Json integer(int64_t value) { return Json(Json::Value(value)); }
Json object(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) result.emplace(value.first, value.second);
    return Json(Json::Value(std::move(result)));
}

Json graph(const std::vector<TensorMapping>& mappings) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.source_name, text(mapping.destination_name));
    Json::Array outputs;
    for (int scale = 0; scale < 6; ++scale) {
        outputs.push_back(text("scale_" + std::to_string(scale) + "_confidence_logits"));
        outputs.push_back(text("scale_" + std::to_string(scale) + "_localization"));
    }
    Json::Array primitives;
    for (const char* value : {"conv2d", "relu", "max_pool2d", "l2_normalize",
                              "mul", "split", "maximum", "concat"})
        primitives.push_back(text(value));
    return object({
        {"schema_version", integer(1)},
        {"kind", text("vision_detector_multiscale")},
        {"entry_point", text("execute")},
        {"config", object({{"dtype", text("float32")},
                            {"input_channels", integer(3)},
                            {"scale_count", integer(6)},
                            {"confidence_channels", integer(2)},
                            {"localization_channels", integer(4)}})},
        {"semantic_outputs", Json(Json::Value(std::move(outputs)))},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))},
    });
}

Json metadata() {
    return object({
        {"architecture", text("vision-detector")},
        {"profile", text("component")},
        {"default_dtype", text("float32")},
        {"component", object({
            {"kind", text("vision_detector_multiscale")},
            {"implementation", text("generic.multiscale_conv_detector.v1")},
            {"source_url", text("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth")},
            {"reference_revision", text("9deb9bea0d46b085d171d9a417671f6ef1af2f69")},
            {"license", text("unresolved")},
            {"checkpoint_bytes", integer(kCheckpointBytes)},
            {"checkpoint_sha256", text(kCheckpointSha256)},
            {"tensor_count", integer(65)},
            {"raw_tensor_bytes", integer(kRawTensorBytes)},
        })},
    });
}

}  // namespace

FrozenComponentConversionResult convert_s3fd_face_detector_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "vision detector conversion interrupted; source retained");
    const fs::path checkpoint = source_directory / "s3fd-619a316812.pth";
    std::error_code error;
    if (!fs::is_regular_file(checkpoint, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen vision detector source");
    if (fs::file_size(checkpoint, error) != kCheckpointBytes || error ||
        sha256_file(checkpoint, progress) != kCheckpointSha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen vision detector source identity mismatch");
    MappingSet mapping = load_map(tensor_map);
    FrozenTensorSource source({checkpoint}, std::move(mapping.tensors));
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "vision detector component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create vision detector output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        VrmWriteResult result = write_vrm_streaming(temporary, "component",
            "vision-detector", metadata(), graph(mapping.mappings),
            source, mapping.mappings, cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish vision detector component");
        return {output, kCheckpointBytes, kRawTensorBytes, 65,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
