#include "vrhino/product/converter.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unistd.h>

#include "vrhino/product/tflite.h"

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr uint64_t kCheckpointBytes = 229746;
constexpr uint64_t kSourceLearnedBytes = 202780;
constexpr uint64_t kConvertedTensorBytes = 405560;
constexpr const char* kCheckpointSha256 =
    "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f";
constexpr const char* kTensorMapSha256 =
    "a9d325cdfe8a99ddd56f23ef75b5e9da2a9fed7905726e742c8238de10cbfaeb";

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
                           "invalid dense detector tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid dense detector tensor dimension");
        shape.push_back(dimension);
    }
    return shape;
}

uint64_t elements(const std::vector<int64_t>& shape) {
    uint64_t count = 1;
    for (int64_t dimension : shape) {
        if (dimension != 0 && count >
            std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector tensor shape overflow");
        count *= static_cast<uint64_t>(dimension);
    }
    return count;
}

struct AuditMapping {
    std::string source_name;
    std::vector<int64_t> source_shape;
    uint64_t source_offset = 0;
    std::string transformation;
    std::string destination_name;
    std::vector<int64_t> destination_shape;
};

std::vector<AuditMapping> load_map(const fs::path& path) {
    if (sha256_file(path) != kTensorMapSha256)
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector frozen tensor-map identity drift");
    std::ifstream input(path);
    const std::string header =
        "source_file\tsource_name\tsource_dtype\tsource_shape\tsource_offset\t"
        "transformation\tdestination_name\tdestination_dtype\t"
        "destination_shape\tcomponent\trole";
    std::string line;
    if (!input || !std::getline(input, line) || line != header)
        fail(ModelPackageErrorCode::PackageInvalid,
             "invalid dense detector tensor-map header");
    std::vector<AuditMapping> result;
    std::set<std::string> source_names, destination_names;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 || fields[0] != "blaze_face_short_range.tflite" ||
            fields[2] != "F16" || fields[7] != "F32" ||
            fields[9] != "vision_detector_dense" || fields[10] != "weight" ||
            (fields[5] != "fp16_to_fp32" &&
             fields[5] != "fp16_ohwi_to_fp32_oihw" &&
             fields[5] != "fp16_1hwc_to_fp32_c1hw"))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported dense detector tensor-map row");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid dense detector tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid dense detector tensor offset");
        AuditMapping mapping{fields[1], parse_shape(fields[3]), offset,
                             fields[5], fields[6], parse_shape(fields[8])};
        if (!source_names.insert(mapping.source_name).second ||
            !destination_names.insert(mapping.destination_name).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate dense detector tensor mapping");
        if (mapping.transformation == "fp16_to_fp32" &&
            mapping.source_shape != mapping.destination_shape)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector scalar layout drift");
        result.push_back(std::move(mapping));
    }
    if (result.size() != 74)
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector frozen tensor inventory drift");
    return result;
}

void validate_inventory(const TfliteModelInventory& inventory,
                        const std::vector<AuditMapping>& mappings) {
    if (inventory.schema_version != 3 || inventory.file_bytes != kCheckpointBytes ||
        inventory.subgraph_count != 1 || inventory.buffer_count != 89 ||
        inventory.tensors.size() != 250 || inventory.operators.size() != 164 ||
        inventory.inputs != std::vector<int32_t>{0} ||
        inventory.outputs != std::vector<int32_t>({175, 174}))
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector TFLite envelope drift");
    const auto& input = inventory.tensors.at(0);
    const auto& regressors = inventory.tensors.at(175);
    const auto& classifications = inventory.tensors.at(174);
    if (input.name != "input" || input.dtype != "F32" ||
        input.shape != std::vector<int64_t>({1,128,128,3}) ||
        regressors.name != "regressors" || regressors.dtype != "F32" ||
        regressors.shape != std::vector<int64_t>({1,896,16}) ||
        classifications.name != "classificators" || classifications.dtype != "F32" ||
        classifications.shape != std::vector<int64_t>({1,896,1}))
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector input/output schema drift");

    std::map<std::string, size_t> counts;
    for (const auto& operation : inventory.operators) {
        ++counts[operation.type];
        const int32_t expected_version = operation.type == "DEQUANTIZE" ? 2 : 1;
        if (operation.version != expected_version)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector operator version drift");
        if ((operation.type == "CONV_2D" &&
             (operation.options.size() != 6 || operation.options[5] != 0)) ||
            (operation.type == "DEPTHWISE_CONV_2D" &&
             (operation.options.size() != 7 || operation.options[3] != 1 ||
              operation.options[6] != 0)) ||
            (operation.type == "MAX_POOL_2D" &&
             operation.options != std::vector<int64_t>({0,2,2,2,2,0})) ||
            (operation.type == "ADD" &&
             operation.options != std::vector<int64_t>({0})) ||
            (operation.type == "CONCATENATION" &&
             operation.options != std::vector<int64_t>({1,0})))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector operator option drift");
    }
    const std::map<std::string, size_t> expected{
        {"ADD",16}, {"CONCATENATION",2}, {"CONV_2D",21},
        {"DEPTHWISE_CONV_2D",16}, {"DEQUANTIZE",74},
        {"MAX_POOL_2D",3}, {"PAD",11}, {"RELU",17}, {"RESHAPE",4}};
    if (counts != expected)
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector operator inventory drift");

    std::map<std::string, const TfliteTensorRecord*> tensors;
    uint64_t learned_bytes = 0;
    size_t learned_count = 0, padding_count = 0;
    for (const auto& tensor : inventory.tensors) {
        if (!tensors.emplace(tensor.name, &tensor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate TFLite tensor name");
        if (tensor.data_bytes != 0 && tensor.dtype == "F16") {
            ++learned_count;
            learned_bytes += tensor.data_bytes;
        } else if (tensor.data_bytes != 0 && tensor.dtype == "I32") {
            ++padding_count;
        }
    }
    if (learned_count != 74 || learned_bytes != kSourceLearnedBytes ||
        padding_count != 11)
        fail(ModelPackageErrorCode::PackageInvalid,
             "dense detector constant inventory drift");
    for (const auto& mapping : mappings) {
        const auto found = tensors.find(mapping.source_name);
        if (found == tensors.end() || found->second->dtype != "F16" ||
            found->second->shape != mapping.source_shape ||
            found->second->data_offset != mapping.source_offset ||
            found->second->data_bytes != elements(mapping.source_shape) * 2)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector fixed tensor range drift: " + mapping.source_name);
    }
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(113 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | mantissa << 13;
    }
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

class ConvertedConstantSource final : public TensorSource {
public:
    ConvertedConstantSource(const fs::path& path,
                            const std::vector<AuditMapping>& mappings) {
        std::ifstream input(path, std::ios::binary);
        std::vector<uint8_t> file(static_cast<size_t>(kCheckpointBytes));
        input.read(reinterpret_cast<char*>(file.data()), file.size());
        if (!input) fail(ModelPackageErrorCode::PackageInvalid,
                         "cannot read dense detector constants");
        for (const auto& mapping : mappings) {
            const uint64_t count = elements(mapping.source_shape);
            if (mapping.source_offset > file.size() || count * 2 >
                file.size() - mapping.source_offset)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "dense detector tensor range exceeds source");
            std::vector<float> source(static_cast<size_t>(count));
            for (uint64_t index = 0; index < count; ++index) {
                const size_t offset = static_cast<size_t>(mapping.source_offset + 2 * index);
                const uint16_t half = static_cast<uint16_t>(file[offset]) |
                    static_cast<uint16_t>(file[offset + 1]) << 8;
                source[index] = half_to_float(half);
            }
            std::vector<float> destination(static_cast<size_t>(count));
            if (mapping.transformation == "fp16_to_fp32") {
                destination = std::move(source);
            } else if (mapping.transformation == "fp16_ohwi_to_fp32_oihw") {
                const int64_t out = mapping.source_shape[0];
                const int64_t height = mapping.source_shape[1];
                const int64_t width = mapping.source_shape[2];
                const int64_t in = mapping.source_shape[3];
                for (int64_t o = 0; o < out; ++o)
                    for (int64_t i = 0; i < in; ++i)
                        for (int64_t y = 0; y < height; ++y)
                            for (int64_t x = 0; x < width; ++x)
                                destination[((o * in + i) * height + y) * width + x] =
                                    source[((o * height + y) * width + x) * in + i];
            } else if (mapping.transformation == "fp16_1hwc_to_fp32_c1hw") {
                const int64_t height = mapping.source_shape[1];
                const int64_t width = mapping.source_shape[2];
                const int64_t channels = mapping.source_shape[3];
                for (int64_t c = 0; c < channels; ++c)
                    for (int64_t y = 0; y < height; ++y)
                        for (int64_t x = 0; x < width; ++x)
                            destination[(c * height + y) * width + x] =
                                source[(y * width + x) * channels + c];
            } else {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "unsupported dense detector constant transformation");
            }
            SourceTensorDescriptor descriptor{mapping.destination_name, DType::F32,
                "F32", mapping.destination_shape, 0, 0,
                static_cast<uint64_t>(destination.size() * sizeof(float))};
            descriptors_.emplace(mapping.destination_name, std::move(descriptor));
            values_.emplace(mapping.destination_name, std::move(destination));
        }
    }

    const std::map<std::string, SourceTensorDescriptor>& tensors() const noexcept override {
        return descriptors_;
    }

    void read_tensor(const SourceTensorDescriptor& tensor, uint64_t offset,
                     void* destination, size_t bytes) const override {
        const auto found = values_.find(tensor.name);
        if (found == values_.end() || offset > tensor.byte_length ||
            bytes > tensor.byte_length - offset)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "dense detector converted tensor read exceeds range");
        std::memcpy(destination,
                    reinterpret_cast<const uint8_t*>(found->second.data()) + offset,
                    bytes);
    }

private:
    std::map<std::string, SourceTensorDescriptor> descriptors_;
    std::map<std::string, std::vector<float>> values_;
};

Json text(const std::string& value) { return Json(Json::Value(value)); }
Json integer(int64_t value) { return Json(Json::Value(value)); }
Json object(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) result.emplace(value.first, value.second);
    return Json(Json::Value(std::move(result)));
}

Json graph(const std::vector<AuditMapping>& mappings) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.destination_name, text(mapping.destination_name));
    Json::Array primitives;
    for (const char* value : {"conv2d", "pad", "add", "relu", "max_pool2d",
                              "permute", "reshape", "concat"})
        primitives.push_back(text(value));
    return object({
        {"schema_version", integer(1)},
        {"kind", text("vision_detector_dense_anchors")},
        {"entry_point", text("execute")},
        {"config", object({{"dtype", text("float32")},
                            {"input_channels", integer(3)},
                            {"input_height", integer(128)},
                            {"input_width", integer(128)},
                            {"block_count", integer(16)},
                            {"anchor_count", integer(896)},
                            {"regression_coordinates", integer(16)}})},
        {"semantic_outputs", Json(Json::Value(Json::Array{
            text("regressors"), text("classification_logits")}))},
        {"required_primitives", Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings", Json(Json::Value(std::move(bindings)))},
    });
}

Json metadata() {
    return object({
        {"architecture", text("vision-detector")},
        {"profile", text("component")},
        {"default_dtype", text("float32")},
        {"component_kind", text("vision_detector_dense_anchors")},
        {"graph_identity", text("vision_detector_dense_anchors.v1")},
        {"source_format", text("tflite-flatbuffer-fixed")},
        {"source_constant_dtype", text("float16")},
        {"native_constant_dtype", text("float32")},
    });
}

}  // namespace

FrozenComponentConversionResult convert_blazeface_short_range_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "dense detector conversion interrupted; source retained");
    const fs::path checkpoint = source_directory / "blaze_face_short_range.tflite";
    std::error_code error;
    if (!fs::is_regular_file(checkpoint, error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen dense detector source");
    if (fs::file_size(checkpoint, error) != kCheckpointBytes || error ||
        sha256_file(checkpoint, progress) != kCheckpointSha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen dense detector source identity mismatch");
    const std::vector<AuditMapping> audit_mappings = load_map(tensor_map);
    const TfliteModelInventory inventory = inspect_tflite_flatbuffer(checkpoint);
    validate_inventory(inventory, audit_mappings);
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "dense detector conversion interrupted; source retained");
    ConvertedConstantSource source(checkpoint, audit_mappings);
    std::vector<TensorMapping> mappings;
    mappings.reserve(audit_mappings.size());
    for (const auto& mapping : audit_mappings)
        mappings.push_back({mapping.destination_name, DType::F32,
            mapping.destination_shape, "identity_bytes", mapping.destination_name,
            DType::F32, mapping.destination_shape, "vision_detector_dense", "weight"});
    if (fs::exists(output, error))
        fail(ModelPackageErrorCode::InstallFailed,
             "dense detector component output already exists");
    fs::create_directories(output.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create dense detector output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string() + ".partial-" + std::to_string(getpid()) +
         "-" + std::to_string(sequence.fetch_add(1)));
    try {
        const VrmWriteResult result = write_vrm_streaming(
            temporary, "component", "vision-detector", metadata(),
            graph(audit_mappings), source, mappings,
            cancellation_requested, progress);
        fs::rename(temporary, output, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish dense detector component");
        return {output, kCheckpointBytes, kConvertedTensorBytes, 74,
            result.file_size, result.largest_buffer_bytes, result.seconds,
            sha256_file(output), result.payload_blake2b128};
    } catch (...) {
        fs::remove(temporary, error);
        throw;
    }
}

}  // namespace vrhino::product
