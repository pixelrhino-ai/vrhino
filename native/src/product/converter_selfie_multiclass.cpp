#include "vrhino/product/converter.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif

#include "vrhino/product/tflite.h"

namespace vrhino::product {
namespace {
namespace fs = std::filesystem;

constexpr uint64_t kCheckpointBytes = 16371837;
constexpr uint64_t kSourceConstantBytes = 16281136;
constexpr const char* kCheckpointSha256 =
    "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0";
constexpr const char* kTensorMapSha256 =
    "7539cee94bc193eb575659d74c46ccc10de0df3ebd1dd18abf82de365a5a2326";

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
                           "invalid semantic segmenter tensor shape"); }
        if (consumed != field.size() || dimension < 0)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid semantic segmenter tensor dimension");
        shape.push_back(dimension);
    }
    return shape;
}

uint64_t elements(const std::vector<int64_t>& shape) {
    uint64_t count = 1;
    for (int64_t dimension : shape) {
        if (dimension != 0 && count > std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(dimension))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "semantic segmenter tensor shape overflow");
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
    std::vector<AuditMapping> result;
    std::set<std::string> source_names, destination_names;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 11 ||
            fields[0] != "selfie_multiclass_256x256.tflite" ||
            fields[2] != "F32" || fields[7] != "F32" ||
            fields[9] != "semantic_segmenter_xenoformer" ||
            fields[10] != "weight" ||
            (fields[5] != "identity_bytes" &&
             fields[5] != "f32_ohwi_to_oihw" &&
             fields[5] != "f32_1hwc_to_c1hw" &&
             fields[5] != "f32_ohwi_to_iohw"))
            fail(ModelPackageErrorCode::PackageInvalid,
                 "unsupported semantic segmenter tensor-map row");
        size_t consumed = 0;
        uint64_t offset = 0;
        try { offset = std::stoull(fields[4], &consumed); }
        catch (...) { fail(ModelPackageErrorCode::PackageInvalid,
                           "invalid semantic segmenter tensor offset"); }
        if (consumed != fields[4].size())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "invalid semantic segmenter tensor offset");
        AuditMapping mapping{fields[1], parse_shape(fields[3]), offset,
                             fields[5], fields[6], parse_shape(fields[8])};
        if (!source_names.insert(mapping.source_name).second ||
            !destination_names.insert(mapping.destination_name).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate semantic segmenter tensor mapping");
        result.push_back(std::move(mapping));
    }
    if (result.size() != 181)
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter frozen tensor inventory drift");
    return result;
}

void validate_inventory(const TfliteModelInventory& inventory,
                        const std::vector<AuditMapping>& mappings) {
    if (inventory.schema_version != 3 || inventory.file_bytes != kCheckpointBytes ||
        inventory.subgraph_count != 1 || inventory.buffer_count != 377 ||
        inventory.tensors.size() != 373 || inventory.operators.size() != 175 ||
        inventory.inputs != std::vector<int32_t>{0} ||
        inventory.outputs != std::vector<int32_t>{372})
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter TFLite envelope drift");
    const auto& input = inventory.tensors.at(0);
    const auto& output = inventory.tensors.at(372);
    if (input.name != "input_29" || input.dtype != "F32" ||
        input.shape != std::vector<int64_t>({1,256,256,3}) ||
        output.name != "Identity" || output.dtype != "F32" ||
        output.shape != std::vector<int64_t>({1,256,256,6}))
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter input/output schema drift");
    const std::map<std::string, size_t> expected{
        {"ADD",25},{"CONV_2D",67},{"DEPTHWISE_CONV_2D",18},{"MUL",18},
        {"RESHAPE",20},{"RESIZE_BILINEAR",3},{"RESIZE_NEAREST_NEIGHBOR",3},
        {"SOFTMAX",6},{"SUM",6},{"TRANSPOSE",8},{"TRANSPOSE_CONV",1}};
    std::map<std::string, size_t> counts;
    for (const auto& operation : inventory.operators) {
        ++counts[operation.type];
        const int expected_version =
            (operation.type == "RESIZE_BILINEAR" ||
             operation.type == "RESIZE_NEAREST_NEIGHBOR" ||
             operation.type == "TRANSPOSE_CONV") ? 3 : 1;
        if (operation.version != expected_version)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "semantic segmenter operator version drift");
    }
    if (counts != expected)
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter operator inventory drift");
    std::map<std::string, const TfliteTensorRecord*> tensors;
    size_t f32_count = 0, i32_count = 0;
    uint64_t f32_bytes = 0, i32_bytes = 0;
    for (const auto& tensor : inventory.tensors) {
        if (!tensors.emplace(tensor.name, &tensor).second)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "duplicate semantic segmenter TFLite tensor name");
        if (tensor.data_bytes && tensor.dtype == "F32") {
            ++f32_count; f32_bytes += tensor.data_bytes;
        } else if (tensor.data_bytes && tensor.dtype == "I32") {
            ++i32_count; i32_bytes += tensor.data_bytes;
        }
    }
    if (f32_count != 181 || f32_bytes != kSourceConstantBytes ||
        i32_count != 16 || i32_bytes != 220)
        fail(ModelPackageErrorCode::PackageInvalid,
             "semantic segmenter constant inventory drift");
    for (const auto& mapping : mappings) {
        const auto found = tensors.find(mapping.source_name);
        if (found == tensors.end() || found->second->dtype != "F32" ||
            found->second->shape != mapping.source_shape ||
            found->second->data_offset != mapping.source_offset ||
            found->second->data_bytes != elements(mapping.source_shape) * 4)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "semantic segmenter fixed tensor range drift: " +
                 mapping.source_name);
    }
}

class ConvertedConstantSource final : public TensorSource {
public:
    ConvertedConstantSource(const fs::path& path,
                            const std::vector<AuditMapping>& mappings) {
        std::ifstream input(path, std::ios::binary);
        std::vector<uint8_t> file(static_cast<size_t>(kCheckpointBytes));
        input.read(reinterpret_cast<char*>(file.data()), file.size());
        if (!input) fail(ModelPackageErrorCode::PackageInvalid,
                         "cannot read semantic segmenter constants");
        for (const auto& mapping : mappings) {
            const uint64_t count = elements(mapping.source_shape);
            if (mapping.source_offset > file.size() || count * 4 >
                    file.size() - mapping.source_offset)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "semantic segmenter tensor range exceeds source");
            std::vector<float> source(static_cast<size_t>(count));
            std::memcpy(source.data(), file.data() + mapping.source_offset,
                        static_cast<size_t>(count * 4));
            std::vector<float> destination(static_cast<size_t>(count));
            if (mapping.transformation == "identity_bytes") {
                destination = std::move(source);
            } else if (mapping.transformation == "f32_ohwi_to_oihw") {
                const auto& s = mapping.source_shape;
                for (int64_t o=0;o<s[0];++o) for(int64_t i=0;i<s[3];++i)
                    for(int64_t y=0;y<s[1];++y) for(int64_t x=0;x<s[2];++x)
                        destination[((o*s[3]+i)*s[1]+y)*s[2]+x] =
                            source[((o*s[1]+y)*s[2]+x)*s[3]+i];
            } else if (mapping.transformation == "f32_1hwc_to_c1hw") {
                const auto& s = mapping.source_shape;
                for(int64_t c=0;c<s[3];++c) for(int64_t y=0;y<s[1];++y)
                    for(int64_t x=0;x<s[2];++x)
                        destination[(c*s[1]+y)*s[2]+x] =
                            source[(y*s[2]+x)*s[3]+c];
            } else if (mapping.transformation == "f32_ohwi_to_iohw") {
                const auto& s = mapping.source_shape;
                for(int64_t i=0;i<s[3];++i) for(int64_t o=0;o<s[0];++o)
                    for(int64_t y=0;y<s[1];++y) for(int64_t x=0;x<s[2];++x)
                        destination[((i*s[0]+o)*s[1]+y)*s[2]+x] =
                            source[((o*s[1]+y)*s[2]+x)*s[3]+i];
            }
            SourceTensorDescriptor descriptor{mapping.destination_name,DType::F32,
                "F32",mapping.destination_shape,0,0,count*4};
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
                 "semantic segmenter converted tensor read exceeds range");
        std::memcpy(destination,
            reinterpret_cast<const uint8_t*>(found->second.data()) + offset, bytes);
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
Json integer_array(const std::vector<int32_t>& values) {
    Json::Array output;
    for (int32_t value : values) output.push_back(integer(value));
    return Json(Json::Value(std::move(output)));
}
Json integer_array64(const std::vector<int64_t>& values) {
    Json::Array output;
    for (int64_t value : values) output.push_back(integer(value));
    return Json(Json::Value(std::move(output)));
}

int32_t read_i32(const std::vector<uint8_t>& file, uint64_t offset) {
    if (offset > file.size() || 4 > file.size() - offset)
        fail(ModelPackageErrorCode::PackageInvalid,
             "integer graph constant exceeds source range");
    uint32_t bits = static_cast<uint32_t>(file[offset]) |
        static_cast<uint32_t>(file[offset+1]) << 8 |
        static_cast<uint32_t>(file[offset+2]) << 16 |
        static_cast<uint32_t>(file[offset+3]) << 24;
    int32_t value = 0; std::memcpy(&value, &bits, sizeof(value)); return value;
}

Json graph(const TfliteModelInventory& inventory,
           const std::vector<AuditMapping>& mappings,
           const fs::path& source_path) {
    Json::Object bindings;
    for (const auto& mapping : mappings)
        bindings.emplace(mapping.destination_name, text(mapping.destination_name));
    Json::Array shapes;
    for (const auto& tensor : inventory.tensors)
        shapes.push_back(integer_array64(tensor.shape));
    Json::Array operators;
    for (const auto& operation : inventory.operators)
        operators.push_back(object({{"type",text(operation.type)},
            {"inputs",integer_array(operation.inputs)},
            {"outputs",integer_array(operation.outputs)},
            {"options",integer_array64(operation.options)}}));
    std::ifstream input(source_path, std::ios::binary);
    std::vector<uint8_t> file(static_cast<size_t>(kCheckpointBytes));
    input.read(reinterpret_cast<char*>(file.data()), file.size());
    if (!input) fail(ModelPackageErrorCode::PackageInvalid,
                     "cannot read semantic segmenter graph constants");
    Json::Object integer_constants;
    for (size_t index=0; index<inventory.tensors.size(); ++index) {
        const auto& tensor = inventory.tensors[index];
        if (tensor.dtype != "I32" || tensor.data_bytes == 0) continue;
        Json::Array values;
        for (uint64_t offset=0; offset<tensor.data_bytes; offset+=4)
            values.push_back(integer(read_i32(file,tensor.data_offset+offset)));
        integer_constants.emplace(std::to_string(index),
            Json(Json::Value(std::move(values))));
    }
    Json::Array primitives;
    for (const char* value : {"conv2d","conv_transpose2d","add","mul","reshape",
            "permute","reduce_sum","softmax","bilinear_resize2d",
            "nearest_resize","relu"}) primitives.push_back(text(value));
    return object({{"schema_version",integer(1)},
        {"kind",text("semantic_segmenter_2d")},{"entry_point",text("execute")},
        {"config",object({{"dtype",text("float32")},
            {"architecture",text("xenoformer_fpn")},{"input_height",integer(256)},
            {"input_width",integer(256)},{"class_count",integer(6)},
            {"tensor_count",integer(373)},{"operator_count",integer(175)}})},
        {"tensor_shapes",Json(Json::Value(std::move(shapes)))},
        {"integer_constants",Json(Json::Value(std::move(integer_constants)))},
        {"operators",Json(Json::Value(std::move(operators)))},
        // The fixed graph's six Softmax operators are internal attention
        // operations. Tensor 372 is the final unnormalized semantic score map.
        {"semantic_outputs",Json(Json::Value(Json::Array{text("semantic_logits")}))},
        {"required_primitives",Json(Json::Value(std::move(primitives)))},
        {"runtime_tensor_bindings",Json(Json::Value(std::move(bindings)))}});
}

Json metadata() {
    return object({{"architecture",text("segmenter-2d")},
        {"profile",text("component")},{"default_dtype",text("float32")},
        {"component_kind",text("semantic_segmenter_2d")},
        {"graph_identity",text("semantic_segmenter_2d.v1")},
        {"source_format",text("tflite-flatbuffer-fixed")},
        {"source_constant_dtype",text("float32")},
        {"native_constant_dtype",text("float32")}});
}

}  // namespace

FrozenComponentConversionResult convert_selfie_multiclass_component(
        const fs::path& source_directory, const fs::path& tensor_map,
        const fs::path& output,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "semantic segmenter conversion interrupted; source retained");
    const fs::path checkpoint = source_directory /
        "selfie_multiclass_256x256.tflite";
    std::error_code error;
    if (!fs::is_regular_file(checkpoint,error) || error)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "missing frozen semantic segmenter source");
    if (fs::file_size(checkpoint,error) != kCheckpointBytes || error ||
        sha256_file(checkpoint,progress) != kCheckpointSha256)
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "frozen semantic segmenter source identity mismatch");
    const auto mappings = load_map(tensor_map);
    const auto inventory = inspect_tflite_flatbuffer(checkpoint);
    validate_inventory(inventory,mappings);
    if (cancellation_requested && cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "semantic segmenter conversion interrupted; source retained");
    ConvertedConstantSource source(checkpoint,mappings);
    std::vector<TensorMapping> converted;
    for (const auto& mapping : mappings)
        converted.push_back({mapping.destination_name,DType::F32,
            mapping.destination_shape,"identity_bytes",mapping.destination_name,
            DType::F32,mapping.destination_shape,"semantic_segmenter_xenoformer",
            "weight"});
    if (fs::exists(output,error))
        fail(ModelPackageErrorCode::InstallFailed,
             "semantic segmenter component output already exists");
    fs::create_directories(output.parent_path(),error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create semantic segmenter output directory");
    static std::atomic<uint64_t> sequence{0};
    const fs::path temporary = output.parent_path() /
        (output.filename().string()+".partial-"+std::to_string(getpid())+"-"+
         std::to_string(sequence.fetch_add(1)));
    try {
        const VrmWriteResult result = write_vrm_streaming(
            temporary,"component","segmenter-2d",metadata(),
            graph(inventory,mappings,checkpoint),source,converted,
            cancellation_requested,progress);
        fs::rename(temporary,output,error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish semantic segmenter component");
        return {output,kCheckpointBytes,kSourceConstantBytes,181,result.file_size,
            result.largest_buffer_bytes,result.seconds,sha256_file(output),
            result.payload_blake2b128};
    } catch (...) { fs::remove(temporary,error); throw; }
}

}  // namespace vrhino::product
