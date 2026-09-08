#include "vrhino/product/tflite.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <type_traits>

#include "vrhino/product/model_package.h"

namespace vrhino::product {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,
                            "invalid bounded TFLite FlatBuffer: " + message);
}

class BufferView {
public:
    explicit BufferView(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw ModelPackageError(ModelPackageErrorCode::ArtifactMissing,
                                             "missing TFLite source");
        input.seekg(0, std::ios::end);
        const auto length = input.tellg();
        if (length < 8 || length > static_cast<std::streamoff>(64 * 1024 * 1024))
            invalid("file length is outside the bounded reader contract");
        bytes_.resize(static_cast<size_t>(length));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(bytes_.data()), length);
        if (!input) invalid("truncated file read");
        if (std::memcmp(bytes_.data() + 4, "TFL3", 4) != 0)
            invalid("missing TFL3 identifier");
    }

    size_t size() const noexcept { return bytes_.size(); }

    template <typename T> T scalar(size_t offset) const {
        static_assert(std::is_integral_v<T>);
        if (offset > bytes_.size() || sizeof(T) > bytes_.size() - offset)
            invalid("scalar exceeds file range");
        using U = std::make_unsigned_t<T>;
        U value = 0;
        for (size_t index = 0; index < sizeof(T); ++index)
            value |= static_cast<U>(bytes_[offset + index]) << (8 * index);
        T output;
        std::memcpy(&output, &value, sizeof(output));
        return output;
    }

    uint32_t float_bits(size_t offset) const { return scalar<uint32_t>(offset); }

    size_t reference(size_t location) const {
        const uint32_t relative = scalar<uint32_t>(location);
        if (relative == 0 || relative > bytes_.size() - location)
            invalid("invalid relative object offset");
        return location + relative;
    }

    const std::vector<uint8_t>& bytes() const noexcept { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
};

class Table {
public:
    Table(const BufferView& buffer, size_t position)
        : buffer_(buffer), position_(position) {
        const int32_t distance = buffer_.scalar<int32_t>(position_);
        if (distance == 0)
            invalid("invalid vtable distance");
        const int64_t signed_vtable = static_cast<int64_t>(position_) - distance;
        if (signed_vtable < 0 ||
            static_cast<uint64_t>(signed_vtable) > buffer_.size())
            invalid("invalid vtable distance");
        vtable_ = static_cast<size_t>(signed_vtable);
        vtable_bytes_ = buffer_.scalar<uint16_t>(vtable_);
        object_bytes_ = buffer_.scalar<uint16_t>(vtable_ + 2);
        if (vtable_bytes_ < 4 || object_bytes_ < 4 ||
            vtable_ + vtable_bytes_ > buffer_.size() ||
            position_ + object_bytes_ > buffer_.size())
            invalid("invalid table extent");
    }

    size_t field(uint16_t index) const {
        const size_t entry = vtable_ + 4 + 2 * static_cast<size_t>(index);
        if (entry + 2 > vtable_ + vtable_bytes_) return 0;
        const uint16_t relative = buffer_.scalar<uint16_t>(entry);
        if (relative == 0) return 0;
        if (relative >= object_bytes_) invalid("table field exceeds object");
        return position_ + relative;
    }

    template <typename T> T scalar(uint16_t index, T fallback = T{}) const {
        const size_t location = field(index);
        return location == 0 ? fallback : buffer_.scalar<T>(location);
    }

    size_t object(uint16_t index) const {
        const size_t location = field(index);
        return location == 0 ? 0 : buffer_.reference(location);
    }

    size_t vector(uint16_t index) const {
        return object(index);
    }

private:
    const BufferView& buffer_;
    size_t position_ = 0;
    size_t vtable_ = 0;
    uint16_t vtable_bytes_ = 0;
    uint16_t object_bytes_ = 0;
};

uint32_t vector_length(const BufferView& buffer, size_t vector) {
    if (vector == 0) return 0;
    return buffer.scalar<uint32_t>(vector);
}

size_t vector_data(const BufferView& buffer, size_t vector, size_t element_bytes) {
    const uint64_t bytes = static_cast<uint64_t>(vector_length(buffer, vector)) *
                           element_bytes;
    if (vector + 4 > buffer.size() || bytes > buffer.size() - (vector + 4))
        invalid("vector exceeds file range");
    return vector + 4;
}

size_t table_at(const BufferView& buffer, size_t vector, uint32_t index) {
    if (index >= vector_length(buffer, vector)) invalid("table vector index out of range");
    const size_t location = vector_data(buffer, vector, 4) + 4 * index;
    return buffer.reference(location);
}

std::string string_at(const BufferView& buffer, size_t position) {
    if (position == 0) return {};
    const uint32_t length = vector_length(buffer, position);
    const size_t data = vector_data(buffer, position, 1);
    if (data + length >= buffer.size() || buffer.bytes()[data + length] != 0)
        invalid("string is not zero terminated");
    return std::string(reinterpret_cast<const char*>(buffer.bytes().data() + data),
                       length);
}

std::vector<int32_t> int_vector(const BufferView& buffer, size_t vector) {
    std::vector<int32_t> result;
    const uint32_t length = vector_length(buffer, vector);
    const size_t data = vector_data(buffer, vector, 4);
    result.reserve(length);
    for (uint32_t index = 0; index < length; ++index)
        result.push_back(buffer.scalar<int32_t>(data + 4 * index));
    return result;
}

std::string tensor_type(int8_t value) {
    switch (value) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "I32";
        case 4: return "U8";
        case 9: return "I8";
        case 16: return "U32";
        default: invalid("unsupported tensor dtype " + std::to_string(value));
    }
}

uint64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16") return 2;
    if (dtype == "U8" || dtype == "I8") return 1;
    invalid("unknown tensor dtype");
}

std::string builtin_name(int32_t code) {
    switch (code) {
        case 0: return "ADD";
        case 2: return "CONCATENATION";
        case 3: return "CONV_2D";
        case 4: return "DEPTHWISE_CONV_2D";
        case 6: return "DEQUANTIZE";
        case 17: return "MAX_POOL_2D";
        case 18: return "MUL";
        case 19: return "RELU";
        case 22: return "RESHAPE";
        case 23: return "RESIZE_BILINEAR";
        case 25: return "SOFTMAX";
        case 34: return "PAD";
        case 39: return "TRANSPOSE";
        case 67: return "TRANSPOSE_CONV";
        case 74: return "SUM";
        case 97: return "RESIZE_NEAREST_NEIGHBOR";
        default: invalid("unsupported builtin operator " + std::to_string(code));
    }
}

std::vector<int64_t> parse_options(const BufferView& buffer,
                                   const Table& operation,
                                   const std::string& type) {
    const int8_t options_type = operation.scalar<int8_t>(3, 0);
    const size_t position = operation.object(4);
    if (type == "DEQUANTIZE" || type == "RELU" || type == "PAD" ||
        type == "TRANSPOSE") {
        if (options_type != 0 || position != 0)
            invalid(type + " unexpectedly carries builtin options");
        return {};
    }
    // Newer graphs may carry the reshape target exclusively as the second
    // input tensor instead of duplicating it in ReshapeOptions.
    if (type == "RESHAPE" && options_type == 0 && position == 0) return {};
    int8_t expected = 0;
    if (type == "CONV_2D") expected = 1;
    else if (type == "DEPTHWISE_CONV_2D") expected = 2;
    else if (type == "MAX_POOL_2D") expected = 5;
    else if (type == "CONCATENATION") expected = 10;
    else if (type == "ADD") expected = 11;
    else if (type == "RESHAPE") expected = 17;
    else if (type == "MUL") expected = 21;
    else if (type == "RESIZE_BILINEAR") expected = 15;
    else if (type == "SOFTMAX") expected = 9;
    else if (type == "TRANSPOSE_CONV") expected = 49;
    else if (type == "SUM") expected = 27;
    else if (type == "RESIZE_NEAREST_NEIGHBOR") expected = 74;
    if (position == 0 || options_type != expected)
        invalid(type + " builtin option table mismatch");
    const Table options(buffer, position);
    if (type == "CONV_2D")
        return {options.scalar<int8_t>(0), options.scalar<int32_t>(2, 1),
                options.scalar<int32_t>(1, 1), options.scalar<int32_t>(5, 1),
                options.scalar<int32_t>(4, 1), options.scalar<int8_t>(3)};
    if (type == "DEPTHWISE_CONV_2D")
        return {options.scalar<int8_t>(0), options.scalar<int32_t>(2, 1),
                options.scalar<int32_t>(1, 1), options.scalar<int32_t>(3, 1),
                options.scalar<int32_t>(6, 1), options.scalar<int32_t>(5, 1),
                options.scalar<int8_t>(4)};
    if (type == "MAX_POOL_2D")
        return {options.scalar<int8_t>(0), options.scalar<int32_t>(2, 1),
                options.scalar<int32_t>(1, 1), options.scalar<int32_t>(4, 1),
                options.scalar<int32_t>(3, 1), options.scalar<int8_t>(5)};
    if (type == "ADD" || type == "MUL")
        return {options.scalar<int8_t>(0)};
    if (type == "CONCATENATION")
        return {options.scalar<int32_t>(0), options.scalar<int8_t>(1)};
    if (type == "RESHAPE") {
        std::vector<int64_t> result;
        for (int32_t value : int_vector(buffer, options.vector(0)))
            result.push_back(value);
        return result;
    }
    if (type == "RESIZE_BILINEAR")
        return {options.scalar<uint8_t>(2, 0), options.scalar<uint8_t>(3, 0)};
    if (type == "RESIZE_NEAREST_NEIGHBOR")
        return {options.scalar<uint8_t>(0, 0), options.scalar<uint8_t>(1, 0)};
    if (type == "SOFTMAX") {
        const size_t beta = options.field(0);
        return {static_cast<int64_t>(beta == 0 ? 0x3f800000u :
                                     buffer.float_bits(beta))};
    }
    if (type == "SUM") return {options.scalar<uint8_t>(0, 0)};
    if (type == "TRANSPOSE_CONV")
        return {options.scalar<int8_t>(0), options.scalar<int32_t>(2, 1),
                options.scalar<int32_t>(1, 1), options.scalar<int8_t>(3, 0)};
    invalid("unhandled builtin option table");
}

}  // namespace

TfliteModelInventory inspect_tflite_flatbuffer(const std::filesystem::path& path) {
    const BufferView buffer(path);
    const Table model(buffer, buffer.reference(0));
    TfliteModelInventory result;
    result.file_bytes = buffer.size();
    result.schema_version = model.scalar<int32_t>(0);

    const size_t opcode_vector = model.vector(1);
    const size_t subgraph_vector = model.vector(2);
    const size_t buffer_vector = model.vector(4);
    result.subgraph_count = vector_length(buffer, subgraph_vector);
    result.buffer_count = vector_length(buffer, buffer_vector);
    if (result.schema_version != 3 || result.subgraph_count != 1 ||
        vector_length(buffer, opcode_vector) == 0 || result.buffer_count == 0)
        invalid("unsupported model envelope");

    struct Opcode { std::string name; int32_t version; };
    std::vector<Opcode> opcodes;
    for (uint32_t index = 0; index < vector_length(buffer, opcode_vector); ++index) {
        const Table opcode(buffer, table_at(buffer, opcode_vector, index));
        const std::string custom = string_at(buffer, opcode.object(1));
        if (!custom.empty()) invalid("custom operators are forbidden");
        int32_t code = opcode.scalar<int32_t>(3, -1);
        if (code < 0) code = opcode.scalar<int8_t>(0, 0);
        opcodes.push_back({builtin_name(code), opcode.scalar<int32_t>(2, 1)});
    }

    struct BufferRange { uint64_t offset; uint64_t bytes; };
    std::vector<BufferRange> ranges;
    ranges.reserve(result.buffer_count);
    for (uint32_t index = 0; index < result.buffer_count; ++index) {
        const Table item(buffer, table_at(buffer, buffer_vector, index));
        if (item.scalar<uint64_t>(1, 0) != 0 || item.scalar<uint64_t>(2, 0) != 0)
            invalid("external buffers are forbidden");
        const size_t data = item.vector(0);
        ranges.push_back({data == 0 ? 0 : vector_data(buffer, data, 1),
                          vector_length(buffer, data)});
    }

    const Table subgraph(buffer, table_at(buffer, subgraph_vector, 0));
    const size_t tensor_vector = subgraph.vector(0);
    for (uint32_t index = 0; index < vector_length(buffer, tensor_vector); ++index) {
        const Table tensor(buffer, table_at(buffer, tensor_vector, index));
        TfliteTensorRecord record;
        record.name = string_at(buffer, tensor.object(3));
        record.dtype = tensor_type(tensor.scalar<int8_t>(1));
        for (int32_t dimension : int_vector(buffer, tensor.vector(0))) {
            if (dimension < 0) invalid("negative tensor dimension");
            record.shape.push_back(dimension);
        }
        record.buffer_index = tensor.scalar<uint32_t>(2);
        if (record.buffer_index >= ranges.size()) invalid("tensor buffer index out of range");
        record.data_offset = ranges[record.buffer_index].offset;
        record.data_bytes = ranges[record.buffer_index].bytes;
        if (record.data_bytes != 0) {
            uint64_t elements = 1;
            for (int64_t dimension : record.shape) {
                if (dimension != 0 && elements >
                    std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(dimension))
                    invalid("tensor element count overflow");
                elements *= static_cast<uint64_t>(dimension);
            }
            if (elements > std::numeric_limits<uint64_t>::max() /
                               dtype_bytes(record.dtype) ||
                elements * dtype_bytes(record.dtype) != record.data_bytes)
                invalid("constant tensor byte count mismatch");
        }
        result.tensors.push_back(std::move(record));
    }
    result.inputs = int_vector(buffer, subgraph.vector(1));
    result.outputs = int_vector(buffer, subgraph.vector(2));
    for (int32_t index : result.inputs)
        if (index < 0 || static_cast<size_t>(index) >= result.tensors.size())
            invalid("input tensor index out of range");
    for (int32_t index : result.outputs)
        if (index < 0 || static_cast<size_t>(index) >= result.tensors.size())
            invalid("output tensor index out of range");

    const size_t operation_vector = subgraph.vector(3);
    for (uint32_t index = 0; index < vector_length(buffer, operation_vector); ++index) {
        const Table operation(buffer, table_at(buffer, operation_vector, index));
        const uint32_t opcode_index = operation.scalar<uint32_t>(0);
        if (opcode_index >= opcodes.size()) invalid("operator code index out of range");
        TfliteOperatorRecord record;
        record.type = opcodes[opcode_index].name;
        record.version = opcodes[opcode_index].version;
        record.inputs = int_vector(buffer, operation.vector(1));
        record.outputs = int_vector(buffer, operation.vector(2));
        for (int32_t tensor : record.inputs)
            if (tensor < -1 || static_cast<size_t>(tensor) >= result.tensors.size())
                invalid("operator input index out of range");
        for (int32_t tensor : record.outputs)
            if (tensor < 0 || static_cast<size_t>(tensor) >= result.tensors.size())
                invalid("operator output index out of range");
        record.options = parse_options(buffer, operation, record.type);
        result.operators.push_back(std::move(record));
    }
    return result;
}

}  // namespace vrhino::product
