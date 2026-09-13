#include "vrhino/product/converter.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif

#include "vrhino/error.h"
#ifdef _WIN32
#include "vrhino/product/windows_process.h"
#endif

namespace vrhino::product {
#ifdef _WIN32
using namespace windows_converter_io;
static_assert(sizeof(off_t) == 8 && sizeof(ssize_t) == 8);
#endif
namespace {

constexpr uint64_t kMaxSafeTensorHeader = 64ULL * 1024 * 1024;
constexpr size_t kVrmHeaderSize = 128;
constexpr size_t kVrmAlignment = 64;
constexpr size_t kCopyBufferBytes = 8ULL * 1024 * 1024;

[[noreturn]] void package_error(const ModelPackageErrorCode code,
                                const std::string& message) {
    throw ModelPackageError(code, message);
}

uint64_t little_u64(const uint8_t* bytes) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; --index) value = (value << 8) | bytes[index];
    return value;
}

void put_u16(uint8_t* output, const uint16_t value) {
    output[0] = static_cast<uint8_t>(value & 0xff);
    output[1] = static_cast<uint8_t>((value >> 8) & 0xff);
}

void put_u64(uint8_t* output, const uint64_t value) {
    for (int index = 0; index < 8; ++index)
        output[index] = static_cast<uint8_t>((value >> (8 * index)) & 0xff);
}

uint64_t align_up(const uint64_t value) {
    require(value <= std::numeric_limits<uint64_t>::max() - (kVrmAlignment - 1),
            "VRM size overflow");
    return (value + kVrmAlignment - 1) / kVrmAlignment * kVrmAlignment;
}

uint64_t checked_add(const uint64_t left, const uint64_t right,
                     const char* message) {
    require(right <= std::numeric_limits<uint64_t>::max() - left, message);
    return left + right;
}

uint64_t checked_numel(const std::vector<int64_t>& shape);

uint64_t checked_tensor_bytes(const std::vector<int64_t>& shape, const DType dtype) {
    const uint64_t elements = checked_numel(shape);
    require(elements <= std::numeric_limits<uint64_t>::max() / dtype_size(dtype),
            "Tensor byte size overflow");
    return elements * dtype_size(dtype);
}

DType safe_dtype(const std::string& value) {
    if (value == "F32") return DType::F32;
    if (value == "F16") return DType::F16;
    if (value == "BF16") return DType::BF16;
    if (value == "I64") return DType::I64;
    if (value == "I32") return DType::I32;
    if (value == "U8") return DType::U8;
    if (value == "I8") return DType::I8;
    if (value == "BOOL") return DType::Bool;
    package_error(ModelPackageErrorCode::PackageVersionUnsupported,
                  "unsupported safetensors dtype: " + value);
}

uint64_t checked_numel(const std::vector<int64_t>& shape) {
    uint64_t result = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "negative safetensors dimension");
        const uint64_t size = static_cast<uint64_t>(dimension);
        if (size != 0 && result > std::numeric_limits<uint64_t>::max() / size)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "safetensors shape element count overflow");
        result *= size;
    }
    return result;
}

void read_exact(const int fd, void* destination, const size_t bytes,
                const uint64_t offset, const std::string& context) {
    auto* output = static_cast<uint8_t*>(destination);
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t count = pread(fd, output + completed, bytes - completed,
                                    static_cast<off_t>(offset + completed));
        if (count <= 0)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "truncated read while " + context);
        completed += static_cast<size_t>(count);
    }
}

void write_exact(const int fd, const void* source, const size_t bytes,
                 const std::string& context) {
    const auto* input = static_cast<const uint8_t*>(source);
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t count = write(fd, input + completed, bytes - completed);
        if (count <= 0)
            package_error(ModelPackageErrorCode::InstallFailed,
                          "write failed while " + context);
        completed += static_cast<size_t>(count);
    }
}

void pwrite_exact(const int fd, const void* source, const size_t bytes,
                  const uint64_t offset, const std::string& context) {
    const auto* input = static_cast<const uint8_t*>(source);
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t count = pwrite(fd, input + completed, bytes - completed,
                                     static_cast<off_t>(offset + completed));
        if (count <= 0)
            package_error(ModelPackageErrorCode::InstallFailed,
                          "write failed while " + context);
        completed += static_cast<size_t>(count);
    }
}

std::string hex(const uint8_t* bytes, const size_t count) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t index = 0; index < count; ++index)
        output << std::setw(2) << static_cast<unsigned>(bytes[index]);
    return output.str();
}

class Blake2b128 {
public:
    Blake2b128() : state_(kIv) { state_[0] ^= 0x01010000U ^ 16U; }

    void update(const uint8_t* data, size_t length) {
        if (length == 0) return;
        if (buffered_ != 0) {
            const size_t fill = std::min(length, buffer_.size() - buffered_);
            std::memcpy(buffer_.data() + buffered_, data, fill);
            buffered_ += fill;
            data += fill;
            length -= fill;
            if (buffered_ == buffer_.size() && length != 0) {
                advance(buffer_.size());
                compress(buffer_.data(), false);
                buffered_ = 0;
            }
        }
        while (length > buffer_.size()) {
            advance(buffer_.size());
            compress(data, false);
            data += buffer_.size();
            length -= buffer_.size();
        }
        if (length != 0) {
            std::memcpy(buffer_.data() + buffered_, data, length);
            buffered_ += length;
        }
    }

    std::array<uint8_t, 16> finish() {
        advance(buffered_);
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                  buffer_.end(), 0);
        compress(buffer_.data(), true);
        std::array<uint8_t, 16> output{};
        for (int word = 0; word < 2; ++word)
            for (int byte = 0; byte < 8; ++byte)
                output[word * 8 + byte] =
                    static_cast<uint8_t>((state_[word] >> (8 * byte)) & 0xff);
        return output;
    }

private:
    static constexpr std::array<uint64_t, 8> kIv = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    static constexpr uint8_t kSigma[12][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
        {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
        {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
        {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
        {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
        {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
        {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
        {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
        {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    };

    static uint64_t rotate(const uint64_t value, const int bits) {
        return (value >> bits) | (value << (64 - bits));
    }

    void advance(const uint64_t bytes) {
        const uint64_t before = low_;
        low_ += bytes;
        if (low_ < before) ++high_;
    }

    void compress(const uint8_t* block, const bool last) {
        uint64_t message[16];
        for (int index = 0; index < 16; ++index)
            message[index] = little_u64(block + index * 8);
        uint64_t values[16];
        for (int index = 0; index < 8; ++index) {
            values[index] = state_[index];
            values[index + 8] = kIv[index];
        }
        values[12] ^= low_;
        values[13] ^= high_;
        if (last) values[14] = ~values[14];
        auto mix = [&](const int a, const int b, const int c, const int d,
                       const uint64_t x, const uint64_t y) {
            values[a] = values[a] + values[b] + x;
            values[d] = rotate(values[d] ^ values[a], 32);
            values[c] += values[d];
            values[b] = rotate(values[b] ^ values[c], 24);
            values[a] = values[a] + values[b] + y;
            values[d] = rotate(values[d] ^ values[a], 16);
            values[c] += values[d];
            values[b] = rotate(values[b] ^ values[c], 63);
        };
        for (int round = 0; round < 12; ++round) {
            const uint8_t* sigma = kSigma[round];
            mix(0,4,8,12,message[sigma[0]],message[sigma[1]]);
            mix(1,5,9,13,message[sigma[2]],message[sigma[3]]);
            mix(2,6,10,14,message[sigma[4]],message[sigma[5]]);
            mix(3,7,11,15,message[sigma[6]],message[sigma[7]]);
            mix(0,5,10,15,message[sigma[8]],message[sigma[9]]);
            mix(1,6,11,12,message[sigma[10]],message[sigma[11]]);
            mix(2,7,8,13,message[sigma[12]],message[sigma[13]]);
            mix(3,4,9,14,message[sigma[14]],message[sigma[15]]);
        }
        for (int index = 0; index < 8; ++index)
            state_[index] ^= values[index] ^ values[index + 8];
    }

    std::array<uint64_t, 8> state_{};
    std::array<uint8_t, 128> buffer_{};
    size_t buffered_ = 0;
    uint64_t low_ = 0;
    uint64_t high_ = 0;
};

void append_json_string(std::string& output, const std::string& value) {
    output += Json(Json::Value(value)).serialize();
}

std::string shape_json(const std::vector<int64_t>& shape) {
    std::string output = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) output.push_back(',');
        output += std::to_string(shape[index]);
    }
    output.push_back(']');
    return output;
}

std::string vrm_dtype_code(const DType dtype) {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::I64: return "i64";
        case DType::I32: return "i32";
        case DType::U8: return "u8";
        case DType::Bool: return "bool";
        case DType::I8: return "i8";
    }
    require(false, "Unsupported VRM tensor dtype");
    return {};
}

std::string safetensors_dtype_code(const DType dtype) {
    switch (dtype) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I64: return "I64";
        case DType::I32: return "I32";
        case DType::U8: return "U8";
        case DType::Bool: return "BOOL";
        case DType::I8: return "I8";
    }
    require(false, "Unsupported safetensors tensor dtype");
    return {};
}

std::string tensor_table(const std::vector<TensorMapping>& mappings,
                         const TensorSource& source,
                         std::vector<uint64_t>& relative_offsets,
                         uint64_t& final_relative) {
    std::string output = "{\"schema_version\":1,\"tensors\":[";
    uint64_t relative = 0;
    for (size_t index = 0; index < mappings.size(); ++index) {
        const TensorMapping& mapping = mappings[index];
        const auto source_found = source.tensors().find(mapping.source_name);
        require(source_found != source.tensors().end(),
                "Missing source tensor while building VRM table");
        const SourceTensorDescriptor& tensor = source_found->second;
        relative = align_up(relative);
        relative_offsets.push_back(relative);
        if (index != 0) output.push_back(',');
        output += "{\"alignment\":64,\"byte_length\":" +
                  std::to_string(tensor.byte_length) +
                  ",\"component\":";
        append_json_string(output, mapping.component);
        output += ",\"dtype\":";
        append_json_string(output, vrm_dtype_code(mapping.destination_dtype));
        output += ",\"layout\":\"contiguous\",\"name\":";
        append_json_string(output, mapping.destination_name);
        output += ",\"offset\":" + std::to_string(relative) +
                  ",\"quantization\":";
        if (mapping.quantization.is_null()) {
            output += "{\"block_size\":0,\"group\":null,\"packed_layout\":null,"
                      "\"scale_tensor\":null,\"type\":\"none\","
                      "\"zero_point_tensor\":null}";
        } else {
            require(mapping.quantization.is_object(),
                    "Tensor quantization metadata must be an object");
            output += canonical_json(mapping.quantization);
        }
        output += ",\"role\":";
        append_json_string(output, mapping.role);
        output += ",\"shape\":" + shape_json(mapping.destination_shape) + "}";
        relative = checked_add(relative, tensor.byte_length,
                               "VRM tensor payload size overflow");
    }
    output += "]}";
    final_relative = relative;
    return output;
}

void set_identifier(uint8_t* destination, const std::string& value) {
    require(value.size() <= 16, "VRM identifier is too long");
    for (const unsigned char character : value)
        require(character >= 0x20 && character <= 0x7e,
                "VRM identifier contains a non-ASCII byte");
    std::memcpy(destination, value.data(), value.size());
}

}  // namespace

SafeTensorReader::SafeTensorReader(const std::filesystem::path& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0)
        package_error(ModelPackageErrorCode::ArtifactMissing,
                      "cannot open safetensors source: " + path.string());
    try {
        struct stat information {};
        if (fstat(fd_, &information) != 0 || information.st_size < 8)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "truncated safetensors file: " + path.string());
        file_size_ = static_cast<uint64_t>(information.st_size);
        std::array<uint8_t, 8> prefix{};
        read_exact(fd_, prefix.data(), prefix.size(), 0, "reading safetensors header length");
        header_bytes_ = little_u64(prefix.data());
        if (header_bytes_ == 0 || header_bytes_ > kMaxSafeTensorHeader ||
            header_bytes_ > file_size_ - 8)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "invalid safetensors header length");
        std::string header(static_cast<size_t>(header_bytes_), '\0');
        read_exact(fd_, header.data(), header.size(), 8, "reading safetensors header");
        const Json root = Json::parse(header);
        if (!root.is_object())
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "safetensors header must be an object");
        data_offset_ = 8 + header_bytes_;
        if (const Json* value = root.find("__metadata__"); value != nullptr) {
            if (!value->is_object())
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "safetensors metadata must be an object");
            metadata_ = *value;
        }
        struct Range { uint64_t begin; uint64_t end; std::string name; };
        std::vector<Range> ranges;
        for (const auto& [name, descriptor] : root.object()) {
            if (name == "__metadata__") continue;
            if (!descriptor.is_object())
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "safetensors tensor descriptor is not an object: " + name);
            const std::string dtype_text = descriptor.at("dtype").string();
            const DType dtype = safe_dtype(dtype_text);
            std::vector<int64_t> shape;
            for (const Json& dimension : descriptor.at("shape").array())
                shape.push_back(dimension.integer());
            const uint64_t elements = checked_numel(shape);
            if (elements > std::numeric_limits<uint64_t>::max() / dtype_size(dtype))
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "safetensors tensor byte count overflow: " + name);
            const auto& offsets = descriptor.at("data_offsets").array();
            if (offsets.size() != 2 || !offsets[0].is_int() || !offsets[1].is_int() ||
                offsets[0].integer() < 0 || offsets[1].integer() < 0)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "invalid safetensors data offsets: " + name);
            const uint64_t begin = static_cast<uint64_t>(offsets[0].integer());
            const uint64_t end = static_cast<uint64_t>(offsets[1].integer());
            const uint64_t expected = elements * dtype_size(dtype);
            if (begin > end || end - begin != expected ||
                begin > file_size_ - data_offset_ || end > file_size_ - data_offset_)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "invalid safetensors tensor payload range: " + name);
            SourceTensorDescriptor tensor{name, dtype, dtype_text, std::move(shape),
                                          0, begin, expected};
            if (!tensors_.emplace(name, std::move(tensor)).second)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "duplicate safetensors tensor: " + name);
            ranges.push_back({begin, end, name});
        }
        if (tensors_.empty())
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "safetensors file contains no tensors");
        std::sort(ranges.begin(), ranges.end(), [](const Range& left, const Range& right) {
            return left.begin < right.begin;
        });
        uint64_t expected_begin = 0;
        for (const Range& range : ranges) {
            if (range.begin != expected_begin)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "non-canonical gap/overlap in safetensors payload near: " +
                                  range.name);
            expected_begin = range.end;
        }
        if (data_offset_ + expected_begin != file_size_)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "safetensors tensor table does not cover payload");
    } catch (const ModelPackageError&) {
        close(fd_);
        fd_ = -1;
        throw;
    } catch (const std::exception& error) {
        close(fd_);
        fd_ = -1;
        package_error(ModelPackageErrorCode::PackageInvalid,
                      "malformed safetensors header: " + std::string(error.what()));
    }
}

SafeTensorReader::~SafeTensorReader() {
    if (fd_ >= 0) close(fd_);
}

void SafeTensorReader::read_tensor(const SourceTensorDescriptor& tensor,
                                   const uint64_t relative_offset,
                                   void* destination, const size_t bytes) const {
    if (relative_offset > tensor.byte_length ||
        bytes > tensor.byte_length - relative_offset)
        package_error(ModelPackageErrorCode::PackageInvalid,
                      "safetensors tensor read exceeds declared range: " + tensor.name);
    read_exact(fd_, destination, bytes,
               data_offset_ + tensor.data_offset + relative_offset,
               "reading tensor " + tensor.name);
}

FrozenTensorSource::FrozenTensorSource(
        std::vector<std::filesystem::path> paths,
        std::map<std::string, SourceTensorDescriptor> tensors)
    : paths_(std::move(paths)), tensors_(std::move(tensors)) {
    if (paths_.empty())
        package_error(ModelPackageErrorCode::PackageInvalid,
                      "frozen tensor source has no files");
    try {
        descriptors_.reserve(paths_.size());
        file_sizes_.reserve(paths_.size());
        for (const std::filesystem::path& path : paths_) {
            const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor < 0)
                package_error(ModelPackageErrorCode::ArtifactMissing,
                              "cannot open frozen tensor source: " + path.string());
            descriptors_.push_back(descriptor);
            struct stat information {};
            if (fstat(descriptor, &information) != 0 || information.st_size < 0)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "cannot inspect frozen tensor source: " + path.string());
            file_sizes_.push_back(static_cast<uint64_t>(information.st_size));
        }
        if (tensors_.empty())
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "frozen tensor source has no tensors");
        for (const auto& [name, tensor] : tensors_) {
            if (name != tensor.name || tensor.source_index >= descriptors_.size() ||
                tensor.data_offset > file_sizes_.at(tensor.source_index) ||
                tensor.byte_length > file_sizes_.at(tensor.source_index) - tensor.data_offset)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "invalid frozen tensor range: " + name);
            const uint64_t elements = checked_numel(tensor.shape);
            if (elements > std::numeric_limits<uint64_t>::max() / dtype_size(tensor.dtype) ||
                elements * dtype_size(tensor.dtype) != tensor.byte_length)
                package_error(ModelPackageErrorCode::PackageInvalid,
                              "frozen tensor byte count mismatch: " + name);
        }
    } catch (...) {
        for (const int descriptor : descriptors_) close(descriptor);
        descriptors_.clear();
        throw;
    }
}

FrozenTensorSource::~FrozenTensorSource() {
    for (const int descriptor : descriptors_) close(descriptor);
}

void FrozenTensorSource::read_tensor(const SourceTensorDescriptor& tensor,
                                     const uint64_t relative_offset,
                                     void* destination, const size_t bytes) const {
    if (tensor.source_index >= descriptors_.size() ||
        relative_offset > tensor.byte_length ||
        bytes > tensor.byte_length - relative_offset)
        package_error(ModelPackageErrorCode::PackageInvalid,
                      "frozen tensor read exceeds declared range: " + tensor.name);
    read_exact(descriptors_.at(tensor.source_index), destination, bytes,
               tensor.data_offset + relative_offset,
               "reading frozen tensor " + tensor.name);
}

std::string canonical_json(const Json& value) {
    return value.serialize();
}

void require_conversion_disk_space(const uint64_t required_bytes,
                                   const uint64_t available_bytes,
                                   const uint64_t safety_margin_bytes,
                                   const std::filesystem::path& cache_root) {
    if (available_bytes < required_bytes ||
        available_bytes - required_bytes < safety_margin_bytes) {
        const uint64_t minimum = required_bytes >
                std::numeric_limits<uint64_t>::max() - safety_margin_bytes
            ? std::numeric_limits<uint64_t>::max()
            : required_bytes + safety_margin_bytes;
        const std::string location = cache_root.empty()
            ? "cache filesystem"
            : "cache root " + cache_root.string();
        package_error(ModelPackageErrorCode::InsufficientDiskSpace,
                      location + " has " + std::to_string(available_bytes) +
                          " bytes available; at least " + std::to_string(minimum) +
                          " bytes are required for native conversion and installation");
    }
}

VrmWriteResult write_vrm_streaming(
        const std::filesystem::path& output,
        const std::string& profile_id,
        const std::string& architecture_id,
        const Json& metadata,
        const Json& graph,
        const TensorSource& source,
        const std::vector<TensorMapping>& input_mappings,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<TensorMapping> mappings = input_mappings;
    std::sort(mappings.begin(), mappings.end(), [](const TensorMapping& left,
                                                   const TensorMapping& right) {
        return left.destination_name < right.destination_name;
    });
    for (size_t index = 0; index < mappings.size(); ++index) {
        if (index != 0 && mappings[index - 1].destination_name ==
                              mappings[index].destination_name)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "duplicate VRM destination tensor: " +
                              mappings[index].destination_name);
        const auto source_found = source.tensors().find(mappings[index].source_name);
        if (source_found == source.tensors().end())
            package_error(ModelPackageErrorCode::ArtifactMissing,
                          "missing mapped source tensor: " + mappings[index].source_name);
        const SourceTensorDescriptor& tensor = source_found->second;
        if (tensor.dtype != mappings[index].source_dtype ||
            tensor.shape != mappings[index].source_shape ||
            mappings[index].source_dtype != mappings[index].destination_dtype ||
            mappings[index].source_shape != mappings[index].destination_shape ||
            mappings[index].transformation != "identity_bytes")
            package_error(ModelPackageErrorCode::PackageVersionUnsupported,
                          "unsupported tensor mapping/layout drift: " + tensor.name);
        if (mappings[index].quantization.is_null() &&
            tensor.byte_length != checked_tensor_bytes(tensor.shape, tensor.dtype))
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "source tensor byte count mismatch: " + tensor.name);
    }

    const std::string metadata_bytes = canonical_json(metadata);
    std::vector<uint64_t> relative_offsets;
    relative_offsets.reserve(mappings.size());
    uint64_t final_relative = 0;
    const std::string table_bytes =
        tensor_table(mappings, source, relative_offsets, final_relative);
    const std::string graph_bytes = canonical_json(graph);
    const uint64_t metadata_offset = kVrmHeaderSize;
    const uint64_t table_offset = checked_add(metadata_offset, metadata_bytes.size(),
                                              "VRM metadata size overflow");
    const uint64_t graph_offset = checked_add(table_offset, table_bytes.size(),
                                              "VRM tensor table size overflow");
    const uint64_t data_offset = align_up(checked_add(
        graph_offset, graph_bytes.size(), "VRM graph size overflow"));
    const uint64_t file_size = checked_add(data_offset, final_relative,
                                           "VRM file size overflow");

    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    if (error)
        package_error(ModelPackageErrorCode::CacheError,
                      "cannot create conversion staging directory: " + error.message());
    const int fd = open(output.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
        package_error(ModelPackageErrorCode::InstallFailed,
                      "cannot create conversion output: " + output.string());
    try {
        std::array<uint8_t, kVrmHeaderSize> empty_header{};
        write_exact(fd, empty_header.data(), empty_header.size(), "reserving VRM header");
        if (progress) progress(empty_header.size());
        Blake2b128 digest;
        auto write_payload = [&](const void* data, const size_t bytes,
                                 const std::string& context) {
            write_exact(fd, data, bytes, context);
            digest.update(static_cast<const uint8_t*>(data), bytes);
            if (progress) progress(bytes);
        };
        write_payload(metadata_bytes.data(), metadata_bytes.size(), "writing VRM metadata");
        write_payload(table_bytes.data(), table_bytes.size(), "writing VRM tensor table");
        write_payload(graph_bytes.data(), graph_bytes.size(), "writing VRM graph");
        std::vector<uint8_t> zeros(kCopyBufferBytes, 0);
        uint64_t current = graph_offset + graph_bytes.size();
        while (current < data_offset) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(zeros.size(), data_offset - current));
            write_payload(zeros.data(), count, "writing VRM section padding");
            current += count;
        }
        std::vector<uint8_t> buffer(kCopyBufferBytes);
        uint64_t current_relative = 0;
        for (size_t index = 0; index < mappings.size(); ++index) {
            if (cancellation_requested && cancellation_requested())
                package_error(ModelPackageErrorCode::Cancelled,
                              "Conversion interrupted. Downloaded source retained.");
            const uint64_t tensor_offset = relative_offsets[index];
            while (current_relative < tensor_offset) {
                const size_t count = static_cast<size_t>(
                    std::min<uint64_t>(zeros.size(), tensor_offset - current_relative));
                write_payload(zeros.data(), count, "writing VRM tensor padding");
                current_relative += count;
            }
            const SourceTensorDescriptor& tensor =
                source.tensors().at(mappings[index].source_name);
            uint64_t copied = 0;
            while (copied < tensor.byte_length) {
                if (cancellation_requested && cancellation_requested())
                    package_error(ModelPackageErrorCode::Cancelled,
                                  "Conversion interrupted. Downloaded source retained.");
                const size_t count = static_cast<size_t>(
                    std::min<uint64_t>(buffer.size(), tensor.byte_length - copied));
                source.read_tensor(tensor, copied, buffer.data(), count);
                write_payload(buffer.data(), count, "writing VRM tensor payload");
                copied += count;
            }
            current_relative += tensor.byte_length;
        }
        if (data_offset + current_relative != file_size)
            package_error(ModelPackageErrorCode::InstallFailed,
                          "VRM writer size invariant failed");
        const std::array<uint8_t, 16> checksum = digest.finish();
        std::array<uint8_t, kVrmHeaderSize> header{};
        const std::array<uint8_t, 8> magic = {'V','R','H','I','N','O',0,1};
        std::copy(magic.begin(), magic.end(), header.begin());
        put_u16(header.data() + 8, 0);
        put_u16(header.data() + 10, 1);
        header[12] = 1;
        header[13] = 6;
        put_u16(header.data() + 14, 0);
        set_identifier(header.data() + 16, profile_id);
        set_identifier(header.data() + 32, architecture_id);
        put_u64(header.data() + 48, metadata_offset);
        put_u64(header.data() + 56, metadata_bytes.size());
        put_u64(header.data() + 64, table_offset);
        put_u64(header.data() + 72, table_bytes.size());
        put_u64(header.data() + 80, graph_offset);
        put_u64(header.data() + 88, graph_bytes.size());
        put_u64(header.data() + 96, data_offset);
        put_u64(header.data() + 104, file_size);
        std::copy(checksum.begin(), checksum.end(), header.begin() + 112);
        pwrite_exact(fd, header.data(), header.size(), 0, "publishing VRM header");
        if (fsync(fd) != 0)
            package_error(ModelPackageErrorCode::InstallFailed,
                          "cannot sync converted VRM");
        close(fd);
        return VrmWriteResult{
            file_size,
            static_cast<uint64_t>(mappings.size()),
            kCopyBufferBytes,
            static_cast<uint64_t>(metadata_bytes.size()),
            static_cast<uint64_t>(table_bytes.size()),
            static_cast<uint64_t>(graph_bytes.size()),
            data_offset,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
            hex(checksum.data(), checksum.size()),
        };
    } catch (...) {
        close(fd);
        std::filesystem::remove(output, error);
        throw;
    }
}

SafeTensorWriteResult write_safetensors_streaming(
        const std::filesystem::path& output,
        const TensorSource& source,
        const std::vector<TensorMapping>& input_mappings,
        const std::vector<std::pair<std::string, std::string>>& metadata,
        const std::function<bool()>& cancellation_requested,
        const WorkProgressCallback& progress) {
    std::vector<TensorMapping> mappings = input_mappings;
    std::sort(mappings.begin(), mappings.end(), [](const TensorMapping& left,
                                                   const TensorMapping& right) {
        return left.destination_name < right.destination_name;
    });
    std::string header = "{\"__metadata__\":{";
    for (size_t index = 0; index < metadata.size(); ++index) {
        if (index != 0) header.push_back(',');
        append_json_string(header, metadata[index].first);
        header.push_back(':');
        append_json_string(header, metadata[index].second);
    }
    header.push_back('}');
    uint64_t relative = 0;
    for (size_t index = 0; index < mappings.size(); ++index) {
        const TensorMapping& mapping = mappings[index];
        if (index != 0 && mappings[index - 1].destination_name == mapping.destination_name)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "duplicate safetensors destination tensor: " +
                              mapping.destination_name);
        const auto found = source.tensors().find(mapping.source_name);
        if (found == source.tensors().end())
            package_error(ModelPackageErrorCode::ArtifactMissing,
                          "missing mapped source tensor: " + mapping.source_name);
        const SourceTensorDescriptor& tensor = found->second;
        if (tensor.dtype != mapping.source_dtype ||
            tensor.shape != mapping.source_shape ||
            mapping.source_dtype != mapping.destination_dtype ||
            mapping.source_shape != mapping.destination_shape ||
            mapping.transformation != "identity_bytes")
            package_error(ModelPackageErrorCode::PackageVersionUnsupported,
                          "unsupported safetensors mapping/layout drift: " + tensor.name);
        if (relative > std::numeric_limits<uint64_t>::max() - tensor.byte_length)
            package_error(ModelPackageErrorCode::PackageInvalid,
                          "safetensors output size overflow");
        header.push_back(',');
        append_json_string(header, mapping.destination_name);
        header += ":{\"dtype\":";
        append_json_string(header, safetensors_dtype_code(mapping.destination_dtype));
        header += ",\"shape\":" + shape_json(mapping.destination_shape) +
                  ",\"data_offsets\":[" + std::to_string(relative) + "," +
                  std::to_string(relative + tensor.byte_length) + "]}";
        relative += tensor.byte_length;
    }
    header.push_back('}');
    while (header.size() % 8 != 0) header.push_back(' ');
    if (header.size() > kMaxSafeTensorHeader)
        package_error(ModelPackageErrorCode::PackageInvalid,
                      "generated safetensors header is too large");

    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    if (error)
        package_error(ModelPackageErrorCode::CacheError,
                      "cannot create conversion staging directory: " + error.message());
    const int fd = open(output.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (fd < 0)
        package_error(ModelPackageErrorCode::InstallFailed,
                      "cannot create safetensors output: " + output.string());
    try {
        std::array<uint8_t, 8> prefix{};
        put_u64(prefix.data(), header.size());
        write_exact(fd, prefix.data(), prefix.size(), "writing safetensors header length");
        write_exact(fd, header.data(), header.size(), "writing safetensors header");
        if (progress) progress(prefix.size() + header.size());
        std::vector<uint8_t> buffer(kCopyBufferBytes);
        for (const TensorMapping& mapping : mappings) {
            if (cancellation_requested && cancellation_requested())
                package_error(ModelPackageErrorCode::Cancelled,
                              "Conversion interrupted. Downloaded source retained.");
            const SourceTensorDescriptor& tensor = source.tensors().at(mapping.source_name);
            uint64_t copied = 0;
            while (copied < tensor.byte_length) {
                if (cancellation_requested && cancellation_requested())
                    package_error(ModelPackageErrorCode::Cancelled,
                                  "Conversion interrupted. Downloaded source retained.");
                const size_t count = static_cast<size_t>(
                    std::min<uint64_t>(buffer.size(), tensor.byte_length - copied));
                source.read_tensor(tensor, copied, buffer.data(), count);
                write_exact(fd, buffer.data(), count, "writing safetensors payload");
                if (progress) progress(count);
                copied += count;
            }
        }
        if (fsync(fd) != 0)
            package_error(ModelPackageErrorCode::InstallFailed,
                          "cannot sync converted safetensors");
        close(fd);
        return SafeTensorWriteResult{
            8 + static_cast<uint64_t>(header.size()) + relative,
            static_cast<uint64_t>(mappings.size()),
            kCopyBufferBytes,
        };
    } catch (...) {
        close(fd);
        std::filesystem::remove(output, error);
        throw;
    }
}

std::filesystem::path discover_converter_spec_root(
        const std::filesystem::path& executable_path) {
#ifdef _WIN32
    if (const wchar_t* configured = _wgetenv(L"VRHINO_CONVERTER_SPEC_ROOT"); configured && *configured)
        return configured;
#else
    if (const char* configured = std::getenv("VRHINO_CONVERTER_SPEC_ROOT");
        configured != nullptr && *configured != '\0')
        return configured;
#endif
    std::error_code error;
    std::filesystem::path executable = executable_path;
#ifdef _WIN32
    if (executable.empty()) executable = windows_process::executable_path();
#else
    if (executable.empty()) executable = std::filesystem::read_symlink("/proc/self/exe", error);
#endif
    if (error || executable.empty())
        package_error(ModelPackageErrorCode::CacheError,
                      "cannot discover converter specification root");
    executable = std::filesystem::weakly_canonical(executable, error);
    if (error) executable = std::filesystem::absolute(executable);
    const std::array<std::filesystem::path, 2> candidates = {
        executable.parent_path() / "share/vrhino/converters",
        executable.parent_path() / "../share/vrhino/converters",
    };
    for (const auto& candidate : candidates)
        if (std::filesystem::is_directory(candidate))
            return std::filesystem::canonical(candidate);
    package_error(ModelPackageErrorCode::CacheError,
                  "converter specifications are not installed beside the VRhino executable");
}

}  // namespace vrhino::product
