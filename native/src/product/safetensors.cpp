#include "vrhino/product/safetensors.h"

#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::product {
namespace {

uint64_t little_u64(const uint8_t* bytes) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; --index) value = (value << 8) | bytes[index];
    return value;
}

DType safetensors_dtype(const std::string& value) {
    if (value == "F32") return DType::F32;
    if (value == "F16") return DType::F16;
    if (value == "BF16") return DType::BF16;
    if (value == "I64") return DType::I64;
    if (value == "I32") return DType::I32;
    if (value == "U8") return DType::U8;
    if (value == "I8") return DType::I8;
    if (value == "BOOL") return DType::Bool;
    throw Error("Unsupported safetensors dtype: " + value);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Cannot open JSON: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

}  // namespace

SafeTensorAsset SafeTensorAsset::single(const std::filesystem::path& path) {
    SafeTensorAsset result;
    result.add_file(path, path.filename().string());
    return result;
}

SafeTensorAsset SafeTensorAsset::indexed(
        const std::filesystem::path& index_path,
        const std::map<std::string, std::filesystem::path>& shard_paths) {
    const Json index = Json::parse(read_text(index_path));
    std::map<std::string, std::string> owners;
    std::set<std::string> required_shards;
    for (const auto& [name, value] : index.at("weight_map").object()) {
        owners.emplace(name, value.string());
        required_shards.insert(value.string());
    }
    SafeTensorAsset result;
    for (const std::string& shard : required_shards) {
        const auto found = shard_paths.find(shard);
        require(found != shard_paths.end(), "Package does not resolve safetensors shard: " + shard);
        result.add_file(found->second, shard, &owners);
    }
    require(result.tensors_.size() == owners.size(),
            "Indexed safetensors tensor count mismatch");
    return result;
}

void SafeTensorAsset::add_file(
        const std::filesystem::path& path, const std::string& logical_name,
        const std::map<std::string, std::string>* owners) {
    Mapping mapping;
    mapping.path = path;
    mapping.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(mapping.fd >= 0, "Cannot open safetensors: " + path.string());
    struct stat info {};
    require(fstat(mapping.fd, &info) == 0 && info.st_size >= 8,
            "Truncated safetensors: " + path.string());
    require(static_cast<uintmax_t>(info.st_size) <=
                std::numeric_limits<size_t>::max(),
            "safetensors file is too large for this host");
    mapping.bytes = static_cast<size_t>(info.st_size);
    mapping.data = mmap(nullptr, mapping.bytes, PROT_READ, MAP_PRIVATE, mapping.fd, 0);
    require(mapping.data != MAP_FAILED, "safetensors mmap failed: " + path.string());
    const auto* bytes = static_cast<const uint8_t*>(mapping.data);
    const uint64_t header_length_u64 = little_u64(bytes);
    require(header_length_u64 > 0 && header_length_u64 <= mapping.bytes - 8,
            "Invalid safetensors header length: " + path.string());
    const size_t header_length = static_cast<size_t>(header_length_u64);
    const Json header = Json::parse(
        std::string(reinterpret_cast<const char*>(bytes + 8), header_length));
    const size_t data_start = 8 + header_length;
    for (const auto& [name, descriptor] : header.object()) {
        if (name == "__metadata__") continue;
        if (owners) {
            const auto found = owners->find(name);
            if (found == owners->end() || found->second != logical_name) continue;
        }
        std::vector<int64_t> shape;
        for (const Json& dimension : descriptor.at("shape").array())
            shape.push_back(dimension.integer());
        const auto& offsets = descriptor.at("data_offsets").array();
        require(offsets.size() == 2 && offsets[0].is_int() && offsets[1].is_int() &&
                    offsets[0].integer() >= 0 && offsets[1].integer() >= 0,
                "Invalid safetensors data offsets");
        const size_t begin = static_cast<size_t>(offsets[0].integer());
        const size_t end = static_cast<size_t>(offsets[1].integer());
        const DType dtype = safetensors_dtype(descriptor.at("dtype").string());
        const int64_t elements = shape_numel(shape);
        require(static_cast<uint64_t>(elements) <=
                    std::numeric_limits<size_t>::max() / dtype_size(dtype),
                "safetensors tensor byte count overflow: " + name);
        const size_t expected = static_cast<size_t>(elements) * dtype_size(dtype);
        require(begin <= end && end <= mapping.bytes - data_start &&
                    end - begin == expected,
                "Invalid safetensors tensor range: " + name);
        Tensor tensor = Tensor::borrowed(const_cast<uint8_t*>(bytes + data_start + begin),
                                         end - begin, shape, dtype);
        require(tensors_.emplace(name, std::move(tensor)).second,
                "Duplicate safetensors tensor: " + name);
    }
    mappings_.push_back(std::move(mapping));
}

SafeTensorAsset::~SafeTensorAsset() { clear(); }
SafeTensorAsset::SafeTensorAsset(SafeTensorAsset&& other) noexcept
    : mappings_(std::move(other.mappings_)), tensors_(std::move(other.tensors_)) {
    other.mappings_.clear();
    other.tensors_.clear();
}
SafeTensorAsset& SafeTensorAsset::operator=(SafeTensorAsset&& other) noexcept {
    if (this != &other) {
        clear();
        mappings_ = std::move(other.mappings_);
        tensors_ = std::move(other.tensors_);
        other.mappings_.clear();
        other.tensors_.clear();
    }
    return *this;
}
void SafeTensorAsset::clear() {
    tensors_.clear();
    for (Mapping& mapping : mappings_) {
        if (mapping.data != nullptr && mapping.data != MAP_FAILED)
            munmap(mapping.data, mapping.bytes);
        if (mapping.fd >= 0) close(mapping.fd);
    }
    mappings_.clear();
}
WeightMap SafeTensorAsset::weights() const {
    std::map<std::string, const Tensor*> values;
    for (const auto& [name, tensor] : tensors_) values.emplace(name, &tensor);
    return WeightMap(std::move(values));
}
size_t SafeTensorAsset::mapped_bytes() const {
    size_t total = 0;
    for (const Mapping& mapping : mappings_) {
        require(mapping.bytes <= std::numeric_limits<size_t>::max() - total,
                "Mapped safetensors byte count overflow");
        total += mapping.bytes;
    }
    return total;
}

}  // namespace vrhino::product
