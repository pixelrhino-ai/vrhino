#include "phase13_safetensors.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::phase13 {
namespace {

uint64_t little_u64(const uint8_t* bytes) {
    uint64_t value = 0;
    for (int index = 7; index >= 0; --index) value = (value << 8) | bytes[index];
    return value;
}

DType dtype(const std::string& value) {
    if (value == "F32") return DType::F32;
    if (value == "F16") return DType::F16;
    if (value == "BF16") return DType::BF16;
    if (value == "I64") return DType::I64;
    if (value == "I32") return DType::I32;
    if (value == "U8") return DType::U8;
    if (value == "BOOL") return DType::Bool;
    throw Error("Unsupported safetensors dtype: " + value);
}

std::string read_text(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "Cannot open JSON: " + path);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::string directory(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

}  // namespace

SafeTensorSet SafeTensorSet::single(const std::string& path) {
    SafeTensorSet result; result.add_file(path); return result;
}

SafeTensorSet SafeTensorSet::indexed(const std::string& index_path) {
    const Json index = Json::parse(read_text(index_path));
    std::map<std::string, std::string> owners;
    std::set<std::string> shards;
    for (const auto& [name, value] : index.at("weight_map").object()) {
        owners.emplace(name, value.string()); shards.insert(value.string());
    }
    SafeTensorSet result;
    const std::string root = directory(index_path);
    for (const std::string& shard : shards) result.add_file(root + "/" + shard, &owners);
    require(result.tensors_.size() == owners.size(), "Indexed safetensors tensor count mismatch");
    return result;
}

void SafeTensorSet::add_file(const std::string& path,
                             const std::map<std::string, std::string>* owners) {
    Mapping mapping; mapping.path = path;
    mapping.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(mapping.fd >= 0, "Cannot open safetensors: " + path);
    struct stat info{};
    require(fstat(mapping.fd, &info) == 0 && info.st_size >= 8, "Truncated safetensors: " + path);
    mapping.bytes = static_cast<size_t>(info.st_size);
    mapping.data = mmap(nullptr, mapping.bytes, PROT_READ, MAP_PRIVATE, mapping.fd, 0);
    require(mapping.data != MAP_FAILED, "safetensors mmap failed: " + path);
    const auto* bytes = static_cast<const uint8_t*>(mapping.data);
    const size_t header_length = static_cast<size_t>(little_u64(bytes));
    require(header_length > 0 && 8 + header_length <= mapping.bytes,
            "Invalid safetensors header length: " + path);
    const Json header = Json::parse(std::string(reinterpret_cast<const char*>(bytes + 8), header_length));
    const size_t data_start = 8 + header_length;
    const std::string filename = path.substr(path.find_last_of('/') + 1);
    for (const auto& [name, descriptor] : header.object()) {
        if (name == "__metadata__") continue;
        if (owners) {
            const auto found = owners->find(name);
            if (found == owners->end() || found->second != filename) continue;
        }
        std::vector<int64_t> shape;
        for (const Json& dimension : descriptor.at("shape").array()) shape.push_back(dimension.integer());
        const auto& offsets = descriptor.at("data_offsets").array();
        require(offsets.size() == 2, "Invalid safetensors data offsets");
        const size_t begin = static_cast<size_t>(offsets[0].integer());
        const size_t end = static_cast<size_t>(offsets[1].integer());
        const DType type = dtype(descriptor.at("dtype").string());
        require(begin <= end && data_start + end <= mapping.bytes &&
                end - begin == static_cast<size_t>(shape_numel(shape)) * dtype_size(type),
                "Invalid safetensors tensor range: " + name);
        Tensor tensor = Tensor::borrowed(const_cast<uint8_t*>(bytes + data_start + begin),
                                         end - begin, shape, type);
        require(tensors_.emplace(name, std::move(tensor)).second,
                "Duplicate safetensors tensor: " + name);
    }
    mappings_.push_back(std::move(mapping));
}

SafeTensorSet::~SafeTensorSet() { clear(); }
SafeTensorSet::SafeTensorSet(SafeTensorSet&& other) noexcept
    : mappings_(std::move(other.mappings_)), tensors_(std::move(other.tensors_)) {
    other.mappings_.clear(); other.tensors_.clear();
}
SafeTensorSet& SafeTensorSet::operator=(SafeTensorSet&& other) noexcept {
    if (this != &other) {
        clear(); mappings_ = std::move(other.mappings_); tensors_ = std::move(other.tensors_);
        other.mappings_.clear(); other.tensors_.clear();
    }
    return *this;
}
void SafeTensorSet::clear() {
    tensors_.clear();
    for (Mapping& mapping : mappings_) {
        if (mapping.data && mapping.data != MAP_FAILED) munmap(mapping.data, mapping.bytes);
        if (mapping.fd >= 0) close(mapping.fd);
    }
    mappings_.clear();
}
WeightMap SafeTensorSet::weights() const {
    std::map<std::string, const Tensor*> values;
    for (const auto& [name, tensor] : tensors_) values.emplace(name, &tensor);
    return WeightMap(std::move(values));
}
size_t SafeTensorSet::mapped_bytes() const {
    size_t total = 0; for (const Mapping& mapping : mappings_) total += mapping.bytes; return total;
}

}  // namespace vrhino::phase13
