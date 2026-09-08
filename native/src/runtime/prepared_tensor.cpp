#include "vrhino/prepared_tensor.h"

#include <tuple>

#include "vrhino/error.h"

namespace vrhino {

bool PreparedTensorKey::operator<(const PreparedTensorKey& other) const {
    return std::tie(operation, input_shape, input_strides, input_dtype,
                    input_device_type, input_device_index, input_content_hash,
                    input_identity,
                    parameter_words, target_dtype, target_backend,
                    target_device_index) <
           std::tie(other.operation, other.input_shape, other.input_strides,
                    other.input_dtype, other.input_device_type,
                    other.input_device_index, other.input_content_hash,
                    other.input_identity,
                    other.parameter_words, other.target_dtype,
                    other.target_backend, other.target_device_index);
}

PreparedTensorHandle PreparedTensorCache::prepare(
        const PreparedTensorKey& key, const Builder& builder) {
    require(!key.operation.empty(),
            "Prepared tensor operation semantic must not be empty");
    require(!key.target_backend.empty(),
            "Prepared tensor target backend must not be empty");
    if (entries_.find(key) != entries_.end()) {
        ++stats_.prepare_hits;
        return {key};
    }
    require(entries_.size() < kMaximumEntries,
            "Prepared tensor cache capacity reached");
    ++stats_.prepare_misses;
    PreparedTensorMaterialization materialized = builder();
    require(!materialized.tensors.empty(),
            "Prepared tensor builder returned no tensors");
    uint64_t resident_bytes = 0;
    for (const Tensor& tensor : materialized.tensors) {
        require(tensor.defined(),
                "Prepared tensor builder returned an undefined tensor");
        require(tensor.bytes() <= UINT64_MAX - resident_bytes,
                "Prepared tensor resident-byte accounting overflow");
        resident_bytes += tensor.bytes();
    }
    const uint64_t tensor_count = materialized.tensors.size();
    entries_.emplace(key, std::move(materialized.tensors));
    stats_.host_materializations += materialized.host_materializations;
    stats_.host_to_device_transfers += materialized.host_to_device_transfers;
    stats_.host_to_device_bytes += materialized.host_to_device_bytes;
    stats_.resident_entries = entries_.size();
    stats_.resident_tensors += tensor_count;
    stats_.resident_bytes += resident_bytes;
    return {key};
}

const std::vector<Tensor>& PreparedTensorCache::reuse(
        const PreparedTensorHandle& handle) {
    const auto found = entries_.find(handle.key);
    require(found != entries_.end(),
            "Prepared tensor handle is not resident in this cache");
    ++stats_.reuses;
    return found->second;
}

uint64_t hash_host_tensor_content(const Tensor& tensor) {
    require(tensor.defined() && tensor.device().is_host(),
            "Prepared tensor content hashing requires a host tensor");
    constexpr uint64_t offset = 1469598103934665603ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t value = offset;
    const auto* bytes = static_cast<const uint8_t*>(tensor.data());
    for (size_t index = 0; index < tensor.bytes(); ++index) {
        value ^= bytes[index];
        value *= prime;
    }
    return value;
}

}  // namespace vrhino
