#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "vrhino/tensor.h"

namespace vrhino {

// A complete, model-neutral identity for a tensor whose value is invariant
// across repeated graph evaluations.  The operation and parameter words
// describe mathematics; the input fields protect against shape/content reuse;
// and the target fields protect backend, device, and precision boundaries.
struct PreparedTensorKey {
    std::string operation;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> input_strides;
    DType input_dtype = DType::F32;
    DeviceType input_device_type = DeviceType::CPU;
    int32_t input_device_index = 0;
    uintptr_t input_identity = 0;
    uint64_t input_content_hash = 0;
    std::vector<uint64_t> parameter_words;
    DType target_dtype = DType::F32;
    std::string target_backend;
    int32_t target_device_index = 0;

    bool operator<(const PreparedTensorKey& other) const;
};

struct PreparedTensorHandle {
    PreparedTensorKey key;
};

struct PreparedTensorMaterialization {
    std::vector<Tensor> tensors;
    uint64_t host_materializations = 0;
    uint64_t host_to_device_transfers = 0;
    uint64_t host_to_device_bytes = 0;
};

struct PreparedTensorCacheStats {
    uint64_t prepare_hits = 0;
    uint64_t prepare_misses = 0;
    uint64_t reuses = 0;
    uint64_t host_materializations = 0;
    uint64_t host_to_device_transfers = 0;
    uint64_t host_to_device_bytes = 0;
    uint64_t resident_entries = 0;
    uint64_t resident_tensors = 0;
    uint64_t resident_bytes = 0;
};

class PreparedTensorCache {
public:
    using Builder = std::function<PreparedTensorMaterialization()>;

    PreparedTensorHandle prepare(const PreparedTensorKey& key,
                                 const Builder& builder);
    const std::vector<Tensor>& reuse(const PreparedTensorHandle& handle);
    const PreparedTensorCacheStats& stats() const { return stats_; }

private:
    static constexpr size_t kMaximumEntries = 64;
    std::map<PreparedTensorKey, std::vector<Tensor>> entries_;
    PreparedTensorCacheStats stats_;
};

uint64_t hash_host_tensor_content(const Tensor& tensor);

}  // namespace vrhino
