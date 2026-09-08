#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cudnn.h>

#include "vrhino/tensor.h"

namespace vrhino {

// Backend-only descriptor for the shared Attention semantic. Dimensions are
// logical BHSD while input/output storage is contiguous BSHD. No architecture
// or model identity is accepted by this interface.
struct CudnnSdpaDescriptor {
    int64_t batch = 0;
    int64_t query_tokens = 0;
    int64_t key_tokens = 0;
    int64_t heads = 0;
    int64_t head_width = 0;
    float scale = 1.0f;
    bool causal = false;
    bool has_padding_mask = false;
    bool has_additive_bias = false;
    int bias_rank = 0;
    int64_t bias_dimensions[4] = {1, 1, 1, 1};
    int64_t bias_strides[4] = {1, 1, 1, 1};
};

// Exact, content-based canonical form for a broadcast Boolean BQK mask.  A
// mask is representable only when every batch is a rectangular prefix:
//   mask[b,q,k] == (q < query_lengths[b] && k < key_value_lengths[b]).
// No tensor/model/architecture identity participates in this decision.
struct PrefixMaskCanonicalization {
    bool representable = false;
    std::vector<int32_t> query_lengths;
    std::vector<int32_t> key_value_lengths;
    std::string reason;
};

PrefixMaskCanonicalization canonicalize_boolean_prefix_mask(
    const Tensor& host_mask, int64_t batch, int64_t query_tokens,
    int64_t key_tokens);

struct CudnnSdpaExecution {
    bool executed = false;
    size_t workspace_bytes = 0;
    std::string reason;
};

// Caches built cuDNN graphs/plans by the generic operation descriptor. A
// rejected descriptor is negatively cached and transparently uses the
// existing bounded Attention fallback.
class CudnnSdpaPlanCache {
public:
    explicit CudnnSdpaPlanCache(cudnnHandle_t handle);
    ~CudnnSdpaPlanCache();
    CudnnSdpaPlanCache(const CudnnSdpaPlanCache&) = delete;
    CudnnSdpaPlanCache& operator=(const CudnnSdpaPlanCache&) = delete;

    CudnnSdpaExecution execute(const CudnnSdpaDescriptor& descriptor,
                               const void* q, const void* k, const void* v,
                               const void* additive_bias,
                               const int32_t* query_lengths,
                               const int32_t* key_value_lengths,
                               void* output);
    uint64_t build_count() const;
    uint64_t cache_hit_count() const;
    size_t cached_plan_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vrhino
