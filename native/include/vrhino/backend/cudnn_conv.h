#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cudnn.h>

#include "vrhino/tensor.h"

namespace vrhino {

// Model-neutral CUDA convolution descriptor. Every tensor dimension and
// stride remains INT64 until a candidate has proved that narrowing is safe.
struct CudnnConvDescriptor {
    int device = 0;
    int compute_major = 0;
    DType io_dtype = DType::F32;
    DType compute_dtype = DType::F32;
    int spatial_rank = 0;
    std::vector<int64_t> x_dimensions;
    std::vector<int64_t> x_strides;
    std::vector<int64_t> w_dimensions;
    std::vector<int64_t> w_strides;
    std::vector<int64_t> y_dimensions;
    std::vector<int64_t> y_strides;
    std::vector<int64_t> padding;
    std::vector<int64_t> convolution_strides;
    std::vector<int64_t> dilation;
    int64_t groups = 1;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
};

// Legacy fixed-function descriptors accept int dimensions/strides and have a
// 2-Giga-element addressable-extent limit. This admission runs before any
// narrowing conversion.
bool cudnn_legacy_conv_admitted(const CudnnConvDescriptor& descriptor,
                                std::string* reason = nullptr);

// Semantic validation for the modern cuDNN Frontend/Backend INT64 candidate.
// Plan support is still checked fail-closed by cuDNN for the current device.
bool cudnn_backend_conv_admitted(const CudnnConvDescriptor& descriptor,
                                 std::string* reason = nullptr);

struct CudnnConvExecution {
    bool executed = false;
    bool cache_hit = false;
    size_t workspace_bytes = 0;
    double plan_build_seconds = 0.0;
    std::string reason;
};

class CudnnConvPlanCache {
public:
    explicit CudnnConvPlanCache(cudnnHandle_t handle);
    ~CudnnConvPlanCache();
    CudnnConvPlanCache(const CudnnConvPlanCache&) = delete;
    CudnnConvPlanCache& operator=(const CudnnConvPlanCache&) = delete;

    CudnnConvExecution execute(const CudnnConvDescriptor& descriptor,
                               const void* x, const void* weight,
                               void* output);
    uint64_t build_count() const;
    uint64_t cache_hit_count() const;
    size_t cached_plan_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vrhino
