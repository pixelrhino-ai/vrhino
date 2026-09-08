#pragma once

#include "vrhino/backend.h"

namespace vrhino {

class CudaBackend final : public Backend {
public:
    struct Impl;
    CudaBackend();
    ~CudaBackend() override;
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    std::string name() const override;
    int device_count() const override;
    DeviceCapability device_capability(int device = 0) const override;
    BackendMemoryCapabilities memory_capabilities() const override;
    bool supports(DType dtype) const override;
    void set_execution_dtype(DType dtype) override;
    DType execution_dtype() const override;
    Tensor allocate_host(const std::vector<int64_t>&, DType) override;
    Tensor allocate_device(const std::vector<int64_t>&, DType) override;
    TransferFence copy(const Tensor&, Tensor&, CopyMode = CopyMode::Synchronous) override;
    TransferFence create_fence(int device = 0) override;
    void record_fence(TransferFence) override;
    void wait_fence(TransferFence) override;
    bool query_fence(TransferFence) override;
    void destroy_fence(TransferFence) override;
    Tensor copy_to_device(const Tensor& input, DType dtype) override;
    Tensor copy_to_host(const Tensor& input) override;
    void synchronize() override;
    size_t peak_device_bytes() const override;
    size_t weight_upload_bytes() const override;
    double weight_upload_seconds() const override;
    void enable_profiling(bool enabled) override;
    bool profiling_enabled() const override;
    std::map<std::string, ProfileStat> profile_stats() override;
    void profile_region_begin(const std::string& name) override;
    void profile_region_end() override;
    uint64_t weight_cache_hits() const override;
    uint64_t weight_cache_misses() const override;
    size_t weight_cache_resident_bytes() const override;
    size_t weight_cache_capacity_bytes() const override;
    void enable_weight_cache(bool enabled) override;
    void enable_true_quant_compute(bool enabled) override;
    bool true_quant_compute_enabled() const override;
    QuantComputeStats quant_compute_stats() const override;
    void reset_quant_compute_stats() override;
    void configure_memory_runtime(const MemoryBudget&, const MemoryRuntimeOptions&) override;
    bool memory_runtime_enabled() const override;
    void set_vrm_mapped_bytes(size_t) override;
    void begin_memory_trace() override;
    std::vector<MemoryAccess> end_memory_trace() override;
    void set_memory_trace(std::vector<MemoryAccess>) override;
    MemoryRuntimeStats memory_runtime_stats() const override;

    Tensor linear(const Tensor&, const Tensor&, const Tensor* = nullptr,
                  DType = DType::F16, DType = DType::F16) override;
    Tensor add(const Tensor&, const Tensor&) override;
    Tensor mul(const Tensor&, const Tensor&) override;
    Tensor div(const Tensor&, const Tensor&) override;
    Tensor maximum(const Tensor&, const Tensor&) override;
    Tensor batched_matmul(const Tensor&, const Tensor&) override;
    Tensor reshape(const Tensor&, const std::vector<int64_t>&) override;
    Tensor permute(const Tensor&, const std::vector<int64_t>&) override;
    Tensor concat(const std::vector<Tensor>&, int64_t) override;
    std::vector<Tensor> split(const Tensor&, const std::vector<int64_t>&, int64_t) override;
    Tensor slice(const Tensor&, int64_t, int64_t, int64_t) override;
    Tensor cast(const Tensor&, DType) override;
    Tensor indexed_gather(const Tensor&, const Tensor&) override;
    Tensor layer_norm(const Tensor&, const Tensor*, const Tensor*, float) override;
    Tensor rms_norm(const Tensor&, const Tensor*, float, int64_t = -1,
                    DType = DType::F16) override;
    Tensor activation(const Tensor&, Activation) override;
    Tensor sinusoidal_embedding(const Tensor&, int64_t, bool, double, bool) override;
    Tensor rope_nd(const Tensor&, const Tensor&, const Tensor&) override;
    Tensor attention(const Tensor&, const Tensor&, const Tensor&, const Tensor*, bool, float,
                     const Tensor* = nullptr, AttentionObservation* = nullptr) override;
    Tensor conv3d(const Tensor&, const Tensor&, const Tensor*, const std::vector<int>&,
                  const std::vector<int>&, const std::vector<int>&, int) override;
    Tensor pad(const Tensor&, const std::vector<int64_t>&, float, PadMode) override;
    Tensor reduce_sum(const Tensor&, int64_t, bool) override;
    Tensor softmax(const Tensor&, int64_t) override;
    Tensor conv2d(const Tensor&, const Tensor&, const Tensor*, const std::vector<int>&,
                  const std::vector<int>&, const std::vector<int>&, int) override;
    Tensor conv_transpose2d(const Tensor&, const Tensor&, const Tensor*,
                            const std::vector<int>&, const std::vector<int>&,
                            const std::vector<int>&, const std::vector<int>&,
                            int) override;
    Tensor max_pool2d(const Tensor&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&) override;
    Tensor group_norm(const Tensor&, int, const Tensor*, const Tensor*, float) override;
    Tensor interpolate_nearest(const Tensor&, const std::vector<double>&) override;
    Tensor interpolate_bilinear_2d(const Tensor&, int64_t, int64_t, bool) override;
    Tensor pixel_norm(const Tensor&, int64_t, float) override;
    Tensor l2_normalize(const Tensor&, int64_t, float) override;
    Tensor pixel_shuffle_nd(const Tensor&, const std::vector<int64_t>&) override;
    Tensor rng_normal(RngState&, const std::vector<int64_t>&, DType) override;
    Tensor clamp(const Tensor&, float, float) override;
    Tensor exp(const Tensor&) override;
    Tensor sqrt(const Tensor&) override;

private:
    Impl* impl_;
};

}  // namespace vrhino
