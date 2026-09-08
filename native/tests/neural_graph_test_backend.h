#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "vrhino/backend.h"
#include "vrhino/error.h"

namespace vrhino::neural_graph::test {
// Deliberately test-only. All unrelated Backend methods fail; no CPU product path.
class TinyBackend : public Backend {
public:
    enum class Fault { None, Undefined, Shape, Dtype, Extent, NullData, Device };
    DType dtype = DType::F32;
    Fault fault = Fault::None;
    size_t fail_call = 0, fault_call = 1;
    bool fail_sync = false, corrupt_capture = false, throw_nonstandard = false;
    size_t syncs = 0;
    std::vector<std::string> calls;
    std::vector<std::weak_ptr<Storage>> produced;
    DType execution_dtype() const override { return dtype; }
    int device_count() const override { return 0; }
    Tensor owned(const std::vector<int64_t>& shape, DType type = DType::F32) {
        auto storage = std::make_shared<Storage>();
        auto payload = std::make_shared<std::vector<float>>(shape_numel(shape));
        storage->data = payload->data();
        storage->bytes = payload->size() * sizeof(float);
        storage->owner = true;
        storage->native_owner = payload;
        produced.push_back(storage);
        return Tensor(storage, 0, shape, type);
    }
    void begin(const char* name) {
        calls.emplace_back(name);
        if (calls.size() == fail_call) {
            if (throw_nonstandard) throw 7;
            throw Error("injected primitive failure");
        }
    }
    Tensor finish(Tensor t) {
        if (calls.size() != fault_call) return t;
        switch (fault) {
            case Fault::None: break;
            case Fault::Undefined: produced.pop_back(); return {};
            case Fault::Shape: return t.reshape({t.numel()});
            case Fault::Dtype: produced.pop_back(); return owned(t.shape(), DType::I32);
            case Fault::Extent: produced.back().lock()->bytes = 0; break;
            case Fault::NullData: produced.back().lock()->data = nullptr; break;
            case Fault::Device:
                produced.back().lock()->device = DeviceId::accelerator(0);
                produced.back().lock()->domain = MemoryDomain::DeviceLocal;
                break;
        }
        return t;
    }
    Tensor binary(const Tensor& a, const Tensor& b, bool multiply) {
        begin(multiply ? "mul" : "add");
        const size_t rank = std::max(a.shape().size(), b.shape().size());
        std::vector<int64_t> shape(rank, 1);
        for (const Tensor* t : {&a, &b})
            for (size_t j = 0; j < t->shape().size(); ++j)
                shape[rank - t->shape().size() + j] =
                    std::max(shape[rank - t->shape().size() + j], t->shape()[j]);
        Tensor out = owned(shape);
        for (int64_t flat = 0; flat < out.numel(); ++flat) {
            auto offset = [&](const Tensor& t) {
                int64_t rem = flat, pos = 0;
                for (size_t j = rank; j-- > 0;) {
                    const int64_t coordinate = rem % shape[j];
                    rem /= shape[j];
                    if (j >= rank - t.shape().size()) {
                        const size_t k = j - (rank - t.shape().size());
                        if (t.shape()[k] != 1) pos += coordinate * t.strides()[k];
                    }
                }
                return pos;
            };
            const float x = a.data_as<float>()[offset(a)], y = b.data_as<float>()[offset(b)];
            out.data_as<float>()[flat] = multiply ? x * y : x + y;
        }
        return finish(std::move(out));
    }
    Tensor add(const Tensor& a, const Tensor& b) override { return binary(a, b, false); }
    Tensor mul(const Tensor& a, const Tensor& b) override { return binary(a, b, true); }
    Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape) override {
        begin("reshape");
        return finish(x.reshape(shape));
    }
    void synchronize() override {
        ++syncs;
        for (const auto& weak : produced)
            require(!weak.expired(), "temporary released before synchronization");
        if (corrupt_capture && !produced.empty()) produced.back().lock()->bytes = 0;
        if (fail_sync) throw Error("injected completion failure");
    }
    std::string name() const override { throw Error("Unused test Backend method: name"); }
    DeviceCapability device_capability(int) const override { throw Error("Unused test Backend method: device_capability"); }
    bool supports(DType) const override { throw Error("Unused test Backend method: supports"); }
    void set_execution_dtype(DType) override { throw Error("Unused test Backend method: set_execution_dtype"); }
    Tensor allocate_host(const std::vector<int64_t>& , DType) override { throw Error("Unused test Backend method: allocate_host"); }
    Tensor allocate_device(const std::vector<int64_t>& , DType) override { throw Error("Unused test Backend method: allocate_device"); }
    TransferFence copy(const Tensor& , Tensor& , CopyMode) override { throw Error("Unused test Backend method: copy"); }
    TransferFence create_fence(int) override { throw Error("Unused test Backend method: create_fence"); }
    void record_fence(TransferFence) override { throw Error("Unused test Backend method: record_fence"); }
    void wait_fence(TransferFence) override { throw Error("Unused test Backend method: wait_fence"); }
    bool query_fence(TransferFence) override { throw Error("Unused test Backend method: query_fence"); }
    void destroy_fence(TransferFence) override { throw Error("Unused test Backend method: destroy_fence"); }
    Tensor copy_to_device(const Tensor& , DType) override { throw Error("Unused test Backend method: copy_to_device"); }
    Tensor copy_to_host(const Tensor&) override { throw Error("Unused test Backend method: copy_to_host"); }
    size_t peak_device_bytes() const override { throw Error("Unused test Backend method: peak_device_bytes"); }
    size_t weight_upload_bytes() const override { throw Error("Unused test Backend method: weight_upload_bytes"); }
    double weight_upload_seconds() const override { throw Error("Unused test Backend method: weight_upload_seconds"); }
    void enable_profiling(bool) override { throw Error("Unused test Backend method: enable_profiling"); }
    bool profiling_enabled() const override { throw Error("Unused test Backend method: profiling_enabled"); }
    std::map<std::string, ProfileStat> profile_stats() override { throw Error("Unused test Backend method: profile_stats"); }
    void profile_region_begin(const std::string&) override { throw Error("Unused test Backend method: profile_region_begin"); }
    void profile_region_end() override { throw Error("Unused test Backend method: profile_region_end"); }
    uint64_t weight_cache_hits() const override { throw Error("Unused test Backend method: weight_cache_hits"); }
    uint64_t weight_cache_misses() const override { throw Error("Unused test Backend method: weight_cache_misses"); }
    size_t weight_cache_resident_bytes() const override { throw Error("Unused test Backend method: weight_cache_resident_bytes"); }
    size_t weight_cache_capacity_bytes() const override { throw Error("Unused test Backend method: weight_cache_capacity_bytes"); }
    void enable_weight_cache(bool) override { throw Error("Unused test Backend method: enable_weight_cache"); }
    void enable_true_quant_compute(bool) override { throw Error("Unused test Backend method: enable_true_quant_compute"); }
    bool true_quant_compute_enabled() const override { throw Error("Unused test Backend method: true_quant_compute_enabled"); }
    QuantComputeStats quant_compute_stats() const override { throw Error("Unused test Backend method: quant_compute_stats"); }
    void reset_quant_compute_stats() override { throw Error("Unused test Backend method: reset_quant_compute_stats"); }
    Tensor linear(const Tensor& , const Tensor& , const Tensor* , DType , DType) override { throw Error("Unused test Backend method: linear"); }
    Tensor div(const Tensor& , const Tensor&) override { throw Error("Unused test Backend method: div"); }
    Tensor maximum(const Tensor& , const Tensor&) override { throw Error("Unused test Backend method: maximum"); }
    Tensor batched_matmul(const Tensor& , const Tensor&) override { throw Error("Unused test Backend method: batched_matmul"); }
    Tensor permute(const Tensor& , const std::vector<int64_t>&) override { throw Error("Unused test Backend method: permute"); }
    Tensor concat(const std::vector<Tensor>& , int64_t) override { throw Error("Unused test Backend method: concat"); }
    std::vector<Tensor> split(const Tensor& , const std::vector<int64_t>& , int64_t) override { throw Error("Unused test Backend method: split"); }
    Tensor slice(const Tensor& , int64_t , int64_t , int64_t) override { throw Error("Unused test Backend method: slice"); }
    Tensor cast(const Tensor& , DType) override { throw Error("Unused test Backend method: cast"); }
    Tensor indexed_gather(const Tensor& , const Tensor&) override { throw Error("Unused test Backend method: indexed_gather"); }
    Tensor layer_norm(const Tensor& , const Tensor* , const Tensor* , float) override { throw Error("Unused test Backend method: layer_norm"); }
    Tensor rms_norm(const Tensor& , const Tensor* , float , int64_t , DType) override { throw Error("Unused test Backend method: rms_norm"); }
    Tensor activation(const Tensor& , Activation) override { throw Error("Unused test Backend method: activation"); }
    Tensor sinusoidal_embedding(const Tensor& , int64_t , bool , double , bool) override { throw Error("Unused test Backend method: sinusoidal_embedding"); }
    Tensor rope_nd(const Tensor& , const Tensor& , const Tensor&) override { throw Error("Unused test Backend method: rope_nd"); }
    Tensor attention(const Tensor& , const Tensor& , const Tensor& , const Tensor* , bool , float , const Tensor* , AttentionObservation*) override { throw Error("Unused test Backend method: attention"); }
    Tensor conv3d(const Tensor& , const Tensor& , const Tensor* , const std::vector<int>& , const std::vector<int>& , const std::vector<int>& , int) override { throw Error("Unused test Backend method: conv3d"); }
    Tensor pad(const Tensor& , const std::vector<int64_t>& , float , PadMode) override { throw Error("Unused test Backend method: pad"); }
    Tensor reduce_sum(const Tensor& , int64_t , bool) override { throw Error("Unused test Backend method: reduce_sum"); }
    Tensor softmax(const Tensor& , int64_t) override { throw Error("Unused test Backend method: softmax"); }
    Tensor conv2d(const Tensor& , const Tensor& , const Tensor* , const std::vector<int>& , const std::vector<int>& , const std::vector<int>& , int) override { throw Error("Unused test Backend method: conv2d"); }
    Tensor conv_transpose2d(const Tensor& , const Tensor& , const Tensor* , const std::vector<int>& , const std::vector<int>& , const std::vector<int>& , const std::vector<int>& , int) override { throw Error("Unused test Backend method: conv_transpose2d"); }
    Tensor max_pool2d(const Tensor& , const std::vector<int>& , const std::vector<int>& , const std::vector<int>&) override { throw Error("Unused test Backend method: max_pool2d"); }
    Tensor group_norm(const Tensor& , int , const Tensor* , const Tensor* , float) override { throw Error("Unused test Backend method: group_norm"); }
    Tensor interpolate_nearest(const Tensor& , const std::vector<double>&) override { throw Error("Unused test Backend method: interpolate_nearest"); }
    Tensor interpolate_bilinear_2d(const Tensor& , int64_t , int64_t , bool) override { throw Error("Unused test Backend method: interpolate_bilinear_2d"); }
    Tensor pixel_norm(const Tensor& , int64_t , float) override { throw Error("Unused test Backend method: pixel_norm"); }
    Tensor l2_normalize(const Tensor& , int64_t , float) override { throw Error("Unused test Backend method: l2_normalize"); }
    Tensor pixel_shuffle_nd(const Tensor& , const std::vector<int64_t>&) override { throw Error("Unused test Backend method: pixel_shuffle_nd"); }
    Tensor rng_normal(RngState& , const std::vector<int64_t>& , DType) override { throw Error("Unused test Backend method: rng_normal"); }
    Tensor clamp(const Tensor& , float , float) override { throw Error("Unused test Backend method: clamp"); }
    Tensor exp(const Tensor&) override { throw Error("Unused test Backend method: exp"); }
    Tensor sqrt(const Tensor&) override { throw Error("Unused test Backend method: sqrt"); }
    BackendMemoryCapabilities memory_capabilities() const override { throw Error("Unused test Backend method: memory_capabilities"); }
    void configure_memory_runtime(const MemoryBudget& , const MemoryRuntimeOptions&) override { throw Error("Unused test Backend method: configure_memory_runtime"); }
    bool memory_runtime_enabled() const override { throw Error("Unused test Backend method: memory_runtime_enabled"); }
    void set_vrm_mapped_bytes(size_t) override { throw Error("Unused test Backend method: set_vrm_mapped_bytes"); }
    void begin_memory_trace() override { throw Error("Unused test Backend method: begin_memory_trace"); }
    std::vector<MemoryAccess> end_memory_trace() override { throw Error("Unused test Backend method: end_memory_trace"); }
    void set_memory_trace(std::vector<MemoryAccess>) override { throw Error("Unused test Backend method: set_memory_trace"); }
    MemoryRuntimeStats memory_runtime_stats() const override { throw Error("Unused test Backend method: memory_runtime_stats"); }
};
}  // namespace vrhino::neural_graph::test
