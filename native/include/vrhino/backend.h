#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vrhino/tensor.h"
#include "vrhino/memory.h"

namespace vrhino {

enum class Activation { Silu, Gelu, GeluTanh, Tanh, Relu };
enum class PadMode { Constant, Replicate };

struct RngState {
    uint64_t seed = 0;
    uint64_t offset = 0;
    std::string algorithm_id = "pytorch_compat.v1";
};

struct DeviceCapability {
    struct Quantization {
        bool fp8_storage = false;
        bool fp8_packed_compute = false;
        bool int8_packed_compute = false;
        bool int4_packed_compute = false;
        bool backend_repack_supported = false;
    };
    std::string name;
    std::string family;
    size_t device_memory_bytes = 0;
    bool supports_fp16_storage = false;
    bool supports_fp16_arithmetic = false;
    bool supports_bf16_storage = false;
    bool supports_bf16_arithmetic = false;
    enum class F64Mode : uint8_t { Unavailable, SoftwareEmulated, Native };
    F64Mode f64_mode = F64Mode::Unavailable;
    BackendMemoryCapabilities memory;
    Quantization quantization;
};

struct ProfileStat {
    uint64_t calls = 0;
    double device_milliseconds = 0.0;
    double mean_milliseconds = 0.0;
    double p50_milliseconds = 0.0;
    double p95_milliseconds = 0.0;
    double minimum_milliseconds = 0.0;
    double maximum_milliseconds = 0.0;
};

struct QuantComputeStats {
    uint64_t true_quant_calls = 0;
    uint64_t fallback_calls = 0;
    uint64_t fp8_calls = 0;
    uint64_t int8_calls = 0;
    uint64_t int4_calls = 0;
    size_t workspace_bytes = 0;
    size_t scratch_peak_bytes = 0;
    size_t device_copy_bytes = 0;
    std::string last_dispatch;
};

// Optional, observation-only tensors from the arithmetic performed by the
// generic attention primitive. Normal Runtime execution always passes null.
struct AttentionObservation {
    Tensor score;
    Tensor biased_score;
    Tensor softmax;
};

class Backend : public MemoryBackend {
public:
    enum class CopyMode : uint8_t { Synchronous, Asynchronous };
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual int device_count() const = 0;
    virtual DeviceCapability device_capability(int device = 0) const = 0;
    virtual bool supports(DType dtype) const = 0;
    // Execution dtype is a backend-wide, architecture-neutral compute policy.
    // Scheduler scalars may remain FP32, while floating activations and weights
    // are converted to this dtype at primitive boundaries.
    virtual void set_execution_dtype(DType dtype) = 0;
    virtual DType execution_dtype() const = 0;
    virtual Tensor allocate_host(const std::vector<int64_t>& shape, DType dtype) = 0;
    virtual Tensor allocate_device(const std::vector<int64_t>& shape, DType dtype) = 0;
    virtual TransferFence copy(const Tensor& source, Tensor& destination,
                               CopyMode mode = CopyMode::Synchronous) = 0;
    virtual TransferFence create_fence(int device = 0) = 0;
    virtual void record_fence(TransferFence fence) = 0;
    virtual void wait_fence(TransferFence fence) = 0;
    virtual bool query_fence(TransferFence fence) = 0;
    virtual void destroy_fence(TransferFence fence) = 0;
    virtual Tensor copy_to_device(const Tensor& input, DType dtype) = 0;
    virtual Tensor copy_to_host(const Tensor& input) = 0;
    virtual void synchronize() = 0;
    virtual size_t peak_device_bytes() const = 0;
    virtual size_t weight_upload_bytes() const = 0;
    virtual double weight_upload_seconds() const = 0;
    virtual void enable_profiling(bool enabled) = 0;
    virtual bool profiling_enabled() const = 0;
    virtual std::map<std::string, ProfileStat> profile_stats() = 0;
    virtual void profile_region_begin(const std::string& name) = 0;
    virtual void profile_region_end() = 0;
    virtual uint64_t weight_cache_hits() const = 0;
    virtual uint64_t weight_cache_misses() const = 0;
    virtual size_t weight_cache_resident_bytes() const = 0;
    virtual size_t weight_cache_capacity_bytes() const = 0;
    virtual void enable_weight_cache(bool enabled) = 0;
    virtual void enable_true_quant_compute(bool enabled) = 0;
    virtual bool true_quant_compute_enabled() const = 0;
    virtual QuantComputeStats quant_compute_stats() const = 0;
    virtual void reset_quant_compute_stats() = 0;

    virtual Tensor linear(const Tensor& x, const Tensor& weight,
                          const Tensor* bias = nullptr,
                          DType compute = DType::F16,
                          DType output = DType::F16) = 0;
    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor mul(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor div(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor maximum(const Tensor& a, const Tensor& b) = 0;
    // Generic rank-3 batched matrix multiplication: [B,M,K] x [B,K,N].
    virtual Tensor batched_matmul(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape) = 0;
    virtual Tensor permute(const Tensor& x, const std::vector<int64_t>& dims) = 0;
    virtual Tensor concat(const std::vector<Tensor>& tensors, int64_t dim) = 0;
    virtual std::vector<Tensor> split(const Tensor& x, const std::vector<int64_t>& sections,
                                      int64_t dim) = 0;
    virtual Tensor slice(const Tensor& x, int64_t dim, int64_t start, int64_t stop) = 0;
    virtual Tensor cast(const Tensor& x, DType dtype) = 0;
    // Generic indexed row gather.  A rank-2 table accepts arbitrary-rank
    // indices; a rank-3 table performs a batch-preserving gather where the
    // first index dimension must equal the table batch dimension.
    virtual Tensor indexed_gather(const Tensor& table, const Tensor& indices) = 0;
    virtual Tensor layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias,
                              float eps) = 0;
    virtual Tensor rms_norm(const Tensor& x, const Tensor* weight, float eps,
                            int64_t axis = -1, DType output = DType::F16) = 0;
    virtual Tensor activation(const Tensor& x, Activation kind) = 0;
    virtual Tensor sinusoidal_embedding(const Tensor& positions, int64_t width,
                                         bool flip, double downscale_shift,
                                         bool require_f64_semantics) = 0;
    virtual Tensor rope_nd(const Tensor& x, const Tensor& cosine, const Tensor& sine) = 0;
    virtual Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor* mask = nullptr, bool causal = false,
                             float scale = 0.0f,
                             const Tensor* additive_bias = nullptr,
                             AttentionObservation* observation = nullptr) = 0;
    virtual Tensor conv3d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                          const std::vector<int>& stride, const std::vector<int>& padding,
                          const std::vector<int>& dilation = {1, 1, 1}, int groups = 1) = 0;
    virtual Tensor pad(const Tensor& x, const std::vector<int64_t>& padding,
                       float value = 0.0f, PadMode mode = PadMode::Constant) = 0;
    virtual Tensor reduce_sum(const Tensor& x, int64_t dim, bool keepdim = false) = 0;
    // Stable axis-wise softmax. The axis is explicit and FP32 accumulation is
    // used independently of the storage/execution dtype.
    virtual Tensor softmax(const Tensor& x, int64_t axis) = 0;
    virtual Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                          const std::vector<int>& stride, const std::vector<int>& padding,
                          const std::vector<int>& dilation = {1, 1}, int groups = 1) = 0;
    // Generic NCHW transposed convolution. Weight layout is
    // [input_channels, output_channels/groups, kernel_height, kernel_width].
    virtual Tensor conv_transpose2d(
        const Tensor& x, const Tensor& weight, const Tensor* bias,
        const std::vector<int>& stride, const std::vector<int>& padding,
        const std::vector<int>& output_padding = {0, 0},
        const std::vector<int>& dilation = {1, 1}, int groups = 1) = 0;
    virtual Tensor max_pool2d(const Tensor& x, const std::vector<int>& kernel,
                              const std::vector<int>& stride,
                              const std::vector<int>& padding = {0, 0}) = 0;
    virtual Tensor group_norm(const Tensor& x, int groups, const Tensor* weight,
                              const Tensor* bias, float eps) = 0;
    virtual Tensor interpolate_nearest(const Tensor& x,
                                       const std::vector<double>& factors) = 0;
    // Generic NCHW bilinear resize. Coordinate mapping is explicit so graph
    // identity does not depend on a framework-specific interpolation default.
    virtual Tensor interpolate_bilinear_2d(const Tensor& x,
                                           int64_t output_height,
                                           int64_t output_width,
                                           bool align_corners) = 0;
    virtual Tensor pixel_norm(const Tensor& x, int64_t dim, float eps) = 0;
    // Axis-wise Euclidean normalization: x / (sqrt(sum(x*x, dim)) + eps).
    // Learned scaling, when required, remains an ordinary broadcast multiply.
    virtual Tensor l2_normalize(const Tensor& x, int64_t dim, float eps) = 0;
    virtual Tensor pixel_shuffle_nd(const Tensor& x,
                                    const std::vector<int64_t>& factors) = 0;
    virtual Tensor rng_normal(RngState& state, const std::vector<int64_t>& shape,
                              DType dtype) = 0;
    virtual Tensor clamp(const Tensor& x, float minimum, float maximum) = 0;
    // Generic elementwise exponential. Component graphs use this for normal
    // mathematical semantics such as distribution reparameterization; it has
    // no architecture or model identity.
    virtual Tensor exp(const Tensor& x) = 0;
    virtual Tensor sqrt(const Tensor& x) = 0;
};

class BackendProfileRegion {
public:
    BackendProfileRegion(Backend& backend, std::string name) : backend_(backend) {
        backend_.profile_region_begin(name);
    }
    ~BackendProfileRegion() { backend_.profile_region_end(); }
    BackendProfileRegion(const BackendProfileRegion&) = delete;
    BackendProfileRegion& operator=(const BackendProfileRegion&) = delete;
private:
    Backend& backend_;
};

}  // namespace vrhino
