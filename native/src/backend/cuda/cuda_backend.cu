#include "vrhino/backend/cuda_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <map>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <mma.h>

#include "vrhino/backend/cuda_attention_config.h"
#include "vrhino/backend/cudnn_conv.h"
#include "vrhino/backend/cudnn_sdpa.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

#define CUDA_CHECK(expr) do { cudaError_t status_ = (expr); require(status_ == cudaSuccess, std::string("CUDA error: ") + cudaGetErrorString(status_)); } while (0)
#define CUBLAS_CHECK(expr) do { cublasStatus_t status_ = (expr); require(status_ == CUBLAS_STATUS_SUCCESS, "cuBLAS error: " + std::to_string(status_)); } while (0)
#define CUDNN_CHECK(expr) do { cudnnStatus_t status_ = (expr); require(status_ == CUDNN_STATUS_SUCCESS, std::string("cuDNN error: ") + cudnnGetErrorString(status_)); } while (0)

// Stream-ordered, exact-size reuse for generic CUDA operation temporaries.
// Every consumer of this pool runs on the legacy default compute stream; a
// buffer returned after its last enqueue can therefore be reused by a later
// enqueue without a host synchronization. Host-transfer destinations, model
// weights, and public/unknown-lifetime allocations deliberately remain direct.
class CudaTemporaryPool final
    : public std::enable_shared_from_this<CudaTemporaryPool> {
    struct PooledAllocation;
public:
    CudaTemporaryPool() {
        int device = 0;
        int supported = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported,
                                   device) == cudaSuccess && supported) {
            cudaMemPoolProps properties{};
            properties.allocType = cudaMemAllocationTypePinned;
            properties.handleTypes = cudaMemHandleTypeNone;
            properties.location.type = cudaMemLocationTypeDevice;
            properties.location.id = device;
            if (cudaMemPoolCreate(&stream_pool_, &properties) != cudaSuccess) {
                (void)cudaGetLastError();
                stream_pool_ = nullptr;
            }
        }
    }
    ~CudaTemporaryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [bytes, pointers] : free_) {
            (void)bytes;
            for (void* pointer : pointers) cudaFree(pointer);
        }
        if (handoff_pointer_) cudaFree(handoff_pointer_);
        if (stream_pool_) cudaMemPoolDestroy(stream_pool_);
    }

    Tensor tensor(std::vector<int64_t> shape, DType dtype,
                  bool workspace = false) {
        const size_t bytes = static_cast<size_t>(shape_numel(shape)) * dtype_size(dtype);
        void* pointer = nullptr;
        void* expired_handoff = nullptr;
        size_t expired_handoff_bytes = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++request_count_;
            request_bytes_ += bytes;
            if (handoff_pointer_) {
                if (handoff_bytes_ == bytes) {
                    pointer = handoff_pointer_;
                    handoff_pointer_ = nullptr;
                    handoff_bytes_ = 0;
                    ++reuse_count_;
                    ++handoff_reuse_count_;
                    last_handoff_reuse_bytes_.store(bytes,
                                                    std::memory_order_relaxed);
                    size_t largest = largest_handoff_reuse_bytes_.load(
                        std::memory_order_relaxed);
                    while (largest < bytes &&
                           !largest_handoff_reuse_bytes_.compare_exchange_weak(
                               largest, bytes, std::memory_order_relaxed)) {}
                } else {
                    // Oversize handoff is deliberately one-shot. It never
                    // becomes an unbounded cache or competes with a differently
                    // sized next allocation.
                    expired_handoff = handoff_pointer_;
                    expired_handoff_bytes = handoff_bytes_;
                    note_uncached_release_locked(handoff_bytes_);
                    handoff_pointer_ = nullptr;
                    handoff_bytes_ = 0;
                }
            }
            auto found = free_.find(bytes);
            if (!pointer && found != free_.end() && !found->second.empty()) {
                pointer = found->second.back();
                found->second.pop_back();
                if (found->second.empty()) free_.erase(found);
                cached_bytes_ -= bytes;
                ++reuse_count_;
            }
            if (!pointer) {
                ++miss_count_;
                miss_bytes_ += bytes;
                if (uncached_releases_by_size_[bytes] > 0) {
                    ++capacity_recurrence_count_;
                    capacity_recurrence_bytes_ += bytes;
                } else if (active_by_size_[bytes] > 0) {
                    ++lifetime_overlap_count_;
                    lifetime_overlap_bytes_ += bytes;
                } else {
                    const auto compatible = free_.lower_bound(bytes);
                    if (compatible != free_.end() ||
                        (handoff_pointer_ && handoff_bytes_ >= bytes)) {
                        ++size_mismatch_count_;
                        size_mismatch_bytes_ += bytes;
                    } else {
                        ++first_use_count_;
                        first_use_bytes_ += bytes;
                    }
                }
                if (workspace) {
                    ++workspace_miss_count_;
                    workspace_miss_bytes_ += bytes;
                }
            }
            ++active_by_size_[bytes];
        }
        if (expired_handoff) {
            release_native(expired_handoff);
            physical_bytes_.fetch_sub(expired_handoff_bytes,
                                      std::memory_order_relaxed);
            ++driver_free_count_;
        }
        if (!pointer) {
            cudaError_t status = allocate_native(&pointer, bytes);
            if (status == cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                trim();
                status = allocate_native(&pointer, bytes);
            }
            require(status == cudaSuccess,
                    std::string("CUDA temporary allocation failed: ") +
                        cudaGetErrorString(status) +
                        ", requested_bytes=" + std::to_string(bytes) +
                        ", active_bytes=" + std::to_string(active_bytes()) +
                        ", handoff_reuses=" +
                        std::to_string(handoff_reuse_count()) +
                        ", largest_handoff_bytes=" +
                        std::to_string(largest_handoff_reuse_bytes()));
            physical_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            ++driver_allocation_count_;
        }
        const size_t active = active_bytes_.fetch_add(
            bytes, std::memory_order_relaxed) + bytes;
        size_t observed = peak_active_bytes_.load(std::memory_order_relaxed);
        while (observed < active && !peak_active_bytes_.compare_exchange_weak(
                   observed, active, std::memory_order_relaxed)) {}

        auto storage = std::make_shared<Storage>();
        storage->data = pointer;
        storage->bytes = bytes;
        storage->device = DeviceId::accelerator();
        storage->domain = MemoryDomain::DeviceLocal;
        storage->owner = true;
        storage->native_owner = std::make_shared<PooledAllocation>(
            shared_from_this(), pointer, bytes);
        return Tensor(std::move(storage), 0, std::move(shape), dtype);
    }

    void set_cache_limit(size_t bytes, size_t retention_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_limit_bytes_ = bytes;
        if (stream_pool_) {
            // The exact-size host-side cache is bounded by the declared
            // workspace reserve.  The CUDA stream pool is a different layer:
            // it should retain whatever generic steady-state working set it
            // has already grown to (within the generic device budget),
            // otherwise every execution-boundary sync
            // releases physical pages and the next execution pays for them
            // again.  No memory is preallocated, and explicit pressure/OOM
            // handling below can still trim unused pool storage.
            uint64_t threshold = retention_bytes;
            const cudaError_t status = cudaMemPoolSetAttribute(
                stream_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
            require(status == cudaSuccess,
                    std::string("CUDA temporary pool threshold failed: ") +
                        cudaGetErrorString(status));
        }
        trim_locked();
    }

    void trim_cached() noexcept { trim(); }

    size_t active_bytes() const {
        return active_bytes_.load(std::memory_order_relaxed);
    }
    size_t peak_active_bytes() const {
        return peak_active_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t reuse_count() const {
        return reuse_count_.load(std::memory_order_relaxed);
    }
    uint64_t handoff_release_count() const {
        return handoff_release_count_.load(std::memory_order_relaxed);
    }
    uint64_t handoff_reuse_count() const {
        return handoff_reuse_count_.load(std::memory_order_relaxed);
    }
    size_t last_handoff_reuse_bytes() const {
        return last_handoff_reuse_bytes_.load(std::memory_order_relaxed);
    }
    size_t largest_handoff_reuse_bytes() const {
        return largest_handoff_reuse_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t request_count() const { return request_count_; }
    uint64_t request_bytes() const { return request_bytes_; }
    uint64_t miss_count() const { return miss_count_; }
    uint64_t miss_bytes() const { return miss_bytes_; }
    uint64_t driver_allocation_count() const {
        return driver_allocation_count_.load(std::memory_order_relaxed);
    }
    uint64_t driver_free_count() const {
        return driver_free_count_.load(std::memory_order_relaxed);
    }
    uint64_t size_mismatch_count() const { return size_mismatch_count_; }
    uint64_t size_mismatch_bytes() const { return size_mismatch_bytes_; }
    uint64_t lifetime_overlap_count() const {
        return lifetime_overlap_count_;
    }
    uint64_t lifetime_overlap_bytes() const {
        return lifetime_overlap_bytes_;
    }
    uint64_t capacity_recurrence_count() const {
        return capacity_recurrence_count_;
    }
    uint64_t capacity_recurrence_bytes() const {
        return capacity_recurrence_bytes_;
    }
    uint64_t first_use_count() const { return first_use_count_; }
    uint64_t first_use_bytes() const { return first_use_bytes_; }
    uint64_t workspace_miss_count() const { return workspace_miss_count_; }
    uint64_t workspace_miss_bytes() const { return workspace_miss_bytes_; }
    uint64_t direct_allocation_count() const {
        return direct_allocation_count_;
    }
    uint64_t direct_allocation_bytes() const {
        return direct_allocation_bytes_;
    }
    uint64_t stream_ordered_allocation_count() const {
        return stream_ordered_allocation_count_;
    }
    uint64_t stream_ordered_free_count() const {
        return stream_ordered_free_count_;
    }
    uint64_t legacy_allocation_count() const {
        return legacy_allocation_count_;
    }
    uint64_t legacy_free_count() const { return legacy_free_count_; }
    void note_direct_allocation(size_t bytes) {
        ++direct_allocation_count_;
        direct_allocation_bytes_ += bytes;
    }

private:
    struct PooledAllocation {
        PooledAllocation(std::shared_ptr<CudaTemporaryPool> owner,
                         void* allocation, size_t allocation_bytes)
            : pool(std::move(owner)), pointer(allocation),
              bytes(allocation_bytes) {}
        PooledAllocation(const PooledAllocation&) = delete;
        PooledAllocation& operator=(const PooledAllocation&) = delete;
        std::shared_ptr<CudaTemporaryPool> pool;
        void* pointer = nullptr;
        size_t bytes = 0;
        ~PooledAllocation() {
            if (pointer) pool->release(pointer, bytes);
        }
    };

    void release(void* pointer, size_t bytes) noexcept {
        active_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
        void* displaced = nullptr;
        size_t displaced_bytes = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto active = active_by_size_.find(bytes);
            if (active != active_by_size_.end() && --active->second == 0)
                active_by_size_.erase(active);
            if (bytes <= cache_limit_bytes_ &&
                cached_bytes_ <= cache_limit_bytes_ - bytes) {
                free_[bytes].push_back(pointer);
                cached_bytes_ += bytes;
                return;
            }
            if (bytes > cache_limit_bytes_) {
                displaced = handoff_pointer_;
                displaced_bytes = handoff_bytes_;
                if (displaced) note_uncached_release_locked(displaced_bytes);
                handoff_pointer_ = pointer;
                handoff_bytes_ = bytes;
                ++handoff_release_count_;
                pointer = nullptr;
            }
            if (pointer) note_uncached_release_locked(bytes);
        }
        if (displaced) {
            release_native(displaced);
            physical_bytes_.fetch_sub(displaced_bytes,
                                      std::memory_order_relaxed);
            ++driver_free_count_;
        }
        if (pointer) {
            release_native(pointer);
            physical_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
            ++driver_free_count_;
        }
    }

    void trim() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [bytes, pointers] : free_) {
            for (void* pointer : pointers) {
                note_uncached_release_locked(bytes);
                release_native(pointer);
                physical_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
                ++driver_free_count_;
            }
        }
        free_.clear();
        cached_bytes_ = 0;
        if (handoff_pointer_) {
            note_uncached_release_locked(handoff_bytes_);
            release_native(handoff_pointer_);
            physical_bytes_.fetch_sub(handoff_bytes_, std::memory_order_relaxed);
            ++driver_free_count_;
            handoff_pointer_ = nullptr;
            handoff_bytes_ = 0;
        }
        if (stream_pool_) (void)cudaMemPoolTrimTo(stream_pool_, 0);
    }

    void trim_locked() noexcept {
        while (cached_bytes_ > cache_limit_bytes_ && !free_.empty()) {
            auto found = std::prev(free_.end());
            void* pointer = found->second.back();
            found->second.pop_back();
            cached_bytes_ -= found->first;
            physical_bytes_.fetch_sub(found->first, std::memory_order_relaxed);
            note_uncached_release_locked(found->first);
            release_native(pointer);
            ++driver_free_count_;
            if (found->second.empty()) free_.erase(found);
        }
    }

    void note_uncached_release_locked(size_t bytes) {
        ++uncached_releases_by_size_[bytes];
    }

    cudaError_t allocate_native(void** pointer, size_t bytes) {
        if (stream_pool_) {
            ++stream_ordered_allocation_count_;
            return cudaMallocFromPoolAsync(pointer, bytes, stream_pool_, nullptr);
        }
        ++legacy_allocation_count_;
        return cudaMalloc(pointer, bytes);
    }

    void release_native(void* pointer) noexcept {
        if (stream_pool_) {
            ++stream_ordered_free_count_;
            (void)cudaFreeAsync(pointer, nullptr);
        } else {
            ++legacy_free_count_;
            (void)cudaFree(pointer);
        }
    }

    mutable std::mutex mutex_;
    std::map<size_t, std::vector<void*>> free_;
    std::map<size_t, uint64_t> active_by_size_;
    std::map<size_t, uint64_t> uncached_releases_by_size_;
    size_t cache_limit_bytes_ = 0;
    size_t cached_bytes_ = 0;
    void* handoff_pointer_ = nullptr;
    size_t handoff_bytes_ = 0;
    std::atomic<size_t> active_bytes_{0};
    std::atomic<size_t> peak_active_bytes_{0};
    std::atomic<size_t> physical_bytes_{0};
    std::atomic<uint64_t> driver_allocation_count_{0};
    std::atomic<uint64_t> driver_free_count_{0};
    std::atomic<uint64_t> reuse_count_{0};
    std::atomic<uint64_t> handoff_release_count_{0};
    std::atomic<uint64_t> handoff_reuse_count_{0};
    std::atomic<size_t> last_handoff_reuse_bytes_{0};
    std::atomic<size_t> largest_handoff_reuse_bytes_{0};
    cudaMemPool_t stream_pool_ = nullptr;
    uint64_t stream_ordered_allocation_count_ = 0;
    uint64_t stream_ordered_free_count_ = 0;
    uint64_t legacy_allocation_count_ = 0;
    uint64_t legacy_free_count_ = 0;
    uint64_t request_count_ = 0;
    uint64_t request_bytes_ = 0;
    uint64_t miss_count_ = 0;
    uint64_t miss_bytes_ = 0;
    uint64_t size_mismatch_count_ = 0;
    uint64_t size_mismatch_bytes_ = 0;
    uint64_t lifetime_overlap_count_ = 0;
    uint64_t lifetime_overlap_bytes_ = 0;
    uint64_t capacity_recurrence_count_ = 0;
    uint64_t capacity_recurrence_bytes_ = 0;
    uint64_t first_use_count_ = 0;
    uint64_t first_use_bytes_ = 0;
    uint64_t workspace_miss_count_ = 0;
    uint64_t workspace_miss_bytes_ = 0;
    uint64_t direct_allocation_count_ = 0;
    uint64_t direct_allocation_bytes_ = 0;
};

Tensor direct_device_tensor(std::vector<int64_t> shape, DType dtype,
                            CudaTemporaryPool* reclaim_pool = nullptr) {
    auto storage = std::make_shared<Storage>();
    storage->bytes = static_cast<size_t>(shape_numel(shape)) * dtype_size(dtype);
    if (reclaim_pool) reclaim_pool->note_direct_allocation(storage->bytes);
    cudaError_t status = cudaMalloc(&storage->data, storage->bytes);
    if (status == cudaErrorMemoryAllocation && reclaim_pool) {
        (void)cudaGetLastError();
        reclaim_pool->trim_cached();
        status = cudaMalloc(&storage->data, storage->bytes);
    }
    require(status == cudaSuccess,
            std::string("CUDA allocation failed: ") + cudaGetErrorString(status));
    storage->device = DeviceId::accelerator();
    storage->domain = MemoryDomain::DeviceLocal;
    storage->owner = true;
    storage->deleter = [](void* pointer) { cudaFree(pointer); };
    return Tensor(std::move(storage), 0, std::move(shape), dtype);
}

size_t temporary_cache_limit(const MemoryBudget& budget) {
    // Cached (inactive) blocks share the device reserve with live operation
    // workspaces. The direct-allocation path trims this reclaimable cache and
    // retries on pressure, so the full declared workspace reserve can serve as
    // the steady-state reuse ceiling without weakening OOM safety.
    return budget.reserved_device_workspace_bytes;
}

size_t temporary_pool_retention_limit(const MemoryBudget& budget) {
    // Retention is lazy and reclaimable: this is an upper bound, not a
    // reservation.  Keep it inside the declared device budget and preserve
    // the safety margin for allocations outside the temporary pool.
    return budget.device_budget_bytes - budget.safety_margin_bytes;
}

constexpr int kThreads = 256;
constexpr int kMaxDims = 8;

struct Meta {
    int rank = 0;
    int64_t shape[kMaxDims]{};
    int64_t strides[kMaxDims]{};
};

Meta meta(const std::vector<int64_t>& shape) {
    require(shape.size() <= kMaxDims, "CUDA primitive rank exceeds 8");
    Meta result; result.rank = static_cast<int>(shape.size());
    const auto strides = contiguous_strides(shape);
    for (int index = 0; index < result.rank; ++index) {
        result.shape[index] = shape[index]; result.strides[index] = strides[index];
    }
    return result;
}

__device__ int64_t broadcast_offset(int64_t linear, const Meta& output, const Meta& input) {
    int64_t offset = 0;
    const int shift = output.rank - input.rank;
    for (int dimension = output.rank - 1; dimension >= 0; --dimension) {
        const int64_t coordinate = linear % output.shape[dimension]; linear /= output.shape[dimension];
        const int input_dimension = dimension - shift;
        if (input_dimension >= 0 && input.shape[input_dimension] != 1)
            offset += coordinate * input.strides[input_dimension];
    }
    return offset;
}

template <typename T> __device__ float load_value(T value) { return static_cast<float>(value); }
template <> __device__ float load_value(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ T store_value(float value) { return static_cast<T>(value); }
template <> __device__ __nv_bfloat16 store_value(float value) { return __float2bfloat16(value); }

template <typename T, int Kind>
__global__ void binary_kernel(const T* a, const T* b, T* output,
                              int64_t count, Meta out_meta, Meta a_meta, Meta b_meta) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float av = load_value(a[broadcast_offset(index, out_meta, a_meta)]);
        const float bv = load_value(b[broadcast_offset(index, out_meta, b_meta)]);
        if constexpr (Kind == 0) output[index] = store_value<T>(av + bv);
        else if constexpr (Kind == 1) output[index] = store_value<T>(av * bv);
        else if constexpr (Kind == 2) output[index] = store_value<T>(av / bv);
        else output[index] = store_value<T>(fmaxf(av, bv));
    }
}

__global__ void bf16_to_float_kernel(const __nv_bfloat16* input, float* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __bfloat162float(input[index]);
}
__global__ void float_to_bf16_kernel(const float* input, __nv_bfloat16* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __float2bfloat16(input[index]);
}
__global__ void f16_to_float_kernel(const __half* input, float* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __half2float(input[index]);
}
__global__ void f16_to_bf16_kernel(const __half* input, __nv_bfloat16* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __float2bfloat16(__half2float(input[index]));
}
template <typename T>
__global__ void numeric_to_float_kernel(const T* input, float* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = static_cast<float>(input[index]);
}
template <typename T>
__global__ void numeric_to_bf16_kernel(const T* input, __nv_bfloat16* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = __float2bfloat16(static_cast<float>(input[index]));
}

__device__ float decode_fp8_e4m3fn(uint8_t raw) {
    const int sign = raw >> 7;
    const int exponent = (raw >> 3) & 0xf;
    const int mantissa = raw & 0x7;
    float value;
    if (exponent == 0) value = ldexpf(static_cast<float>(mantissa), -9);
    else value = ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    return sign ? -value : value;
}

template <typename T, int Kind>
__global__ void dequantize_kernel(const uint8_t* packed, const float* scales, T* output,
                                  int64_t count, int64_t inner, int64_t groups,
                                  int64_t group_size) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float quantized;
        if constexpr (Kind == 0) quantized = decode_fp8_e4m3fn(packed[index]);
        else if constexpr (Kind == 1) quantized = static_cast<float>(static_cast<int8_t>(packed[index]));
        else {
            const uint8_t byte = packed[index / 2];
            const uint8_t nibble = index & 1 ? byte >> 4 : byte & 0xf;
            quantized = static_cast<float>(nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble);
        }
        const int64_t outer = index / inner;
        const int64_t group = (index % inner) / group_size;
        output[index] = store_value<T>(quantized * scales[outer * groups + group]);
    }
}

// One model-neutral weight-only GEMM skeleton serves FP8 E4M3FN, signed INT8,
// and packed signed INT4.  Quantized values are decoded only into the current
// 16x16 shared-memory B tile; a full dense weight tensor is never materialized.
// Eight warps reuse that B tile for 128 activation rows and execute BF16 Tensor
// Core MMA with FP32 accumulation. Kind: 0=FP8, 1=INT8, 2=INT4.
template <int Kind>
__global__ void quantized_bf16_gemm_kernel(
    const __nv_bfloat16* activation, const uint8_t* packed_weight,
    const float* scales, const __nv_bfloat16* bias, __nv_bfloat16* output,
    int64_t m, int64_t n, int64_t k, int64_t groups, int64_t group_size) {
    constexpr int kTile = 16;
    constexpr int kWarps = 8;
    constexpr int kRowsPerBlock = kTile * kWarps;
    __shared__ __align__(16) __nv_bfloat16 shared_a[kWarps][kTile * kTile];
    __shared__ __align__(16) __nv_bfloat16 shared_b[kTile * kTile];
    __shared__ __align__(16) float shared_c[kWarps][kTile * kTile];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock + warp * kTile;
    const int64_t column_base = static_cast<int64_t>(blockIdx.x) * kTile;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTile, kTile, kTile, float> accumulator;
    nvcuda::wmma::fill_fragment(accumulator, 0.0f);
    for (int64_t k_base = 0; k_base < k; k_base += kTile) {
        // B is stored column-major as [K,N], which is the canonical row-major
        // [N,K] VRM weight without a persistent backend repack.
        const int b_index = threadIdx.x;
        if (b_index < kTile * kTile) {
            const int local_k = b_index % kTile;
            const int local_n = b_index / kTile;
            const int64_t global_k = k_base + local_k;
            const int64_t global_n = column_base + local_n;
            float value = 0.0f;
            if (global_k < k && global_n < n) {
                const int64_t linear = global_n * k + global_k;
                float quantized;
                if constexpr (Kind == 0) quantized = decode_fp8_e4m3fn(packed_weight[linear]);
                else if constexpr (Kind == 1)
                    quantized = static_cast<float>(static_cast<int8_t>(packed_weight[linear]));
                else {
                    const uint8_t byte = packed_weight[linear / 2];
                    const uint8_t nibble = linear & 1 ? byte >> 4 : byte & 0xf;
                    quantized = static_cast<float>(nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble);
                }
                value = quantized * scales[global_n * groups + global_k / group_size];
            }
            shared_b[local_n * kTile + local_k] = __float2bfloat16(value);
        }
        for (int index = lane; index < kTile * kTile; index += 32) {
            const int local_m = index / kTile;
            const int local_k = index % kTile;
            const int64_t global_m = row_base + local_m;
            const int64_t global_k = k_base + local_k;
            shared_a[warp][index] = global_m < m && global_k < k
                ? activation[global_m * k + global_k] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTile, kTile, kTile,
                               __nv_bfloat16, nvcuda::wmma::row_major> a_fragment;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTile, kTile, kTile,
                               __nv_bfloat16, nvcuda::wmma::col_major> b_fragment;
        nvcuda::wmma::load_matrix_sync(a_fragment, shared_a[warp], kTile);
        nvcuda::wmma::load_matrix_sync(b_fragment, shared_b, kTile);
        nvcuda::wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(shared_c[warp], accumulator, kTile,
                                    nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTile * kTile; index += 32) {
        const int local_m = index / kTile;
        const int local_n = index % kTile;
        const int64_t global_m = row_base + local_m;
        const int64_t global_n = column_base + local_n;
        if (global_m < m && global_n < n) {
            const float value = shared_c[warp][index] +
                (bias ? __bfloat162float(bias[global_n]) : 0.0f);
            output[global_m * n + global_n] = __float2bfloat16(value);
        }
    }
}

template <typename T>
__global__ void bias_kernel(T* output, const T* bias, int64_t count, int64_t width) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = store_value<T>(load_value(output[index]) + load_value(bias[index % width]));
}

// PyTorch CUDA autocast linear uses BF16 operands with FP32 accumulation and
// applies the BF16-rounded bias before the single BF16 output rounding. Keep
// this distinct from convolution, whose observed cuDNN contract rounds the
// convolution output before its BF16 bias add.
__global__ void fused_bf16_linear_output_kernel(const float* accumulation,
                                                 const __nv_bfloat16* bias,
                                                 __nv_bfloat16* output,
                                                 int64_t count, int64_t width) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float value = accumulation[index] +
            (bias ? __bfloat162float(bias[index % width]) : 0.0f);
        output[index] = __float2bfloat16(value);
    }
}

template <typename T>
__global__ void permute_kernel(const T* input, T* output, int64_t count,
                               Meta input_meta, Meta output_meta, Meta dimensions) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, source = 0;
        for (int dimension = output_meta.rank - 1; dimension >= 0; --dimension) {
            const int64_t coordinate = remaining % output_meta.shape[dimension];
            remaining /= output_meta.shape[dimension];
            source += coordinate * input_meta.strides[dimensions.shape[dimension]];
        }
        output[index] = input[source];
    }
}

template <typename T>
__global__ void slice_kernel(const T* input, T* output, int64_t count,
                             Meta input_meta, Meta output_meta, int sliced_dim, int64_t start) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, source = 0;
        for (int dimension = output_meta.rank - 1; dimension >= 0; --dimension) {
            int64_t coordinate = remaining % output_meta.shape[dimension];
            remaining /= output_meta.shape[dimension];
            if (dimension == sliced_dim) coordinate += start;
            source += coordinate * input_meta.strides[dimension];
        }
        output[index] = input[source];
    }
}

template <typename T>
__global__ void copy_into_concat_kernel(const T* input, T* output, int64_t count,
                                        Meta input_meta, Meta output_meta, int concat_dim,
                                        int64_t dim_offset) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, target = 0;
        for (int dimension = input_meta.rank - 1; dimension >= 0; --dimension) {
            int64_t coordinate = remaining % input_meta.shape[dimension];
            remaining /= input_meta.shape[dimension];
            if (dimension == concat_dim) coordinate += dim_offset;
            target += coordinate * output_meta.strides[dimension];
        }
        output[target] = input[index];
    }
}

template <typename T>
__global__ void layer_norm_kernel(const T* input, const T* weight, const T* bias,
                                  T* output, int64_t rows, int64_t width, float eps) {
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        if (threadIdx.x == 0) {
            double sum = 0.0, square = 0.0;
            for (int64_t column = 0; column < width; ++column) {
                const float value = load_value(input[row * width + column]); sum += value; square += static_cast<double>(value) * value;
            }
            const float mean = static_cast<float>(sum / width);
            const float inverse = rsqrtf(static_cast<float>(square / width - static_cast<double>(mean) * mean) + eps);
            for (int64_t column = 0; column < width; ++column) {
                float value = (load_value(input[row * width + column]) - mean) * inverse;
                if (weight) value *= load_value(weight[column]); if (bias) value += load_value(bias[column]);
                output[row * width + column] = store_value<T>(value);
            }
        }
    }
}

constexpr int kNormThreads = 256;

template <typename T>
__global__ void layer_norm_parallel_kernel(
        const T* input, const T* weight, const T* bias, T* output,
        int64_t rows, int64_t width, float eps) {
    __shared__ double sum_workspace[kNormThreads];
    __shared__ double square_workspace[kNormThreads];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0, square = 0.0;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = load_value(input[row * width + column]);
            sum += value;
            square += static_cast<double>(value) * value;
        }
        sum_workspace[threadIdx.x] = sum;
        square_workspace[threadIdx.x] = square;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                sum_workspace[threadIdx.x] += sum_workspace[threadIdx.x + stride];
                square_workspace[threadIdx.x] += square_workspace[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float mean = static_cast<float>(sum_workspace[0] / width);
        const float inverse = rsqrtf(static_cast<float>(
            square_workspace[0] / width - static_cast<double>(mean) * mean) + eps);
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            float value = (load_value(input[row * width + column]) - mean) * inverse;
            if (weight) value *= load_value(weight[column]);
            if (bias) value += load_value(bias[column]);
            output[row * width + column] = store_value<T>(value);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void rms_norm_kernel(const T* input, const T* weight, T* output,
                                int64_t rows, int64_t width, float eps) {
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) if (threadIdx.x == 0) {
        double square = 0.0;
        for (int64_t column = 0; column < width; ++column) {
            const float value = load_value(input[row * width + column]); square += static_cast<double>(value) * value;
        }
        const float inverse = rsqrtf(static_cast<float>(square / width) + eps);
        for (int64_t column = 0; column < width; ++column) {
            float value = load_value(input[row * width + column]) * inverse;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) value = load_value(store_value<T>(value));
            if (weight) value *= load_value(weight[column]); output[row * width + column] = store_value<T>(value);
        }
    }
}

template <typename T>
__global__ void rms_norm_parallel_kernel(
        const T* input, const T* weight, T* output,
        int64_t rows, int64_t width, float eps) {
    __shared__ double square_workspace[kNormThreads];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double square = 0.0;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = load_value(input[row * width + column]);
            square += static_cast<double>(value) * value;
        }
        square_workspace[threadIdx.x] = square;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                square_workspace[threadIdx.x] += square_workspace[threadIdx.x + stride];
            __syncthreads();
        }
        const float inverse = rsqrtf(static_cast<float>(
            square_workspace[0] / width) + eps);
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            float value = load_value(input[row * width + column]) * inverse;
            if constexpr (std::is_same_v<T, __nv_bfloat16>)
                value = load_value(store_value<T>(value));
            if (weight) value *= load_value(weight[column]);
            output[row * width + column] = store_value<T>(value);
        }
        __syncthreads();
    }
}

__global__ void rms_norm_bf16_input_f32_weight_kernel(
        const __nv_bfloat16* input, const float* weight, float* output,
        int64_t rows, int64_t width, float eps) {
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) if (threadIdx.x == 0) {
        double square = 0.0;
        for (int64_t column = 0; column < width; ++column) {
            const float value = __bfloat162float(input[row * width + column]);
            square += static_cast<double>(value) * value;
        }
        const float inverse = rsqrtf(static_cast<float>(square / width) + eps);
        for (int64_t column = 0; column < width; ++column) {
            // Preserve contracts that round the normalized value to the input
            // dtype before multiplying an FP32 affine parameter.
            const __nv_bfloat16 normalized = __float2bfloat16(
                __bfloat162float(input[row * width + column]) * inverse);
            output[row * width + column] = __bfloat162float(normalized) * weight[column];
        }
    }
}

__global__ void rms_norm_bf16_input_f32_weight_parallel_kernel(
        const __nv_bfloat16* input, const float* weight, float* output,
        int64_t rows, int64_t width, float eps) {
    __shared__ double square_workspace[kNormThreads];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double square = 0.0;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = __bfloat162float(input[row * width + column]);
            square += static_cast<double>(value) * value;
        }
        square_workspace[threadIdx.x] = square;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride)
                square_workspace[threadIdx.x] += square_workspace[threadIdx.x + stride];
            __syncthreads();
        }
        const float inverse = rsqrtf(static_cast<float>(
            square_workspace[0] / width) + eps);
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
            const __nv_bfloat16 normalized = __float2bfloat16(
                __bfloat162float(input[row * width + column]) * inverse);
            output[row * width + column] = __bfloat162float(normalized) * weight[column];
        }
        __syncthreads();
    }
}

constexpr int kAxisNormInnerTile = 32;
constexpr int kAxisNormReductionThreads = kNormThreads / kAxisNormInnerTile;

template <typename T>
__global__ void rms_norm_axis_parallel_kernel(
        const T* input, const T* weight, T* output,
        int64_t outer, int64_t axis_size, int64_t inner, float eps) {
    static_assert(kNormThreads % kAxisNormInnerTile == 0);
    __shared__ double square_workspace[kNormThreads];
    __shared__ float inverse_workspace[kAxisNormInnerTile];
    const int inner_lane = threadIdx.x % kAxisNormInnerTile;
    const int axis_lane = threadIdx.x / kAxisNormInnerTile;
    const int64_t inner_tiles = (inner + kAxisNormInnerTile - 1) /
                                kAxisNormInnerTile;
    const int64_t total_tiles = outer * inner_tiles;
    for (int64_t logical_block = blockIdx.x; logical_block < total_tiles;
         logical_block += gridDim.x) {
        const int64_t outer_index = logical_block / inner_tiles;
        const int64_t inner_index = (logical_block % inner_tiles) *
                                    kAxisNormInnerTile + inner_lane;
        double square = 0.0;
        if (inner_index < inner) {
            const int64_t base = outer_index * axis_size * inner + inner_index;
            for (int64_t coordinate = axis_lane; coordinate < axis_size;
                 coordinate += kAxisNormReductionThreads) {
                const float value = load_value(input[base + coordinate * inner]);
                square += static_cast<double>(value) * value;
            }
        }
        square_workspace[axis_lane * kAxisNormInnerTile + inner_lane] = square;
        __syncthreads();
        if (axis_lane == 0 && inner_index < inner) {
            double total = 0.0;
            for (int lane = 0; lane < kAxisNormReductionThreads; ++lane)
                total += square_workspace[lane * kAxisNormInnerTile + inner_lane];
            inverse_workspace[inner_lane] = rsqrtf(
                static_cast<float>(total / axis_size) + eps);
        }
        __syncthreads();
        if (inner_index < inner) {
            const int64_t base = outer_index * axis_size * inner + inner_index;
            const float inverse = inverse_workspace[inner_lane];
            for (int64_t coordinate = axis_lane; coordinate < axis_size;
                 coordinate += kAxisNormReductionThreads) {
                const int64_t index = base + coordinate * inner;
                float result = load_value(input[index]) * inverse;
                if constexpr (std::is_same_v<T, __nv_bfloat16>)
                    result = load_value(store_value<T>(result));
                if (weight) result *= load_value(weight[coordinate]);
                output[index] = store_value<T>(result);
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void activation_kernel(const T* input, T* output, int64_t count, int kind) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float x = load_value(input[index]); float value;
        if (kind == 0) value = x / (1.0f + expf(-x));
        else if (kind == 1) value = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        else if (kind == 2) value = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        else if (kind == 3) value = tanhf(x);
        else value = fmaxf(x, 0.0f);
        output[index] = store_value<T>(value);
    }
}

template <typename T>
__global__ void max_pool2d_kernel(const T* input, T* output, int64_t count,
                                  int channels, int input_height,
                                  int input_width, int output_height,
                                  int output_width, int kernel_height,
                                  int kernel_width, int stride_height,
                                  int stride_width, int padding_height,
                                  int padding_width) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        const int output_x = remaining % output_width; remaining /= output_width;
        const int output_y = remaining % output_height; remaining /= output_height;
        const int channel = remaining % channels; const int item = remaining / channels;
        float maximum = -3.402823466e+38F;
        for (int ky = 0; ky < kernel_height; ++ky) {
            const int input_y = output_y * stride_height - padding_height + ky;
            if (input_y < 0 || input_y >= input_height) continue;
            for (int kx = 0; kx < kernel_width; ++kx) {
                const int input_x = output_x * stride_width - padding_width + kx;
                if (input_x < 0 || input_x >= input_width) continue;
                const int64_t source = ((static_cast<int64_t>(item) * channels + channel) *
                    input_height + input_y) * input_width + input_x;
                maximum = fmaxf(maximum, load_value(input[source]));
            }
        }
        output[index] = store_value<T>(maximum);
    }
}

template <typename T>
__global__ void l2_normalize_kernel(const T* input, T* output, int64_t units,
                                    Meta tensor_meta, int normalized_dim,
                                    float eps) {
    __shared__ float workspace[kThreads];
    __shared__ float denominator;
    const int64_t unit = blockIdx.x;
    if (unit >= units) return;
    int64_t remaining = unit, base = 0;
    for (int dimension = tensor_meta.rank - 1; dimension >= 0; --dimension) {
        if (dimension == normalized_dim) continue;
        const int64_t coordinate = remaining % tensor_meta.shape[dimension];
        remaining /= tensor_meta.shape[dimension];
        base += coordinate * tensor_meta.strides[dimension];
    }
    const int64_t channels = tensor_meta.shape[normalized_dim];
    const int64_t channel_stride = tensor_meta.strides[normalized_dim];
    float square = 0.0f;
    for (int64_t channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        const float item = load_value(input[base + channel * channel_stride]);
        square += item * item;
    }
    workspace[threadIdx.x] = square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) workspace[threadIdx.x] += workspace[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denominator = sqrtf(workspace[0]) + eps;
    __syncthreads();
    for (int64_t channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        const int64_t index = base + channel * channel_stride;
        output[index] = store_value<T>(load_value(input[index]) / denominator);
    }
}

template <typename T>
__global__ void sinusoidal_kernel(const T* positions, T* output, int64_t count,
                                  int width, bool flip, double downscale, bool use_double) {
    const int half = width / 2;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count * width; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int column = index % width; const int64_t row = index / width;
        if (column >= 2 * half) { output[index] = store_value<T>(0.0f); continue; }
        const int logical = flip ? (column + half) % (2 * half) : column;
        const int frequency = logical % half;
        if (use_double) {
            const double exponent = -log(10000.0) * frequency / (half - downscale);
            const double phase = static_cast<double>(load_value(positions[row])) * exp(exponent);
            output[index] = store_value<T>(static_cast<float>(logical < half ? sin(phase) : cos(phase)));
        } else {
            const float exponent = -logf(10000.0f) * frequency / (half - static_cast<float>(downscale));
            const float phase = load_value(positions[row]) * expf(exponent);
            output[index] = store_value<T>(logical < half ? sinf(phase) : cosf(phase));
        }
    }
}

template <typename T, typename F>
__global__ void rope_kernel(const T* input, const F* cosine, const F* sine,
                            T* output, int64_t count, Meta x_meta, Meta frequency_meta) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t pair = (index % x_meta.shape[x_meta.rank - 1]) ^ 1;
        const int64_t paired_index = index - index % x_meta.shape[x_meta.rank - 1] + pair;
        const float paired = load_value(input[paired_index]);
        const float rotated = (index & 1) ? paired : -paired;
        const int64_t f = broadcast_offset(index, x_meta, frequency_meta);
        output[index] = store_value<T>(load_value(input[index]) * load_value(cosine[f]) + rotated * load_value(sine[f]));
    }
}

template <typename T, typename O = T>
__global__ void attention_kernel(const T* q, const T* k, const T* v,
                                 const uint8_t* mask, const T* bias, O* output,
                                 int batch, int q_tokens, int k_tokens, int heads,
                                 int width, int output_tiles, bool causal, float scale,
                                 Meta mask_meta, Meta bias_meta) {
    using namespace cuda_attention_config;
    const int lane = threadIdx.x % kWarpThreads;
    const int warp = threadIdx.x / kWarpThreads;
    const int output_tile = blockIdx.x % output_tiles;
    int64_t group = blockIdx.x / output_tiles;
    const int head = group % heads; group /= heads;
    const int query_tile = group % ((q_tokens + kQueryTile - 1) / kQueryTile);
    const int b = group / ((q_tokens + kQueryTile - 1) / kQueryTile);
    const int query = query_tile * kQueryTile + warp;
    const bool active_query = query < q_tokens;

    extern __shared__ float workspace[];
    float* shared_k = workspace;
    float* shared_v = workspace + kKeyTile * kHeadTile;
    const bool cache_kv = width <= kHeadTile;

    float accumulator[kOutputsPerLane]{};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    const int output_base = output_tile * kHeadTile + lane;

    Meta mask_logical{};
    mask_logical.rank = 3;
    mask_logical.shape[0] = batch;
    mask_logical.shape[1] = q_tokens;
    mask_logical.shape[2] = k_tokens;
    Meta bias_logical{};
    bias_logical.rank = 4;
    bias_logical.shape[0] = batch;
    bias_logical.shape[1] = heads;
    bias_logical.shape[2] = q_tokens;
    bias_logical.shape[3] = k_tokens;

    for (int key_base = 0; key_base < k_tokens; key_base += kKeyTile) {
        const int key_count = min(kKeyTile, k_tokens - key_base);
        if (cache_kv) {
            for (int index = threadIdx.x; index < kKeyTile * kHeadTile;
                 index += blockDim.x) {
                const int local_key = index / kHeadTile;
                const int dimension = index % kHeadTile;
                float key_value = 0.0f, value_value = 0.0f;
                if (local_key < key_count && dimension < width) {
                    const int key = key_base + local_key;
                    const int64_t offset =
                        (((static_cast<int64_t>(b) * k_tokens + key) * heads + head) *
                         width + dimension);
                    key_value = load_value(k[offset]);
                    value_value = load_value(v[offset]);
                }
                shared_k[index] = key_value;
                shared_v[index] = value_value;
            }
        }
        __syncthreads();

        if (active_query) {
            for (int local_key = 0; local_key < key_count; ++local_key) {
                const int key = key_base + local_key;
                float partial = 0.0f;
                for (int dimension = lane; dimension < width;
                     dimension += kWarpThreads) {
                    const int64_t qi =
                        (((static_cast<int64_t>(b) * q_tokens + query) * heads + head) *
                         width + dimension);
                    const int64_t ki =
                        (((static_cast<int64_t>(b) * k_tokens + key) * heads + head) *
                         width + dimension);
                    const float key_value = cache_kv
                        ? shared_k[local_key * kHeadTile + dimension]
                        : load_value(k[ki]);
                    partial += load_value(q[qi]) * key_value;
                }
                for (int offset = kWarpThreads / 2; offset > 0; offset /= 2)
                    partial += __shfl_down_sync(0xffffffffU, partial, offset);

                float score = -INFINITY;
                if (lane == 0) {
                    bool valid = !causal || key <= query;
                    if (mask) {
                        const int64_t logical =
                            ((static_cast<int64_t>(b) * q_tokens + query) * k_tokens + key);
                        valid = valid && mask[broadcast_offset(logical, mask_logical,
                                                               mask_meta)];
                    }
                    if (valid) {
                        score = partial * scale;
                        if (bias) {
                            const int64_t logical =
                                (((static_cast<int64_t>(b) * heads + head) * q_tokens +
                                  query) * k_tokens + key);
                            score += load_value(bias[broadcast_offset(logical, bias_logical,
                                                                      bias_meta)]);
                        }
                    }
                }
                score = __shfl_sync(0xffffffffU, score, 0);

                if (isfinite(score)) {
                    const float updated_maximum = fmaxf(maximum, score);
                    const float previous_scale = isfinite(maximum)
                        ? expf(maximum - updated_maximum) : 0.0f;
                    const float current_scale = expf(score - updated_maximum);
                    denominator = denominator * previous_scale + current_scale;
                    for (int item = 0; item < kOutputsPerLane; ++item) {
                        const int dimension = output_base + item * kWarpThreads;
                        if (dimension < width) {
                            const int64_t vi =
                                (((static_cast<int64_t>(b) * k_tokens + key) * heads + head) *
                                 width + dimension);
                            const float value_value = cache_kv
                                ? shared_v[local_key * kHeadTile + dimension]
                                : load_value(v[vi]);
                            accumulator[item] = accumulator[item] * previous_scale +
                                                current_scale * value_value;
                        }
                    }
                    maximum = updated_maximum;
                }
            }
        }
        __syncthreads();
    }

    if (active_query) {
        for (int item = 0; item < kOutputsPerLane; ++item) {
            const int dimension = output_base + item * kWarpThreads;
            if (dimension < width) {
                const int64_t oi =
                    (((static_cast<int64_t>(b) * q_tokens + query) * heads + head) * width +
                     dimension);
                output[oi] = store_value<O>(accumulator[item] / denominator);
            }
        }
    }
}

__global__ void attention_ordered_initialize_kernel(float* maximum, float* denominator,
                                                    float* output, int64_t rows,
                                                    int width) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < rows * width;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        output[index] = 0.0f;
        if (index < rows) {
            maximum[index] = -INFINITY;
            denominator[index] = 0.0f;
        }
    }
}

template <typename B>
__global__ void attention_ordered_state_kernel(
    float* scores_or_current, float* previous, float* maximum, float* denominator,
    const uint8_t* mask, const B* bias, int batch, int q_tokens, int k_tokens,
    int heads, int key_base, int key_count, float scale, bool causal,
    Meta mask_meta, Meta bias_meta) {
    const int64_t rows = static_cast<int64_t>(batch) * q_tokens * heads;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int head = row % heads;
        const int64_t bq = row / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t score_base =
            (static_cast<int64_t>(b) * heads + head) * q_tokens * key_count +
            static_cast<int64_t>(query) * key_count;
        float row_maximum = maximum[row];
        float row_denominator = denominator[row];

        Meta mask_logical{};
        mask_logical.rank = 3;
        mask_logical.shape[0] = batch;
        mask_logical.shape[1] = q_tokens;
        mask_logical.shape[2] = k_tokens;
        Meta bias_logical{};
        bias_logical.rank = 4;
        bias_logical.shape[0] = batch;
        bias_logical.shape[1] = heads;
        bias_logical.shape[2] = q_tokens;
        bias_logical.shape[3] = k_tokens;

        for (int local_key = 0; local_key < key_count; ++local_key) {
            const int key = key_base + local_key;
            bool valid = !causal || key <= query;
            if (mask) {
                const int64_t logical =
                    (static_cast<int64_t>(b) * q_tokens + query) * k_tokens + key;
                valid = valid && mask[broadcast_offset(logical, mask_logical, mask_meta)];
            }
            float previous_scale = 1.0f;
            float current_scale = 0.0f;
            if (valid) {
                float score = scores_or_current[score_base + local_key] * scale;
                if (bias) {
                    const int64_t logical =
                        ((static_cast<int64_t>(b) * heads + head) * q_tokens + query) *
                            k_tokens + key;
                    score += load_value(bias[broadcast_offset(logical, bias_logical, bias_meta)]);
                }
                const float updated_maximum = fmaxf(row_maximum, score);
                previous_scale = isfinite(row_maximum)
                    ? expf(row_maximum - updated_maximum) : 0.0f;
                current_scale = expf(score - updated_maximum);
                row_denominator = row_denominator * previous_scale + current_scale;
                row_maximum = updated_maximum;
            }
            previous[score_base + local_key] = previous_scale;
            scores_or_current[score_base + local_key] = current_scale;
        }
        maximum[row] = row_maximum;
        denominator[row] = row_denominator;
    }
}

template <typename V>
__global__ void attention_ordered_pv_kernel(
    const float* current, const float* previous, const V* v, float* output,
    int batch, int q_tokens, int k_tokens, int heads, int width,
    int key_base, int key_count) {
    const int64_t count = static_cast<int64_t>(batch) * q_tokens * heads * width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int dimension = index % width;
        const int64_t row = index / width;
        const int head = row % heads;
        const int64_t bq = row / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t score_base =
            (static_cast<int64_t>(b) * heads + head) * q_tokens * key_count +
            static_cast<int64_t>(query) * key_count;
        float accumulator = output[index];
        for (int local_key = 0; local_key < key_count; ++local_key) {
            const int key = key_base + local_key;
            const int64_t vi =
                ((static_cast<int64_t>(b) * k_tokens + key) * heads + head) * width +
                dimension;
            accumulator = accumulator * previous[score_base + local_key] +
                          current[score_base + local_key] * load_value(v[vi]);
        }
        output[index] = accumulator;
    }
}

__global__ void attention_ordered_pv_float4_kernel(
    const float* current, const float* previous, const float* v, float* output,
    int batch, int q_tokens, int k_tokens, int heads, int width,
    int key_base, int key_count) {
    const int vectors = width / 4;
    const int64_t count = static_cast<int64_t>(batch) * q_tokens * heads * vectors;
    const float4* values = reinterpret_cast<const float4*>(v);
    float4* outputs = reinterpret_cast<float4*>(output);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int vector = index % vectors;
        const int64_t row = index / vectors;
        const int head = row % heads;
        const int64_t bq = row / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t score_base =
            (static_cast<int64_t>(b) * heads + head) * q_tokens * key_count +
            static_cast<int64_t>(query) * key_count;
        float4 accumulator = outputs[index];
        for (int local_key = 0; local_key < key_count; ++local_key) {
            const int key = key_base + local_key;
            const int64_t vi =
                ((static_cast<int64_t>(b) * k_tokens + key) * heads + head) * vectors +
                vector;
            const float4 value = values[vi];
            const float previous_scale = previous[score_base + local_key];
            const float current_scale = current[score_base + local_key];
            accumulator.x = accumulator.x * previous_scale + current_scale * value.x;
            accumulator.y = accumulator.y * previous_scale + current_scale * value.y;
            accumulator.z = accumulator.z * previous_scale + current_scale * value.z;
            accumulator.w = accumulator.w * previous_scale + current_scale * value.w;
        }
        outputs[index] = accumulator;
    }
}

__global__ void attention_ordered_pv_bf16x2_kernel(
    const float* current, const float* previous, const __nv_bfloat16* v, float* output,
    int batch, int q_tokens, int k_tokens, int heads, int width,
    int key_base, int key_count) {
    const int pairs = width / 2;
    const int64_t count = static_cast<int64_t>(batch) * q_tokens * heads * pairs;
    const __nv_bfloat162* values = reinterpret_cast<const __nv_bfloat162*>(v);
    float2* outputs = reinterpret_cast<float2*>(output);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int pair = index % pairs;
        const int64_t row = index / pairs;
        const int head = row % heads;
        const int64_t bq = row / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t score_base =
            (static_cast<int64_t>(b) * heads + head) * q_tokens * key_count +
            static_cast<int64_t>(query) * key_count;
        float2 accumulator = outputs[index];
        for (int local_key = 0; local_key < key_count; ++local_key) {
            const int key = key_base + local_key;
            const int64_t vi =
                ((static_cast<int64_t>(b) * k_tokens + key) * heads + head) * pairs + pair;
            const float2 value = __bfloat1622float2(values[vi]);
            const float previous_scale = previous[score_base + local_key];
            const float current_scale = current[score_base + local_key];
            accumulator.x = accumulator.x * previous_scale + current_scale * value.x;
            accumulator.y = accumulator.y * previous_scale + current_scale * value.y;
        }
        outputs[index] = accumulator;
    }
}

template <typename T>
__global__ void attention_ordered_finalize_kernel(const float* accumulation,
                                                  const float* denominator, T* output,
                                                  int64_t rows, int width) {
    const int64_t count = rows * width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = store_value<T>(accumulation[index] / denominator[index / width]);
}

__global__ void attention_two_pass_initialize_kernel(float* maximum, float* denominator,
                                                     float* accumulation,
                                                     int64_t rows, int width) {
    const int64_t count = rows * width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        accumulation[index] = 0.0f;
        if (index < rows) {
            maximum[index] = -INFINITY;
            denominator[index] = 0.0f;
        }
    }
}

template <typename B>
__global__ void attention_two_pass_statistics_kernel(
    const float* scores, float* maximum, float* denominator,
    const uint8_t* mask, const B* bias, int batch, int q_tokens, int k_tokens,
    int heads, int key_base, int key_count, float scale, bool causal,
    Meta mask_meta, Meta bias_meta) {
    const int64_t rows = static_cast<int64_t>(batch) * q_tokens * heads;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int head = row % heads;
        const int64_t bq = row / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t score_base =
            (static_cast<int64_t>(b) * heads + head) * q_tokens * key_count +
            static_cast<int64_t>(query) * key_count;
        float row_maximum = maximum[row];
        float row_denominator = denominator[row];
        Meta mask_logical{};
        mask_logical.rank = 3;
        mask_logical.shape[0] = batch;
        mask_logical.shape[1] = q_tokens;
        mask_logical.shape[2] = k_tokens;
        Meta bias_logical{};
        bias_logical.rank = 4;
        bias_logical.shape[0] = batch;
        bias_logical.shape[1] = heads;
        bias_logical.shape[2] = q_tokens;
        bias_logical.shape[3] = k_tokens;
        for (int local_key = 0; local_key < key_count; ++local_key) {
            const int key = key_base + local_key;
            bool valid = !causal || key <= query;
            if (mask) {
                const int64_t logical =
                    (static_cast<int64_t>(b) * q_tokens + query) * k_tokens + key;
                valid = valid && mask[broadcast_offset(logical, mask_logical, mask_meta)];
            }
            if (!valid) continue;
            float score = scores[score_base + local_key] * scale;
            if (bias) {
                const int64_t logical =
                    ((static_cast<int64_t>(b) * heads + head) * q_tokens + query) *
                        k_tokens + key;
                score += load_value(bias[broadcast_offset(logical, bias_logical,
                                                           bias_meta)]);
            }
            const float updated_maximum = fmaxf(row_maximum, score);
            const float previous_scale = isfinite(row_maximum)
                ? expf(row_maximum - updated_maximum) : 0.0f;
            row_denominator = row_denominator * previous_scale +
                              expf(score - updated_maximum);
            row_maximum = updated_maximum;
        }
        maximum[row] = row_maximum;
        denominator[row] = row_denominator;
    }
}

template <typename B>
__global__ void attention_two_pass_probability_kernel(
    const float* scores, __nv_bfloat16* probabilities,
    const float* maximum, const float* denominator,
    const uint8_t* mask, const B* bias, int batch, int q_tokens, int k_tokens,
    int heads, int key_base, int key_count, float scale, bool causal,
    Meta mask_meta, Meta bias_meta) {
    const int64_t count = static_cast<int64_t>(batch) * heads * q_tokens * key_count;
    Meta mask_logical{};
    mask_logical.rank = 3;
    mask_logical.shape[0] = batch;
    mask_logical.shape[1] = q_tokens;
    mask_logical.shape[2] = k_tokens;
    Meta bias_logical{};
    bias_logical.rank = 4;
    bias_logical.shape[0] = batch;
    bias_logical.shape[1] = heads;
    bias_logical.shape[2] = q_tokens;
    bias_logical.shape[3] = k_tokens;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int local_key = index % key_count;
        const int64_t row_bhq = index / key_count;
        const int query = row_bhq % q_tokens;
        const int64_t bh = row_bhq / q_tokens;
        const int head = bh % heads;
        const int b = bh / heads;
        const int key = key_base + local_key;
        bool valid = !causal || key <= query;
        if (mask) {
            const int64_t logical =
                (static_cast<int64_t>(b) * q_tokens + query) * k_tokens + key;
            valid = valid && mask[broadcast_offset(logical, mask_logical, mask_meta)];
        }
        float probability = 0.0f;
        const int64_t state_row =
            (static_cast<int64_t>(b) * q_tokens + query) * heads + head;
        if (valid && denominator[state_row] > 0.0f) {
            float score = scores[index] * scale;
            if (bias) {
                const int64_t logical =
                    ((static_cast<int64_t>(b) * heads + head) * q_tokens + query) *
                        k_tokens + key;
                score += load_value(bias[broadcast_offset(logical, bias_logical,
                                                           bias_meta)]);
            }
            // Preserve dynamic range by rounding the unnormalized exponential
            // to BF16.  Normalization remains an FP32 boundary operation after
            // the Tensor-Op P×V accumulation.
            probability = expf(score - maximum[state_row]);
        }
        probabilities[index] = __float2bfloat16_rn(probability);
    }
}

template <typename O>
__global__ void attention_two_pass_finalize_kernel(
    const float* head_major, const float* denominator, O* output,
    int batch, int q_tokens,
    int heads, int width) {
    const int64_t count = static_cast<int64_t>(batch) * q_tokens * heads * width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int dimension = index % width;
        const int64_t bqh = index / width;
        const int head = bqh % heads;
        const int64_t bq = bqh / heads;
        const int query = bq % q_tokens;
        const int b = bq / q_tokens;
        const int64_t source =
            ((static_cast<int64_t>(b) * heads + head) * q_tokens + query) * width +
            dimension;
        const int64_t state_row =
            (static_cast<int64_t>(b) * q_tokens + query) * heads + head;
        output[index] = store_value<O>(head_major[source] / denominator[state_row]);
    }
}

template <typename T, typename I>
__global__ void indexed_gather_kernel(const T* table, const I* indices, T* output,
                                      int64_t output_count, int64_t index_count,
                                      int64_t rows, int64_t width, bool batched) {
    for (int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < output_count;
         output_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t logical_index = output_index / width;
        const int64_t column = output_index % width;
        const int64_t batch = batched ? logical_index / index_count : 0;
        const int64_t local_index = batched ? logical_index % index_count : logical_index;
        const int64_t row = static_cast<int64_t>(indices[batched ? batch * index_count + local_index
                                                                : local_index]);
        output[output_index] = table[(batch * rows + row) * width + column];
    }
}

template <typename T>
__global__ void pad_kernel(const T* input, T* output, int64_t count,
                           Meta input_meta, Meta output_meta, Meta before, float value, bool replicate) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, source = 0; bool valid = true;
        for (int dimension = output_meta.rank - 1; dimension >= 0; --dimension) {
            int64_t coordinate = remaining % output_meta.shape[dimension]; remaining /= output_meta.shape[dimension];
            coordinate -= before.shape[dimension];
            if (coordinate < 0 || coordinate >= input_meta.shape[dimension]) {
                if (!replicate) valid = false;
                coordinate = max(static_cast<int64_t>(0), min(input_meta.shape[dimension] - 1, coordinate));
            }
            source += coordinate * input_meta.strides[dimension];
        }
        output[index] = valid ? input[source] : store_value<T>(value);
    }
}

template <typename T>
__global__ void reduce_sum_kernel(const T* input, T* output, int64_t count,
                                  Meta input_meta, Meta output_meta, int reduced_dim) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, base = 0;
        for (int dimension = output_meta.rank - 1; dimension >= 0; --dimension) {
            const int64_t coordinate = remaining % output_meta.shape[dimension]; remaining /= output_meta.shape[dimension];
            const int input_dim = output_meta.rank == input_meta.rank ? dimension : (dimension < reduced_dim ? dimension : dimension + 1);
            if (input_dim != reduced_dim) base += coordinate * input_meta.strides[input_dim];
        }
        float sum = 0.0f;
        for (int64_t coordinate = 0; coordinate < input_meta.shape[reduced_dim]; ++coordinate)
            sum += load_value(input[base + coordinate * input_meta.strides[reduced_dim]]);
        output[index] = store_value<T>(sum);
    }
}

template <typename T>
__global__ void softmax_axis_kernel(const T* input, T* output,
                                    int64_t rows, int64_t axis_size,
                                    int64_t inner) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t outer_index = row / inner;
        const int64_t inner_index = row % inner;
        const int64_t base = outer_index * axis_size * inner + inner_index;
        float maximum = -INFINITY;
        for (int64_t coordinate = 0; coordinate < axis_size; ++coordinate)
            maximum = fmaxf(maximum,
                load_value(input[base + coordinate * inner]));
        float denominator = 0.0f;
        for (int64_t coordinate = 0; coordinate < axis_size; ++coordinate)
            denominator += expf(load_value(input[base + coordinate * inner]) - maximum);
        for (int64_t coordinate = 0; coordinate < axis_size; ++coordinate)
            output[base + coordinate * inner] = store_value<T>(
                expf(load_value(input[base + coordinate * inner]) - maximum) /
                denominator);
    }
}

template <typename T>
__global__ void conv_transpose2d_kernel(
        const T* input, const T* weight, const T* bias, T* output,
        int64_t count, int batch, int input_channels, int output_channels,
        int input_height, int input_width, int output_height, int output_width,
        int kernel_height, int kernel_width, int stride_height, int stride_width,
        int padding_height, int padding_width, int dilation_height,
        int dilation_width, int groups, bool has_bias) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        const int output_x = remaining % output_width; remaining /= output_width;
        const int output_y = remaining % output_height; remaining /= output_height;
        const int output_channel = remaining % output_channels;
        const int batch_index = remaining / output_channels;
        const int output_per_group = output_channels / groups;
        const int input_per_group = input_channels / groups;
        const int group = output_channel / output_per_group;
        const int local_output = output_channel % output_per_group;
        float value = has_bias ? load_value(bias[output_channel]) : 0.0f;
        for (int local_input = 0; local_input < input_per_group; ++local_input) {
            const int input_channel = group * input_per_group + local_input;
            for (int ky = 0; ky < kernel_height; ++ky) {
                const int numerator_y = output_y + padding_height -
                    ky * dilation_height;
                if (numerator_y < 0 || numerator_y % stride_height != 0) continue;
                const int input_y = numerator_y / stride_height;
                if (input_y >= input_height) continue;
                for (int kx = 0; kx < kernel_width; ++kx) {
                    const int numerator_x = output_x + padding_width -
                        kx * dilation_width;
                    if (numerator_x < 0 || numerator_x % stride_width != 0) continue;
                    const int input_x = numerator_x / stride_width;
                    if (input_x >= input_width) continue;
                    const int64_t input_index =
                        ((static_cast<int64_t>(batch_index) * input_channels +
                          input_channel) * input_height + input_y) * input_width + input_x;
                    const int64_t weight_index =
                        ((static_cast<int64_t>(input_channel) * output_per_group +
                          local_output) * kernel_height + ky) * kernel_width + kx;
                    value += load_value(input[input_index]) *
                             load_value(weight[weight_index]);
                }
            }
        }
        output[index] = store_value<T>(value);
    }
}

constexpr int kGroupNormThreads = 256;
constexpr int64_t kGroupNormDirectMaximum = 1LL << 18;
constexpr int64_t kGroupNormElementsPerPartial = 1LL << 20;
constexpr int kGroupNormMaximumPartials = 32;

__device__ float group_norm_block_sum(float value, float* workspace) {
    workspace[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            workspace[threadIdx.x] += workspace[threadIdx.x + stride];
        __syncthreads();
    }
    return workspace[0];
}

// Stable two-pass moments avoid E[x^2] - E[x]^2 cancellation. The frozen
// normalization contract declares FP32 statistics; FP64 was an implementation
// detail of the retired scalar kernel, not a semantic requirement.
template <typename T>
__global__ void group_norm_direct_kernel(
        const T* input, const T* weight, const T* bias, T* output,
        int channels, int64_t spatial, int groups, float eps) {
    __shared__ float workspace[kGroupNormThreads];
    __shared__ float shared_anchor;
    __shared__ float shared_mean;
    __shared__ float shared_inverse;
    const int unit = blockIdx.x;
    const int group = unit % groups;
    const int channels_per = channels / groups;
    const int64_t count = static_cast<int64_t>(channels_per) * spatial;
    const int64_t base = static_cast<int64_t>(unit) * count;
    if (threadIdx.x == 0) shared_anchor = load_value(input[base]);
    __syncthreads();
    float sum = 0.0f;
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x)
        sum += load_value(input[base + index]) - shared_anchor;
    const float total = group_norm_block_sum(sum, workspace);
    if (threadIdx.x == 0) shared_mean = shared_anchor + total / count;
    __syncthreads();
    float square = 0.0f;
    for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float delta = load_value(input[base + index]) - shared_mean;
        square += delta * delta;
    }
    const float square_total = group_norm_block_sum(square, workspace);
    if (threadIdx.x == 0)
        shared_inverse = rsqrtf(square_total / count + eps);
    __syncthreads();
    for (int channel = 0; channel < channels_per; ++channel) {
        const int absolute_channel = group * channels_per + channel;
        const float scale = weight ? load_value(weight[absolute_channel]) : 1.0f;
        const float shift = bias ? load_value(bias[absolute_channel]) : 0.0f;
        const int64_t channel_base = base + static_cast<int64_t>(channel) * spatial;
        for (int64_t position = threadIdx.x; position < spatial;
             position += blockDim.x) {
            const int64_t index = channel_base + position;
            const float normalized =
                (load_value(input[index]) - shared_mean) * shared_inverse;
            output[index] = store_value<T>(normalized * scale + shift);
        }
    }
}

template <typename T>
__global__ void group_norm_partial_sum_kernel(
        const T* input, float* partial, int64_t count, int partials) {
    __shared__ float workspace[kGroupNormThreads];
    const int unit = blockIdx.x / partials;
    const int partial_index = blockIdx.x % partials;
    const int64_t base = static_cast<int64_t>(unit) * count;
    const float anchor = load_value(input[base]);
    float sum = 0.0f;
    for (int64_t index = static_cast<int64_t>(partial_index) * blockDim.x +
                         threadIdx.x;
         index < count; index += static_cast<int64_t>(partials) * blockDim.x)
        sum += load_value(input[base + index]) - anchor;
    const float total = group_norm_block_sum(sum, workspace);
    if (threadIdx.x == 0)
        partial[static_cast<int64_t>(unit) * partials + partial_index] = total;
}

template <typename T>
__global__ void group_norm_mean_kernel(
        const T* input, const float* partial, float* mean,
        int64_t count, int partials) {
    __shared__ float workspace[kGroupNormThreads];
    float sum = 0.0f;
    for (int index = threadIdx.x; index < partials; index += blockDim.x)
        sum += partial[static_cast<int64_t>(blockIdx.x) * partials + index];
    const float total = group_norm_block_sum(sum, workspace);
    if (threadIdx.x == 0)
        mean[blockIdx.x] = load_value(input[static_cast<int64_t>(blockIdx.x) * count]) +
                           total / count;
}

template <typename T>
__global__ void group_norm_partial_variance_kernel(
        const T* input, const float* mean, float* partial,
        int64_t count, int partials) {
    __shared__ float workspace[kGroupNormThreads];
    const int unit = blockIdx.x / partials;
    const int partial_index = blockIdx.x % partials;
    const int64_t base = static_cast<int64_t>(unit) * count;
    const float unit_mean = mean[unit];
    float square = 0.0f;
    for (int64_t index = static_cast<int64_t>(partial_index) * blockDim.x +
                         threadIdx.x;
         index < count; index += static_cast<int64_t>(partials) * blockDim.x) {
        const float delta = load_value(input[base + index]) - unit_mean;
        square += delta * delta;
    }
    const float total = group_norm_block_sum(square, workspace);
    if (threadIdx.x == 0)
        partial[static_cast<int64_t>(unit) * partials + partial_index] = total;
}

__global__ void group_norm_inverse_kernel(
        const float* partial, float* inverse, int64_t count, int partials,
        float eps) {
    __shared__ float workspace[kGroupNormThreads];
    float square = 0.0f;
    for (int index = threadIdx.x; index < partials; index += blockDim.x)
        square += partial[static_cast<int64_t>(blockIdx.x) * partials + index];
    const float total = group_norm_block_sum(square, workspace);
    if (threadIdx.x == 0)
        inverse[blockIdx.x] = rsqrtf(total / count + eps);
}

template <typename T>
__global__ void group_norm_normalize_kernel(
        const T* input, const T* weight, const T* bias, T* output,
        const float* mean, const float* inverse, int channels,
        int64_t spatial, int groups) {
    const int channels_per = channels / groups;
    const int unit = blockIdx.x / channels_per;
    const int local_channel = blockIdx.x % channels_per;
    const int group = unit % groups;
    const int absolute_channel = group * channels_per + local_channel;
    const int64_t count = static_cast<int64_t>(channels_per) * spatial;
    const int64_t base = static_cast<int64_t>(unit) * count +
                         static_cast<int64_t>(local_channel) * spatial;
    const float scale = weight ? load_value(weight[absolute_channel]) : 1.0f;
    const float shift = bias ? load_value(bias[absolute_channel]) : 0.0f;
    for (int64_t position = threadIdx.x; position < spatial;
         position += blockDim.x) {
        const int64_t index = base + position;
        const float normalized =
            (load_value(input[index]) - mean[unit]) * inverse[unit];
        output[index] = store_value<T>(normalized * scale + shift);
    }
}

template <typename T>
__global__ void interpolate_kernel(const T* input, T* output, int64_t count,
                                   Meta input_meta, Meta output_meta) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index, source = 0;
        for (int dimension = output_meta.rank - 1; dimension >= 0; --dimension) {
            const int64_t coordinate = remaining % output_meta.shape[dimension]; remaining /= output_meta.shape[dimension];
            const int64_t input_coordinate = coordinate * input_meta.shape[dimension] / output_meta.shape[dimension];
            source += input_coordinate * input_meta.strides[dimension];
        }
        output[index] = input[source];
    }
}

template <typename T>
__global__ void interpolate_bilinear_2d_kernel(
        const T* input, T* output, int64_t count, int channels,
        int input_height, int input_width, int output_height,
        int output_width, bool align_corners) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        const int output_x = remaining % output_width; remaining /= output_width;
        const int output_y = remaining % output_height; remaining /= output_height;
        const int channel = remaining % channels;
        const int batch = remaining / channels;
        float source_y = align_corners && output_height > 1
            ? static_cast<float>(output_y) * (input_height - 1) / (output_height - 1)
            : (static_cast<float>(output_y) + 0.5f) * input_height / output_height - 0.5f;
        float source_x = align_corners && output_width > 1
            ? static_cast<float>(output_x) * (input_width - 1) / (output_width - 1)
            : (static_cast<float>(output_x) + 0.5f) * input_width / output_width - 0.5f;
        source_y = fminf(fmaxf(source_y, 0.0f), static_cast<float>(input_height - 1));
        source_x = fminf(fmaxf(source_x, 0.0f), static_cast<float>(input_width - 1));
        const int y0 = static_cast<int>(floorf(source_y));
        const int x0 = static_cast<int>(floorf(source_x));
        const int y1 = min(y0 + 1, input_height - 1);
        const int x1 = min(x0 + 1, input_width - 1);
        const float ly = source_y - y0, lx = source_x - x0;
        const int64_t base = (static_cast<int64_t>(batch) * channels + channel) *
                             input_height * input_width;
        const float top = load_value(input[base + y0 * input_width + x0]) * (1.0f - lx) +
                          load_value(input[base + y0 * input_width + x1]) * lx;
        const float bottom = load_value(input[base + y1 * input_width + x0]) * (1.0f - lx) +
                             load_value(input[base + y1 * input_width + x1]) * lx;
        output[index] = store_value<T>(top * (1.0f - ly) + bottom * ly);
    }
}

__device__ int64_t pixel_norm_unit_base(int64_t unit, const Meta& tensor_meta,
                                        int normalized_dim) {
    int64_t base = 0;
    for (int dimension = tensor_meta.rank - 1; dimension >= 0; --dimension) {
        if (dimension == normalized_dim) continue;
        const int64_t coordinate = unit % tensor_meta.shape[dimension];
        unit /= tensor_meta.shape[dimension];
        base += coordinate * tensor_meta.strides[dimension];
    }
    return base;
}

// Contiguous-axis topology: one CTA owns one independent normalization unit.
// Threads cooperatively reduce the channel/feature axis, then write it back.
template <typename T>
__global__ void pixel_norm_contiguous_axis_kernel(
        const T* input, T* output, int64_t units, Meta tensor_meta,
        int normalized_dim, float eps) {
    __shared__ float workspace[kThreads];
    __shared__ float denominator;
    const int64_t unit = blockIdx.x;
    if (unit >= units) return;
    const int64_t channels = tensor_meta.shape[normalized_dim];
    const int64_t base = pixel_norm_unit_base(unit, tensor_meta, normalized_dim);
    float square = 0.0f;
    for (int64_t channel = threadIdx.x; channel < channels;
         channel += blockDim.x) {
        const float value = load_value(input[base + channel]);
        square += value * value;
    }
    const float total = group_norm_block_sum(square, workspace);
    if (threadIdx.x == 0)
        denominator = sqrtf(total / channels + eps);
    __syncthreads();
    for (int64_t channel = threadIdx.x; channel < channels;
         channel += blockDim.x)
        output[base + channel] =
            store_value<T>(load_value(input[base + channel]) / denominator);
}

constexpr int kPixelNormPositions = 32;
constexpr int kPixelNormChannelLanes = 8;

// Strided-axis topology for dense channel-first and arbitrary-axis tensors.
// A CTA covers 32 adjacent non-axis positions. Eight cooperative channel
// lanes reduce each position while preserving coalesced access across the
// position dimension. Each channel value is read exactly once for statistics
// and once for normalization instead of once per output channel.
template <typename T>
__global__ void pixel_norm_strided_axis_kernel(
        const T* input, T* output, int64_t units, Meta tensor_meta,
        int normalized_dim, float eps) {
    __shared__ float workspace[kPixelNormChannelLanes][kPixelNormPositions];
    const int position_lane = threadIdx.x;
    const int channel_lane = threadIdx.y;
    const int64_t unit = static_cast<int64_t>(blockIdx.x) *
                             kPixelNormPositions + position_lane;
    const int64_t channels = tensor_meta.shape[normalized_dim];
    const int64_t channel_stride = tensor_meta.strides[normalized_dim];
    const int64_t base = unit < units
        ? pixel_norm_unit_base(unit, tensor_meta, normalized_dim) : 0;
    float square = 0.0f;
    if (unit < units) {
        for (int64_t channel = channel_lane; channel < channels;
             channel += kPixelNormChannelLanes) {
            const float value =
                load_value(input[base + channel * channel_stride]);
            square += value * value;
        }
    }
    workspace[channel_lane][position_lane] = square;
    __syncthreads();
    for (int stride = kPixelNormChannelLanes / 2; stride > 0; stride /= 2) {
        if (channel_lane < stride)
            workspace[channel_lane][position_lane] +=
                workspace[channel_lane + stride][position_lane];
        __syncthreads();
    }
    if (unit >= units) return;
    const float denominator =
        sqrtf(workspace[0][position_lane] / channels + eps);
    for (int64_t channel = channel_lane; channel < channels;
         channel += kPixelNormChannelLanes) {
        const int64_t index = base + channel * channel_stride;
        output[index] = store_value<T>(load_value(input[index]) / denominator);
    }
}

template <typename T>
__global__ void pixel_shuffle_kernel(const T* input, T* output, int64_t count,
                                     int b, int c, int t, int h, int w, int ft, int fh, int fw) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t remaining = index;
        const int ow = remaining % (w * fw); remaining /= (w * fw);
        const int oh = remaining % (h * fh); remaining /= (h * fh);
        const int ot = remaining % (t * ft); remaining /= (t * ft);
        const int oc = remaining % c; const int ob = remaining / c;
        const int iw = ow / fw, pw = ow % fw, ih = oh / fh, ph = oh % fh, it = ot / ft, pt = ot % ft;
        const int packed_channel = (((oc * ft + pt) * fh + ph) * fw + pw);
        const int64_t source = ((((static_cast<int64_t>(ob) * (c * ft * fh * fw) + packed_channel) * t + it) * h + ih) * w + iw);
        output[index] = input[source];
    }
}

template <typename T>
__global__ void clamp_kernel(const T* input, T* output, int64_t count, float low, float high) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = store_value<T>(fminf(high, fmaxf(low, load_value(input[index]))));
}

template <typename T>
__global__ void exp_kernel(const T* input, T* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = store_value<T>(expf(load_value(input[index])));
}

template <typename T>
__global__ void sqrt_kernel(const T* input, T* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = store_value<T>(sqrtf(load_value(input[index])));
}

template <typename T>
__global__ void batched_matmul_kernel(const T* a, const T* b, T* output,
                                      int64_t count, int64_t m,
                                      int64_t n, int64_t k) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t column = index % n;
        const int64_t row = (index / n) % m;
        const int64_t batch = index / (m * n);
        float sum = 0.0f;
        for (int64_t inner = 0; inner < k; ++inner)
            sum += load_value(a[(batch * m + row) * k + inner]) *
                   load_value(b[(batch * k + inner) * n + column]);
        output[index] = store_value<T>(sum);
    }
}

__global__ void rng_normal_kernel(float* output, int64_t count, uint64_t seed, uint64_t offset) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, index, offset, &state);
    output[index] = curand_normal4(&state).x;
}

int blocks(int64_t count) { return static_cast<int>(std::min<int64_t>((count + kThreads - 1) / kThreads, 65535)); }

std::vector<int64_t> broadcast_shape(const Tensor& a, const Tensor& b) {
    const int rank = std::max(a.ndim(), b.ndim()); std::vector<int64_t> output(rank, 1);
    for (int index = 0; index < rank; ++index) {
        const int ai = index - (rank - a.ndim()), bi = index - (rank - b.ndim());
        const int64_t av = ai < 0 ? 1 : a.shape()[ai], bv = bi < 0 ? 1 : b.shape()[bi];
        require(av == bv || av == 1 || bv == 1, "Incompatible broadcast shapes"); output[index] = std::max(av, bv);
    }
    return output;
}

}  // namespace

struct CudaBackend::Impl {
    struct PendingProfile { std::string name; cudaEvent_t start{}; cudaEvent_t end{}; };
    struct ActiveProfile { std::string name; cudaEvent_t start{}; };
    struct ResolvedProfile {
        uint64_t calls = 0;
        double device_milliseconds = 0.0;
        std::vector<float> samples_milliseconds;
    };
    cublasHandle_t cublas{};
    cublasLtHandle_t cublaslt{};
    cudnnHandle_t cudnn{};
    std::unique_ptr<CudnnConvPlanCache> cudnn_conv;
    std::unique_ptr<CudnnSdpaPlanCache> cudnn_sdpa;
    struct AttentionMaskCacheKey {
        const void* identity = nullptr;
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;
        DType dtype = DType::Bool;
        DeviceId device = DeviceId::host();
        size_t bytes = 0;
        uint64_t content_hash = 0;
        int64_t batch = 0;
        int64_t query_tokens = 0;
        int64_t key_tokens = 0;
    };
    struct AttentionMaskCacheEntry {
        AttentionMaskCacheKey key;
        // Retaining the exact Tensor storage prevents device-pointer ABA while
        // the entry is reusable. Backend writes invalidate overlapping ranges.
        Tensor identity_guard;
        PrefixMaskCanonicalization canonical;
        Tensor query_lengths_device;
        Tensor key_value_lengths_device;
        uint64_t last_use = 0;
    };
    std::vector<AttentionMaskCacheEntry> attention_mask_cache;
    size_t attention_mask_cache_bytes = 0;
    uint64_t attention_mask_cache_clock = 0;
    uint64_t attention_mask_cache_hits = 0;
    uint64_t attention_mask_cache_misses = 0;
    uint64_t attention_mask_canonicalization_builds = 0;
    uint64_t attention_mask_descriptor_builds = 0;
    uint64_t attention_mask_d2h_copies = 0;
    uint64_t attention_mask_cache_invalidations = 0;
    uint64_t attention_mask_cache_evictions = 0;
    static constexpr size_t kMaximumAttentionMaskCacheEntries = 64;
    static constexpr size_t kMaximumAttentionMaskBytes = 1ULL << 20;
    static constexpr size_t kAttentionMaskCacheByteLimit = 4ULL << 20;
    cudaStream_t transfer_stream{};
    size_t baseline_free = 0;
    size_t peak = 0;
    size_t upload_bytes = 0;
    double upload_seconds = 0.0;
    DType execution_dtype = DType::F32;
    bool profiling = false;
    std::vector<PendingProfile> pending_profiles;
    std::vector<ActiveProfile> active_profiles;
    std::map<std::string, ResolvedProfile> resolved_profiles;
    size_t pending_profile_peak = 0;
    uint64_t profiler_forced_drains = 0;
    struct WeightCacheEntry {
        Tensor value;
        size_t bytes = 0;
        uint64_t last_use = 0;
        bool prefetched = false;
        cudaEvent_t ready{};
    };
    std::map<std::pair<const void*, DType>, WeightCacheEntry> weight_cache;
    struct QuantizedCacheEntry {
        Tensor packed;
        Tensor scales;
        size_t bytes = 0;
        uint64_t last_use = 0;
        bool prefetched = false;
        cudaEvent_t ready{};
    };
    std::map<const void*, QuantizedCacheEntry> quantized_weight_cache;
    struct SmallValueKey {
        DType source_dtype = DType::F32;
        DType target_dtype = DType::F32;
        std::vector<int64_t> shape;
        std::vector<uint8_t> payload;
        bool operator<(const SmallValueKey& other) const {
            return std::tie(source_dtype, target_dtype, shape, payload) <
                   std::tie(other.source_dtype, other.target_dtype,
                            other.shape, other.payload);
        }
    };
    std::map<SmallValueKey, Tensor> small_value_cache;
    size_t small_value_cache_bytes = 0;
    static constexpr size_t kMaximumCachedValueBytes = 64;
    static constexpr size_t kSmallValueCacheLimit = 1ULL << 20;
    struct PinnedCacheEntry {
        void* data = nullptr;
        size_t bytes = 0;
        uint64_t last_use = 0;
        cudaEvent_t last_transfer{};
    };
    std::map<std::pair<const void*, size_t>, PinnedCacheEntry> pinned_cache;
    size_t weight_cache_capacity = 0;
    size_t weight_cache_resident = 0;
    int compute_major = 0;
    int multiprocessor_count = 0;
    uint64_t weight_cache_hit_count = 0;
    uint64_t weight_cache_miss_count = 0;
    bool weight_cache_enabled = true;
    bool true_quant_compute = true;
    QuantComputeStats quant_compute;
    int compute_minor = 0;
    int cuda_runtime_version = 0;
    int cublas_version = 0;
    MemoryBudget memory_budget;
    MemoryRuntimeOptions memory_options;
    MemoryRuntimeStats memory_stats;
    MemoryPlanner memory_planner;
    uint64_t memory_clock = 0;
    bool trace_recording = false;
    std::vector<MemoryAccess> recorded_trace;
    std::vector<MemoryAccess> execution_trace;
    size_t trace_cursor = 0;
    struct PendingTransfer {
        cudaEvent_t start{};
        cudaEvent_t end{};
        size_t bytes = 0;
        bool prefetch = false;
        bool overlapped = false;
    };
    std::vector<PendingTransfer> pending_transfers;
    EventFenceTracker fence_tracker;
    std::map<uint64_t, cudaEvent_t> fences;
    std::shared_ptr<CudaTemporaryPool> temporary_pool =
        std::make_shared<CudaTemporaryPool>();

    struct Bf16LinearPlan {
        cublasLtMatmulDesc_t operation{};
        cublasLtMatrixLayout_t weight_layout{};
        cublasLtMatrixLayout_t input_layout{};
        cublasLtMatrixLayout_t output_layout{};
        cublasLtMatmulAlgo_t algorithm{};
        size_t workspace_bytes = 0;
        uint64_t numerical_implementation_flags = 0;
    };
    using Bf16LinearPlanKey = std::tuple<int64_t, int64_t, int64_t>;
    std::map<Bf16LinearPlanKey, Bf16LinearPlan> bf16_linear_plans;
    void* bf16_linear_workspace = nullptr;
    size_t bf16_linear_workspace_bytes = 0;
    Tensor bf16_linear_accumulation;
    int64_t bf16_linear_accumulation_elements = 0;
    static constexpr size_t kBf16LinearWorkspaceLimit = 32ULL << 20;

    Impl() {
        CUBLAS_CHECK(cublasCreate(&cublas)); CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
        CUBLAS_CHECK(cublasLtCreate(&cublaslt));
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
        CUDNN_CHECK(cudnnCreate(&cudnn));
        cudnn_conv = std::make_unique<CudnnConvPlanCache>(cudnn);
        cudnn_sdpa = std::make_unique<CudnnSdpaPlanCache>(cudnn);
        size_t total = 0; CUDA_CHECK(cudaMemGetInfo(&baseline_free, &total));
        int device = 0; cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        compute_major = properties.major;
        compute_minor = properties.minor;
        multiprocessor_count = properties.multiProcessorCount;
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUBLAS_CHECK(cublasGetVersion(cublas, &cublas_version));
        // Admission-only cache: retain a bounded prefix of immutable mmap
        // tensors and reserve 32% of device memory for activations/workspaces.
        weight_cache_capacity = static_cast<size_t>(total * 0.68);
        memory_budget = {static_cast<size_t>(total * 0.90), 1024ULL << 20,
                         64ULL << 30, static_cast<size_t>(total * 0.12),
                         static_cast<size_t>(total * 0.10)};
        memory_planner.set_budget(memory_budget);
        temporary_pool->set_cache_limit(
            temporary_cache_limit(memory_budget),
            temporary_pool_retention_limit(memory_budget));
    }
    ~Impl() {
        for (auto& item : pending_profiles) { cudaEventDestroy(item.start); cudaEventDestroy(item.end); }
        for (auto& item : active_profiles) cudaEventDestroy(item.start);
        for (auto& item : pending_transfers) { cudaEventDestroy(item.start); cudaEventDestroy(item.end); }
        for (auto& item : pinned_cache) {
            if (item.second.last_transfer) cudaEventDestroy(item.second.last_transfer);
            cudaFreeHost(item.second.data);
        }
        for (auto& [id, event] : fences) {
            (void)id;
            cudaEventDestroy(event);
        }
        fence_tracker.shutdown();
        for (auto& [key, plan] : bf16_linear_plans) {
            (void)key;
            if (plan.output_layout) cublasLtMatrixLayoutDestroy(plan.output_layout);
            if (plan.input_layout) cublasLtMatrixLayoutDestroy(plan.input_layout);
            if (plan.weight_layout) cublasLtMatrixLayoutDestroy(plan.weight_layout);
            if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
        }
        if (bf16_linear_workspace) cudaFree(bf16_linear_workspace);
        cudaStreamDestroy(transfer_stream);
        cudnn_sdpa.reset();
        cudnn_conv.reset();
        cudnnDestroy(cudnn); cublasLtDestroy(cublaslt); cublasDestroy(cublas);
    }
    Tensor temporary(std::vector<int64_t> shape, DType dtype,
                     bool workspace = false) {
        return temporary_pool->tensor(std::move(shape), dtype, workspace);
    }
    Tensor direct(std::vector<int64_t> shape, DType dtype) {
        return direct_device_tensor(
            std::move(shape), dtype, temporary_pool.get());
    }
    static uint64_t hash_host_mask(const Tensor& mask) {
        require(mask.device().is_host() && mask.dtype() == DType::Bool,
                "Attention mask hashing requires a host Boolean tensor");
        constexpr uint64_t offset = 1469598103934665603ULL;
        constexpr uint64_t prime = 1099511628211ULL;
        uint64_t value = offset;
        const auto* bytes = static_cast<const uint8_t*>(mask.data());
        for (size_t index = 0; index < mask.bytes(); ++index) {
            value ^= bytes[index];
            value *= prime;
        }
        return value;
    }
    static bool same_mask_identity(const AttentionMaskCacheKey& key,
                                   const Tensor& mask, int64_t batch,
                                   int64_t query_tokens, int64_t key_tokens) {
        return key.identity == mask.data() && key.shape == mask.shape() &&
            key.strides == mask.strides() && key.dtype == mask.dtype() &&
            key.device == mask.device() && key.bytes == mask.bytes() &&
            key.batch == batch && key.query_tokens == query_tokens &&
            key.key_tokens == key_tokens;
    }
    AttentionMaskCacheEntry* find_attention_mask(
            const Tensor& mask, int64_t batch, int64_t query_tokens,
            int64_t key_tokens) {
        if (mask.dtype() != DType::Bool || mask.bytes() == 0 ||
            mask.bytes() > kMaximumAttentionMaskBytes) {
            ++attention_mask_cache_misses;
            return nullptr;
        }
        const bool validate_content = mask.device().is_host();
        const uint64_t content_hash = validate_content ? hash_host_mask(mask) : 0;
        for (auto& entry : attention_mask_cache) {
            if (!same_mask_identity(entry.key, mask, batch, query_tokens,
                                    key_tokens))
                continue;
            if (validate_content && entry.key.content_hash != content_hash)
                continue;
            entry.last_use = ++attention_mask_cache_clock;
            ++attention_mask_cache_hits;
            return &entry;
        }
        ++attention_mask_cache_misses;
        return nullptr;
    }
    void erase_attention_mask(size_t index, bool eviction) {
        attention_mask_cache_bytes -= attention_mask_cache[index].key.bytes;
        attention_mask_cache.erase(
            attention_mask_cache.begin() + static_cast<std::ptrdiff_t>(index));
        if (eviction) ++attention_mask_cache_evictions;
    }
    void insert_attention_mask(
            const Tensor& identity, uint64_t content_hash, int64_t batch,
            int64_t query_tokens, int64_t key_tokens,
            PrefixMaskCanonicalization canonical,
            Tensor query_lengths_device, Tensor key_value_lengths_device) {
        if (identity.dtype() != DType::Bool || identity.bytes() == 0 ||
            identity.bytes() > kMaximumAttentionMaskBytes)
            return;
        for (size_t index = attention_mask_cache.size(); index-- > 0;) {
            if (same_mask_identity(attention_mask_cache[index].key, identity,
                                   batch, query_tokens, key_tokens))
                erase_attention_mask(index, false);
        }
        while (!attention_mask_cache.empty() &&
               (attention_mask_cache.size() >=
                    kMaximumAttentionMaskCacheEntries ||
                attention_mask_cache_bytes + identity.bytes() >
                    kAttentionMaskCacheByteLimit)) {
            const auto oldest = std::min_element(
                attention_mask_cache.begin(), attention_mask_cache.end(),
                [](const auto& left, const auto& right) {
                    return left.last_use < right.last_use;
                });
            erase_attention_mask(static_cast<size_t>(
                oldest - attention_mask_cache.begin()), true);
        }
        AttentionMaskCacheKey key;
        key.identity = identity.data();
        key.shape = identity.shape();
        key.strides = identity.strides();
        key.dtype = identity.dtype();
        key.device = identity.device();
        key.bytes = identity.bytes();
        key.content_hash = content_hash;
        key.batch = batch;
        key.query_tokens = query_tokens;
        key.key_tokens = key_tokens;
        attention_mask_cache_bytes += identity.bytes();
        attention_mask_cache.push_back({
            std::move(key), identity, std::move(canonical),
            std::move(query_lengths_device),
            std::move(key_value_lengths_device),
            ++attention_mask_cache_clock});
    }
    void invalidate_attention_masks_overlapping(const Tensor& destination) {
        if (!destination.defined() || destination.bytes() == 0) return;
        const uintptr_t destination_begin =
            reinterpret_cast<uintptr_t>(destination.data());
        const uintptr_t destination_end = destination_begin + destination.bytes();
        for (size_t index = attention_mask_cache.size(); index-- > 0;) {
            const auto& entry = attention_mask_cache[index];
            if (entry.key.device != destination.device()) continue;
            const uintptr_t mask_begin =
                reinterpret_cast<uintptr_t>(entry.key.identity);
            const uintptr_t mask_end = mask_begin + entry.key.bytes;
            if (destination_begin < mask_end && mask_begin < destination_end) {
                erase_attention_mask(index, false);
                ++attention_mask_cache_invalidations;
            }
        }
    }
    void record() {
        const size_t active = temporary_pool->active_bytes();
        const size_t peak_active = temporary_pool->peak_active_bytes();
        memory_stats.accounting.device_activation_bytes = active;
        memory_stats.accounting.peak_device_activation_bytes = std::max(
            memory_stats.accounting.peak_device_activation_bytes, peak_active);
        memory_stats.accounting.device_resident_weight_bytes = weight_cache_resident;
        memory_stats.accounting.peak_device_resident_weight_bytes = std::max(
            memory_stats.accounting.peak_device_resident_weight_bytes,
            weight_cache_resident);
    }
    void refresh_device_accounting() {
        size_t free = 0, total = 0; CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const size_t used = baseline_free > free ? baseline_free - free : 0;
        peak = std::max(peak, used);
        memory_stats.accounting.peak_device_bytes = std::max(memory_stats.accounting.peak_device_bytes, used);
    }
};

namespace {
bool bf16_linear_lt_eligible(int64_t m, int64_t n, int64_t k, DType output_dtype,
                             int compute_major) {
    if (output_dtype != DType::BF16 || compute_major < 8 || k % 8 != 0 || n % 8 != 0)
        return false;
    // Amortize plan lookup and a Tensor-Op epilogue only for non-trivial work.
    // This admission rule is expressed solely by operation shape and capability.
    return static_cast<long double>(m) * n * k >= static_cast<long double>(1ULL << 20);
}

CudaBackend::Impl::Bf16LinearPlan* find_bf16_linear_lt_plan(
    CudaBackend::Impl* impl, int64_t m, int64_t n, int64_t k) {
    const CudaBackend::Impl::Bf16LinearPlanKey key{m, n, k};
    const auto found = impl->bf16_linear_plans.find(key);
    if (found != impl->bf16_linear_plans.end()) return &found->second;

    CudaBackend::Impl::Bf16LinearPlan plan;
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) return nullptr;
    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    const cublasOperation_t transpose_input = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &transpose_weight, sizeof(transpose_weight));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulDescSetAttribute(
            plan.operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &transpose_input, sizeof(transpose_input));
    // Row-major [n,k], [m,k], and [m,n] buffers are described through their
    // equivalent column-major views; no transpose/materialization is added.
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &plan.weight_layout, CUDA_R_16BF, k, n, k);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &plan.input_layout, CUDA_R_16BF, k, m, k);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &plan.output_layout, CUDA_R_32F, n, m, n);

    cublasLtMatmulPreference_t preference{};
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulPreferenceCreate(&preference);
    if (status == CUBLAS_STATUS_SUCCESS) {
        const size_t workspace_limit = CudaBackend::Impl::kBf16LinearWorkspaceLimit;
        status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit));
    }

    cublasLtMatmulHeuristicResult_t candidates[16]{};
    int candidate_count = 0;
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(
            impl->cublaslt, plan.operation,
            plan.weight_layout, plan.input_layout,
            plan.output_layout, plan.output_layout,
            preference, 16, candidates, &candidate_count);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);

    const uint64_t required_flags = CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA |
        CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F |
        CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF;
    bool selected = false;
    if (status == CUBLAS_STATUS_SUCCESS) {
        for (int index = 0; index < candidate_count; ++index) {
            if (candidates[index].state != CUBLAS_STATUS_SUCCESS) continue;
            uint64_t flags = 0;
            size_t written = 0;
            if (cublasLtMatmulAlgoCapGetAttribute(
                    &candidates[index].algo,
                    CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                    &flags, sizeof(flags), &written) != CUBLAS_STATUS_SUCCESS ||
                written != sizeof(flags) || (flags & required_flags) != required_flags)
                continue;
            plan.algorithm = candidates[index].algo;
            plan.workspace_bytes = candidates[index].workspaceSize;
            plan.numerical_implementation_flags = flags;
            selected = true;
            break;
        }
    }
    if (!selected) {
        if (plan.output_layout) cublasLtMatrixLayoutDestroy(plan.output_layout);
        if (plan.input_layout) cublasLtMatrixLayoutDestroy(plan.input_layout);
        if (plan.weight_layout) cublasLtMatrixLayoutDestroy(plan.weight_layout);
        if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
        return nullptr;
    }
    auto [inserted, ok] = impl->bf16_linear_plans.emplace(key, plan);
    require(ok, "BF16 Linear plan cache insertion failed");
    return &inserted->second;
}

bool execute_bf16_linear_lt(CudaBackend::Impl* impl,
                            CudaBackend::Impl::Bf16LinearPlan* plan,
                            const Tensor& weight, const Tensor& input, void* output) {
    if (!plan) return false;
    if (plan->workspace_bytes > impl->bf16_linear_workspace_bytes) {
        if (impl->bf16_linear_workspace) CUDA_CHECK(cudaFree(impl->bf16_linear_workspace));
        impl->bf16_linear_workspace = nullptr;
        impl->bf16_linear_workspace_bytes = 0;
        CUDA_CHECK(cudaMalloc(&impl->bf16_linear_workspace, plan->workspace_bytes));
        impl->bf16_linear_workspace_bytes = plan->workspace_bytes;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        impl->cublaslt, plan->operation,
        &alpha,
        weight.data(), plan->weight_layout,
        input.data(), plan->input_layout,
        &beta,
        output, plan->output_layout,
        output, plan->output_layout,
        &plan->algorithm,
        impl->bf16_linear_workspace, plan->workspace_bytes,
        nullptr));
    return true;
}

bool memory_cache_eligible(const Tensor& input, DType dtype) {
    return input.device() == DeviceId::host() && !input.owns_storage() &&
        ((input.is_quantized() && (dtype == DType::F32 || dtype == DType::BF16)) ||
         ((input.dtype() == DType::F32 || input.dtype() == DType::BF16) &&
          (dtype == DType::F32 || dtype == DType::BF16)));
}

void note_memory_access(CudaBackend::Impl* impl, const Tensor& input, DType dtype,
                        size_t backend_bytes) {
    if (!impl->memory_options.enabled || !memory_cache_eligible(input, dtype)) return;
    MemoryAccess access{input, dtype, input.bytes(), backend_bytes, input.is_quantized()};
    if (impl->trace_recording) impl->recorded_trace.push_back(access);
    if (!impl->execution_trace.empty()) {
        size_t match = impl->trace_cursor;
        while (match < impl->execution_trace.size() &&
               (impl->execution_trace[match].tensor.data() != input.data() ||
                impl->execution_trace[match].target_dtype != dtype)) ++match;
        if (match < impl->execution_trace.size()) impl->trace_cursor = match + 1;
    }
}

uint64_t next_use(CudaBackend::Impl* impl, const void* identity) {
    if (impl->execution_trace.empty()) return std::numeric_limits<uint64_t>::max();
    for (size_t index = impl->trace_cursor; index < impl->execution_trace.size(); ++index)
        if (impl->execution_trace[index].tensor.data() == identity) return index;
    return std::numeric_limits<uint64_t>::max();
}

std::vector<ResidencyRecord> residency_candidates(CudaBackend::Impl* impl,
                                                  const void* protected_identity = nullptr) {
    std::vector<ResidencyRecord> result;
    for (const auto& [key, entry] : impl->weight_cache)
        result.push_back({key.first, entry.bytes, ResidencyState::Evictable,
                          entry.last_use, next_use(impl, key.first),
                          key.first == protected_identity, entry.ready != nullptr});
    for (const auto& [key, entry] : impl->quantized_weight_cache)
        result.push_back({key, entry.bytes, ResidencyState::DevicePackedReady,
                          entry.last_use, next_use(impl, key),
                          key == protected_identity, entry.ready != nullptr});
    return result;
}

bool ensure_weight_capacity(CudaBackend::Impl* impl, size_t requested,
                            const void* protected_identity = nullptr) {
    if (!impl->weight_cache_enabled || requested > impl->weight_cache_capacity) return false;
    if (impl->weight_cache_resident <= impl->weight_cache_capacity &&
        requested <= impl->weight_cache_capacity - impl->weight_cache_resident) return true;
    if (!impl->memory_options.enabled) return false;
    const AdmissionPlan plan = impl->memory_planner.plan_admission(
        impl->weight_cache_resident, requested,
        residency_candidates(impl, protected_identity),
        impl->memory_options.eviction_policy);
    if (!plan.admit) {
        ++impl->memory_stats.unsafe_eviction_rejections;
        return false;
    }
    for (const void* identity : plan.evict) {
        bool removed = false;
        for (auto it = impl->weight_cache.begin(); it != impl->weight_cache.end();) {
            if (it->first.first == identity) {
                impl->weight_cache_resident -= it->second.bytes;
                it = impl->weight_cache.erase(it);
                removed = true;
            } else ++it;
        }
        const auto quant = impl->quantized_weight_cache.find(identity);
        if (quant != impl->quantized_weight_cache.end()) {
            impl->weight_cache_resident -= quant->second.bytes;
            impl->memory_stats.accounting.quantized_packed_bytes -= quant->second.bytes;
            impl->quantized_weight_cache.erase(quant);
            removed = true;
        }
        if (removed) ++impl->memory_stats.evictions;
        if (impl->weight_cache_resident <= impl->weight_cache_capacity &&
            requested <= impl->weight_cache_capacity - impl->weight_cache_resident) break;
    }
    return impl->weight_cache_resident <= impl->weight_cache_capacity &&
        requested <= impl->weight_cache_capacity - impl->weight_cache_resident;
}

struct StagedSource {
    const void* data = nullptr;
    CudaBackend::Impl::PinnedCacheEntry* entry = nullptr;
};

StagedSource pinned_source(CudaBackend::Impl* impl, const void* source, size_t bytes,
                           bool cacheable_source) {
    // Host addresses owned by transient tensors are routinely recycled.  Keeping
    // their staged copy under an address-only key would turn allocator reuse into
    // stale data.  Only immutable model storage is eligible for the persistent
    // pinned cache; transient sources safely use the pageable path.
    if (!impl->memory_options.enabled || !impl->memory_options.host_staging ||
        !cacheable_source || bytes == 0)
        return {source, nullptr};
    const auto key = std::make_pair(source, bytes);
    const auto hit = impl->pinned_cache.find(key);
    if (hit != impl->pinned_cache.end()) {
        ++impl->memory_stats.staging_hits;
        hit->second.last_use = ++impl->memory_clock;
        return {hit->second.data, &hit->second};
    }
    ++impl->memory_stats.staging_misses;
    const size_t budget = impl->memory_budget.host_staging_budget_bytes;
    if (bytes > budget || impl->memory_stats.accounting.vrm_mapped_bytes >
        impl->memory_budget.host_total_budget_bytes - std::min(bytes, impl->memory_budget.host_total_budget_bytes))
        return {source, nullptr};
    while (impl->memory_stats.accounting.host_staging_bytes > budget - bytes) {
        auto victim = impl->pinned_cache.end();
        for (auto it = impl->pinned_cache.begin(); it != impl->pinned_cache.end(); ++it) {
            const bool ready = !it->second.last_transfer ||
                cudaEventQuery(it->second.last_transfer) == cudaSuccess;
            if (ready && (victim == impl->pinned_cache.end() ||
                          it->second.last_use < victim->second.last_use)) victim = it;
        }
        if (victim == impl->pinned_cache.end()) return {source, nullptr};
        if (victim->second.last_transfer) cudaEventDestroy(victim->second.last_transfer);
        CUDA_CHECK(cudaFreeHost(victim->second.data));
        impl->memory_stats.accounting.host_staging_bytes -= victim->second.bytes;
        impl->pinned_cache.erase(victim);
        ++impl->memory_stats.staging_evictions;
    }
    void* pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocPortable));
    std::memcpy(pinned, source, bytes);
    auto [inserted, ok] = impl->pinned_cache.emplace(
        key, CudaBackend::Impl::PinnedCacheEntry{pinned, bytes, ++impl->memory_clock, nullptr});
    require(ok, "Pinned cache duplicate insertion");
    impl->memory_stats.accounting.host_staging_bytes += bytes;
    impl->memory_stats.accounting.peak_host_staging_bytes = std::max(
        impl->memory_stats.accounting.peak_host_staging_bytes,
        impl->memory_stats.accounting.host_staging_bytes);
    impl->memory_stats.accounting.peak_host_total_bytes = std::max(
        impl->memory_stats.accounting.peak_host_total_bytes,
        impl->memory_stats.accounting.vrm_mapped_bytes +
        impl->memory_stats.accounting.host_staging_bytes);
    return {pinned, &inserted->second};
}

cudaEvent_t enqueue_h2d(CudaBackend::Impl* impl, void* destination,
                        const void* source, size_t bytes, bool wait_for_use,
                        bool prefetched, bool cacheable_source) {
    StagedSource staged = pinned_source(impl, source, bytes, cacheable_source);
    cudaEvent_t start{}, end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, impl->transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, staged.data, bytes, cudaMemcpyHostToDevice,
                               impl->transfer_stream));
    CUDA_CHECK(cudaEventRecord(end, impl->transfer_stream));
    if (staged.entry) {
        if (staged.entry->last_transfer) cudaEventDestroy(staged.entry->last_transfer);
        CUDA_CHECK(cudaEventCreate(&staged.entry->last_transfer));
        CUDA_CHECK(cudaEventRecord(staged.entry->last_transfer, impl->transfer_stream));
    }
    impl->pending_transfers.push_back({start, end, bytes, prefetched, false});
    ++impl->memory_stats.upload_copies;
    impl->memory_stats.upload_bytes += bytes;
    if (wait_for_use) {
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, end));
        ++impl->memory_stats.stream_waits;
        ++impl->memory_stats.event_waits;
    }
    return end;
}

void resolve_transfers(CudaBackend::Impl* impl) {
    for (auto& transfer : impl->pending_transfers) {
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, transfer.start, transfer.end));
        impl->memory_stats.upload_seconds += milliseconds / 1000.0;
        if (transfer.overlapped) impl->memory_stats.overlapped_upload_seconds += milliseconds / 1000.0;
        CUDA_CHECK(cudaEventDestroy(transfer.start));
        CUDA_CHECK(cudaEventDestroy(transfer.end));
    }
    impl->pending_transfers.clear();
}

void consume_prefetch(CudaBackend::Impl* impl, bool& prefetched, cudaEvent_t& ready) {
    if (!ready) return;
    const cudaError_t status = cudaEventQuery(ready);
    if (status == cudaSuccess) {
        if (prefetched) {
            ++impl->memory_stats.prefetch_hits;
            for (size_t index = 0; index < impl->pending_transfers.size(); ++index) {
                if (impl->pending_transfers[index].end != ready) continue;
                impl->pending_transfers[index].overlapped = true;
                if (index > 0 && impl->pending_transfers[index - 1].prefetch)
                    impl->pending_transfers[index - 1].overlapped = true;
                break;
            }
        }
    } else {
        require(status == cudaErrorNotReady,
                std::string("CUDA prefetch event query failed: ") + cudaGetErrorString(status));
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, ready));
        ++impl->memory_stats.stream_waits;
        ++impl->memory_stats.event_waits;
    }
    prefetched = false;
    ready = nullptr;
}

void schedule_prefetch(CudaBackend::Impl* impl, const void* protected_identity) {
    if (!impl->memory_options.enabled || !impl->memory_options.prefetch ||
        impl->execution_trace.empty() || impl->trace_cursor >= impl->execution_trace.size()) return;
    const size_t stop = std::min(impl->execution_trace.size(),
        impl->trace_cursor + std::max<size_t>(1, impl->memory_options.prefetch_lookahead));
    for (size_t index = impl->trace_cursor; index < stop; ++index) {
        const MemoryAccess& access = impl->execution_trace[index];
        const Tensor& input = access.tensor;
        const void* identity = input.data();
        if (identity == protected_identity) continue;
        if (input.is_quantized()) {
            if (impl->quantized_weight_cache.find(identity) != impl->quantized_weight_cache.end()) continue;
            ++impl->memory_stats.prefetch_requests;
            const QuantizationInfo& quantization = input.quantization();
            const size_t resident = input.bytes() + quantization.scales.bytes();
            if (!ensure_weight_capacity(impl, resident, protected_identity)) {
                ++impl->memory_stats.prefetch_dropped;
                continue;
            }
            Tensor packed = impl->direct(
                {static_cast<int64_t>(input.bytes())}, DType::U8);
            Tensor scales = impl->direct(
                quantization.scales.shape(), DType::F32);
            enqueue_h2d(impl, packed.data(), input.data(), input.bytes(), false, true, true);
            cudaEvent_t ready = enqueue_h2d(impl, scales.data(), quantization.scales.data(),
                                            quantization.scales.bytes(), false, true, true);
            impl->weight_cache_resident += resident;
            impl->quantized_weight_cache.emplace(identity,
                CudaBackend::Impl::QuantizedCacheEntry{
                    packed, scales, resident, ++impl->memory_clock, true, ready});
            impl->memory_stats.accounting.quantized_packed_bytes += resident;
            impl->upload_bytes += resident;
            continue;
        }
        if (input.dtype() != access.target_dtype) continue;
        const auto key = std::make_pair(identity, access.target_dtype);
        if (impl->weight_cache.find(key) != impl->weight_cache.end()) continue;
        ++impl->memory_stats.prefetch_requests;
        const size_t resident = input.bytes();
        if (!ensure_weight_capacity(impl, resident, protected_identity)) {
            ++impl->memory_stats.prefetch_dropped;
            continue;
        }
        Tensor value = impl->direct(input.shape(), input.dtype());
        cudaEvent_t ready = enqueue_h2d(impl, value.data(), input.data(), input.bytes(),
                                        false, true, true);
        impl->weight_cache_resident += resident;
        impl->weight_cache.emplace(key, CudaBackend::Impl::WeightCacheEntry{
            value, resident, ++impl->memory_clock, true, ready});
        impl->upload_bytes += resident;
    }
}

class ProfileScope {
public:
    ProfileScope(CudaBackend::Impl* impl, std::string name)
        : impl_(impl), name_(std::move(name)) {
        if (!impl_->profiling) return;
        CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&end_));
        CUDA_CHECK(cudaEventRecord(start_));
    }
    ~ProfileScope() {
        if (!impl_->profiling) return;
        cudaEventRecord(end_);
        impl_->pending_profiles.push_back({name_, start_, end_});
        impl_->pending_profile_peak = std::max(
            impl_->pending_profile_peak, impl_->pending_profiles.size());
        constexpr size_t kMaximumPendingProfiles = 8192;
        if (impl_->pending_profiles.size() >= kMaximumPendingProfiles) {
            CUDA_CHECK(cudaEventSynchronize(impl_->pending_profiles.front().end));
            ++impl_->profiler_forced_drains;
            size_t completed = 0;
            for (auto& item : impl_->pending_profiles) {
                const cudaError_t status = cudaEventQuery(item.end);
                if (status == cudaErrorNotReady) break;
                CUDA_CHECK(status);
                float milliseconds = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, item.start, item.end));
                auto& aggregate = impl_->resolved_profiles[item.name];
                ++aggregate.calls;
                aggregate.device_milliseconds += milliseconds;
                aggregate.samples_milliseconds.push_back(milliseconds);
                CUDA_CHECK(cudaEventDestroy(item.start));
                CUDA_CHECK(cudaEventDestroy(item.end));
                ++completed;
            }
            impl_->pending_profiles.erase(
                impl_->pending_profiles.begin(),
                impl_->pending_profiles.begin() + static_cast<std::ptrdiff_t>(completed));
        }
    }
private:
    CudaBackend::Impl* impl_;
    std::string name_;
    cudaEvent_t start_{};
    cudaEvent_t end_{};
};
}  // namespace

#define VRHINO_PROFILE(name) ProfileScope profile_scope_(impl_, name)

CudaBackend::CudaBackend() : impl_(new Impl()) {}
CudaBackend::~CudaBackend() { delete impl_; }
std::string CudaBackend::name() const { return "native-cuda-correctness"; }
int CudaBackend::device_count() const { int count = 0; CUDA_CHECK(cudaGetDeviceCount(&count)); return count; }
DeviceCapability CudaBackend::device_capability(int device) const {
    require(device >= 0 && device < device_count(), "CUDA device index out of range");
    cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    DeviceCapability result;
    result.name = properties.name;
    result.family = "compute_" + std::to_string(properties.major) +
        std::to_string(properties.minor);
    result.device_memory_bytes = properties.totalGlobalMem;
    result.supports_fp16_storage = true;
    result.supports_fp16_arithmetic = properties.major >= 5;
    result.supports_bf16_storage = properties.major >= 8;
    result.supports_bf16_arithmetic = properties.major >= 8;
    result.f64_mode = DeviceCapability::F64Mode::Native;
    result.memory = memory_capabilities();
    result.quantization = {true, properties.major >= 8, properties.major >= 8,
                           properties.major >= 8, true};
    return result;
}
BackendMemoryCapabilities CudaBackend::memory_capabilities() const {
    return {false, false, false, true, true, true};
}
bool CudaBackend::supports(DType dtype) const { return dtype == DType::F32 || dtype == DType::BF16 || dtype == DType::U8 || dtype == DType::Bool || dtype == DType::I64 || dtype == DType::I32; }
void CudaBackend::set_execution_dtype(DType dtype) {
    require(dtype == DType::F32 || dtype == DType::BF16, "Execution dtype must be float32 or bfloat16");
    require(dtype != DType::BF16 || impl_->compute_major >= 8,
            "Native BF16 requires CUDA compute capability 8.0+");
    impl_->execution_dtype = dtype;
}
DType CudaBackend::execution_dtype() const { return impl_->execution_dtype; }
Tensor CudaBackend::allocate_host(const std::vector<int64_t>& shape, DType dtype) { return Tensor::host(shape, dtype); }
Tensor CudaBackend::allocate_device(const std::vector<int64_t>& shape, DType dtype) { Tensor value = impl_->direct(shape, dtype); impl_->record(); return value; }
TransferFence CudaBackend::copy(const Tensor& source, Tensor& destination, CopyMode mode) {
    require(source.defined() && destination.defined() &&
            source.dtype() == destination.dtype() && source.shape() == destination.shape(),
            "copy contract violation");
    // Public copies are the only backend API that mutates an existing Tensor.
    // Invalidate content-derived Attention metadata before overlapping writes.
    impl_->invalidate_attention_masks_overlapping(destination);
    cudaMemcpyKind kind;
    if (source.device().is_host() && !destination.device().is_host())
        kind = cudaMemcpyHostToDevice;
    else if (!source.device().is_host() && destination.device().is_host())
        kind = cudaMemcpyDeviceToHost;
    else if (!source.device().is_host() && !destination.device().is_host())
        kind = cudaMemcpyDeviceToDevice;
    else
        kind = cudaMemcpyHostToHost;
    if (mode == CopyMode::Synchronous) {
        CUDA_CHECK(cudaMemcpy(destination.data(), source.data(), source.bytes(), kind));
        impl_->record();
        return {};
    }
    TransferFence fence = create_fence(destination.device().index);
    CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data(), source.bytes(), kind));
    record_fence(fence);
    impl_->record();
    return fence;
}
TransferFence CudaBackend::create_fence(int device) {
    require(device == 0, "CUDA fence device index out of range");
    TransferFence fence = impl_->fence_tracker.create();
    cudaEvent_t event{};
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    require(impl_->fences.emplace(fence.id, event).second, "duplicate CUDA fence");
    return fence;
}
void CudaBackend::record_fence(TransferFence fence) {
    impl_->fence_tracker.record(fence);
    CUDA_CHECK(cudaEventRecord(impl_->fences.at(fence.id)));
}
void CudaBackend::wait_fence(TransferFence fence) {
    impl_->fence_tracker.wait(fence);
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, impl_->fences.at(fence.id)));
}
bool CudaBackend::query_fence(TransferFence fence) {
    if (!impl_->fence_tracker.query(fence)) return false;
    const cudaError_t status = cudaEventQuery(impl_->fences.at(fence.id));
    if (status == cudaErrorNotReady) return false;
    CUDA_CHECK(status);
    return true;
}
void CudaBackend::destroy_fence(TransferFence fence) {
    impl_->fence_tracker.destroy(fence);
    const auto found = impl_->fences.find(fence.id);
    require(found != impl_->fences.end(), "destroyed or stale CUDA fence");
    CUDA_CHECK(cudaEventDestroy(found->second));
    impl_->fences.erase(found);
}

Tensor CudaBackend::copy_to_device(const Tensor& input, DType dtype) {
    require(input.defined(), "Cannot upload undefined tensor");
    if (input.device() == DeviceId::accelerator() && input.dtype() == dtype) return input;
    const bool cacheable_small_value = input.device() == DeviceId::host() &&
        input.owns_storage() && !input.is_quantized() && input.bytes() > 0 &&
        input.bytes() <= Impl::kMaximumCachedValueBytes;
    Impl::SmallValueKey small_value_key;
    if (cacheable_small_value) {
        small_value_key.source_dtype = input.dtype();
        small_value_key.target_dtype = dtype;
        small_value_key.shape = input.shape();
        const auto* begin = static_cast<const uint8_t*>(input.data());
        small_value_key.payload.assign(begin, begin + input.bytes());
        const auto found = impl_->small_value_cache.find(small_value_key);
        if (found != impl_->small_value_cache.end()) return found->second;
    }
    if (input.is_quantized()) {
        require(input.device() == DeviceId::host() && (dtype == DType::F32 || dtype == DType::BF16),
                "Quantized tensors require CPU storage and float compute output");
        const QuantizationInfo& quantization = input.quantization();
        require(impl_->compute_major >= 8,
                "Quantized CUDA execution requires SM80+");
        require(quantization.axis == 1 && quantization.group_size > 0 &&
                quantization.block_size == 0 && quantization.symmetric &&
                (quantization.granularity == "per_group" ||
                 quantization.granularity == "per_channel") &&
                quantization.zero_point_mode == "none" &&
                quantization.dequantization == DequantizationSemantics::ScaleThenCast,
                "Unsupported quantized tensor layout/policy");
        require((quantization.type == QuantType::FP8E4M3FN &&
                 quantization.packing_layout == "byte_e4m3fn") ||
                (quantization.type == QuantType::INT8Symmetric &&
                 quantization.packing_layout == "byte_twos_complement") ||
                (quantization.type == QuantType::INT4Symmetric &&
                 quantization.packing_layout == "nibble_low_first_twos_complement"),
                "Unsupported quantized tensor packing");
        const void* key = input.data();
        const size_t resident_bytes = input.bytes() + quantization.scales.bytes();
        note_memory_access(impl_, input, dtype, resident_bytes);
        Tensor packed, scales;
        const auto found = impl_->quantized_weight_cache.find(key);
        if (found != impl_->quantized_weight_cache.end()) {
            ++impl_->weight_cache_hit_count;
            ++impl_->memory_stats.cache_hits;
            found->second.last_use = ++impl_->memory_clock;
            consume_prefetch(impl_, found->second.prefetched, found->second.ready);
            packed = found->second.packed;
            scales = found->second.scales;
        } else {
            ++impl_->weight_cache_miss_count;
            ++impl_->memory_stats.cache_misses;
            VRHINO_PROFILE("quantized_h2d");
            const auto started = std::chrono::steady_clock::now();
            packed = impl_->direct(
                {static_cast<int64_t>(input.bytes())}, DType::U8);
            scales = impl_->direct(
                quantization.scales.shape(), DType::F32);
            if (impl_->memory_options.enabled) {
                enqueue_h2d(impl_, packed.data(), input.data(), input.bytes(), true, false, true);
                enqueue_h2d(impl_, scales.data(), quantization.scales.data(),
                            quantization.scales.bytes(), true, false, true);
            } else {
                CUDA_CHECK(cudaMemcpy(packed.data(), input.data(), input.bytes(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(scales.data(), quantization.scales.data(), quantization.scales.bytes(),
                                      cudaMemcpyHostToDevice));
            }
            const size_t resident = packed.bytes() + scales.bytes();
            impl_->upload_bytes += resident;
            impl_->upload_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            if (ensure_weight_capacity(impl_, resident)) {
                impl_->weight_cache_resident += resident;
                impl_->quantized_weight_cache.emplace(
                    key, Impl::QuantizedCacheEntry{packed, scales, resident,
                                                   ++impl_->memory_clock, false, nullptr});
                impl_->memory_stats.accounting.quantized_packed_bytes += resident;
            }
        }
        schedule_prefetch(impl_, key);
        VRHINO_PROFILE("quantized_dequant");
        Tensor output = impl_->temporary(input.shape(), dtype);
        const int64_t inner = input.numel() / input.dim(0);
        const int64_t groups = (inner + quantization.group_size - 1) / quantization.group_size;
        const int grid = blocks(input.numel());
        auto launch = [&](auto* output_pointer) {
            using Output = std::remove_pointer_t<decltype(output_pointer)>;
            if (quantization.type == QuantType::FP8E4M3FN)
                dequantize_kernel<Output, 0><<<grid, kThreads>>>(packed.data_as<uint8_t>(), scales.data_as<float>(), output_pointer, input.numel(), inner, groups, quantization.group_size);
            else if (quantization.type == QuantType::INT8Symmetric)
                dequantize_kernel<Output, 1><<<grid, kThreads>>>(packed.data_as<uint8_t>(), scales.data_as<float>(), output_pointer, input.numel(), inner, groups, quantization.group_size);
            else if (quantization.type == QuantType::INT4Symmetric)
                dequantize_kernel<Output, 2><<<grid, kThreads>>>(packed.data_as<uint8_t>(), scales.data_as<float>(), output_pointer, input.numel(), inner, groups, quantization.group_size);
            else throw Error("Unsupported quantized CUDA dispatch");
        };
        if (dtype == DType::BF16) launch(output.data_as<__nv_bfloat16>());
        else launch(output.data_as<float>());
        CUDA_CHECK(cudaGetLastError());
        impl_->record();
        return output;
    }
    const bool immutable_mmap = impl_->weight_cache_enabled && input.device() == DeviceId::host() && !input.owns_storage() &&
        (input.dtype() == DType::F32 || input.dtype() == DType::BF16) &&
        (dtype == DType::F32 || dtype == DType::BF16);
    const auto cache_key = std::make_pair(static_cast<const void*>(input.data()), dtype);
    if (immutable_mmap) {
        note_memory_access(impl_, input, dtype,
                           static_cast<size_t>(input.numel()) * dtype_size(dtype));
        const auto found = impl_->weight_cache.find(cache_key);
        if (found != impl_->weight_cache.end()) {
            ++impl_->weight_cache_hit_count;
            ++impl_->memory_stats.cache_hits;
            found->second.last_use = ++impl_->memory_clock;
            consume_prefetch(impl_, found->second.prefetched, found->second.ready);
            schedule_prefetch(impl_, cache_key.first);
            return found->second.value;
        }
        ++impl_->weight_cache_miss_count;
        ++impl_->memory_stats.cache_misses;
    }
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "h2d_or_cast|bytes=" + std::to_string(input.bytes()) +
          "|from=" + dtype_name(input.dtype()) + "|to=" + dtype_name(dtype)
        : "h2d_or_cast");
    const auto started = std::chrono::steady_clock::now();
    Tensor output = input.device() == DeviceId::host()
        ? impl_->direct(input.shape(), dtype)
        : impl_->temporary(input.shape(), dtype);
    if (input.dtype() == dtype) {
        if (input.device() == DeviceId::host() && impl_->memory_options.enabled)
            enqueue_h2d(impl_, output.data(), input.data(), input.bytes(),
                        true, false, immutable_mmap);
        else CUDA_CHECK(cudaMemcpy(output.data(), input.data(), input.bytes(), input.device() == DeviceId::host() ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));
    }
    else {
        Tensor source = input;
        if (source.device() == DeviceId::host()) {
            source = impl_->direct(input.shape(), input.dtype());
            if (impl_->memory_options.enabled)
                enqueue_h2d(impl_, source.data(), input.data(), input.bytes(),
                            true, false, immutable_mmap);
            else CUDA_CHECK(cudaMemcpy(source.data(), input.data(), input.bytes(), cudaMemcpyHostToDevice));
            impl_->memory_stats.accounting.temporary_bytes = input.bytes();
            impl_->memory_stats.accounting.peak_temporary_bytes = std::max(
                impl_->memory_stats.accounting.peak_temporary_bytes, input.bytes());
        }
        if (input.dtype() == DType::BF16 && dtype == DType::F32)
            bf16_to_float_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<__nv_bfloat16>(), output.data_as<float>(), input.numel());
        else if (input.dtype() == DType::F16 && dtype == DType::F32)
            f16_to_float_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<__half>(), output.data_as<float>(), input.numel());
        else if (input.dtype() == DType::F32 && dtype == DType::BF16)
            float_to_bf16_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<float>(), output.data_as<__nv_bfloat16>(), input.numel());
        else if (input.dtype() == DType::F16 && dtype == DType::BF16)
            f16_to_bf16_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<__half>(), output.data_as<__nv_bfloat16>(), input.numel());
        else if (dtype == DType::F32 && input.dtype() == DType::I64)
            numeric_to_float_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<int64_t>(), output.data_as<float>(), input.numel());
        else if (dtype == DType::F32 && input.dtype() == DType::I32)
            numeric_to_float_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<int32_t>(), output.data_as<float>(), input.numel());
        else if (dtype == DType::F32 && (input.dtype() == DType::Bool || input.dtype() == DType::U8))
            numeric_to_float_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<uint8_t>(), output.data_as<float>(), input.numel());
        else if (dtype == DType::BF16 && input.dtype() == DType::I64)
            numeric_to_bf16_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<int64_t>(), output.data_as<__nv_bfloat16>(), input.numel());
        else if (dtype == DType::BF16 && input.dtype() == DType::I32)
            numeric_to_bf16_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<int32_t>(), output.data_as<__nv_bfloat16>(), input.numel());
        else if (dtype == DType::BF16 && (input.dtype() == DType::Bool || input.dtype() == DType::U8))
            numeric_to_bf16_kernel<<<blocks(input.numel()), kThreads>>>(source.data_as<uint8_t>(), output.data_as<__nv_bfloat16>(), input.numel());
        else throw Error("Unsupported CUDA dtype conversion: " + dtype_name(input.dtype()) + " -> " + dtype_name(dtype));
        CUDA_CHECK(cudaGetLastError());
    }
    if (input.device() == DeviceId::host()) {
        impl_->upload_bytes += input.bytes();
        impl_->upload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    if (immutable_mmap && ensure_weight_capacity(impl_, output.bytes())) {
        impl_->weight_cache_resident += output.bytes();
        impl_->weight_cache.emplace(cache_key,
            Impl::WeightCacheEntry{output, output.bytes(), ++impl_->memory_clock, false, nullptr});
    }
    if (immutable_mmap) schedule_prefetch(impl_, cache_key.first);
    if (cacheable_small_value &&
        output.bytes() <= Impl::kSmallValueCacheLimit -
            std::min(impl_->small_value_cache_bytes,
                     Impl::kSmallValueCacheLimit)) {
        const auto [found, inserted] = impl_->small_value_cache.emplace(
            std::move(small_value_key), output);
        if (inserted) impl_->small_value_cache_bytes += output.bytes();
        else output = found->second;
    }
    impl_->memory_stats.accounting.temporary_bytes = 0;
    impl_->record(); return output;
}

Tensor CudaBackend::copy_to_host(const Tensor& input) {
    require(input.device() == DeviceId::accelerator(), "copy_to_host expects CUDA tensor");
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "d2h|bytes=" + std::to_string(input.bytes()) + "|dtype=" +
          dtype_name(input.dtype())
        : "d2h");
    Tensor output = Tensor::host(input.shape(), input.dtype());
    CUDA_CHECK(cudaMemcpy(output.data(), input.data(), input.bytes(), cudaMemcpyDeviceToHost)); return output;
}
void CudaBackend::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
    if (impl_->memory_options.enabled) resolve_transfers(impl_);
}
size_t CudaBackend::peak_device_bytes() const {
    impl_->refresh_device_accounting();
    return impl_->peak;
}
size_t CudaBackend::weight_upload_bytes() const { return impl_->upload_bytes; }
double CudaBackend::weight_upload_seconds() const { return impl_->upload_seconds; }
void CudaBackend::enable_profiling(bool enabled) {
    require(impl_->pending_profiles.empty(), "Cannot change profiling mode with unresolved CUDA events");
    impl_->profiling = enabled;
}
bool CudaBackend::profiling_enabled() const { return impl_->profiling; }
std::map<std::string, ProfileStat> CudaBackend::profile_stats() {
    synchronize(); std::map<std::string, ProfileStat> result;
    for (auto& item : impl_->pending_profiles) {
        float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, item.start, item.end));
        auto& aggregate = impl_->resolved_profiles[item.name];
        ++aggregate.calls;
        aggregate.device_milliseconds += milliseconds;
        aggregate.samples_milliseconds.push_back(milliseconds);
        CUDA_CHECK(cudaEventDestroy(item.start)); CUDA_CHECK(cudaEventDestroy(item.end));
    }
    impl_->pending_profiles.clear();
    for (auto& [name, aggregate] : impl_->resolved_profiles) {
        auto& stat = result[name];
        stat.calls = aggregate.calls;
        stat.device_milliseconds = aggregate.device_milliseconds;
        stat.mean_milliseconds = aggregate.calls
            ? aggregate.device_milliseconds / static_cast<double>(aggregate.calls) : 0.0;
        if (!aggregate.samples_milliseconds.empty()) {
            std::sort(aggregate.samples_milliseconds.begin(),
                      aggregate.samples_milliseconds.end());
            const auto percentile = [&](double fraction) {
                const size_t index = std::min(
                    aggregate.samples_milliseconds.size() - 1,
                    static_cast<size_t>(std::ceil(
                        fraction * aggregate.samples_milliseconds.size())) - 1);
                return static_cast<double>(aggregate.samples_milliseconds[index]);
            };
            stat.minimum_milliseconds = aggregate.samples_milliseconds.front();
            stat.maximum_milliseconds = aggregate.samples_milliseconds.back();
            stat.p50_milliseconds = percentile(0.50);
            stat.p95_milliseconds = percentile(0.95);
        }
    }
    if (impl_->profiler_forced_drains) {
        result["profiler.forced_drain"].calls = impl_->profiler_forced_drains;
    }
    result["attention.mask_cache.hit"].calls =
        impl_->attention_mask_cache_hits;
    result["attention.mask_cache.miss"].calls =
        impl_->attention_mask_cache_misses;
    result["attention.mask_cache.canonicalization_build"].calls =
        impl_->attention_mask_canonicalization_builds;
    result["attention.mask_cache.descriptor_build"].calls =
        impl_->attention_mask_descriptor_builds;
    result["attention.mask_cache.d2h"].calls =
        impl_->attention_mask_d2h_copies;
    result["attention.mask_cache.invalidation"].calls =
        impl_->attention_mask_cache_invalidations;
    result["attention.mask_cache.eviction"].calls =
        impl_->attention_mask_cache_evictions;
    result["profiler.pending_peak"].calls = impl_->pending_profile_peak;
    result["allocator.temporary.request"].calls =
        impl_->temporary_pool->request_count();
    result["allocator.temporary.request_bytes"].calls =
        impl_->temporary_pool->request_bytes();
    result["allocator.temporary.miss"].calls =
        impl_->temporary_pool->miss_count();
    result["allocator.temporary.miss_bytes"].calls =
        impl_->temporary_pool->miss_bytes();
    result["allocator.temporary.driver_allocation"].calls =
        impl_->temporary_pool->driver_allocation_count();
    result["allocator.temporary.driver_free"].calls =
        impl_->temporary_pool->driver_free_count();
    result["allocator.temporary.stream_ordered_allocation"].calls =
        impl_->temporary_pool->stream_ordered_allocation_count();
    result["allocator.temporary.stream_ordered_free"].calls =
        impl_->temporary_pool->stream_ordered_free_count();
    result["allocator.temporary.legacy_allocation"].calls =
        impl_->temporary_pool->legacy_allocation_count();
    result["allocator.temporary.legacy_free"].calls =
        impl_->temporary_pool->legacy_free_count();
    result["allocator.temporary.size_mismatch"].calls =
        impl_->temporary_pool->size_mismatch_count();
    result["allocator.temporary.size_mismatch_bytes"].calls =
        impl_->temporary_pool->size_mismatch_bytes();
    result["allocator.temporary.lifetime_overlap"].calls =
        impl_->temporary_pool->lifetime_overlap_count();
    result["allocator.temporary.lifetime_overlap_bytes"].calls =
        impl_->temporary_pool->lifetime_overlap_bytes();
    result["allocator.temporary.capacity_recurrence"].calls =
        impl_->temporary_pool->capacity_recurrence_count();
    result["allocator.temporary.capacity_recurrence_bytes"].calls =
        impl_->temporary_pool->capacity_recurrence_bytes();
    result["allocator.temporary.first_use"].calls =
        impl_->temporary_pool->first_use_count();
    result["allocator.temporary.first_use_bytes"].calls =
        impl_->temporary_pool->first_use_bytes();
    result["allocator.temporary.workspace_miss"].calls =
        impl_->temporary_pool->workspace_miss_count();
    result["allocator.temporary.workspace_miss_bytes"].calls =
        impl_->temporary_pool->workspace_miss_bytes();
    result["allocator.direct.allocation"].calls =
        impl_->temporary_pool->direct_allocation_count();
    result["allocator.direct.allocation_bytes"].calls =
        impl_->temporary_pool->direct_allocation_bytes();
    impl_->resolved_profiles.clear();
    impl_->profiler_forced_drains = 0;
    impl_->pending_profile_peak = 0;
    impl_->attention_mask_cache_hits = 0;
    impl_->attention_mask_cache_misses = 0;
    impl_->attention_mask_canonicalization_builds = 0;
    impl_->attention_mask_descriptor_builds = 0;
    impl_->attention_mask_d2h_copies = 0;
    impl_->attention_mask_cache_invalidations = 0;
    impl_->attention_mask_cache_evictions = 0;
    return result;
}
void CudaBackend::profile_region_begin(const std::string& name) {
    if (!impl_->profiling) return;
    cudaEvent_t start{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventRecord(start));
    impl_->active_profiles.push_back({name, start});
}
void CudaBackend::profile_region_end() {
    if (!impl_->profiling) return;
    require(!impl_->active_profiles.empty(), "CUDA profile region stack underflow");
    auto active = std::move(impl_->active_profiles.back());
    impl_->active_profiles.pop_back();
    cudaEvent_t end{};
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(end));
    impl_->pending_profiles.push_back({std::move(active.name), active.start, end});
}
uint64_t CudaBackend::weight_cache_hits() const { return impl_->weight_cache_hit_count; }
uint64_t CudaBackend::weight_cache_misses() const { return impl_->weight_cache_miss_count; }
size_t CudaBackend::weight_cache_resident_bytes() const { return impl_->weight_cache_resident; }
size_t CudaBackend::weight_cache_capacity_bytes() const { return impl_->weight_cache_capacity; }
void CudaBackend::enable_weight_cache(bool enabled) {
    require(impl_->weight_cache.empty() && impl_->quantized_weight_cache.empty(),
            "Weight cache mode must be selected before execution");
    impl_->weight_cache_enabled = enabled;
}
void CudaBackend::configure_memory_runtime(const MemoryBudget& budget,
                                           const MemoryRuntimeOptions& options) {
    require(impl_->weight_cache.empty() && impl_->quantized_weight_cache.empty() &&
            impl_->pinned_cache.empty(),
            "Memory Runtime must be configured before execution");
    budget.validate();
    const size_t device_bytes = device_capability().device_memory_bytes;
    require(budget.device_budget_bytes <= device_bytes,
            "GPU memory budget exceeds device capacity");
    impl_->memory_budget = budget;
    impl_->memory_options = options;
    impl_->memory_planner.set_budget(budget);
    impl_->memory_planner.set_capabilities(memory_capabilities());
    impl_->weight_cache_capacity = options.enabled ? budget.device_weight_budget_bytes() :
        static_cast<size_t>(device_bytes * 0.68);
    impl_->temporary_pool->set_cache_limit(
        temporary_cache_limit(budget), temporary_pool_retention_limit(budget));
    impl_->memory_stats = {};
    impl_->memory_stats.accounting.device_workspace_bytes = budget.reserved_device_workspace_bytes;
}
bool CudaBackend::memory_runtime_enabled() const { return impl_->memory_options.enabled; }
void CudaBackend::set_vrm_mapped_bytes(size_t bytes) {
    require(!impl_->memory_options.enabled || bytes <= impl_->memory_budget.host_total_budget_bytes,
            "VRM mapping exceeds CPU total memory budget");
    impl_->memory_stats.accounting.vrm_mapped_bytes = bytes;
    impl_->memory_stats.accounting.host_pageable_bytes = bytes;
    impl_->memory_stats.accounting.peak_host_total_bytes = std::max(
        impl_->memory_stats.accounting.peak_host_total_bytes,
        bytes + impl_->memory_stats.accounting.host_staging_bytes);
}
void CudaBackend::begin_memory_trace() {
    impl_->recorded_trace.clear();
    impl_->trace_recording = true;
}
std::vector<MemoryAccess> CudaBackend::end_memory_trace() {
    impl_->trace_recording = false;
    return impl_->recorded_trace;
}
void CudaBackend::set_memory_trace(std::vector<MemoryAccess> trace) {
    impl_->execution_trace = std::move(trace);
    impl_->trace_cursor = 0;
}
MemoryRuntimeStats CudaBackend::memory_runtime_stats() const {
    impl_->refresh_device_accounting();
    MemoryRuntimeStats result = impl_->memory_stats;
    result.cache_hits = impl_->weight_cache_hit_count;
    result.cache_misses = impl_->weight_cache_miss_count;
    result.accounting.device_resident_weight_bytes = impl_->weight_cache_resident;
    result.accounting.peak_device_bytes = impl_->peak;
    result.temporary_pool_reuses = impl_->temporary_pool->reuse_count();
    result.stream_ordered_handoff_releases =
        impl_->temporary_pool->handoff_release_count();
    result.stream_ordered_handoff_reuses =
        impl_->temporary_pool->handoff_reuse_count();
    result.last_stream_ordered_handoff_bytes =
        impl_->temporary_pool->last_handoff_reuse_bytes();
    result.largest_stream_ordered_handoff_bytes =
        impl_->temporary_pool->largest_handoff_reuse_bytes();
    return result;
}
void CudaBackend::enable_true_quant_compute(bool enabled) { impl_->true_quant_compute = enabled; }
bool CudaBackend::true_quant_compute_enabled() const { return impl_->true_quant_compute; }
QuantComputeStats CudaBackend::quant_compute_stats() const { return impl_->quant_compute; }
void CudaBackend::reset_quant_compute_stats() { impl_->quant_compute = {}; }

Tensor CudaBackend::linear(const Tensor& x_raw, const Tensor& weight_raw,
                           const Tensor* bias_raw, DType compute,
                           DType requested_output) {
    require(x_raw.ndim() >= 1 && weight_raw.ndim() == 2, "linear shape rank mismatch");
    require(x_raw.dim(-1) == weight_raw.dim(1), "linear K mismatch");
    const DType dtype = compute == DType::F16 ? impl_->execution_dtype : compute;
    require(dtype == DType::F32 || dtype == DType::BF16, "linear compute dtype unsupported");
    DType output_dtype = requested_output == DType::F16 ? dtype : requested_output;
    require(output_dtype == DType::F32 || output_dtype == DType::BF16,
            "linear producer output dtype unsupported");
    require(dtype == DType::BF16 || output_dtype == DType::F32,
            "FP32 Linear compute cannot directly produce BF16 output");
    Tensor x = copy_to_device(x_raw, dtype);
    const int64_t k = x.dim(-1), n = weight_raw.dim(0), m = x.numel() / k;
    std::vector<int64_t> shape = x.shape(); shape.back() = n;

    if (weight_raw.is_quantized() && impl_->true_quant_compute && dtype == DType::BF16 &&
        output_dtype == DType::BF16 &&
        impl_->compute_major >= 8 && impl_->cuda_runtime_version >= 11000 &&
        impl_->cublas_version >= 11000) {
        const QuantizationInfo& quantization = weight_raw.quantization();
        const bool supported_layout = quantization.axis == 1 && quantization.group_size > 0 &&
            quantization.block_size == 0 && quantization.symmetric &&
            (quantization.granularity == "per_group" || quantization.granularity == "per_channel") &&
            quantization.zero_point_mode == "none" &&
            quantization.dequantization == DequantizationSemantics::ScaleThenCast &&
            ((quantization.type == QuantType::FP8E4M3FN && quantization.packing_layout == "byte_e4m3fn") ||
             (quantization.type == QuantType::INT8Symmetric && quantization.packing_layout == "byte_twos_complement") ||
             (quantization.type == QuantType::INT4Symmetric && quantization.packing_layout == "nibble_low_first_twos_complement"));
        if (supported_layout) {
            const int64_t groups = (k + quantization.group_size - 1) / quantization.group_size;
            require(quantization.scales.shape() == std::vector<int64_t>({n, groups}),
                    "Quantized Linear scale shape mismatch");
            Tensor packed, scales;
            const void* key = weight_raw.data();
            const size_t resident_bytes = weight_raw.bytes() + quantization.scales.bytes();
            note_memory_access(impl_, weight_raw, DType::BF16, resident_bytes);
            const auto found = impl_->quantized_weight_cache.find(key);
            if (found != impl_->quantized_weight_cache.end()) {
                ++impl_->weight_cache_hit_count;
                ++impl_->memory_stats.cache_hits;
                found->second.last_use = ++impl_->memory_clock;
                consume_prefetch(impl_, found->second.prefetched, found->second.ready);
                packed = found->second.packed;
                scales = found->second.scales;
            } else {
                ++impl_->weight_cache_miss_count;
                ++impl_->memory_stats.cache_misses;
                {
                    VRHINO_PROFILE("quantized_h2d");
                    const auto started = std::chrono::steady_clock::now();
                    packed = impl_->direct(
                        {static_cast<int64_t>(weight_raw.bytes())}, DType::U8);
                    scales = impl_->direct(
                        quantization.scales.shape(), DType::F32);
                    if (impl_->memory_options.enabled) {
                        enqueue_h2d(impl_, packed.data(), weight_raw.data(), weight_raw.bytes(),
                                    true, false, true);
                        enqueue_h2d(impl_, scales.data(), quantization.scales.data(),
                                    quantization.scales.bytes(), true, false, true);
                    } else {
                        CUDA_CHECK(cudaMemcpy(packed.data(), weight_raw.data(), weight_raw.bytes(), cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(scales.data(), quantization.scales.data(), quantization.scales.bytes(),
                                              cudaMemcpyHostToDevice));
                    }
                    const size_t resident = packed.bytes() + scales.bytes();
                    impl_->upload_bytes += resident;
                    impl_->upload_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                    if (ensure_weight_capacity(impl_, resident)) {
                        impl_->weight_cache_resident += resident;
                        impl_->quantized_weight_cache.emplace(
                            key, Impl::QuantizedCacheEntry{packed, scales, resident,
                                                          ++impl_->memory_clock, false, nullptr});
                        impl_->memory_stats.accounting.quantized_packed_bytes += resident;
                    }
                }
            }
            schedule_prefetch(impl_, key);
            Tensor bias;
            const __nv_bfloat16* bias_pointer = nullptr;
            if (bias_raw) {
                bias = copy_to_device(*bias_raw, DType::BF16);
                require(bias.numel() == n, "linear bias mismatch");
                bias_pointer = bias.data_as<__nv_bfloat16>();
            }
            Tensor output = impl_->temporary(shape, DType::BF16);
            const dim3 grid(static_cast<unsigned>((n + 15) / 16),
                            static_cast<unsigned>((m + 127) / 128));
            const dim3 block(256);
            if (quantization.type == QuantType::FP8E4M3FN) {
                ProfileScope quant_profile(impl_, "quant_gemm.fp8");
                quantized_bf16_gemm_kernel<0><<<grid, block>>>(
                    x.data_as<__nv_bfloat16>(), packed.data_as<uint8_t>(), scales.data_as<float>(),
                    bias_pointer, output.data_as<__nv_bfloat16>(), m, n, k, groups, quantization.group_size);
                ++impl_->quant_compute.fp8_calls;
                impl_->quant_compute.last_dispatch = "true_quant:fp8_e4m3fn:bf16_wmma";
            } else if (quantization.type == QuantType::INT8Symmetric) {
                ProfileScope quant_profile(impl_, "quant_gemm.int8");
                quantized_bf16_gemm_kernel<1><<<grid, block>>>(
                    x.data_as<__nv_bfloat16>(), packed.data_as<uint8_t>(), scales.data_as<float>(),
                    bias_pointer, output.data_as<__nv_bfloat16>(), m, n, k, groups, quantization.group_size);
                ++impl_->quant_compute.int8_calls;
                impl_->quant_compute.last_dispatch = "true_quant:int8_symmetric:bf16_wmma";
            } else {
                ProfileScope quant_profile(impl_, "quant_gemm.int4");
                quantized_bf16_gemm_kernel<2><<<grid, block>>>(
                    x.data_as<__nv_bfloat16>(), packed.data_as<uint8_t>(), scales.data_as<float>(),
                    bias_pointer, output.data_as<__nv_bfloat16>(), m, n, k, groups, quantization.group_size);
                ++impl_->quant_compute.int4_calls;
                impl_->quant_compute.last_dispatch = "true_quant:int4_symmetric:bf16_wmma";
            }
            CUDA_CHECK(cudaGetLastError());
            ++impl_->quant_compute.true_quant_calls;
            impl_->record();
            return output;
        }
        impl_->quant_compute.last_dispatch = "fallback:unsupported_quant_layout";
    } else if (weight_raw.is_quantized()) {
        if (!impl_->true_quant_compute) impl_->quant_compute.last_dispatch = "fallback:phase8_baseline_selected";
        else if (dtype != DType::BF16) impl_->quant_compute.last_dispatch = "fallback:compute_dtype";
        else impl_->quant_compute.last_dispatch = "fallback:device_or_library_capability";
    }
    if (weight_raw.is_quantized()) ++impl_->quant_compute.fallback_calls;
    Tensor weight = copy_to_device(weight_raw, dtype);
    const char* bf16_output_text = std::getenv("VRHINO_BF16_LINEAR_OUTPUT");
    require(!bf16_output_text || std::strcmp(bf16_output_text, "bf16") == 0 ||
                                    std::strcmp(bf16_output_text, "fp32") == 0,
            "VRHINO_BF16_LINEAR_OUTPUT must be bf16 or fp32");
    // Phase 16R2 observation candidate: preserve BF16 Tensor-Op GEMM operands
    // and FP32 accumulation while optionally retaining the generic Linear
    // boundary in FP32. This switch is architecture/shape neutral and is not
    // the default until the full precision-policy ladder passes.
    if (requested_output == DType::F16 && dtype == DType::BF16 && bf16_output_text &&
        std::strcmp(bf16_output_text, "fp32") == 0)
        output_dtype = DType::F32;
    Tensor output = impl_->temporary(shape, output_dtype);
    Tensor bf16_bias;
    if (dtype == DType::BF16 && output_dtype == DType::BF16 && bias_raw) {
        bf16_bias = copy_to_device(*bias_raw, DType::BF16);
        require(bf16_bias.numel() == n, "linear bias mismatch");
    }
    Impl::Bf16LinearPlan* bf16_lt_plan = nullptr;
    if (dtype == DType::BF16 &&
        bf16_linear_lt_eligible(m, n, k, output_dtype, impl_->compute_major))
        bf16_lt_plan = find_bf16_linear_lt_plan(impl_, m, n, k);
    if (bf16_lt_plan && impl_->bf16_linear_accumulation_elements < output.numel()) {
        impl_->bf16_linear_accumulation = impl_->direct(
            {output.numel()}, DType::F32);
        impl_->bf16_linear_accumulation_elements = output.numel();
    }
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "linear|m=" + std::to_string(m) + "|n=" + std::to_string(n) +
          "|k=" + std::to_string(k) + "|dtype=" + dtype_name(dtype) +
          "|output=" + dtype_name(output_dtype) +
          (bf16_lt_plan ? std::string("|plan=cublaslt_hmma_fp32_epilogue|impl_flags=") +
                              std::to_string(bf16_lt_plan->numerical_implementation_flags)
                        : "|plan=baseline")
        : "linear");
    const float alpha = 1.0f, beta = 0.0f;
    if (dtype == DType::BF16) {
        void* lt_output = bf16_lt_plan
            ? impl_->bf16_linear_accumulation.data() : output.data();
        const bool used_lt = bf16_lt_plan && execute_bf16_linear_lt(
            impl_, bf16_lt_plan, weight, x, lt_output);
        if (used_lt) {
            const __nv_bfloat16* bias_pointer = bias_raw
                ? bf16_bias.data_as<__nv_bfloat16>() : nullptr;
            fused_bf16_linear_output_kernel<<<blocks(output.numel()), kThreads>>>(
                impl_->bf16_linear_accumulation.data_as<float>(), bias_pointer,
                output.data_as<__nv_bfloat16>(), output.numel(), n);
        }
        if (!used_lt) {
            Tensor accumulation = output_dtype == DType::F32
                ? output : impl_->temporary(shape, DType::F32);
            CUBLAS_CHECK(cublasGemmEx(impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                weight.data(), CUDA_R_16BF, static_cast<int>(k), x.data(), CUDA_R_16BF,
                static_cast<int>(k), &beta, accumulation.data(), CUDA_R_32F, static_cast<int>(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if (output_dtype == DType::BF16) {
                const __nv_bfloat16* bias_pointer = bias_raw
                    ? bf16_bias.data_as<__nv_bfloat16>() : nullptr;
                fused_bf16_linear_output_kernel<<<blocks(output.numel()), kThreads>>>(
                    accumulation.data_as<float>(), bias_pointer,
                    output.data_as<__nv_bfloat16>(), output.numel(), n);
            } else if (bias_raw) {
                Tensor bias = copy_to_device(*bias_raw, DType::F32);
                require(bias.numel() == n, "linear bias mismatch");
                bias_kernel<<<blocks(output.numel()), kThreads>>>(
                    output.data_as<float>(), bias.data_as<float>(), output.numel(), n);
            }
        }
    } else {
        CUBLAS_CHECK(cublasSgemm(impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
            weight.data_as<float>(), static_cast<int>(k), x.data_as<float>(),
            static_cast<int>(k), &beta, output.data_as<float>(), static_cast<int>(n)));
    }
    if (bias_raw && dtype == DType::F32) {
        Tensor bias = copy_to_device(*bias_raw, dtype);
        require(bias.numel() == n, "linear bias mismatch");
        bias_kernel<<<blocks(output.numel()), kThreads>>>(output.data_as<float>(), bias.data_as<float>(), output.numel(), n);
    }
    CUDA_CHECK(cudaGetLastError()); impl_->record();
    return output;
}

template <int Kind>
Tensor binary(CudaBackend& backend, CudaBackend::Impl* impl,
              const Tensor& a_raw, const Tensor& b_raw) {
    const std::vector<int64_t> shape = broadcast_shape(a_raw, b_raw);
    DType dtype = backend.execution_dtype();
    if (a_raw.device() == DeviceId::accelerator() &&
        (a_raw.dtype() == DType::F32 || a_raw.dtype() == DType::BF16)) dtype = a_raw.dtype();
    else if (b_raw.device() == DeviceId::accelerator() &&
             (b_raw.dtype() == DType::F32 || b_raw.dtype() == DType::BF16)) dtype = b_raw.dtype();
    Tensor a = backend.copy_to_device(a_raw, dtype), b = backend.copy_to_device(b_raw, dtype);
    Tensor output = impl->temporary(shape, dtype);
    if (dtype == DType::BF16)
        binary_kernel<__nv_bfloat16, Kind><<<blocks(output.numel()), kThreads>>>(a.data_as<__nv_bfloat16>(), b.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(shape), meta(a.shape()), meta(b.shape()));
    else binary_kernel<float, Kind><<<blocks(output.numel()), kThreads>>>(a.data_as<float>(), b.data_as<float>(), output.data_as<float>(), output.numel(), meta(shape), meta(a.shape()), meta(b.shape()));
    CUDA_CHECK(cudaGetLastError()); return output;
}
Tensor CudaBackend::add(const Tensor& a, const Tensor& b) { VRHINO_PROFILE("elementwise.add"); Tensor value = binary<0>(*this, impl_, a, b); impl_->record(); return value; }
Tensor CudaBackend::mul(const Tensor& a, const Tensor& b) { VRHINO_PROFILE("elementwise.mul"); Tensor value = binary<1>(*this, impl_, a, b); impl_->record(); return value; }
Tensor CudaBackend::div(const Tensor& a, const Tensor& b) { VRHINO_PROFILE("elementwise.div"); Tensor value = binary<2>(*this, impl_, a, b); impl_->record(); return value; }
Tensor CudaBackend::maximum(const Tensor& a, const Tensor& b) { VRHINO_PROFILE("elementwise.maximum"); Tensor value = binary<3>(*this, impl_, a, b); impl_->record(); return value; }

Tensor CudaBackend::batched_matmul(const Tensor& a_raw, const Tensor& b_raw) {
    VRHINO_PROFILE("batched_matmul");
    require(a_raw.ndim() == 3 && b_raw.ndim() == 3 &&
            a_raw.dim(0) == b_raw.dim(0) && a_raw.dim(2) == b_raw.dim(1),
            "batched_matmul expects [B,M,K] x [B,K,N]");
    const DType dtype = a_raw.device() == DeviceId::accelerator() &&
            (a_raw.dtype() == DType::F32 || a_raw.dtype() == DType::BF16)
        ? a_raw.dtype() : execution_dtype();
    Tensor a = copy_to_device(a_raw, dtype), b = copy_to_device(b_raw, dtype);
    require(b.dtype() == dtype, "batched_matmul dtype mismatch");
    Tensor output = impl_->temporary({a.dim(0), a.dim(1), b.dim(2)}, dtype);
    if (dtype == DType::BF16)
        batched_matmul_kernel<<<blocks(output.numel()), kThreads>>>(
            a.data_as<__nv_bfloat16>(), b.data_as<__nv_bfloat16>(),
            output.data_as<__nv_bfloat16>(), output.numel(), a.dim(1),
            b.dim(2), a.dim(2));
    else
        batched_matmul_kernel<<<blocks(output.numel()), kThreads>>>(
            a.data_as<float>(), b.data_as<float>(), output.data_as<float>(),
            output.numel(), a.dim(1), b.dim(2), a.dim(2));
    CUDA_CHECK(cudaGetLastError());
    impl_->record();
    return output;
}
Tensor CudaBackend::reshape(const Tensor& x, const std::vector<int64_t>& shape) {
    VRHINO_PROFILE("reshape");
    const DType dtype = x.device() == DeviceId::accelerator() && (x.dtype() == DType::F32 || x.dtype() == DType::BF16) ? x.dtype() : execution_dtype();
    return copy_to_device(x, dtype).reshape(shape);
}

Tensor CudaBackend::permute(const Tensor& x_raw, const std::vector<int64_t>& dims) {
    VRHINO_PROFILE("permute");
    require(static_cast<int64_t>(dims.size()) == x_raw.ndim(), "permute rank mismatch");
    const DType dtype = x_raw.device() == DeviceId::accelerator() && (x_raw.dtype() == DType::F32 || x_raw.dtype() == DType::BF16) ? x_raw.dtype() : execution_dtype(); Tensor x = copy_to_device(x_raw, dtype); std::vector<int64_t> shape(dims.size()); Meta dm{}; dm.rank = dims.size();
    std::vector<bool> seen(dims.size());
    for (size_t index = 0; index < dims.size(); ++index) { require(dims[index] >= 0 && dims[index] < x.ndim() && !seen[dims[index]], "Invalid permutation"); seen[dims[index]] = true; shape[index] = x.shape()[dims[index]]; dm.shape[index] = dims[index]; }
    Tensor output = impl_->temporary(shape, dtype);
    if (dtype == DType::BF16) permute_kernel<<<blocks(output.numel()), kThreads>>>(x.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(x.shape()), meta(shape), dm);
    else permute_kernel<<<blocks(output.numel()), kThreads>>>(x.data_as<float>(), output.data_as<float>(), output.numel(), meta(x.shape()), meta(shape), dm);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::concat(const std::vector<Tensor>& inputs, int64_t dim) {
    VRHINO_PROFILE("concat");
    require(!inputs.empty(), "concat requires inputs"); int rank = inputs[0].ndim(); if (dim < 0) dim += rank;
    require(dim >= 0 && dim < rank, "concat dimension invalid"); std::vector<int64_t> shape = inputs[0].shape(); shape[dim] = 0;
    for (const Tensor& input : inputs) { require(input.ndim() == rank, "concat rank mismatch"); for (int d = 0; d < rank; ++d) if (d != dim) require(input.shape()[d] == inputs[0].shape()[d], "concat shape mismatch"); shape[dim] += input.shape()[dim]; }
    DType dtype = execution_dtype(); if (inputs[0].device() == DeviceId::accelerator() && (inputs[0].dtype() == DType::F32 || inputs[0].dtype() == DType::BF16)) dtype = inputs[0].dtype(); Tensor output = impl_->temporary(shape, dtype); int64_t offset = 0;
    for (const Tensor& raw : inputs) { Tensor input = copy_to_device(raw, dtype); if (dtype == DType::BF16) copy_into_concat_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), input.numel(), meta(input.shape()), meta(shape), dim, offset); else copy_into_concat_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), input.numel(), meta(input.shape()), meta(shape), dim, offset); offset += input.shape()[dim]; }
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::slice(const Tensor& raw, int64_t dim, int64_t start, int64_t stop) {
    VRHINO_PROFILE("slice");
    if (dim < 0) dim += raw.ndim(); require(dim >= 0 && dim < raw.ndim(), "slice dimension invalid");
    const int64_t size = raw.shape()[dim]; if (start < 0) start += size; if (stop < 0) stop += size;
    start = std::clamp<int64_t>(start, 0, size); stop = std::clamp<int64_t>(stop, start, size);
    std::vector<int64_t> shape = raw.shape(); shape[dim] = stop - start; const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(shape, dtype);
    if (dtype == DType::BF16) slice_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(input.shape()), meta(shape), dim, start);
    else slice_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), output.numel(), meta(input.shape()), meta(shape), dim, start);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

std::vector<Tensor> CudaBackend::split(const Tensor& x, const std::vector<int64_t>& sections, int64_t dim) {
    if (dim < 0) dim += x.ndim(); int64_t total = std::accumulate(sections.begin(), sections.end(), int64_t{0}); require(total == x.shape()[dim], "split sections mismatch");
    std::vector<Tensor> result; int64_t offset = 0; for (int64_t section : sections) { result.push_back(slice(x, dim, offset, offset + section)); offset += section; } return result;
}

Tensor CudaBackend::cast(const Tensor& x, DType dtype) { VRHINO_PROFILE("cast"); return copy_to_device(x, dtype); }

Tensor CudaBackend::indexed_gather(const Tensor& table_raw, const Tensor& indices_raw) {
    VRHINO_PROFILE("indexed_gather");
    require(table_raw.ndim() == 2 || table_raw.ndim() == 3,
            "indexed_gather table must have rank 2 or 3");
    require(indices_raw.ndim() >= 1 &&
            (indices_raw.dtype() == DType::I64 || indices_raw.dtype() == DType::I32),
            "indexed_gather indices must be rank >=1 int64/int32");
    const bool batched = table_raw.ndim() == 3;
    if (batched)
        require(indices_raw.dim(0) == table_raw.dim(0),
                "indexed_gather batch dimension mismatch");
    const int64_t rows = table_raw.dim(batched ? 1 : 0);
    const int64_t width = table_raw.dim(-1);
    Tensor host_indices = indices_raw.device().is_host() ? indices_raw : copy_to_host(indices_raw);
    for (int64_t index = 0; index < host_indices.numel(); ++index) {
        const int64_t value = host_indices.dtype() == DType::I64
            ? host_indices.data_as<int64_t>()[index]
            : host_indices.data_as<int32_t>()[index];
        require(value >= 0 && value < rows, "indexed_gather index out of range");
    }
    const DType dtype = execution_dtype();
    Tensor table = copy_to_device(table_raw, dtype);
    Tensor indices = copy_to_device(indices_raw, indices_raw.dtype());
    std::vector<int64_t> shape = indices_raw.shape();
    shape.push_back(width);
    Tensor output = impl_->temporary(shape, dtype);
    const int64_t per_batch = batched ? indices_raw.numel() / indices_raw.dim(0)
                                      : indices_raw.numel();
    if (dtype == DType::BF16) {
        if (indices_raw.dtype() == DType::I64)
            indexed_gather_kernel<<<blocks(output.numel()), kThreads>>>(
                table.data_as<__nv_bfloat16>(), indices.data_as<int64_t>(),
                output.data_as<__nv_bfloat16>(), output.numel(), per_batch, rows, width, batched);
        else
            indexed_gather_kernel<<<blocks(output.numel()), kThreads>>>(
                table.data_as<__nv_bfloat16>(), indices.data_as<int32_t>(),
                output.data_as<__nv_bfloat16>(), output.numel(), per_batch, rows, width, batched);
    } else if (indices_raw.dtype() == DType::I64)
        indexed_gather_kernel<<<blocks(output.numel()), kThreads>>>(
            table.data_as<float>(), indices.data_as<int64_t>(), output.data_as<float>(),
            output.numel(), per_batch, rows, width, batched);
    else
        indexed_gather_kernel<<<blocks(output.numel()), kThreads>>>(
            table.data_as<float>(), indices.data_as<int32_t>(), output.data_as<float>(),
            output.numel(), per_batch, rows, width, batched);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::layer_norm(const Tensor& raw, const Tensor* weight_raw, const Tensor* bias_raw, float eps) {
    const char* bf16_output_text = std::getenv("VRHINO_BF16_NORM_OUTPUT");
    require(!bf16_output_text || std::strcmp(bf16_output_text, "bf16") == 0 ||
                                    std::strcmp(bf16_output_text, "fp32") == 0,
            "VRHINO_BF16_NORM_OUTPUT must be bf16 or fp32");
    DType dtype = raw.device() == DeviceId::accelerator() &&
            (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    if (dtype == DType::BF16 && bf16_output_text &&
        std::strcmp(bf16_output_text, "fp32") == 0) dtype = DType::F32;
    Tensor x = copy_to_device(raw, dtype), output = impl_->temporary(x.shape(), dtype);
    Tensor weight, bias;
    if (weight_raw) { weight = copy_to_device(*weight_raw, dtype); require(weight.numel() == x.dim(-1), "layer norm weight mismatch"); }
    if (bias_raw) { bias = copy_to_device(*bias_raw, dtype); require(bias.numel() == x.dim(-1), "layer norm bias mismatch"); }
    const int64_t width = x.dim(-1), rows = x.numel() / width;
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "norm.layer|rows=" + std::to_string(rows) + "|width=" +
          std::to_string(width) + "|dtype=" + dtype_name(dtype)
        : "norm.layer");
    if (dtype == DType::BF16)
        layer_norm_parallel_kernel<<<std::min<int64_t>(rows, 65535), kNormThreads>>>(
            x.data_as<__nv_bfloat16>(),
            weight_raw ? weight.data_as<__nv_bfloat16>() : nullptr,
            bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr,
            output.data_as<__nv_bfloat16>(), rows, width, eps);
    else
        layer_norm_parallel_kernel<<<std::min<int64_t>(rows, 65535), kNormThreads>>>(
            x.data_as<float>(), weight_raw ? weight.data_as<float>() : nullptr,
            bias_raw ? bias.data_as<float>() : nullptr,
            output.data_as<float>(), rows, width, eps);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::rms_norm(const Tensor& raw, const Tensor* weight_raw, float eps,
                             int64_t axis, DType requested_output) {
    if (axis < 0) axis += raw.ndim(); require(axis >= 0 && axis < raw.ndim(), "rms_norm axis invalid");
    const char* bf16_output_text = std::getenv("VRHINO_BF16_NORM_OUTPUT");
    require(!bf16_output_text || std::strcmp(bf16_output_text, "bf16") == 0 ||
                                    std::strcmp(bf16_output_text, "fp32") == 0,
            "VRHINO_BF16_NORM_OUTPUT must be bf16 or fp32");
    DType dtype = raw.device() == DeviceId::accelerator() &&
            (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    const bool promote = dtype == DType::BF16 && bf16_output_text &&
                         std::strcmp(bf16_output_text, "fp32") == 0;
    if (promote) dtype = DType::F32;
    const DType output_dtype = requested_output == DType::F16 ? dtype : requested_output;
    require(output_dtype == dtype || (dtype == DType::BF16 && output_dtype == DType::F32),
            "rms_norm output dtype unsupported");
    Tensor x = copy_to_device(raw, dtype), output = impl_->temporary(x.shape(), output_dtype);
    Tensor weight; if (weight_raw) { weight = copy_to_device(*weight_raw, output_dtype); require(weight.numel() == x.dim(axis), "rms norm weight mismatch"); }
    const int64_t profiled_width = x.dim(axis);
    const int64_t profiled_rows = x.numel() / profiled_width;
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "norm.rms|rows=" + std::to_string(profiled_rows) + "|width=" +
          std::to_string(profiled_width) + "|axis=" + std::to_string(axis) +
          "|dtype=" + dtype_name(dtype)
        : "norm.rms");
    if (axis == x.ndim() - 1) {
        const int64_t width = x.dim(-1), rows = x.numel() / width;
        if (dtype == DType::BF16 && output_dtype == DType::F32)
            require(weight_raw != nullptr,
                    "BF16-to-FP32 rms_norm requires affine weight");
        if (dtype == DType::BF16 && output_dtype == DType::F32) {
            rms_norm_bf16_input_f32_weight_parallel_kernel<<<
                std::min<int64_t>(rows, 65535), kNormThreads>>>(
                x.data_as<__nv_bfloat16>(), weight.data_as<float>(), output.data_as<float>(),
                rows, width, eps);
        }
        else if (dtype == DType::BF16)
            rms_norm_parallel_kernel<<<std::min<int64_t>(rows, 65535), kNormThreads>>>(
                x.data_as<__nv_bfloat16>(),
                weight_raw ? weight.data_as<__nv_bfloat16>() : nullptr,
                output.data_as<__nv_bfloat16>(), rows, width, eps);
        else
            rms_norm_parallel_kernel<<<std::min<int64_t>(rows, 65535), kNormThreads>>>(
                x.data_as<float>(), weight_raw ? weight.data_as<float>() : nullptr,
                output.data_as<float>(), rows, width, eps);
    } else {
        require(output_dtype == dtype, "mixed-dtype rms_norm currently requires final axis");
        const int64_t width = x.dim(axis);
        int64_t inner = 1;
        for (int64_t dimension = axis + 1; dimension < x.ndim(); ++dimension)
            inner *= x.dim(dimension);
        const int64_t outer = x.numel() / (width * inner);
        const int64_t inner_tiles = (inner + kAxisNormInnerTile - 1) /
                                    kAxisNormInnerTile;
        const int grid = static_cast<int>(std::min<int64_t>(
            outer * inner_tiles, 65535));
        if (dtype == DType::BF16)
            rms_norm_axis_parallel_kernel<<<grid, kNormThreads>>>(
                x.data_as<__nv_bfloat16>(),
                weight_raw ? weight.data_as<__nv_bfloat16>() : nullptr,
                output.data_as<__nv_bfloat16>(), outer, width, inner, eps);
        else
            rms_norm_axis_parallel_kernel<<<grid, kNormThreads>>>(
                x.data_as<float>(), weight_raw ? weight.data_as<float>() : nullptr,
                output.data_as<float>(), outer, width, inner, eps);
    }
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::activation(const Tensor& raw, Activation kind) {
    const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(input.shape(), dtype);
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "activation|elements=" + std::to_string(input.numel()) + "|kind=" +
          std::to_string(static_cast<int>(kind)) + "|dtype=" + dtype_name(dtype)
        : "activation");
    if (dtype == DType::BF16) activation_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), input.numel(), static_cast<int>(kind));
    else activation_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), input.numel(), static_cast<int>(kind)); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::sinusoidal_embedding(const Tensor& raw, int64_t width, bool flip, double downscale, bool frequency_f64) {
    VRHINO_PROFILE("sinusoidal_embedding");
    require(raw.ndim() == 1 && width > 0, "sinusoidal embedding contract violation"); const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor positions = copy_to_device(raw, dtype), output = impl_->temporary({raw.numel(), width}, dtype);
    if (dtype == DType::BF16) sinusoidal_kernel<<<blocks(output.numel()), kThreads>>>(positions.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), raw.numel(), width, flip, downscale, frequency_f64);
    else sinusoidal_kernel<<<blocks(output.numel()), kThreads>>>(positions.data_as<float>(), output.data_as<float>(), raw.numel(), width, flip, downscale, frequency_f64); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::rope_nd(const Tensor& raw, const Tensor& cos_raw, const Tensor& sin_raw) {
    require(raw.dim(-1) % 2 == 0, "RoPE width must be even");
    const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype();
    const DType frequency_dtype = cos_raw.device() == DeviceId::accelerator() && (cos_raw.dtype() == DType::F32 || cos_raw.dtype() == DType::BF16) ? cos_raw.dtype() : dtype;
    Tensor x = copy_to_device(raw, dtype), cosine = copy_to_device(cos_raw, frequency_dtype), sine = copy_to_device(sin_raw, frequency_dtype), output = impl_->temporary(x.shape(), dtype);
    require(cosine.shape() == sine.shape(), "RoPE frequency mismatch"); broadcast_shape(x, cosine);
    ProfileScope profile_scope_(impl_, impl_->profiling
        ? "rope|elements=" + std::to_string(x.numel()) + "|width=" +
          std::to_string(x.dim(-1)) + "|dtype=" + dtype_name(dtype)
        : "rope");
    if (dtype == DType::BF16 && frequency_dtype == DType::F32) rope_kernel<<<blocks(x.numel()), kThreads>>>(x.data_as<__nv_bfloat16>(), cosine.data_as<float>(), sine.data_as<float>(), output.data_as<__nv_bfloat16>(), x.numel(), meta(x.shape()), meta(cosine.shape()));
    else if (dtype == DType::BF16) rope_kernel<<<blocks(x.numel()), kThreads>>>(x.data_as<__nv_bfloat16>(), cosine.data_as<__nv_bfloat16>(), sine.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), x.numel(), meta(x.shape()), meta(cosine.shape()));
    else if (frequency_dtype == DType::BF16) rope_kernel<<<blocks(x.numel()), kThreads>>>(x.data_as<float>(), cosine.data_as<__nv_bfloat16>(), sine.data_as<__nv_bfloat16>(), output.data_as<float>(), x.numel(), meta(x.shape()), meta(cosine.shape()));
    else rope_kernel<<<blocks(x.numel()), kThreads>>>(x.data_as<float>(), cosine.data_as<float>(), sine.data_as<float>(), output.data_as<float>(), x.numel(), meta(x.shape()), meta(cosine.shape())); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::attention(const Tensor& q_raw, const Tensor& k_raw, const Tensor& v_raw,
                              const Tensor* mask_raw, bool causal, float scale,
                              const Tensor* bias_raw, AttentionObservation* observation) {
    VRHINO_PROFILE("attention");
    require(observation == nullptr,
            "CUDA attention observation is available only in the fixed Linux capture harness");
    require(q_raw.ndim() == 4 && k_raw.ndim() == 4 && v_raw.ndim() == 4, "attention requires BSHD");
    require(q_raw.dim(0) == k_raw.dim(0) && k_raw.shape() == v_raw.shape() && q_raw.dim(2) == k_raw.dim(2) && q_raw.dim(3) == k_raw.dim(3), "attention shape mismatch");
    const DType dtype = execution_dtype();
    const char* bf16_output_text = std::getenv("VRHINO_BF16_ATTENTION_OUTPUT");
    require(!bf16_output_text || std::strcmp(bf16_output_text, "bf16") == 0 ||
                                    std::strcmp(bf16_output_text, "fp32") == 0,
            "VRHINO_BF16_ATTENTION_OUTPUT must be bf16 or fp32");
    const DType output_dtype = dtype == DType::BF16 && bf16_output_text &&
            std::strcmp(bf16_output_text, "fp32") == 0
        ? DType::F32 : dtype;
    Tensor q = copy_to_device(q_raw, dtype), k = copy_to_device(k_raw, dtype),
           v = copy_to_device(v_raw, dtype), output = impl_->temporary(q.shape(), output_dtype);
    Tensor mask; const uint8_t* mp = nullptr; Meta mm{};
    if (mask_raw) { mask = copy_to_device(*mask_raw, DType::Bool); mp = mask.data_as<uint8_t>(); mm = meta(mask.shape()); }
    Tensor bias; Meta bm{};
    if (bias_raw) {
        bias = copy_to_device(*bias_raw, dtype); bm = meta(bias.shape());
        const std::vector<int64_t> logical = {q.dim(0), q.dim(2), q.dim(1), k.dim(1)};
        require(bias.ndim() <= 4, "attention additive bias rank exceeds 4");
        const int shift = 4 - bias.ndim();
        for (int index = 0; index < bias.ndim(); ++index)
            require(bias.dim(index) == 1 || bias.dim(index) == logical[index + shift],
                    "attention additive bias is not broadcastable to BHQK");
    }
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(q.dim(3)));
    using namespace cuda_attention_config;
    const int output_tiles = static_cast<int>((q.dim(3) + kHeadTile - 1) / kHeadTile);
    const int64_t query_tiles = (q.dim(1) + kQueryTile - 1) / kQueryTile;
    const int64_t blocks_required = q.dim(0) * query_tiles * q.dim(2) * output_tiles;
    require(blocks_required > 0 && blocks_required <= INT_MAX,
            "attention grid exceeds CUDA launch limits");
    const size_t workspace = workspace_bytes(q.dim(3));
    const dim3 threads(kWarpThreads * kWarpsPerBlock);
    const char* variant = std::getenv("VRHINO_CUDA_ATTENTION_VARIANT");
    require(!variant || std::strcmp(variant, "ordered") == 0 ||
                         std::strcmp(variant, "baseline") == 0 ||
                         std::strcmp(variant, "tensor2pass") == 0,
            "VRHINO_CUDA_ATTENTION_VARIANT must be ordered, baseline, or tensor2pass");
    const char* bf16_qk_text = std::getenv("VRHINO_CUDA_ATTENTION_BF16_QK");
    require(!bf16_qk_text || std::strcmp(bf16_qk_text, "fp32") == 0 ||
                                  std::strcmp(bf16_qk_text, "tensor") == 0,
            "VRHINO_CUDA_ATTENTION_BF16_QK must be fp32 or tensor");
    const char* sdpa_admission_text =
        std::getenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION");
    require(!sdpa_admission_text ||
                std::strcmp(sdpa_admission_text, "auto") == 0 ||
                std::strcmp(sdpa_admission_text, "force") == 0 ||
                std::strcmp(sdpa_admission_text, "off") == 0,
            "VRHINO_CUDA_ATTENTION_SDPA_ADMISSION must be auto, force, or off");
    // FP32 keeps the Phase 15 ordered default. BF16 remains on the established
    // Phase 7 stable kernel unless an explicit research replay requests an
    // ordered candidate. No failed BF16 candidate becomes a runtime default.
    const long double score_elements = static_cast<long double>(q.dim(0)) *
        q.dim(1) * k.dim(1) * q.dim(2);
    const bool tensor_two_pass_requested = dtype == DType::BF16 &&
        ((variant && std::strcmp(variant, "tensor2pass") == 0) ||
         (!variant && !bf16_qk_text &&
          score_elements >= kBf16TensorMinimumScoreElements));
    const size_t tensor_two_pass_workspace = bf16_tensor_two_pass_workspace_bytes(
        q.dim(0), q.dim(1), q.dim(2), q.dim(3), kBf16TensorKeyTile);
    const bool bf16_tensor_two_pass = tensor_two_pass_requested &&
        impl_->compute_major >= 8 && q.dim(3) % 8 == 0 &&
        tensor_two_pass_workspace <=
            impl_->memory_budget.reserved_device_workspace_bytes;
    const bool ordered_candidate = variant
        ? std::strcmp(variant, "ordered") == 0
        : dtype == DType::F32 || bf16_qk_text;
    // cuDNN SDPA is a CUDA implementation candidate for the same shared
    // Attention semantic. Admission depends only on dtype/layout/mask/device
    // capability. Explicit research variants retain their reproducible path.
    // Boolean masks enter cuDNN only after exact, content-based prefix
    // canonicalization; arbitrary patterns remain on the bounded fallback.
    bool cudnn_sdpa_executed = false;
    const bool sdpa_admitted = sdpa_admission_text &&
        std::strcmp(sdpa_admission_text, "force") == 0
        ? true
        : score_elements >= kCudnnSdpaMinimumScoreElements;
    if (dtype == DType::BF16 && output_dtype == DType::BF16 &&
        impl_->compute_major >= 8 && !variant && !bf16_qk_text &&
        (!sdpa_admission_text ||
         std::strcmp(sdpa_admission_text, "off") != 0) && sdpa_admitted) {
        PrefixMaskCanonicalization prefix_mask;
        Tensor query_lengths_device, key_value_lengths_device;
        if (mask_raw) {
            const Tensor& cache_identity = mask_raw->device().is_host()
                ? *mask_raw : mask;
            auto* cached = impl_->find_attention_mask(
                cache_identity, q.dim(0), q.dim(1), k.dim(1));
            if (cached) {
                prefix_mask = cached->canonical;
                query_lengths_device = cached->query_lengths_device;
                key_value_lengths_device = cached->key_value_lengths_device;
            } else {
                const Tensor host_mask = mask_raw->device().is_host()
                    ? *mask_raw : copy_to_host(mask);
                if (!mask_raw->device().is_host())
                    ++impl_->attention_mask_d2h_copies;
                ++impl_->attention_mask_canonicalization_builds;
                prefix_mask = canonicalize_boolean_prefix_mask(
                    host_mask, q.dim(0), q.dim(1), k.dim(1));
                if (prefix_mask.representable) {
                    Tensor query_lengths = Tensor::host(
                        {q.dim(0), 1, 1, 1}, DType::I32);
                    Tensor key_value_lengths = Tensor::host(
                        {q.dim(0), 1, 1, 1}, DType::I32);
                    std::memcpy(query_lengths.data(),
                                prefix_mask.query_lengths.data(),
                                query_lengths.bytes());
                    std::memcpy(key_value_lengths.data(),
                                prefix_mask.key_value_lengths.data(),
                                key_value_lengths.bytes());
                    query_lengths_device =
                        copy_to_device(query_lengths, DType::I32);
                    key_value_lengths_device =
                        copy_to_device(key_value_lengths, DType::I32);
                    ++impl_->attention_mask_descriptor_builds;
                }
                impl_->insert_attention_mask(
                    cache_identity, Impl::hash_host_mask(host_mask),
                    q.dim(0), q.dim(1), k.dim(1), prefix_mask,
                    query_lengths_device, key_value_lengths_device);
            }
        }
        if (!mask_raw || prefix_mask.representable) {
            CudnnSdpaDescriptor descriptor;
            descriptor.batch = q.dim(0);
            descriptor.query_tokens = q.dim(1);
            descriptor.key_tokens = k.dim(1);
            descriptor.heads = q.dim(2);
            descriptor.head_width = q.dim(3);
            descriptor.scale = scale;
            descriptor.causal = causal;
            descriptor.has_padding_mask = mask_raw != nullptr;
            descriptor.has_additive_bias = bias_raw != nullptr;
            if (bias_raw) {
                descriptor.bias_rank = bias.ndim();
                const int shift = 4 - bias.ndim();
                for (int index = 0; index < 4; ++index) {
                    if (index < shift) {
                        descriptor.bias_dimensions[index] = 1;
                        descriptor.bias_strides[index] = bias.numel();
                    } else {
                        descriptor.bias_dimensions[index] = bias.dim(index - shift);
                        descriptor.bias_strides[index] = bias.strides()[index - shift];
                    }
                }
            }
            ProfileScope sdpa_profile(
                impl_, impl_->profiling
                    ? "attention.cudnn_sdpa|b=" + std::to_string(q.dim(0)) +
                          "|q=" + std::to_string(q.dim(1)) +
                          "|k=" + std::to_string(k.dim(1)) +
                          "|h=" + std::to_string(q.dim(2)) +
                          "|d=" + std::to_string(q.dim(3)) +
                          "|causal=" + std::to_string(causal ? 1 : 0) +
                          "|bias=" + std::to_string(bias_raw ? 1 : 0) +
                          "|padding=" + std::to_string(mask_raw ? 1 : 0)
                    : "attention.cudnn_sdpa");
            const CudnnSdpaExecution execution = impl_->cudnn_sdpa->execute(
                descriptor, q.data(), k.data(), v.data(),
                bias_raw ? bias.data() : nullptr,
                mask_raw ? query_lengths_device.data_as<int32_t>() : nullptr,
                mask_raw ? key_value_lengths_device.data_as<int32_t>() : nullptr,
                output.data());
            cudnn_sdpa_executed = execution.executed;
        }
    }
    if (cudnn_sdpa_executed) {
        impl_->record();
    } else if (bf16_tensor_two_pass) {
        const int key_tile = kBf16TensorKeyTile;
        const int64_t rows = q.dim(0) * q.dim(1) * q.dim(2);
        Tensor maximum = impl_->temporary({rows}, DType::F32);
        Tensor denominator = impl_->temporary({rows}, DType::F32);
        Tensor scores = impl_->temporary(
            {q.dim(0), q.dim(2), q.dim(1), key_tile}, DType::F32);
        Tensor probabilities = impl_->temporary(
            {q.dim(0), q.dim(2), q.dim(1), key_tile}, DType::BF16);
        Tensor accumulation = impl_->temporary(
            {q.dim(0), q.dim(2), q.dim(1), q.dim(3)}, DType::F32);
        attention_two_pass_initialize_kernel<<<blocks(accumulation.numel()), kThreads>>>(
            maximum.data_as<float>(), denominator.data_as<float>(),
            accumulation.data_as<float>(), rows, q.dim(3));
        CUDA_CHECK(cudaGetLastError());
        const float alpha = 1.0f;
        const float zero = 0.0f;
        auto launch_qk = [&](int key_base, int key_count) {
            for (int b = 0; b < q.dim(0); ++b) {
                float* score_pointer = scores.data_as<float>() +
                    static_cast<int64_t>(b) * q.dim(2) * q.dim(1) * key_count;
                const int64_t key_offset =
                    (static_cast<int64_t>(b) * k.dim(1) * q.dim(2) * q.dim(3)) +
                    static_cast<int64_t>(key_base) * q.dim(2) * q.dim(3);
                const int64_t query_offset =
                    static_cast<int64_t>(b) * q.dim(1) * q.dim(2) * q.dim(3);
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    key_count, q.dim(1), q.dim(3), &alpha,
                    k.data_as<__nv_bfloat16>() + key_offset, CUDA_R_16BF,
                    q.dim(2) * q.dim(3), q.dim(3),
                    q.data_as<__nv_bfloat16>() + query_offset, CUDA_R_16BF,
                    q.dim(2) * q.dim(3), q.dim(3),
                    &zero, score_pointer, CUDA_R_32F, key_count,
                    static_cast<int64_t>(q.dim(1)) * key_count, q.dim(2),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        };
        for (int key_base = 0; key_base < k.dim(1); key_base += key_tile) {
            const int key_count = std::min<int64_t>(key_tile, k.dim(1) - key_base);
            launch_qk(key_base, key_count);
            attention_two_pass_statistics_kernel<<<blocks(rows), kThreads>>>(
                scores.data_as<float>(), maximum.data_as<float>(),
                denominator.data_as<float>(), mp,
                bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr,
                q.dim(0), q.dim(1), k.dim(1), q.dim(2), key_base, key_count,
                scale, causal, mm, bm);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int key_base = 0; key_base < k.dim(1); key_base += key_tile) {
            const int key_count = std::min<int64_t>(key_tile, k.dim(1) - key_base);
            launch_qk(key_base, key_count);
            const int64_t probability_count =
                q.dim(0) * q.dim(2) * q.dim(1) * key_count;
            attention_two_pass_probability_kernel<<<blocks(probability_count), kThreads>>>(
                scores.data_as<float>(), probabilities.data_as<__nv_bfloat16>(),
                maximum.data_as<float>(), denominator.data_as<float>(), mp,
                bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr,
                q.dim(0), q.dim(1), k.dim(1), q.dim(2), key_base, key_count,
                scale, causal, mm, bm);
            CUDA_CHECK(cudaGetLastError());
            const float beta = key_base == 0 ? 0.0f : 1.0f;
            for (int b = 0; b < q.dim(0); ++b) {
                const int64_t value_offset =
                    static_cast<int64_t>(b) * k.dim(1) * q.dim(2) * q.dim(3) +
                    static_cast<int64_t>(key_base) * q.dim(2) * q.dim(3);
                const int64_t probability_offset =
                    static_cast<int64_t>(b) * q.dim(2) * q.dim(1) * key_count;
                const int64_t accumulation_offset =
                    static_cast<int64_t>(b) * q.dim(2) * q.dim(1) * q.dim(3);
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    impl_->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    q.dim(3), q.dim(1), key_count, &alpha,
                    v.data_as<__nv_bfloat16>() + value_offset, CUDA_R_16BF,
                    q.dim(2) * q.dim(3), q.dim(3),
                    probabilities.data_as<__nv_bfloat16>() + probability_offset,
                    CUDA_R_16BF, key_count,
                    static_cast<int64_t>(q.dim(1)) * key_count,
                    &beta, accumulation.data_as<float>() + accumulation_offset,
                    CUDA_R_32F, q.dim(3),
                    static_cast<int64_t>(q.dim(1)) * q.dim(3), q.dim(2),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        if (output_dtype == DType::BF16) {
            attention_two_pass_finalize_kernel<<<blocks(output.numel()), kThreads>>>(
                accumulation.data_as<float>(), denominator.data_as<float>(),
                output.data_as<__nv_bfloat16>(),
                q.dim(0), q.dim(1), q.dim(2), q.dim(3));
        } else {
            attention_two_pass_finalize_kernel<<<blocks(output.numel()), kThreads>>>(
                accumulation.data_as<float>(), denominator.data_as<float>(),
                output.data_as<float>(),
                q.dim(0), q.dim(1), q.dim(2), q.dim(3));
        }
        CUDA_CHECK(cudaGetLastError());
        impl_->record();
    } else if (ordered_candidate) {
        // Keep the Tensor-Op QK implementation as an independently
        // reproducible failed candidate. The default promotes the already
        // rounded BF16 operands and reuses Phase 15's pedantic FP32 reduction.
        // This is a generic precision policy, never model/architecture/shape
        // dispatch.
        const bool bf16_tensor_qk = dtype == DType::BF16 && bf16_qk_text &&
                                    std::strcmp(bf16_qk_text, "tensor") == 0;
        Tensor q_dot = dtype == DType::BF16 && !bf16_tensor_qk
            ? copy_to_device(q, DType::F32) : q;
        Tensor k_dot = dtype == DType::BF16 && !bf16_tensor_qk
            ? copy_to_device(k, DType::F32) : k;
        int ordered_key_tile = kOrderedCandidateKeyTile;
        if (const char* tile_text = std::getenv("VRHINO_CUDA_ATTENTION_KEY_TILE"))
            ordered_key_tile = static_cast<int>(std::strtol(tile_text, nullptr, 10));
        require(ordered_key_tile >= 8 && ordered_key_tile <= 256 &&
                    ordered_key_tile % 8 == 0,
                "ordered Attention key tile must be a multiple of 8 in [8, 256]");
        const int64_t rows = q.dim(0) * q.dim(1) * q.dim(2);
        Tensor maximum = impl_->temporary({rows}, DType::F32);
        Tensor denominator = impl_->temporary({rows}, DType::F32);
        Tensor scores = impl_->temporary(
            {q.dim(0), q.dim(2), q.dim(1), ordered_key_tile}, DType::F32);
        Tensor previous = impl_->temporary(scores.shape(), DType::F32);
        Tensor accumulation = output_dtype == DType::F32
            ? output : impl_->temporary(q.shape(), DType::F32);
        attention_ordered_initialize_kernel<<<blocks(output.numel()), kThreads>>>(
            maximum.data_as<float>(), denominator.data_as<float>(),
            accumulation.data_as<float>(),
            rows, q.dim(3));
        CUDA_CHECK(cudaGetLastError());
        const float alpha = 1.0f, beta = 0.0f;
        for (int key_base = 0; key_base < k.dim(1);
             key_base += ordered_key_tile) {
            const int key_count = std::min<int64_t>(
                ordered_key_tile, k.dim(1) - key_base);
            for (int b = 0; b < q.dim(0); ++b) {
                float* score_pointer = scores.data_as<float>() +
                    static_cast<int64_t>(b) * q.dim(2) * q.dim(1) * key_count;
                const int64_t key_offset =
                    (static_cast<int64_t>(b) * k.dim(1) * q.dim(2) * q.dim(3)) +
                    static_cast<int64_t>(key_base) * q.dim(2) * q.dim(3);
                const int64_t query_offset =
                    static_cast<int64_t>(b) * q.dim(1) * q.dim(2) * q.dim(3);
                if (bf16_tensor_qk) {
                    const __nv_bfloat16* key_pointer =
                        k.data_as<__nv_bfloat16>() + key_offset;
                    const __nv_bfloat16* query_pointer =
                        q.data_as<__nv_bfloat16>() + query_offset;
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        key_count, q.dim(1), q.dim(3), &alpha,
                        key_pointer, CUDA_R_16BF, q.dim(2) * q.dim(3), q.dim(3),
                        query_pointer, CUDA_R_16BF, q.dim(2) * q.dim(3), q.dim(3),
                        &beta, score_pointer, CUDA_R_32F, key_count,
                        static_cast<int64_t>(q.dim(1)) * key_count, q.dim(2),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                } else {
                    const float* key_pointer = k_dot.data_as<float>() + key_offset;
                    const float* query_pointer = q_dot.data_as<float>() + query_offset;
                    CUBLAS_CHECK(cublasSgemmStridedBatched(
                        impl_->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        key_count, q.dim(1), q.dim(3), &alpha,
                        key_pointer, q.dim(2) * q.dim(3), q.dim(3),
                        query_pointer, q.dim(2) * q.dim(3), q.dim(3),
                        &beta, score_pointer, key_count,
                        static_cast<int64_t>(q.dim(1)) * key_count, q.dim(2)));
                }
            }
            if (dtype == DType::BF16) {
                attention_ordered_state_kernel<<<blocks(rows), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(),
                    maximum.data_as<float>(), denominator.data_as<float>(), mp,
                    bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr, q.dim(0), q.dim(1),
                    k.dim(1), q.dim(2), key_base, key_count, scale, causal, mm, bm);
            } else {
                attention_ordered_state_kernel<<<blocks(rows), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(),
                    maximum.data_as<float>(), denominator.data_as<float>(), mp,
                    bias_raw ? bias.data_as<float>() : nullptr, q.dim(0), q.dim(1),
                    k.dim(1), q.dim(2), key_base, key_count, scale, causal, mm, bm);
            }
            CUDA_CHECK(cudaGetLastError());
            if (dtype == DType::BF16 && q.dim(3) % 2 == 0) {
                attention_ordered_pv_bf16x2_kernel<<<blocks(output.numel() / 2), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(),
                    v.data_as<__nv_bfloat16>(), accumulation.data_as<float>(), q.dim(0),
                    q.dim(1), k.dim(1), q.dim(2), q.dim(3), key_base, key_count);
            } else if (dtype == DType::F32 && q.dim(3) % 4 == 0) {
                attention_ordered_pv_float4_kernel<<<blocks(output.numel() / 4), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(), v.data_as<float>(),
                    accumulation.data_as<float>(), q.dim(0), q.dim(1), k.dim(1),
                    q.dim(2), q.dim(3), key_base, key_count);
            } else if (dtype == DType::BF16) {
                attention_ordered_pv_kernel<<<blocks(output.numel()), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(),
                    v.data_as<__nv_bfloat16>(), accumulation.data_as<float>(), q.dim(0),
                    q.dim(1), k.dim(1), q.dim(2), q.dim(3), key_base, key_count);
            } else {
                attention_ordered_pv_kernel<<<blocks(output.numel()), kThreads>>>(
                    scores.data_as<float>(), previous.data_as<float>(), v.data_as<float>(),
                    accumulation.data_as<float>(), q.dim(0), q.dim(1), k.dim(1),
                    q.dim(2), q.dim(3), key_base, key_count);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (output_dtype == DType::BF16) {
            attention_ordered_finalize_kernel<<<blocks(output.numel()), kThreads>>>(
                accumulation.data_as<float>(), denominator.data_as<float>(),
                output.data_as<__nv_bfloat16>(), rows, q.dim(3));
        } else {
            attention_ordered_finalize_kernel<<<blocks(output.numel()), kThreads>>>(
                accumulation.data_as<float>(), denominator.data_as<float>(),
                output.data_as<float>(), rows, q.dim(3));
        }
        // Account for generic temporary score/state storage while it is live.
        // The existing reserved device-workspace budget covers this allocation;
        // no architecture-specific Memory Runtime policy is introduced.
        impl_->record();
    } else if (dtype == DType::BF16 && output_dtype == DType::F32) {
        attention_kernel<__nv_bfloat16, float><<<static_cast<int>(blocks_required), threads, workspace>>>(q.data_as<__nv_bfloat16>(), k.data_as<__nv_bfloat16>(), v.data_as<__nv_bfloat16>(), mp, bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr, output.data_as<float>(), q.dim(0), q.dim(1), k.dim(1), q.dim(2), q.dim(3), output_tiles, causal, scale, mm, bm);
    } else if (dtype == DType::BF16) {
        attention_kernel<<<static_cast<int>(blocks_required), threads, workspace>>>(q.data_as<__nv_bfloat16>(), k.data_as<__nv_bfloat16>(), v.data_as<__nv_bfloat16>(), mp, bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr, output.data_as<__nv_bfloat16>(), q.dim(0), q.dim(1), k.dim(1), q.dim(2), q.dim(3), output_tiles, causal, scale, mm, bm);
    } else {
        attention_kernel<<<static_cast<int>(blocks_required), threads, workspace>>>(q.data_as<float>(), k.data_as<float>(), v.data_as<float>(), mp, bias_raw ? bias.data_as<float>() : nullptr, output.data_as<float>(), q.dim(0), q.dim(1), k.dim(1), q.dim(2), q.dim(3), output_tiles, causal, scale, mm, bm);
    }
    CUDA_CHECK(cudaGetLastError());

    // Phase 15R observation-only hook. It is disabled unless an explicit path
    // is supplied and never changes Attention dispatch or arithmetic. The
    // token threshold is caller-provided so the hook is model/shape neutral.
    static bool attention_capture_complete = false;
    const char* capture_path = std::getenv("VRHINO_ATTENTION_CAPTURE_PATH");
    const char* minimum_tokens_text = std::getenv("VRHINO_ATTENTION_CAPTURE_MIN_TOKENS");
    const int64_t minimum_tokens = minimum_tokens_text
        ? std::strtoll(minimum_tokens_text, nullptr, 10) : 0;
    if (!attention_capture_complete && capture_path && capture_path[0] != '\0' &&
        q.dim(1) >= minimum_tokens && k.dim(1) >= minimum_tokens) {
        attention_capture_complete = true;
        TensorBundle capture{
            {"q", copy_to_host(q)},
            {"k", copy_to_host(k)},
            {"v", copy_to_host(v)},
            {"output.correct_bounded", copy_to_host(output)},
            {"scale", scalar_f32(scale)},
            {"causal", scalar_i64(causal ? 1 : 0)},
        };
        if (mask_raw) capture.emplace("mask", copy_to_host(mask));
        if (bias_raw) capture.emplace("bias", copy_to_host(bias));
        write_bundle(capture_path, capture);
        if (std::getenv("VRHINO_ATTENTION_CAPTURE_EXIT")) std::exit(0);
    }
    impl_->record(); return output;
}

template <int Spatial>
Tensor convolution(CudaBackend& backend, CudaBackend::Impl* impl, const Tensor& x_raw,
                   const Tensor& weight_raw, const Tensor* bias_raw,
                   const std::vector<int>& stride, const std::vector<int>& padding,
                   const std::vector<int>& dilation, int groups) {
    require(x_raw.ndim() == Spatial + 2 && weight_raw.ndim() == Spatial + 2, "convolution rank mismatch");
    require(static_cast<int>(stride.size()) == Spatial && static_cast<int>(padding.size()) == Spatial && static_cast<int>(dilation.size()) == Spatial, "convolution parameter rank mismatch");
    const DType execution_dtype = backend.execution_dtype();
    const char* bf16_compute_text = std::getenv("VRHINO_BF16_CONV_COMPUTE");
    require(!bf16_compute_text || std::strcmp(bf16_compute_text, "bf16") == 0 ||
                                     std::strcmp(bf16_compute_text, "fp32") == 0,
            "VRHINO_BF16_CONV_COMPUTE must be bf16 or fp32");
    // Generic Phase 16R2 FP32-island experiment for convolution semantics.
    // It remains off by default and never dispatches by architecture or shape.
    const DType dtype = execution_dtype == DType::BF16 && bf16_compute_text &&
            std::strcmp(bf16_compute_text, "fp32") == 0
        ? DType::F32 : execution_dtype;
    Tensor x = backend.copy_to_device(x_raw, dtype), weight = backend.copy_to_device(weight_raw, dtype);
    require(x.dim(1) == weight.dim(1) * groups, "convolution channel mismatch");
    std::vector<int64_t> shape = x.shape(); shape[1] = weight.dim(0);
    for (int d = 0; d < Spatial; ++d) shape[d + 2] = (x.shape()[d + 2] + 2 * padding[d] - dilation[d] * (weight.shape()[d + 2] - 1) - 1) / stride[d] + 1;
    for (int64_t value : shape) require(value > 0, "Invalid convolution output shape");
    Tensor output = impl->temporary(shape, dtype);

    CudnnConvDescriptor descriptor;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    descriptor.device = device;
    descriptor.compute_major = impl->compute_major;
    descriptor.io_dtype = dtype;
    descriptor.compute_dtype = DType::F32;
    descriptor.spatial_rank = Spatial;
    descriptor.x_dimensions = x.shape();
    descriptor.x_strides = x.strides();
    descriptor.w_dimensions = weight.shape();
    descriptor.w_strides = weight.strides();
    descriptor.y_dimensions = output.shape();
    descriptor.y_strides = output.strides();
    descriptor.padding.assign(padding.begin(), padding.end());
    descriptor.convolution_strides.assign(stride.begin(), stride.end());
    descriptor.dilation.assign(dilation.begin(), dilation.end());
    descriptor.groups = groups;
    descriptor.mode = CUDNN_CROSS_CORRELATION;

    std::string legacy_rejection;
    if (!cudnn_legacy_conv_admitted(descriptor, &legacy_rejection)) {
        std::string backend_rejection;
        require(cudnn_backend_conv_admitted(descriptor, &backend_rejection),
                "Convolution is not representable by legacy cuDNN (" +
                    legacy_rejection + ") and is not admitted by the INT64 "
                    "cuDNN Backend candidate (" + backend_rejection + ")");
        ProfileScope profile_scope_(impl, impl->profiling
            ? std::string(Spatial == 3 ? "conv3d.cudnn_backend" :
                                        "conv2d.cudnn_backend") +
              "|b=" + std::to_string(x.dim(0)) +
              "|cin=" + std::to_string(x.dim(1)) +
              "|cout=" + std::to_string(output.dim(1)) +
              "|elements=" + std::to_string(output.numel()) +
              "|dtype=" + dtype_name(dtype)
            : (Spatial == 3 ? "conv3d.cudnn_backend" :
                              "conv2d.cudnn_backend"));
        const CudnnConvExecution execution = impl->cudnn_conv->execute(
            descriptor, x.data(), weight.data(), output.data());
        require(execution.executed,
                "cuDNN Backend INT64 convolution failed closed: " +
                    execution.reason);
        if (bias_raw) {
            Tensor bias = backend.copy_to_device(*bias_raw, dtype);
            std::vector<int64_t> bias_shape(Spatial + 2, 1);
            bias_shape[1] = bias.numel();
            output = backend.add(output, bias.reshape(bias_shape));
        }
        impl->record();
        return output;
    }

    cudnnTensorDescriptor_t xd{}, yd{}; cudnnFilterDescriptor_t wd{}; cudnnConvolutionDescriptor_t cd{};
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xd)); CUDNN_CHECK(cudnnCreateTensorDescriptor(&yd));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wd)); CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cd));
    std::vector<int> xdims(x.ndim()), ydims(output.ndim()), xstrides(x.ndim()), ystrides(output.ndim()), wdims(weight.ndim());
    for (int i = 0; i < x.ndim(); ++i) { xdims[i] = x.shape()[i]; xstrides[i] = x.strides()[i]; ydims[i] = output.shape()[i]; ystrides[i] = output.strides()[i]; wdims[i] = weight.shape()[i]; }
    const cudnnDataType_t cudnn_dtype = dtype == DType::BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xd, cudnn_dtype, x.ndim(), xdims.data(), xstrides.data()));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yd, cudnn_dtype, output.ndim(), ydims.data(), ystrides.data()));
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(wd, cudnn_dtype, CUDNN_TENSOR_NCHW, weight.ndim(), wdims.data()));
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(cd, Spatial, padding.data(), stride.data(), dilation.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(cd, groups));
    CUDNN_CHECK(cudnnSetConvolutionMathType(cd, dtype == DType::BF16 ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH));
    int returned = 0; cudnnConvolutionFwdAlgoPerf_t performance{};
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(impl->cudnn, xd, wd, cd, yd, 1, &returned, &performance)); require(returned == 1 && performance.status == CUDNN_STATUS_SUCCESS, "No cuDNN convolution algorithm");
    size_t workspace_size = 0; CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(impl->cudnn, xd, wd, cd, yd, performance.algo, &workspace_size));
    Tensor workspace_tensor;
    void* workspace = nullptr;
    if (workspace_size) {
        workspace_tensor = impl->temporary(
            {static_cast<int64_t>(workspace_size)}, DType::U8, true);
        workspace = workspace_tensor.data();
    }
    const float alpha = 1.0f, beta = 0.0f;
    ProfileScope profile_scope_(impl, impl->profiling
        ? std::string(Spatial == 3 ? "conv3d" : "conv2d") + "|b=" +
          std::to_string(x.dim(0)) + "|cin=" + std::to_string(x.dim(1)) +
          "|cout=" + std::to_string(output.dim(1)) + "|elements=" +
          std::to_string(output.numel()) + "|dtype=" + dtype_name(dtype)
        : (Spatial == 3 ? "conv3d" : "conv2d"));
    CUDNN_CHECK(cudnnConvolutionForward(impl->cudnn, &alpha, xd, x.data(), wd, weight.data(), cd, performance.algo, workspace, workspace_size, &beta, yd, output.data()));
    cudnnDestroyConvolutionDescriptor(cd); cudnnDestroyFilterDescriptor(wd); cudnnDestroyTensorDescriptor(yd); cudnnDestroyTensorDescriptor(xd);
    if (bias_raw) {
        Tensor bias = backend.copy_to_device(*bias_raw, dtype); std::vector<int64_t> bias_shape(Spatial + 2, 1); bias_shape[1] = bias.numel();
        output = backend.add(output, bias.reshape(bias_shape));
    }
    impl->record(); return output;
}

Tensor CudaBackend::conv3d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                           const std::vector<int>& stride, const std::vector<int>& padding,
                           const std::vector<int>& dilation, int groups) {
    return convolution<3>(*this, impl_, x, weight, bias, stride, padding, dilation, groups);
}
Tensor CudaBackend::conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                           const std::vector<int>& stride, const std::vector<int>& padding,
                           const std::vector<int>& dilation, int groups) {
    return convolution<2>(*this, impl_, x, weight, bias, stride, padding, dilation, groups);
}

Tensor CudaBackend::conv_transpose2d(
        const Tensor& raw, const Tensor& raw_weight, const Tensor* raw_bias,
        const std::vector<int>& stride, const std::vector<int>& padding,
        const std::vector<int>& output_padding, const std::vector<int>& dilation,
        int groups) {
    VRHINO_PROFILE("conv_transpose2d");
    require(raw.ndim() == 4 && raw_weight.ndim() == 4 &&
            stride.size() == 2 && padding.size() == 2 &&
            output_padding.size() == 2 && dilation.size() == 2 && groups > 0 &&
            raw.dim(1) == raw_weight.dim(0) && raw.dim(1) % groups == 0 &&
            stride[0] > 0 && stride[1] > 0 && padding[0] >= 0 &&
            padding[1] >= 0 && output_padding[0] >= 0 && output_padding[1] >= 0 &&
            output_padding[0] < stride[0] && output_padding[1] < stride[1] &&
            dilation[0] > 0 && dilation[1] > 0,
            "conv_transpose2d contract violation");
    const int64_t output_channels = raw_weight.dim(1) * groups;
    if (raw_bias) require(raw_bias->numel() == output_channels,
                          "conv_transpose2d bias mismatch");
    const int64_t output_height = (raw.dim(2) - 1) * stride[0] -
        2 * padding[0] + dilation[0] * (raw_weight.dim(2) - 1) +
        output_padding[0] + 1;
    const int64_t output_width = (raw.dim(3) - 1) * stride[1] -
        2 * padding[1] + dilation[1] * (raw_weight.dim(3) - 1) +
        output_padding[1] + 1;
    require(output_height > 0 && output_width > 0,
            "conv_transpose2d output shape invalid");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    Tensor weight = copy_to_device(raw_weight, dtype);
    Tensor bias;
    if (raw_bias) bias = copy_to_device(*raw_bias, dtype);
    Tensor output = impl_->temporary(
        {raw.dim(0), output_channels, output_height, output_width}, dtype);
    if (dtype == DType::BF16)
        conv_transpose2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<__nv_bfloat16>(), weight.data_as<__nv_bfloat16>(),
            raw_bias ? bias.data_as<__nv_bfloat16>() : nullptr,
            output.data_as<__nv_bfloat16>(), output.numel(), raw.dim(0), raw.dim(1),
            output_channels, raw.dim(2), raw.dim(3), output_height, output_width,
            raw_weight.dim(2), raw_weight.dim(3), stride[0], stride[1], padding[0],
            padding[1], dilation[0], dilation[1], groups, raw_bias != nullptr);
    else
        conv_transpose2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<float>(), weight.data_as<float>(),
            raw_bias ? bias.data_as<float>() : nullptr, output.data_as<float>(),
            output.numel(), raw.dim(0), raw.dim(1), output_channels, raw.dim(2),
            raw.dim(3), output_height, output_width, raw_weight.dim(2),
            raw_weight.dim(3), stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups, raw_bias != nullptr);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::max_pool2d(const Tensor& raw,
                               const std::vector<int>& kernel,
                               const std::vector<int>& stride,
                               const std::vector<int>& padding) {
    VRHINO_PROFILE("max_pool2d");
    require(raw.ndim() == 4 && kernel.size() == 2 && stride.size() == 2 &&
            padding.size() == 2 && kernel[0] > 0 && kernel[1] > 0 &&
            stride[0] > 0 && stride[1] > 0 && padding[0] >= 0 && padding[1] >= 0,
            "max_pool2d contract violation");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    const int64_t output_height =
        (input.dim(2) + 2 * padding[0] - kernel[0]) / stride[0] + 1;
    const int64_t output_width =
        (input.dim(3) + 2 * padding[1] - kernel[1]) / stride[1] + 1;
    require(output_height > 0 && output_width > 0,
            "max_pool2d output shape invalid");
    Tensor output = impl_->temporary(
        {input.dim(0), input.dim(1), output_height, output_width}, dtype);
    if (dtype == DType::BF16)
        max_pool2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            output.numel(), input.dim(1), input.dim(2), input.dim(3),
            output_height, output_width, kernel[0], kernel[1], stride[0],
            stride[1], padding[0], padding[1]);
    else
        max_pool2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), output.numel(),
            input.dim(1), input.dim(2), input.dim(3), output_height,
            output_width, kernel[0], kernel[1], stride[0], stride[1],
            padding[0], padding[1]);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::pad(const Tensor& raw, const std::vector<int64_t>& padding, float value, PadMode mode) {
    VRHINO_PROFILE("pad");
    require(padding.size() % 2 == 0 && padding.size() <= static_cast<size_t>(2 * raw.ndim()), "Invalid pad vector");
    const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype); std::vector<int64_t> shape = input.shape(); Meta before{}; before.rank = input.ndim();
    for (size_t pair = 0; pair < padding.size() / 2; ++pair) { const int dim = input.ndim() - 1 - pair; require(padding[2 * pair] >= 0 && padding[2 * pair + 1] >= 0, "Negative padding unsupported"); before.shape[dim] = padding[2 * pair]; shape[dim] += padding[2 * pair] + padding[2 * pair + 1]; }
    Tensor output = impl_->temporary(shape, dtype); if (dtype == DType::BF16) pad_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(input.shape()), meta(shape), before, value, mode == PadMode::Replicate); else pad_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), output.numel(), meta(input.shape()), meta(shape), before, value, mode == PadMode::Replicate);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::reduce_sum(const Tensor& raw, int64_t dim, bool keepdim) {
    VRHINO_PROFILE("reduce_sum");
    if (dim < 0) dim += raw.ndim(); require(dim >= 0 && dim < raw.ndim(), "reduce dimension invalid"); const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype); std::vector<int64_t> shape = input.shape();
    if (keepdim) shape[dim] = 1; else shape.erase(shape.begin() + dim); Tensor output = impl_->temporary(shape, dtype);
    if (dtype == DType::BF16) reduce_sum_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(input.shape()), meta(shape), dim); else reduce_sum_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), output.numel(), meta(input.shape()), meta(shape), dim); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::softmax(const Tensor& raw, int64_t axis) {
    VRHINO_PROFILE("softmax");
    if (axis < 0) axis += raw.ndim();
    require(axis >= 0 && axis < raw.ndim(), "softmax axis invalid");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    Tensor output = impl_->temporary(input.shape(), dtype);
    int64_t inner = 1;
    for (int dimension = static_cast<int>(axis) + 1;
         dimension < input.ndim(); ++dimension)
        inner *= input.dim(dimension);
    const int64_t rows = input.numel() / input.dim(axis);
    if (dtype == DType::BF16)
        softmax_axis_kernel<<<blocks(rows), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            rows, input.dim(axis), inner);
    else
        softmax_axis_kernel<<<blocks(rows), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), rows,
            input.dim(axis), inner);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::group_norm(const Tensor& raw, int groups, const Tensor* weight_raw, const Tensor* bias_raw, float eps) {
    VRHINO_PROFILE("norm.group");
    require(raw.ndim() >= 3 && groups > 0 && raw.dim(1) % groups == 0,
            "group norm shape/groups mismatch");
    const char* bf16_output_text = std::getenv("VRHINO_BF16_NORM_OUTPUT");
    DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype();
    if (dtype == DType::BF16 && bf16_output_text && std::strcmp(bf16_output_text, "fp32") == 0) dtype = DType::F32;
    Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(input.shape(), dtype), weight, bias;
    if (weight_raw) weight = copy_to_device(*weight_raw, dtype); if (bias_raw) bias = copy_to_device(*bias_raw, dtype);
    const int64_t spatial = input.numel() / (input.dim(0) * input.dim(1));
    const int64_t units = input.dim(0) * groups;
    const int channels = static_cast<int>(input.dim(1));
    const int channels_per = channels / groups;
    const int64_t count = static_cast<int64_t>(channels_per) * spatial;
    require(units <= std::numeric_limits<int>::max() &&
            units * channels_per <= std::numeric_limits<int>::max(),
            "group norm launch extent exceeds CUDA grid capacity");
    if (count <= kGroupNormDirectMaximum) {
        if (dtype == DType::BF16)
            group_norm_direct_kernel<<<static_cast<int>(units), kGroupNormThreads>>>(
                input.data_as<__nv_bfloat16>(),
                weight_raw ? weight.data_as<__nv_bfloat16>() : nullptr,
                bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr,
                output.data_as<__nv_bfloat16>(), channels, spatial, groups, eps);
        else
            group_norm_direct_kernel<<<static_cast<int>(units), kGroupNormThreads>>>(
                input.data_as<float>(), weight_raw ? weight.data_as<float>() : nullptr,
                bias_raw ? bias.data_as<float>() : nullptr, output.data_as<float>(),
                channels, spatial, groups, eps);
    } else {
        const int work_partials = static_cast<int>(
            (count + kGroupNormElementsPerPartial - 1) /
            kGroupNormElementsPerPartial);
        const int occupancy_partials = static_cast<int>(
            (2LL * impl_->multiprocessor_count + units - 1) / units);
        const int useful_partials = static_cast<int>(std::min<int64_t>(
            kGroupNormMaximumPartials,
            (count + kGroupNormThreads - 1) / kGroupNormThreads));
        const int partials = std::min(
            useful_partials,
            std::clamp(std::max({2, work_partials, occupancy_partials}),
                       2, kGroupNormMaximumPartials));
        require(units * partials <= std::numeric_limits<int>::max(),
                "group norm partial launch exceeds CUDA grid capacity");
        Tensor partial = impl_->temporary({units * partials}, DType::F32);
        Tensor moments = impl_->temporary({2, units}, DType::F32);
        float* mean = moments.data_as<float>();
        float* inverse = mean + units;
        const int partial_blocks = static_cast<int>(units * partials);
        if (dtype == DType::BF16)
            group_norm_partial_sum_kernel<<<partial_blocks, kGroupNormThreads>>>(
                input.data_as<__nv_bfloat16>(), partial.data_as<float>(),
                count, partials);
        else
            group_norm_partial_sum_kernel<<<partial_blocks, kGroupNormThreads>>>(
                input.data_as<float>(), partial.data_as<float>(), count, partials);
        if (dtype == DType::BF16) {
            group_norm_mean_kernel<<<static_cast<int>(units), kGroupNormThreads>>>(
                input.data_as<__nv_bfloat16>(), partial.data_as<float>(), mean,
                count, partials);
            group_norm_partial_variance_kernel<<<partial_blocks, kGroupNormThreads>>>(
                input.data_as<__nv_bfloat16>(), mean, partial.data_as<float>(),
                count, partials);
        } else {
            group_norm_mean_kernel<<<static_cast<int>(units), kGroupNormThreads>>>(
                input.data_as<float>(), partial.data_as<float>(), mean, count,
                partials);
            group_norm_partial_variance_kernel<<<partial_blocks, kGroupNormThreads>>>(
                input.data_as<float>(), mean, partial.data_as<float>(), count,
                partials);
        }
        group_norm_inverse_kernel<<<static_cast<int>(units), kGroupNormThreads>>>(
            partial.data_as<float>(), inverse, count, partials, eps);
        const int normalize_blocks = static_cast<int>(units * channels_per);
        if (dtype == DType::BF16)
            group_norm_normalize_kernel<<<normalize_blocks, kGroupNormThreads>>>(
                input.data_as<__nv_bfloat16>(),
                weight_raw ? weight.data_as<__nv_bfloat16>() : nullptr,
                bias_raw ? bias.data_as<__nv_bfloat16>() : nullptr,
                output.data_as<__nv_bfloat16>(), mean, inverse, channels,
                spatial, groups);
        else
            group_norm_normalize_kernel<<<normalize_blocks, kGroupNormThreads>>>(
                input.data_as<float>(), weight_raw ? weight.data_as<float>() : nullptr,
                bias_raw ? bias.data_as<float>() : nullptr, output.data_as<float>(),
                mean, inverse, channels, spatial, groups);
    }
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::interpolate_nearest(const Tensor& raw, const std::vector<double>& factors) {
    VRHINO_PROFILE("interpolate_nearest");
    require(factors.size() <= static_cast<size_t>(raw.ndim() - 2), "interpolate factor rank mismatch"); const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype); std::vector<int64_t> shape = input.shape(); const int start = raw.ndim() - factors.size();
    for (size_t index = 0; index < factors.size(); ++index) { require(factors[index] > 0, "Invalid interpolation factor"); shape[start + index] = static_cast<int64_t>(std::floor(shape[start + index] * factors[index])); }
    Tensor output = impl_->temporary(shape, dtype); if (dtype == DType::BF16) interpolate_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), meta(input.shape()), meta(shape)); else interpolate_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), output.numel(), meta(input.shape()), meta(shape)); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::interpolate_bilinear_2d(
        const Tensor& raw, int64_t output_height, int64_t output_width,
        bool align_corners) {
    VRHINO_PROFILE("interpolate_bilinear_2d");
    require(raw.ndim() == 4 && output_height > 0 && output_width > 0,
            "bilinear resize requires NCHW and positive output size");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    const std::vector<int64_t> shape = {
        input.dim(0), input.dim(1), output_height, output_width};
    Tensor output = impl_->temporary(shape, dtype);
    if (dtype == DType::BF16)
        interpolate_bilinear_2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            output.numel(), static_cast<int>(input.dim(1)),
            static_cast<int>(input.dim(2)), static_cast<int>(input.dim(3)),
            static_cast<int>(output_height), static_cast<int>(output_width),
            align_corners);
    else
        interpolate_bilinear_2d_kernel<<<blocks(output.numel()), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), output.numel(),
            static_cast<int>(input.dim(1)), static_cast<int>(input.dim(2)),
            static_cast<int>(input.dim(3)), static_cast<int>(output_height),
            static_cast<int>(output_width), align_corners);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::pixel_norm(const Tensor& raw, int64_t dim, float eps) {
    VRHINO_PROFILE("norm.pixel");
    if (dim < 0) dim += raw.ndim(); require(dim >= 0 && dim < raw.ndim(), "pixel norm dim invalid");
    const char* bf16_output_text = std::getenv("VRHINO_BF16_NORM_OUTPUT");
    DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype();
    if (dtype == DType::BF16 && bf16_output_text && std::strcmp(bf16_output_text, "fp32") == 0) dtype = DType::F32;
    Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(input.shape(), dtype);
    const int64_t channels = input.dim(dim);
    const int64_t units = input.numel() / channels;
    const Meta input_meta = meta(input.shape());
    require(units <= std::numeric_limits<int>::max(),
            "pixel norm launch extent exceeds CUDA grid capacity");
    if (input_meta.strides[dim] == 1) {
        if (dtype == DType::BF16)
            pixel_norm_contiguous_axis_kernel<<<static_cast<int>(units), kThreads>>>(
                input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
                units, input_meta, dim, eps);
        else
            pixel_norm_contiguous_axis_kernel<<<static_cast<int>(units), kThreads>>>(
                input.data_as<float>(), output.data_as<float>(), units,
                input_meta, dim, eps);
    } else {
        const int64_t block_count =
            (units + kPixelNormPositions - 1) / kPixelNormPositions;
        require(block_count <= std::numeric_limits<int>::max(),
                "pixel norm tiled launch exceeds CUDA grid capacity");
        const dim3 threads(kPixelNormPositions, kPixelNormChannelLanes);
        if (dtype == DType::BF16)
            pixel_norm_strided_axis_kernel<<<static_cast<int>(block_count), threads>>>(
                input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
                units, input_meta, dim, eps);
        else
            pixel_norm_strided_axis_kernel<<<static_cast<int>(block_count), threads>>>(
                input.data_as<float>(), output.data_as<float>(), units,
                input_meta, dim, eps);
    }
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::l2_normalize(const Tensor& raw, int64_t dim, float eps) {
    VRHINO_PROFILE("norm.l2");
    if (dim < 0) dim += raw.ndim();
    require(dim >= 0 && dim < raw.ndim() && eps >= 0.0f,
            "l2 normalize contract violation");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(input.shape(), dtype);
    const int64_t units = input.numel() / input.dim(dim);
    require(units <= std::numeric_limits<int>::max(),
            "l2 normalize launch extent exceeds CUDA grid capacity");
    if (dtype == DType::BF16)
        l2_normalize_kernel<<<static_cast<int>(units), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            units, meta(input.shape()), static_cast<int>(dim), eps);
    else
        l2_normalize_kernel<<<static_cast<int>(units), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), units,
            meta(input.shape()), static_cast<int>(dim), eps);
    CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::pixel_shuffle_nd(const Tensor& raw, const std::vector<int64_t>& factors) {
    VRHINO_PROFILE("pixel_shuffle");
    require(raw.ndim() == 5 && factors.size() == 3, "pixel_shuffle_nd requires 5D and 3 factors"); const int64_t product = factors[0] * factors[1] * factors[2]; require(raw.dim(1) % product == 0, "pixel shuffle channel mismatch");
    const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype); std::vector<int64_t> shape = {raw.dim(0), raw.dim(1) / product, raw.dim(2) * factors[0], raw.dim(3) * factors[1], raw.dim(4) * factors[2]}; Tensor output = impl_->temporary(shape, dtype);
    if (dtype == DType::BF16) pixel_shuffle_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), output.numel(), shape[0], shape[1], raw.dim(2), raw.dim(3), raw.dim(4), factors[0], factors[1], factors[2]); else pixel_shuffle_kernel<<<blocks(output.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), output.numel(), shape[0], shape[1], raw.dim(2), raw.dim(3), raw.dim(4), factors[0], factors[1], factors[2]); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::rng_normal(RngState& state, const std::vector<int64_t>& shape, DType dtype) {
    VRHINO_PROFILE("rng_normal");
    require(state.algorithm_id == "pytorch_compat.v1", "Unsupported RNG algorithm"); require(dtype == DType::F32 || dtype == DType::BF16, "rng_normal dtype unsupported");
    Tensor fp32 = impl_->temporary(shape, DType::F32); const int grid = static_cast<int>((fp32.numel() + 255) / 256);
    rng_normal_kernel<<<grid, 256>>>(fp32.data_as<float>(), fp32.numel(), state.seed, state.offset); CUDA_CHECK(cudaGetLastError()); state.offset += 4; impl_->record(); return dtype == DType::BF16 ? cast(fp32, DType::BF16) : fp32;
}

Tensor CudaBackend::clamp(const Tensor& raw, float minimum, float maximum) {
    VRHINO_PROFILE("clamp");
    require(minimum <= maximum, "Invalid clamp bounds"); const DType dtype = raw.device() == DeviceId::accelerator() && (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16) ? raw.dtype() : execution_dtype(); Tensor input = copy_to_device(raw, dtype), output = impl_->temporary(input.shape(), dtype);
    if (dtype == DType::BF16) clamp_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(), input.numel(), minimum, maximum); else clamp_kernel<<<blocks(input.numel()), kThreads>>>(input.data_as<float>(), output.data_as<float>(), input.numel(), minimum, maximum); CUDA_CHECK(cudaGetLastError()); impl_->record(); return output;
}

Tensor CudaBackend::exp(const Tensor& raw) {
    VRHINO_PROFILE("exp");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    Tensor output = impl_->temporary(input.shape(), dtype);
    if (dtype == DType::BF16)
        exp_kernel<<<blocks(input.numel()), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            input.numel());
    else
        exp_kernel<<<blocks(input.numel()), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), input.numel());
    CUDA_CHECK(cudaGetLastError());
    impl_->record();
    return output;
}

Tensor CudaBackend::sqrt(const Tensor& raw) {
    VRHINO_PROFILE("sqrt");
    const DType dtype = raw.device() == DeviceId::accelerator() &&
        (raw.dtype() == DType::F32 || raw.dtype() == DType::BF16)
        ? raw.dtype() : execution_dtype();
    Tensor input = copy_to_device(raw, dtype);
    Tensor output = impl_->temporary(input.shape(), dtype);
    if (dtype == DType::BF16)
        sqrt_kernel<<<blocks(input.numel()), kThreads>>>(
            input.data_as<__nv_bfloat16>(), output.data_as<__nv_bfloat16>(),
            input.numel());
    else
        sqrt_kernel<<<blocks(input.numel()), kThreads>>>(
            input.data_as<float>(), output.data_as<float>(), input.numel());
    CUDA_CHECK(cudaGetLastError());
    impl_->record();
    return output;
}

}  // namespace vrhino
