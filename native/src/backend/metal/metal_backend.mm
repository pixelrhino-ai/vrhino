#include "vrhino/backend/metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "vrhino/error.h"
#include "metal_kernels.h"

namespace vrhino {
namespace {

constexpr int kMaxDims = 8;
constexpr NSUInteger kThreads = 256;

struct Meta {
    int32_t rank = 0;
    int32_t pad = 0;
    int64_t shape[kMaxDims]{};
    int64_t strides[kMaxDims]{};
};

Meta meta(const std::vector<int64_t>& shape) {
    require(shape.size() <= kMaxDims, "Metal primitive rank exceeds 8");
    Meta result;
    result.rank = static_cast<int32_t>(shape.size());
    const auto strides = contiguous_strides(shape);
    for (int index = 0; index < result.rank; ++index) {
        result.shape[index] = shape[index];
        result.strides[index] = strides[index];
    }
    return result;
}

int dtype_code(DType dtype) {
    switch (dtype) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::BF16: return 2;
        case DType::I64: return 3;
        case DType::I32: return 4;
        case DType::U8:
        case DType::Bool: return 5;
    }
    throw Error("Unsupported Metal dtype");
}

std::string ns_error(NSError* error) {
    if (!error) return "unknown Metal error";
    return std::string([[error localizedDescription] UTF8String]);
}

std::vector<int64_t> broadcast_shape(const Tensor& a, const Tensor& b) {
    const int rank = std::max(a.ndim(), b.ndim());
    std::vector<int64_t> output(rank, 1);
    for (int index = 0; index < rank; ++index) {
        const int ai = index - (rank - a.ndim());
        const int bi = index - (rank - b.ndim());
        const int64_t av = ai < 0 ? 1 : a.shape()[ai];
        const int64_t bv = bi < 0 ? 1 : b.shape()[bi];
        require(av == bv || av == 1 || bv == 1, "Incompatible broadcast shapes");
        output[index] = std::max(av, bv);
    }
    return output;
}

struct AllocationTracker {
    std::atomic<size_t> current{0};
    std::atomic<size_t> peak{0};
};

struct MetalAllocation {
    id<MTLBuffer> buffer = nil;
    size_t logical_bytes = 0;
    std::shared_ptr<AllocationTracker> tracker;

    MetalAllocation(id<MTLBuffer> value, size_t bytes,
                    std::shared_ptr<AllocationTracker> allocation_tracker)
        : buffer(value), logical_bytes(bytes), tracker(std::move(allocation_tracker)) {
        const size_t current = tracker->current.fetch_add(bytes) + bytes;
        size_t peak = tracker->peak.load();
        while (peak < current && !tracker->peak.compare_exchange_weak(peak, current)) {}
    }
    ~MetalAllocation() {
        tracker->current.fetch_sub(logical_bytes);
        [buffer release];
    }
};

struct BufferRef {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
};

struct BufferArg {
    BufferRef ref;
    NSUInteger index = 0;
};

struct CastParams { uint64_t count; int32_t source; int32_t destination; };
struct BinaryParams { uint64_t count; int32_t kind; int32_t pad; Meta output; Meta a; Meta b; };
struct BiasParams { uint64_t count; int64_t width; };
struct MatmulParams { uint64_t count; int64_t m; int64_t n; int64_t k; int32_t has_bias; int32_t pad = 0; };
struct PermuteParams { uint64_t count; Meta input; Meta output; Meta dimensions; };
struct SliceParams { uint64_t count; Meta input; Meta output; int32_t dimension; int32_t pad; int64_t start; };
struct GatherParams { uint64_t count; int64_t indices_per_batch; int64_t rows; int64_t width; int32_t batched; int32_t index_dtype; };
struct ConcatParams { uint64_t count; Meta input; Meta output; int32_t dimension; int32_t pad; int64_t offset; };
struct NormParams { uint64_t count; int64_t rows; int64_t width; float eps; int32_t has_weight; int32_t has_bias; int32_t axis; int32_t pad = 0; };
struct AxisNormParams { uint64_t count; Meta tensor_meta; int32_t axis; float eps; int32_t has_weight; int32_t pad = 0; };
struct UnaryParams { uint64_t count; int32_t kind; float a; float b; };
struct SinParams { uint64_t count; int32_t width; int32_t flip; };
struct RopeParams { uint64_t count; Meta x; Meta frequency; };
struct AttentionParams { int32_t batch; int32_t query_tokens; int32_t key_tokens; int32_t heads; int32_t width; int32_t causal; float scale; int32_t has_mask; int32_t has_bias; int32_t observe; Meta mask; Meta bias; };
struct PadParams { uint64_t count; Meta input; Meta output; Meta before; float value; int32_t replicate; };
struct ReduceParams { uint64_t count; Meta input; Meta output; int32_t dimension; int32_t pad = 0; };
struct SoftmaxParams { uint64_t rows; int64_t axis_size; int64_t inner; };
struct GroupNormParams { int32_t batch; int32_t channels; int64_t spatial; int32_t groups; float eps; int32_t has_weight; int32_t has_bias; int32_t pad = 0; };
struct MetaCountParams { uint64_t count; Meta input; Meta output; int32_t axis; float eps; };
struct ShuffleParams { uint64_t count; int32_t b,c,t,h,w,ft,fh,fw; };
struct Pool2DParams {
    uint64_t count;
    int32_t channels, ih, iw, oh, ow;
    int32_t kh, kw, sh, sw, ph, pw;
};
struct ConvParams { uint64_t count; int32_t spatial; int32_t batch; int32_t in_channels; int32_t out_channels; int32_t groups; int32_t in_size[3]; int32_t out_size[3]; int32_t kernel[3]; int32_t stride[3]; int32_t padding[3]; int32_t dilation[3]; int32_t has_bias; };
struct ConvTranspose2DParams { uint64_t count; int32_t batch,input_channels,output_channels,groups; int32_t input_height,input_width,output_height,output_width; int32_t kernel_height,kernel_width,stride_height,stride_width; int32_t padding_height,padding_width,dilation_height,dilation_width; int32_t has_bias; };
struct RngParams { uint64_t count; uint64_t seed; uint64_t offset; };

}  // namespace

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    std::vector<id<MTLCommandBuffer>> pending;
    std::vector<std::pair<std::string, id<MTLCommandBuffer>>> profiled;
    std::map<uint64_t, id<MTLCommandBuffer>> fences;
    EventFenceTracker fence_tracker;
    std::shared_ptr<AllocationTracker> allocation_tracker = std::make_shared<AllocationTracker>();
    std::map<uintptr_t, std::weak_ptr<MetalAllocation>> allocations;
    std::map<std::pair<const void*, DType>, Tensor> weight_cache;
    size_t weight_cache_bytes = 0;
    size_t weight_cache_capacity = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    size_t upload_bytes = 0;
    double upload_seconds = 0.0;
    DType execution_dtype = DType::F32;
    bool profiling = false;
    bool weight_cache_enabled = true;
    bool true_quant_compute = false;
    QuantComputeStats quant_stats;
    MemoryBudget memory_budget;
    MemoryRuntimeOptions memory_options;
    MemoryRuntimeStats memory_stats;
    bool trace_recording = false;
    std::vector<MemoryAccess> recorded_trace;
    std::vector<MemoryAccess> execution_trace;
    std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> active_profiles;
    std::map<std::string, ProfileStat> region_profiles;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "No Metal device available");
        [device retain];
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue creation failed");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:metal::kMetalKernelSource];
        library = [device newLibraryWithSource:source options:options error:&error];
        [options release];
        require(library != nil, "Metal shader compilation failed: " + ns_error(error));
        const size_t working_set = static_cast<size_t>(device.recommendedMaxWorkingSetSize);
        weight_cache_capacity = working_set ? static_cast<size_t>(working_set * 0.40) : (6ULL << 30);
    }

    ~Impl() {
        try { synchronize(); } catch (...) {}
        for (auto& [id, command] : fences) { (void)id; [command release]; }
        fences.clear();
        fence_tracker.shutdown();
        for (auto& [name, pipeline] : pipelines) { (void)name; [pipeline release]; }
        [library release];
        [queue release];
        [device release];
    }

    id<MTLComputePipelineState> pipeline(const std::string& name) {
        const auto found = pipelines.find(name);
        if (found != pipelines.end()) return found->second;
        NSString* function_name = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        require(function != nil, "Metal shader function not found: " + name);
        NSError* error = nil;
        id<MTLComputePipelineState> value = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        require(value != nil, "Metal pipeline creation failed for " + name + ": " + ns_error(error));
        pipelines.emplace(name, value);
        return value;
    }

    Tensor allocate(const std::vector<int64_t>& shape, DType dtype) {
        const size_t bytes = static_cast<size_t>(shape_numel(shape)) * dtype_size(dtype);
        id<MTLBuffer> buffer = [device newBufferWithLength:std::max<size_t>(bytes, 1)
                                                   options:MTLResourceStorageModeShared];
        require(buffer != nil, "Metal buffer allocation failed for " + std::to_string(bytes) + " bytes");
        auto allocation = std::make_shared<MetalAllocation>(buffer, bytes, allocation_tracker);
        auto storage = std::make_shared<Storage>();
        storage->data = buffer.contents;
        storage->bytes = bytes;
        storage->device = DeviceId::accelerator();
        storage->domain = MemoryDomain::Unified;
        storage->owner = true;
        storage->native_owner = allocation;
        allocations[reinterpret_cast<uintptr_t>(storage->data)] = allocation;
        memory_stats.accounting.peak_device_bytes = std::max(
            memory_stats.accounting.peak_device_bytes, allocation_tracker->peak.load());
        return Tensor(std::move(storage), 0, shape, dtype);
    }

    BufferRef resolve(const Tensor& tensor) {
        require(tensor.defined() && !tensor.device().is_host(), "Metal operation requires accelerator tensor");
        const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data());
        for (auto it = allocations.begin(); it != allocations.end();) {
            auto allocation = it->second.lock();
            if (!allocation) { it = allocations.erase(it); continue; }
            const uintptr_t base = it->first;
            if (address >= base && address < base + allocation->logical_bytes) {
                require(address + tensor.bytes() <= base + allocation->logical_bytes,
                        "Metal tensor view exceeds buffer");
                return {allocation->buffer, static_cast<NSUInteger>(address - base)};
            }
            ++it;
        }
        throw Error("Invalid or foreign Metal buffer");
    }

    void complete(id<MTLCommandBuffer> command) {
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Metal command failed: " + ns_error(command.error));
    }

    void reap(bool force = false) {
        for (auto it = pending.begin(); it != pending.end();) {
            id<MTLCommandBuffer> command = *it;
            if (force || command.status == MTLCommandBufferStatusCompleted ||
                command.status == MTLCommandBufferStatusError) {
                complete(command);
                [command release];
                it = pending.erase(it);
            } else ++it;
        }
        if (pending.size() > 64) {
            id<MTLCommandBuffer> command = pending.front();
            complete(command); [command release]; pending.erase(pending.begin());
        }
    }

    void submit(id<MTLCommandBuffer> command, const std::string& profile_name = {}) {
        [command commit];
        [command retain];
        pending.push_back(command);
        if (profiling && !profile_name.empty()) {
            [command retain];
            profiled.emplace_back(profile_name, command);
        }
        reap(false);
    }

    void dispatch(const std::string& function, uint64_t count,
                  const std::vector<BufferArg>& buffers,
                  NSUInteger parameter_index, const void* parameters, size_t parameter_bytes,
                  const std::string& profile_name = {}) {
        if (count == 0) return;
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command != nil, "Metal command buffer creation failed");
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder != nil, "Metal compute encoder creation failed");
            id<MTLComputePipelineState> state = pipeline(function);
            [encoder setComputePipelineState:state];
            for (const BufferArg& argument : buffers)
                [encoder setBuffer:argument.ref.buffer offset:argument.ref.offset atIndex:argument.index];
            [encoder setBytes:parameters length:parameter_bytes atIndex:parameter_index];
            const NSUInteger width = std::min<NSUInteger>(kThreads, state.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            submit(command, profile_name.empty() ? function : profile_name);
        }
    }

    void synchronize() {
        reap(true);
    }
};

namespace {

struct ProfileScope {
    MetalBackend::Impl* impl;
    std::string name;
    ProfileScope(MetalBackend::Impl* value, std::string label) : impl(value), name(std::move(label)) {}
};

Tensor binary(MetalBackend& backend, MetalBackend::Impl* impl,
              const Tensor& a_raw, const Tensor& b_raw, int kind) {
    Tensor a = backend.copy_to_device(a_raw, DType::F32);
    Tensor b = backend.copy_to_device(b_raw, DType::F32);
    const auto shape = broadcast_shape(a, b);
    Tensor output = backend.allocate_device(shape, DType::F32);
    BinaryParams params{static_cast<uint64_t>(output.numel()), kind, 0,
                        meta(shape), meta(a.shape()), meta(b.shape())};
    impl->dispatch("binary_f32", params.count,
                   {{impl->resolve(a), 0}, {impl->resolve(b), 1}, {impl->resolve(output), 2}},
                   3, &params, sizeof(params), kind == 0 ? "elementwise.add" :
                       (kind == 1 ? "elementwise.mul" : "elementwise.div"));
    return output;
}

}  // namespace

MetalBackend::MetalBackend() : impl_(new Impl()) {}
MetalBackend::~MetalBackend() { delete impl_; }
std::string MetalBackend::name() const { return "native-metal-correctness"; }
int MetalBackend::device_count() const { return impl_->device ? 1 : 0; }

DeviceCapability MetalBackend::device_capability(int device_index) const {
    require(device_index == 0 && impl_->device != nil, "Metal device index out of range");
    DeviceCapability result;
    result.name = std::string([impl_->device.name UTF8String]);
    result.family = "apple-gpu";
    result.device_memory_bytes = static_cast<size_t>(impl_->device.recommendedMaxWorkingSetSize);
    result.supports_fp16_storage = true;
    result.supports_fp16_arithmetic = true;
    result.supports_bf16_storage = true;
    result.supports_bf16_arithmetic = true;
    result.f64_mode = DeviceCapability::F64Mode::SoftwareEmulated;
    result.memory = memory_capabilities();
    return result;
}

BackendMemoryCapabilities MetalBackend::memory_capabilities() const {
    return {true, true, true, true, false, true};
}

bool MetalBackend::supports(DType dtype) const {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16 ||
        dtype == DType::I64 || dtype == DType::I32 || dtype == DType::U8 || dtype == DType::Bool;
}

void MetalBackend::set_execution_dtype(DType dtype) {
    require(dtype == DType::F32, "Phase 12B Metal execution currently supports FP32 only");
    impl_->execution_dtype = dtype;
}
DType MetalBackend::execution_dtype() const { return impl_->execution_dtype; }
Tensor MetalBackend::allocate_host(const std::vector<int64_t>& shape, DType dtype) { return Tensor::host(shape, dtype); }
Tensor MetalBackend::allocate_device(const std::vector<int64_t>& shape, DType dtype) {
    require(supports(dtype), "Unsupported Metal allocation dtype");
    return impl_->allocate(shape, dtype);
}

TransferFence MetalBackend::copy(const Tensor& source, Tensor& destination, CopyMode mode) {
    require(source.defined() && destination.defined() && source.dtype() == destination.dtype() &&
            source.shape() == destination.shape(), "Metal copy contract violation");
    if (source.device().is_host() || destination.device().is_host()) {
        if (!source.device().is_host()) (void)impl_->resolve(source);
        if (!destination.device().is_host()) (void)impl_->resolve(destination);
        if (!source.device().is_host()) impl_->synchronize();
        std::memcpy(destination.data(), source.data(), source.bytes());
        if (mode == CopyMode::Asynchronous) {
            TransferFence fence = create_fence(); record_fence(fence); return fence;
        }
        return {};
    }
    @autoreleasepool {
        BufferRef src = impl_->resolve(source), dst = impl_->resolve(destination);
        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        require(encoder != nil, "Metal blit encoder creation failed");
        [encoder copyFromBuffer:src.buffer sourceOffset:src.offset toBuffer:dst.buffer
              destinationOffset:dst.offset size:source.bytes()];
        [encoder endEncoding];
        impl_->submit(command, "copy");
    }
    if (mode == CopyMode::Synchronous) { impl_->synchronize(); return {}; }
    TransferFence fence = create_fence(); record_fence(fence); return fence;
}

TransferFence MetalBackend::create_fence(int device) {
    require(device == 0, "Metal fence device index out of range");
    return impl_->fence_tracker.create();
}
void MetalBackend::record_fence(TransferFence fence) {
    impl_->fence_tracker.record(fence);
    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    require(command != nil, "Metal fence command creation failed");
    [command commit]; [command retain];
    require(impl_->fences.emplace(fence.id, command).second, "duplicate Metal fence");
}
void MetalBackend::wait_fence(TransferFence fence) {
    impl_->fence_tracker.wait(fence);
    auto found = impl_->fences.find(fence.id);
    require(found != impl_->fences.end(), "destroyed or stale Metal fence");
    impl_->complete(found->second);
}
bool MetalBackend::query_fence(TransferFence fence) {
    if (!impl_->fence_tracker.query(fence)) return false;
    auto found = impl_->fences.find(fence.id);
    require(found != impl_->fences.end(), "destroyed or stale Metal fence");
    const auto status = found->second.status;
    if (status == MTLCommandBufferStatusError) impl_->complete(found->second);
    return status == MTLCommandBufferStatusCompleted;
}
void MetalBackend::destroy_fence(TransferFence fence) {
    impl_->fence_tracker.destroy(fence);
    auto found = impl_->fences.find(fence.id);
    require(found != impl_->fences.end(), "destroyed or stale Metal fence");
    [found->second release]; impl_->fences.erase(found);
}

Tensor MetalBackend::copy_to_device(const Tensor& input, DType dtype) {
    require(input.defined() && supports(dtype), "Unsupported Metal tensor upload");
    if (!input.device().is_host()) {
        (void)impl_->resolve(input);
        if (input.dtype() == dtype) return input;
    }
    require(!input.is_quantized(), "Phase 12B Metal quantized compute is not implemented");
    const bool cacheable = input.device().is_host() && !input.owns_storage();
    const auto key = std::make_pair(input.data(), dtype);
    if (impl_->trace_recording && input.device().is_host())
        impl_->recorded_trace.push_back({input, dtype, input.bytes(),
                                        static_cast<size_t>(input.numel()) * dtype_size(dtype),
                                        input.is_quantized()});
    if (cacheable && impl_->weight_cache_enabled) {
        auto found = impl_->weight_cache.find(key);
        if (found != impl_->weight_cache.end()) {
            ++impl_->cache_hits; ++impl_->memory_stats.cache_hits; return found->second;
        }
        ++impl_->cache_misses; ++impl_->memory_stats.cache_misses;
    }
    const auto started = std::chrono::steady_clock::now();
    Tensor output = impl_->allocate(input.shape(), dtype);
    if (input.dtype() == dtype && input.device().is_host()) {
        std::memcpy(output.data(), input.data(), input.bytes());
    } else {
        Tensor source = input;
        if (input.device().is_host()) {
            source = impl_->allocate(input.shape(), input.dtype());
            std::memcpy(source.data(), input.data(), input.bytes());
        }
        CastParams params{static_cast<uint64_t>(input.numel()), dtype_code(input.dtype()), dtype_code(dtype)};
        impl_->dispatch("cast_kernel", params.count,
                        {{impl_->resolve(source), 0}, {impl_->resolve(output), 1}},
                        2, &params, sizeof(params), "cast");
    }
    if (input.device().is_host()) {
        impl_->upload_bytes += input.bytes();
        impl_->upload_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        ++impl_->memory_stats.upload_copies;
        impl_->memory_stats.upload_bytes += input.bytes();
    }
    if (cacheable && impl_->weight_cache_enabled &&
        output.bytes() <= impl_->weight_cache_capacity - std::min(impl_->weight_cache_bytes, impl_->weight_cache_capacity)) {
        impl_->weight_cache.emplace(key, output);
        impl_->weight_cache_bytes += output.bytes();
    }
    return output;
}

Tensor MetalBackend::copy_to_host(const Tensor& input) {
    require(input.defined() && !input.device().is_host(), "copy_to_host expects Metal tensor");
    (void)impl_->resolve(input); impl_->synchronize();
    Tensor output = Tensor::host(input.shape(), input.dtype());
    std::memcpy(output.data(), input.data(), input.bytes());
    return output;
}

void MetalBackend::synchronize() { impl_->synchronize(); }
size_t MetalBackend::peak_device_bytes() const { return impl_->allocation_tracker->peak.load(); }
size_t MetalBackend::weight_upload_bytes() const { return impl_->upload_bytes; }
double MetalBackend::weight_upload_seconds() const { return impl_->upload_seconds; }
void MetalBackend::enable_profiling(bool enabled) { impl_->profiling = enabled; }
bool MetalBackend::profiling_enabled() const { return impl_->profiling; }

std::map<std::string, ProfileStat> MetalBackend::profile_stats() {
    impl_->synchronize();
    auto result = impl_->region_profiles;
    for (auto& [name, command] : impl_->profiled) {
        impl_->complete(command);
        const double start = command.GPUStartTime, end = command.GPUEndTime;
        auto& stat = result[name]; ++stat.calls;
        if (end >= start) stat.device_milliseconds += (end - start) * 1000.0;
        [command release];
    }
    impl_->profiled.clear();
    return result;
}

void MetalBackend::profile_region_begin(const std::string& name) {
    if (!impl_->profiling) return;
    impl_->synchronize();
    impl_->active_profiles.emplace_back(name, std::chrono::steady_clock::now());
}
void MetalBackend::profile_region_end() {
    if (!impl_->profiling) return;
    require(!impl_->active_profiles.empty(), "Metal profile region stack underflow");
    impl_->synchronize();
    auto active = impl_->active_profiles.back(); impl_->active_profiles.pop_back();
    auto& stat = impl_->region_profiles[active.first]; ++stat.calls;
    stat.device_milliseconds += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - active.second).count();
}

uint64_t MetalBackend::weight_cache_hits() const { return impl_->cache_hits; }
uint64_t MetalBackend::weight_cache_misses() const { return impl_->cache_misses; }
size_t MetalBackend::weight_cache_resident_bytes() const { return impl_->weight_cache_bytes; }
size_t MetalBackend::weight_cache_capacity_bytes() const { return impl_->weight_cache_capacity; }
void MetalBackend::enable_weight_cache(bool enabled) {
    require(impl_->weight_cache.empty(), "Weight cache mode must be selected before execution");
    impl_->weight_cache_enabled = enabled;
}
void MetalBackend::enable_true_quant_compute(bool enabled) { impl_->true_quant_compute = enabled; }
bool MetalBackend::true_quant_compute_enabled() const { return impl_->true_quant_compute; }
QuantComputeStats MetalBackend::quant_compute_stats() const { return impl_->quant_stats; }
void MetalBackend::reset_quant_compute_stats() { impl_->quant_stats = {}; }

void MetalBackend::configure_memory_runtime(const MemoryBudget& budget,
                                            const MemoryRuntimeOptions& options) {
    budget.validate();
    impl_->memory_budget = budget; impl_->memory_options = options;
    impl_->memory_stats = {};
    impl_->memory_stats.accounting.device_workspace_bytes = budget.reserved_device_workspace_bytes;
    if (options.enabled) impl_->weight_cache_capacity = budget.device_weight_budget_bytes();
}
bool MetalBackend::memory_runtime_enabled() const { return impl_->memory_options.enabled; }
void MetalBackend::set_vrm_mapped_bytes(size_t bytes) {
    impl_->memory_stats.accounting.vrm_mapped_bytes = bytes;
    impl_->memory_stats.accounting.host_pageable_bytes = bytes;
    impl_->memory_stats.accounting.peak_host_total_bytes = bytes;
}
void MetalBackend::begin_memory_trace() { impl_->recorded_trace.clear(); impl_->trace_recording = true; }
std::vector<MemoryAccess> MetalBackend::end_memory_trace() { impl_->trace_recording = false; return impl_->recorded_trace; }
void MetalBackend::set_memory_trace(std::vector<MemoryAccess> trace) { impl_->execution_trace = std::move(trace); }
MemoryRuntimeStats MetalBackend::memory_runtime_stats() const {
    MemoryRuntimeStats result = impl_->memory_stats;
    result.accounting.device_resident_weight_bytes = impl_->weight_cache_bytes;
    result.accounting.peak_device_resident_weight_bytes = impl_->weight_cache_bytes;
    result.accounting.peak_device_bytes = impl_->allocation_tracker->peak.load();
    result.upload_bytes = impl_->upload_bytes;
    result.upload_seconds = impl_->upload_seconds;
    return result;
}

Tensor MetalBackend::linear(const Tensor& x_raw, const Tensor& weight_raw,
                            const Tensor* bias_raw, DType compute,
                            DType requested_output) {
    require(x_raw.ndim() >= 1 && weight_raw.ndim() == 2, "linear shape rank mismatch");
    require(x_raw.dim(-1) == weight_raw.dim(1), "linear K mismatch");
    const DType dtype = compute == DType::F16 ? impl_->execution_dtype : compute;
    require(dtype == DType::F32, "Phase 12B Metal Linear supports FP32 only");
    const DType output_dtype = requested_output == DType::F16
        ? dtype : requested_output;
    require(output_dtype == DType::F32,
            "Phase 12B Metal Linear producer output supports FP32 only");
    require(!weight_raw.is_quantized(), "Phase 12B Metal quantized Linear is unsupported");
    Tensor x = copy_to_device(x_raw, DType::F32), weight = copy_to_device(weight_raw, DType::F32), bias;
    if (bias_raw) bias = copy_to_device(*bias_raw, DType::F32);
    const int64_t k = x.dim(-1), n = weight.dim(0), m = x.numel() / k;
    std::vector<int64_t> shape = x.shape(); shape.back() = n;
    Tensor output = impl_->allocate(shape, DType::F32);
    const bool use_mps = k >= 16 && n >= 16 && (k * sizeof(float)) % 16 == 0 &&
        (n * sizeof(float)) % 16 == 0;
    if (use_mps) {
        @autoreleasepool {
            BufferRef xb=impl_->resolve(x), wb=impl_->resolve(weight), ob=impl_->resolve(output);
            MPSMatrixDescriptor* xd=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* wd=[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k*sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* od=[MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrix* xm=[[MPSMatrix alloc] initWithBuffer:xb.buffer offset:xb.offset descriptor:xd];
            MPSMatrix* wm=[[MPSMatrix alloc] initWithBuffer:wb.buffer offset:wb.offset descriptor:wd];
            MPSMatrix* om=[[MPSMatrix alloc] initWithBuffer:ob.buffer offset:ob.offset descriptor:od];
            MPSMatrixMultiplication* gemm=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
            id<MTLCommandBuffer> command=[impl_->queue commandBuffer];
            [gemm encodeToCommandBuffer:command leftMatrix:xm rightMatrix:wm resultMatrix:om];
            impl_->submit(command,"linear");
            [gemm release];[xm release];[wm release];[om release];
        }
        if (bias_raw) {
            BiasParams params{static_cast<uint64_t>(output.numel()), n};
            impl_->dispatch("bias_f32", params.count,
                {{impl_->resolve(output),0},{impl_->resolve(bias),1}},2,&params,sizeof(params),"linear.bias");
        }
    } else {
        MatmulParams params{static_cast<uint64_t>(output.numel()),m,n,k,bias_raw?1:0};
        impl_->dispatch("matmul_f32",params.count,
            {{impl_->resolve(x),0},{impl_->resolve(weight),1},{bias_raw?impl_->resolve(bias):BufferRef{},2},{impl_->resolve(output),3}},
            4,&params,sizeof(params),"linear");
    }
    return output;
}

Tensor MetalBackend::add(const Tensor& a, const Tensor& b) { return binary(*this, impl_, a, b, 0); }
Tensor MetalBackend::mul(const Tensor& a, const Tensor& b) { return binary(*this, impl_, a, b, 1); }
Tensor MetalBackend::div(const Tensor& a, const Tensor& b) { return binary(*this, impl_, a, b, 2); }
Tensor MetalBackend::maximum(const Tensor& a, const Tensor& b) { return binary(*this, impl_, a, b, 3); }

Tensor MetalBackend::batched_matmul(const Tensor& a_raw, const Tensor& b_raw) {
    require(a_raw.ndim()==3&&b_raw.ndim()==3&&a_raw.dim(0)==b_raw.dim(0)&&a_raw.dim(2)==b_raw.dim(1),"batched_matmul expects [B,M,K] x [B,K,N]");
    Tensor a=copy_to_device(a_raw,DType::F32),b=copy_to_device(b_raw,DType::F32),output=impl_->allocate({a.dim(0),a.dim(1),b.dim(2)},DType::F32);
    MatmulParams params{static_cast<uint64_t>(output.numel()),a.dim(1),b.dim(2),a.dim(2),0};
    impl_->dispatch("batched_matmul_f32",params.count,{{impl_->resolve(a),0},{impl_->resolve(b),1},{impl_->resolve(output),2}},3,&params,sizeof(params),"batched_matmul");return output;
}

Tensor MetalBackend::reshape(const Tensor& raw, const std::vector<int64_t>& shape) {
    Tensor input = raw.device().is_host() ? copy_to_device(raw, execution_dtype()) : raw;
    (void)impl_->resolve(input); return input.reshape(shape);
}

Tensor MetalBackend::permute(const Tensor& raw, const std::vector<int64_t>& dims) {
    require(dims.size() == static_cast<size_t>(raw.ndim()), "permute rank mismatch");
    std::vector<bool> seen(dims.size()); std::vector<int64_t> shape(dims.size()); Meta dimensions{};
    dimensions.rank = raw.ndim();
    for (size_t i=0;i<dims.size();++i){require(dims[i]>=0&&dims[i]<raw.ndim()&&!seen[dims[i]],"invalid permutation");seen[dims[i]]=true;shape[i]=raw.shape()[dims[i]];dimensions.shape[i]=dims[i];}
    Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(shape,DType::F32);
    PermuteParams params{static_cast<uint64_t>(output.numel()),meta(input.shape()),meta(shape),dimensions};
    impl_->dispatch("permute_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"permute");return output;
}

Tensor MetalBackend::concat(const std::vector<Tensor>& inputs, int64_t dim) {
    require(!inputs.empty(),"concat requires tensors");if(dim<0)dim+=inputs[0].ndim();require(dim>=0&&dim<inputs[0].ndim(),"concat dimension invalid");
    std::vector<int64_t> shape=inputs[0].shape();shape[dim]=0;for(const Tensor& input:inputs){require(input.ndim()==inputs[0].ndim(),"concat rank mismatch");for(int d=0;d<input.ndim();++d)if(d!=dim)require(input.dim(d)==inputs[0].dim(d),"concat shape mismatch");shape[dim]+=input.dim(dim);}
    Tensor output=impl_->allocate(shape,DType::F32);int64_t offset=0;
    for(const Tensor& raw:inputs){Tensor input=copy_to_device(raw,DType::F32);ConcatParams params{static_cast<uint64_t>(input.numel()),meta(input.shape()),meta(shape),static_cast<int32_t>(dim),0,offset};impl_->dispatch("concat_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"concat");offset+=input.dim(dim);}return output;
}

Tensor MetalBackend::slice(const Tensor& raw, int64_t dim, int64_t start, int64_t stop) {
    if(dim<0)dim+=raw.ndim();require(dim>=0&&dim<raw.ndim()&&start>=0&&start<=stop&&stop<=raw.dim(dim),"slice bounds invalid");Tensor input=copy_to_device(raw,DType::F32);auto shape=input.shape();shape[dim]=stop-start;Tensor output=impl_->allocate(shape,DType::F32);SliceParams params{static_cast<uint64_t>(output.numel()),meta(input.shape()),meta(shape),static_cast<int32_t>(dim),0,start};impl_->dispatch("slice_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"slice");return output;
}

std::vector<Tensor> MetalBackend::split(const Tensor& x,const std::vector<int64_t>& sections,int64_t dim){if(dim<0)dim+=x.ndim();int64_t total=std::accumulate(sections.begin(),sections.end(),int64_t{0});require(dim>=0&&dim<x.ndim()&&total==x.dim(dim),"split sections mismatch");std::vector<Tensor> result;int64_t start=0;for(int64_t section:sections){result.push_back(slice(x,dim,start,start+section));start+=section;}return result;}

Tensor MetalBackend::cast(const Tensor& input, DType dtype) { return copy_to_device(input,dtype); }

Tensor MetalBackend::indexed_gather(const Tensor& table_raw, const Tensor& indices_raw) {
    require(table_raw.ndim()==2||table_raw.ndim()==3,"indexed_gather table must have rank 2 or 3");
    require(indices_raw.ndim()>=1&&(indices_raw.dtype()==DType::I64||indices_raw.dtype()==DType::I32),"indexed_gather indices must be rank >=1 int64/int32");
    const bool batched=table_raw.ndim()==3;
    if(batched)require(indices_raw.dim(0)==table_raw.dim(0),"indexed_gather batch dimension mismatch");
    const int64_t rows=table_raw.dim(batched?1:0),width=table_raw.dim(-1);
    Tensor host_indices=indices_raw.device().is_host()?indices_raw:copy_to_host(indices_raw);
    for(int64_t i=0;i<host_indices.numel();++i){const int64_t value=host_indices.dtype()==DType::I64?host_indices.data_as<int64_t>()[i]:host_indices.data_as<int32_t>()[i];require(value>=0&&value<rows,"indexed_gather index out of range");}
    Tensor table=copy_to_device(table_raw,DType::F32),indices=copy_to_device(indices_raw,indices_raw.dtype());
    auto shape=indices_raw.shape();shape.push_back(width);Tensor output=impl_->allocate(shape,DType::F32);
    const int64_t per_batch=batched?indices_raw.numel()/indices_raw.dim(0):indices_raw.numel();
    GatherParams params{static_cast<uint64_t>(output.numel()),per_batch,rows,width,batched?1:0,dtype_code(indices_raw.dtype())};
    impl_->dispatch("indexed_gather_f32",params.count,{{impl_->resolve(table),0},{impl_->resolve(indices),1},{impl_->resolve(output),2}},3,&params,sizeof(params),"indexed_gather");return output;
}

Tensor MetalBackend::layer_norm(const Tensor& raw,const Tensor* weight_raw,const Tensor* bias_raw,float eps){Tensor input=copy_to_device(raw,DType::F32),weight,bias,output=impl_->allocate(input.shape(),DType::F32);if(weight_raw)weight=copy_to_device(*weight_raw,DType::F32);if(bias_raw)bias=copy_to_device(*bias_raw,DType::F32);int64_t width=input.dim(-1),rows=input.numel()/width;NormParams params{static_cast<uint64_t>(input.numel()),rows,width,eps,weight_raw?1:0,bias_raw?1:0,-1};impl_->dispatch("layer_norm_f32",rows,{{impl_->resolve(input),0},{weight_raw?impl_->resolve(weight):BufferRef{},1},{bias_raw?impl_->resolve(bias):BufferRef{},2},{impl_->resolve(output),3}},4,&params,sizeof(params),"norm.layer");return output;}

Tensor MetalBackend::rms_norm(const Tensor& raw,const Tensor* weight_raw,float eps,int64_t axis,DType requested){Tensor input=copy_to_device(raw,DType::F32);if(axis<0)axis+=input.ndim();require(axis>=0&&axis<input.ndim(),"rms norm axis invalid");DType output_dtype=requested==DType::F16?DType::F32:requested;require(output_dtype==DType::F32,"Phase 12B RMSNorm output supports FP32 only");Tensor weight,output=impl_->allocate(input.shape(),DType::F32);if(weight_raw)weight=copy_to_device(*weight_raw,DType::F32);if(axis==input.ndim()-1){int64_t width=input.dim(-1),rows=input.numel()/width;NormParams params{static_cast<uint64_t>(input.numel()),rows,width,eps,weight_raw?1:0,0,static_cast<int32_t>(axis)};impl_->dispatch("rms_norm_f32",rows,{{impl_->resolve(input),0},{weight_raw?impl_->resolve(weight):BufferRef{},1},{impl_->resolve(output),2}},3,&params,sizeof(params),"norm.rms");}else{AxisNormParams params{static_cast<uint64_t>(input.numel()),meta(input.shape()),static_cast<int32_t>(axis),eps,weight_raw?1:0};impl_->dispatch("rms_norm_axis_f32",params.count,{{impl_->resolve(input),0},{weight_raw?impl_->resolve(weight):BufferRef{},1},{impl_->resolve(output),2}},3,&params,sizeof(params),"norm.rms");}return output;}

Tensor MetalBackend::activation(const Tensor& raw,Activation kind){Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);UnaryParams params{static_cast<uint64_t>(input.numel()),static_cast<int32_t>(kind),0,0};impl_->dispatch("activation_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"activation");return output;}

Tensor MetalBackend::sinusoidal_embedding(const Tensor& raw,int64_t width,bool flip,double downscale,bool require_f64){require(raw.ndim()==1&&width>0,"sinusoidal embedding contract violation");Tensor positions=copy_to_device(raw,DType::F32),inverse=impl_->allocate({width/2},DType::F32),output=impl_->allocate({raw.numel(),width},DType::F32);float* values=inverse.data_as<float>();for(int64_t i=0;i<width/2;++i){double exponent=-std::log(10000.0)*i/(width/2-downscale);values[i]=static_cast<float>(std::exp(exponent));}(void)require_f64;SinParams params{static_cast<uint64_t>(raw.numel()),static_cast<int32_t>(width),flip?1:0};impl_->dispatch("sinusoidal_f32",static_cast<uint64_t>(output.numel()),{{impl_->resolve(positions),0},{impl_->resolve(inverse),1},{impl_->resolve(output),2}},3,&params,sizeof(params),"sinusoidal_embedding");return output;}

Tensor MetalBackend::rope_nd(const Tensor& raw,const Tensor& cos_raw,const Tensor& sin_raw){require(raw.dim(-1)%2==0,"RoPE width must be even");Tensor input=copy_to_device(raw,DType::F32),cosine=copy_to_device(cos_raw,DType::F32),sine=copy_to_device(sin_raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);require(cosine.shape()==sine.shape(),"RoPE frequency mismatch");(void)broadcast_shape(input,cosine);RopeParams params{static_cast<uint64_t>(input.numel()),meta(input.shape()),meta(cosine.shape())};impl_->dispatch("rope_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(cosine),1},{impl_->resolve(sine),2},{impl_->resolve(output),3}},4,&params,sizeof(params),"rope");return output;}

Tensor MetalBackend::attention(const Tensor& q_raw,const Tensor& k_raw,const Tensor& v_raw,const Tensor* mask_raw,bool causal,float scale,const Tensor* bias_raw,AttentionObservation* observation){require(q_raw.ndim()==4&&k_raw.ndim()==4&&v_raw.ndim()==4,"attention requires BSHD");require(q_raw.dim(0)==k_raw.dim(0)&&k_raw.shape()==v_raw.shape()&&q_raw.dim(2)==k_raw.dim(2)&&q_raw.dim(3)==k_raw.dim(3),"attention shape mismatch");Tensor q=copy_to_device(q_raw,DType::F32),k=copy_to_device(k_raw,DType::F32),v=copy_to_device(v_raw,DType::F32),mask,bias,output=impl_->allocate(q.shape(),DType::F32);if(mask_raw)mask=copy_to_device(*mask_raw,DType::Bool);if(bias_raw){bias=copy_to_device(*bias_raw,DType::F32);const std::vector<int64_t> logical={q.dim(0),q.dim(2),q.dim(1),k.dim(1)};require(bias.ndim()<=4,"attention additive bias rank exceeds 4");const int shift=4-bias.ndim();for(int i=0;i<bias.ndim();++i)require(bias.dim(i)==1||bias.dim(i)==logical[i+shift],"attention additive bias is not broadcastable to BHQK");}if(scale==0)scale=1/std::sqrt(static_cast<float>(q.dim(3)));Tensor observed_score,observed_biased,observed_softmax;if(observation){const std::vector<int64_t> shape={q.dim(0),q.dim(2),q.dim(1),k.dim(1)};observed_score=impl_->allocate(shape,DType::F32);observed_biased=impl_->allocate(shape,DType::F32);observed_softmax=impl_->allocate(shape,DType::F32);}AttentionParams params{static_cast<int32_t>(q.dim(0)),static_cast<int32_t>(q.dim(1)),static_cast<int32_t>(k.dim(1)),static_cast<int32_t>(q.dim(2)),static_cast<int32_t>(q.dim(3)),causal?1:0,scale,mask_raw?1:0,bias_raw?1:0,observation?1:0,mask_raw?meta(mask.shape()):Meta{},bias_raw?meta(bias.shape()):Meta{}};uint64_t units=q.dim(0)*q.dim(1)*q.dim(2);impl_->dispatch("attention_f32",units,{{impl_->resolve(q),0},{impl_->resolve(k),1},{impl_->resolve(v),2},{mask_raw?impl_->resolve(mask):BufferRef{},3},{bias_raw?impl_->resolve(bias):BufferRef{},4},{impl_->resolve(output),5},{observation?impl_->resolve(observed_score):BufferRef{},6},{observation?impl_->resolve(observed_biased):BufferRef{},7},{observation?impl_->resolve(observed_softmax):BufferRef{},8}},9,&params,sizeof(params),"attention");if(observation){observation->score=observed_score;observation->biased_score=observed_biased;observation->softmax=observed_softmax;}return output;}

namespace {
template<int Spatial> Tensor convolution(MetalBackend& backend,MetalBackend::Impl* impl,const Tensor& x_raw,const Tensor& weight_raw,const Tensor* bias_raw,const std::vector<int>& stride,const std::vector<int>& padding,const std::vector<int>& dilation,int groups){require(x_raw.ndim()==Spatial+2&&weight_raw.ndim()==Spatial+2,"convolution rank mismatch");require(stride.size()==Spatial&&padding.size()==Spatial&&dilation.size()==Spatial,"convolution parameter rank mismatch");Tensor input=backend.copy_to_device(x_raw,DType::F32),weight=backend.copy_to_device(weight_raw,DType::F32),bias;require(input.dim(1)==weight.dim(1)*groups,"convolution channel mismatch");if(bias_raw)bias=backend.copy_to_device(*bias_raw,DType::F32);auto shape=input.shape();shape[1]=weight.dim(0);for(int d=0;d<Spatial;++d){shape[d+2]=(input.shape()[d+2]+2*padding[d]-dilation[d]*(weight.shape()[d+2]-1)-1)/stride[d]+1;require(shape[d+2]>0&&shape[d+2]<=INT_MAX,"Invalid convolution output shape");}Tensor output=impl->allocate(shape,DType::F32);ConvParams params{};params.count=output.numel();params.spatial=Spatial;params.batch=input.dim(0);params.in_channels=input.dim(1);params.out_channels=weight.dim(0);params.groups=groups;for(int d=0;d<3;++d){params.in_size[d]=params.out_size[d]=params.kernel[d]=params.stride[d]=params.dilation[d]=1;params.padding[d]=0;}for(int d=0;d<Spatial;++d){params.in_size[d]=input.dim(d+2);params.out_size[d]=shape[d+2];params.kernel[d]=weight.dim(d+2);params.stride[d]=stride[d];params.padding[d]=padding[d];params.dilation[d]=dilation[d];}params.has_bias=bias_raw?1:0;impl->dispatch("convolution_f32",params.count,{{impl->resolve(input),0},{impl->resolve(weight),1},{bias_raw?impl->resolve(bias):BufferRef{},2},{impl->resolve(output),3}},4,&params,sizeof(params),Spatial==2?"conv2d":"conv3d");return output;}
}
Tensor MetalBackend::conv3d(const Tensor& x,const Tensor& weight,const Tensor* bias,const std::vector<int>& stride,const std::vector<int>& padding,const std::vector<int>& dilation,int groups){return convolution<3>(*this,impl_,x,weight,bias,stride,padding,dilation,groups);}
Tensor MetalBackend::conv2d(const Tensor& x,const Tensor& weight,const Tensor* bias,const std::vector<int>& stride,const std::vector<int>& padding,const std::vector<int>& dilation,int groups){return convolution<2>(*this,impl_,x,weight,bias,stride,padding,dilation,groups);}
Tensor MetalBackend::conv_transpose2d(const Tensor& raw,const Tensor& raw_weight,const Tensor* raw_bias,const std::vector<int>& stride,const std::vector<int>& padding,const std::vector<int>& output_padding,const std::vector<int>& dilation,int groups){require(raw.ndim()==4&&raw_weight.ndim()==4&&stride.size()==2&&padding.size()==2&&output_padding.size()==2&&dilation.size()==2&&groups>0&&raw.dim(1)==raw_weight.dim(0)&&raw.dim(1)%groups==0&&stride[0]>0&&stride[1]>0&&padding[0]>=0&&padding[1]>=0&&output_padding[0]>=0&&output_padding[1]>=0&&output_padding[0]<stride[0]&&output_padding[1]<stride[1]&&dilation[0]>0&&dilation[1]>0,"conv_transpose2d contract violation");int64_t output_channels=raw_weight.dim(1)*groups;if(raw_bias)require(raw_bias->numel()==output_channels,"conv_transpose2d bias mismatch");int64_t oh=(raw.dim(2)-1)*stride[0]-2*padding[0]+dilation[0]*(raw_weight.dim(2)-1)+output_padding[0]+1,ow=(raw.dim(3)-1)*stride[1]-2*padding[1]+dilation[1]*(raw_weight.dim(3)-1)+output_padding[1]+1;require(oh>0&&ow>0,"conv_transpose2d output shape invalid");Tensor input=copy_to_device(raw,DType::F32),weight=copy_to_device(raw_weight,DType::F32),bias;if(raw_bias)bias=copy_to_device(*raw_bias,DType::F32);Tensor output=impl_->allocate({raw.dim(0),output_channels,oh,ow},DType::F32);ConvTranspose2DParams params{static_cast<uint64_t>(output.numel()),static_cast<int32_t>(raw.dim(0)),static_cast<int32_t>(raw.dim(1)),static_cast<int32_t>(output_channels),groups,static_cast<int32_t>(raw.dim(2)),static_cast<int32_t>(raw.dim(3)),static_cast<int32_t>(oh),static_cast<int32_t>(ow),static_cast<int32_t>(raw_weight.dim(2)),static_cast<int32_t>(raw_weight.dim(3)),stride[0],stride[1],padding[0],padding[1],dilation[0],dilation[1],raw_bias?1:0};impl_->dispatch("conv_transpose2d_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(weight),1},{raw_bias?impl_->resolve(bias):BufferRef{},2},{impl_->resolve(output),3}},4,&params,sizeof(params),"conv_transpose2d");return output;}
Tensor MetalBackend::max_pool2d(const Tensor& raw,const std::vector<int>& kernel,const std::vector<int>& stride,const std::vector<int>& padding){require(raw.ndim()==4&&kernel.size()==2&&stride.size()==2&&padding.size()==2&&kernel[0]>0&&kernel[1]>0&&stride[0]>0&&stride[1]>0&&padding[0]>=0&&padding[1]>=0,"max_pool2d contract violation");Tensor input=copy_to_device(raw,DType::F32);int64_t oh=(input.dim(2)+2*padding[0]-kernel[0])/stride[0]+1,ow=(input.dim(3)+2*padding[1]-kernel[1])/stride[1]+1;require(oh>0&&ow>0,"max_pool2d output shape invalid");Tensor output=impl_->allocate({input.dim(0),input.dim(1),oh,ow},DType::F32);Pool2DParams params{static_cast<uint64_t>(output.numel()),static_cast<int32_t>(input.dim(1)),static_cast<int32_t>(input.dim(2)),static_cast<int32_t>(input.dim(3)),static_cast<int32_t>(oh),static_cast<int32_t>(ow),kernel[0],kernel[1],stride[0],stride[1],padding[0],padding[1]};impl_->dispatch("max_pool2d_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"max_pool2d");return output;}

Tensor MetalBackend::pad(const Tensor& raw,const std::vector<int64_t>& padding,float value,PadMode mode){require(padding.size()%2==0&&padding.size()<=static_cast<size_t>(2*raw.ndim()),"Invalid pad vector");Tensor input=copy_to_device(raw,DType::F32);auto shape=input.shape();Meta before{};before.rank=input.ndim();for(size_t pair=0;pair<padding.size()/2;++pair){int dim=input.ndim()-1-pair;require(padding[2*pair]>=0&&padding[2*pair+1]>=0,"Negative padding unsupported");before.shape[dim]=padding[2*pair];shape[dim]+=padding[2*pair]+padding[2*pair+1];}Tensor output=impl_->allocate(shape,DType::F32);PadParams params{static_cast<uint64_t>(output.numel()),meta(input.shape()),meta(shape),before,value,mode==PadMode::Replicate?1:0};impl_->dispatch("pad_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"pad");return output;}

Tensor MetalBackend::reduce_sum(const Tensor& raw,int64_t dim,bool keepdim){if(dim<0)dim+=raw.ndim();require(dim>=0&&dim<raw.ndim(),"reduce dimension invalid");Tensor input=copy_to_device(raw,DType::F32);auto shape=input.shape();if(keepdim)shape[dim]=1;else shape.erase(shape.begin()+dim);Tensor output=impl_->allocate(shape,DType::F32);ReduceParams params{static_cast<uint64_t>(output.numel()),meta(input.shape()),meta(shape),static_cast<int32_t>(dim)};impl_->dispatch("reduce_sum_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"reduce_sum");return output;}

Tensor MetalBackend::softmax(const Tensor& raw,int64_t axis){if(axis<0)axis+=raw.ndim();require(axis>=0&&axis<raw.ndim(),"softmax axis invalid");Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);int64_t inner=1;for(int dimension=static_cast<int>(axis)+1;dimension<input.ndim();++dimension)inner*=input.dim(dimension);SoftmaxParams params{static_cast<uint64_t>(input.numel()/input.dim(axis)),input.dim(axis),inner};impl_->dispatch("softmax_axis_f32",params.rows,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"softmax");return output;}

Tensor MetalBackend::group_norm(const Tensor& raw,int groups,const Tensor* weight_raw,const Tensor* bias_raw,float eps){require(raw.ndim()>=3&&raw.dim(1)%groups==0,"group norm shape/groups mismatch");Tensor input=copy_to_device(raw,DType::F32),weight,bias,output=impl_->allocate(input.shape(),DType::F32);if(weight_raw)weight=copy_to_device(*weight_raw,DType::F32);if(bias_raw)bias=copy_to_device(*bias_raw,DType::F32);int64_t spatial=input.numel()/(input.dim(0)*input.dim(1));GroupNormParams params{static_cast<int32_t>(input.dim(0)),static_cast<int32_t>(input.dim(1)),spatial,groups,eps,weight_raw?1:0,bias_raw?1:0};uint64_t units=input.dim(0)*groups;impl_->dispatch("group_norm_f32",units,{{impl_->resolve(input),0},{weight_raw?impl_->resolve(weight):BufferRef{},1},{bias_raw?impl_->resolve(bias):BufferRef{},2},{impl_->resolve(output),3}},4,&params,sizeof(params),"norm.group");return output;}

Tensor MetalBackend::interpolate_nearest(const Tensor& raw,const std::vector<double>& factors){require(factors.size()<=static_cast<size_t>(raw.ndim()-2),"interpolate factor rank mismatch");Tensor input=copy_to_device(raw,DType::F32);auto shape=input.shape();int start=raw.ndim()-factors.size();for(size_t i=0;i<factors.size();++i){require(factors[i]>0,"Invalid interpolation factor");shape[start+i]=static_cast<int64_t>(std::floor(shape[start+i]*factors[i]));}Tensor output=impl_->allocate(shape,DType::F32);MetaCountParams params{static_cast<uint64_t>(output.numel()),meta(input.shape()),meta(shape),0,0};impl_->dispatch("interpolate_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"interpolate_nearest");return output;}

Tensor MetalBackend::interpolate_bilinear_2d(const Tensor& raw,int64_t oh,int64_t ow,bool align_corners){require(raw.ndim()==4&&oh>0&&ow>0,"bilinear resize requires NCHW and positive output size");Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate({raw.dim(0),raw.dim(1),oh,ow},DType::F32);Bilinear2DParams params{static_cast<uint64_t>(output.numel()),static_cast<int32_t>(raw.dim(1)),static_cast<int32_t>(raw.dim(2)),static_cast<int32_t>(raw.dim(3)),static_cast<int32_t>(oh),static_cast<int32_t>(ow),align_corners?1:0};impl_->dispatch("interpolate_bilinear_2d_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"interpolate_bilinear_2d");return output;}

Tensor MetalBackend::pixel_norm(const Tensor& raw,int64_t dim,float eps){if(dim<0)dim+=raw.ndim();require(dim>=0&&dim<raw.ndim(),"pixel norm dim invalid");Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);MetaCountParams params{static_cast<uint64_t>(input.numel()),meta(input.shape()),Meta{},static_cast<int32_t>(dim),eps};impl_->dispatch("pixel_norm_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"norm.pixel");return output;}
Tensor MetalBackend::l2_normalize(const Tensor& raw,int64_t dim,float eps){if(dim<0)dim+=raw.ndim();require(dim>=0&&dim<raw.ndim()&&eps>=0,"l2 normalize contract violation");Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);MetaCountParams params{static_cast<uint64_t>(input.numel()),meta(input.shape()),Meta{},static_cast<int32_t>(dim),eps};impl_->dispatch("l2_normalize_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"norm.l2");return output;}

Tensor MetalBackend::pixel_shuffle_nd(const Tensor& raw,const std::vector<int64_t>& factors){require(raw.ndim()==5&&factors.size()==3,"pixel_shuffle_nd requires 5D and 3 factors");int64_t product=factors[0]*factors[1]*factors[2];require(raw.dim(1)%product==0,"pixel shuffle channel mismatch");Tensor input=copy_to_device(raw,DType::F32);std::vector<int64_t> shape={raw.dim(0),raw.dim(1)/product,raw.dim(2)*factors[0],raw.dim(3)*factors[1],raw.dim(4)*factors[2]};Tensor output=impl_->allocate(shape,DType::F32);ShuffleParams params{static_cast<uint64_t>(output.numel()),static_cast<int32_t>(shape[0]),static_cast<int32_t>(shape[1]),static_cast<int32_t>(raw.dim(2)),static_cast<int32_t>(raw.dim(3)),static_cast<int32_t>(raw.dim(4)),static_cast<int32_t>(factors[0]),static_cast<int32_t>(factors[1]),static_cast<int32_t>(factors[2])};impl_->dispatch("pixel_shuffle_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"pixel_shuffle");return output;}

Tensor MetalBackend::rng_normal(RngState& state,const std::vector<int64_t>& shape,DType dtype){require(state.algorithm_id=="pytorch_compat.v1","Unsupported RNG algorithm");require(dtype==DType::F32,"Phase 12B Metal RNG supports FP32 only");Tensor output=impl_->allocate(shape,DType::F32);RngParams params{static_cast<uint64_t>(output.numel()),state.seed,state.offset};impl_->dispatch("rng_normal_f32",params.count,{{impl_->resolve(output),0}},1,&params,sizeof(params),"rng_normal");state.offset+=4;return output;}

Tensor MetalBackend::clamp(const Tensor& raw,float minimum,float maximum){require(minimum<=maximum,"Invalid clamp bounds");Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);UnaryParams params{static_cast<uint64_t>(input.numel()),0,minimum,maximum};impl_->dispatch("clamp_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"clamp");return output;}

Tensor MetalBackend::exp(const Tensor& raw){Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);UnaryParams params{static_cast<uint64_t>(input.numel()),0,0,0};impl_->dispatch("exp_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"exp");return output;}
Tensor MetalBackend::sqrt(const Tensor& raw){Tensor input=copy_to_device(raw,DType::F32),output=impl_->allocate(input.shape(),DType::F32);UnaryParams params{static_cast<uint64_t>(input.numel()),0,0,0};impl_->dispatch("sqrt_f32",params.count,{{impl_->resolve(input),0},{impl_->resolve(output),1}},2,&params,sizeof(params),"sqrt");return output;}

}  // namespace vrhino
