#include "vrhino/backend/cudnn_conv.h"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <chrono>
#include <climits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace vrhino {
namespace {
namespace fe = cudnn_frontend;

constexpr int64_t kXUid = 1;
constexpr int64_t kWUid = 2;
constexpr int64_t kYUid = 3;
constexpr size_t kMaximumCachedPlans = 128;

using PlanKey = std::tuple<
    int, int, DType, DType, int,
    std::vector<int64_t>, std::vector<int64_t>,
    std::vector<int64_t>, std::vector<int64_t>,
    std::vector<int64_t>, std::vector<int64_t>,
    std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
    int64_t, int>;

PlanKey plan_key(const CudnnConvDescriptor& descriptor) {
    return {descriptor.device, descriptor.compute_major,
            descriptor.io_dtype, descriptor.compute_dtype,
            descriptor.spatial_rank,
            descriptor.x_dimensions, descriptor.x_strides,
            descriptor.w_dimensions, descriptor.w_strides,
            descriptor.y_dimensions, descriptor.y_strides,
            descriptor.padding, descriptor.convolution_strides,
            descriptor.dilation, descriptor.groups,
            static_cast<int>(descriptor.mode)};
}

bool fail(std::string* reason, std::string value) {
    if (reason) *reason = std::move(value);
    return false;
}

bool tensor_descriptor_valid(const std::vector<int64_t>& dimensions,
                             const std::vector<int64_t>& strides,
                             int expected_rank, bool legacy,
                             std::string* reason, const char* name) {
    if (static_cast<int>(dimensions.size()) != expected_rank ||
        dimensions.size() != strides.size())
        return fail(reason, std::string(name) + " rank mismatch");
#if defined(_MSC_VER)
    const uint64_t limit = legacy ? INT_MAX : INT64_MAX;
    uint64_t extent = 1;
    bool extent_overflow = false;
#else
    __int128 extent = 1;
#endif
    for (int index = 0; index < expected_rank; ++index) {
        const int64_t dimension = dimensions[static_cast<size_t>(index)];
        const int64_t stride = strides[static_cast<size_t>(index)];
        if (dimension <= 0)
            return fail(reason, std::string(name) + " has non-positive dimension");
        if (stride <= 0)
            return fail(reason, std::string(name) + " has non-positive stride");
        if (legacy && dimension > INT_MAX)
            return fail(reason, std::string(name) + " dimension exceeds legacy INT_MAX");
        if (legacy && stride > INT_MAX)
            return fail(reason, std::string(name) + " stride exceeds legacy INT_MAX");
#if defined(_MSC_VER)
        // Every term is nonnegative. Defer rejection until after the loop to
        // retain the original dimension/stride diagnostic precedence.
        if (!extent_overflow) {
            const auto count = static_cast<uint64_t>(dimension - 1);
            const auto step = static_cast<uint64_t>(stride);
            extent_overflow = count > (limit - extent) / step;
            if (!extent_overflow) extent += count * step;
        }
#else
        extent += static_cast<__int128>(dimension - 1) * stride;
#endif
    }
#if defined(_MSC_VER)
    if (extent_overflow)
#else
    const __int128 limit = legacy ? static_cast<__int128>(INT_MAX)
                                  : static_cast<__int128>(INT64_MAX);
    if (extent > limit)
#endif
        return fail(reason, std::string(name) +
            (legacy ? " addressable extent exceeds legacy INT_MAX"
                    : " addressable extent exceeds INT64_MAX"));
    return true;
}

bool common_semantic_valid(const CudnnConvDescriptor& descriptor,
                           bool legacy, std::string* reason) {
    if (descriptor.spatial_rank != 2 && descriptor.spatial_rank != 3)
        return fail(reason, "only Conv2D and Conv3D are supported");
    const int rank = descriptor.spatial_rank + 2;
    if (!tensor_descriptor_valid(descriptor.x_dimensions, descriptor.x_strides,
                                 rank, legacy, reason, "X") ||
        !tensor_descriptor_valid(descriptor.w_dimensions, descriptor.w_strides,
                                 rank, legacy, reason, "W") ||
        !tensor_descriptor_valid(descriptor.y_dimensions, descriptor.y_strides,
                                 rank, legacy, reason, "Y"))
        return false;
    if (descriptor.padding.size() != static_cast<size_t>(descriptor.spatial_rank) ||
        descriptor.convolution_strides.size() != static_cast<size_t>(descriptor.spatial_rank) ||
        descriptor.dilation.size() != static_cast<size_t>(descriptor.spatial_rank))
        return fail(reason, "convolution parameter rank mismatch");
    if (descriptor.groups <= 0)
        return fail(reason, "group count must be positive");
    if (descriptor.x_dimensions[0] != descriptor.y_dimensions[0] ||
        descriptor.y_dimensions[1] != descriptor.w_dimensions[0] ||
        descriptor.x_dimensions[1] != descriptor.w_dimensions[1] * descriptor.groups)
        return fail(reason, "batch/channel/group contract mismatch");
    if (descriptor.mode != CUDNN_CROSS_CORRELATION)
        return fail(reason, "unsupported convolution mode");
    for (int spatial = 0; spatial < descriptor.spatial_rank; ++spatial) {
        const int64_t pad = descriptor.padding[static_cast<size_t>(spatial)];
        const int64_t stride = descriptor.convolution_strides[static_cast<size_t>(spatial)];
        const int64_t dilation = descriptor.dilation[static_cast<size_t>(spatial)];
        if (pad < 0 || stride <= 0 || dilation <= 0)
            return fail(reason, "invalid padding, stride, or dilation");
#if defined(_MSC_VER)
        // MSVC x64 has no __int128. Retain the wide integer calculation with
        // native carry/multiply/divide intrinsics; never narrow an extent.
        uint64_t available_low = 0;
        const uint64_t available_high = _addcarry_u64(
            0, static_cast<uint64_t>(descriptor.x_dimensions[spatial + 2] - 1),
            2 * static_cast<uint64_t>(pad), &available_low);
        uint64_t kernel_high = 0;
        const uint64_t kernel_low = _umul128(static_cast<uint64_t>(dilation),
            static_cast<uint64_t>(descriptor.w_dimensions[spatial + 2] - 1),
            &kernel_high);
        if (kernel_high > available_high ||
            (kernel_high == available_high && kernel_low > available_low))
            return fail(reason, "convolution has non-positive output extent");
        uint64_t numerator_low = 0;
        const auto borrow = _subborrow_u64(0, available_low, kernel_low, &numerator_low);
        const uint64_t numerator_high = available_high - kernel_high - borrow;
        const auto divisor = static_cast<uint64_t>(stride);
        uint64_t remainder = 0;
        if (numerator_high >= divisor ||
            _udiv128(numerator_high, numerator_low, divisor, &remainder) !=
                static_cast<uint64_t>(descriptor.y_dimensions[spatial + 2] - 1))
#else
        const __int128 numerator =
            static_cast<__int128>(descriptor.x_dimensions[spatial + 2]) + 2 * pad -
            static_cast<__int128>(dilation) *
                (descriptor.w_dimensions[spatial + 2] - 1) - 1;
        if (numerator < 0)
            return fail(reason, "convolution has non-positive output extent");
        const __int128 expected = numerator / stride + 1;
        if (expected != descriptor.y_dimensions[spatial + 2])
#endif
            return fail(reason, "output shape does not match convolution semantic");
    }
    return true;
}

fe::DataType_t frontend_dtype(DType dtype) {
    return dtype == DType::BF16 ? fe::DataType_t::BFLOAT16
                                : fe::DataType_t::FLOAT;
}

struct Plan {
    std::shared_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    double build_seconds = 0.0;
    std::string rejection_reason;
    ~Plan() {
        if (workspace) cudaFree(workspace);
    }
};

std::shared_ptr<Plan> build_plan(cudnnHandle_t handle,
                                 const CudnnConvDescriptor& descriptor) {
    auto plan = std::make_shared<Plan>();
    std::string admission_reason;
    if (!cudnn_backend_conv_admitted(descriptor, &admission_reason)) {
        plan->rejection_reason = admission_reason;
        return plan;
    }
    const auto started = std::chrono::steady_clock::now();
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(frontend_dtype(descriptor.io_dtype))
        .set_intermediate_data_type(frontend_dtype(descriptor.compute_dtype))
        .set_compute_data_type(frontend_dtype(descriptor.compute_dtype));
    auto x = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("X").set_uid(kXUid)
        .set_dim(descriptor.x_dimensions).set_stride(descriptor.x_strides));
    auto weight = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("W").set_uid(kWUid)
        .set_dim(descriptor.w_dimensions).set_stride(descriptor.w_strides));
    auto attributes = fe::graph::Conv_fprop_attributes()
        .set_name("vrhino_generic_conv_forward")
        .set_padding(descriptor.padding)
        .set_stride(descriptor.convolution_strides)
        .set_dilation(descriptor.dilation)
        .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);
    auto output = graph->conv_fprop(x, weight, attributes);
    output->set_name("Y").set_uid(kYUid).set_output(true)
        .set_dim(descriptor.y_dimensions).set_stride(descriptor.y_strides)
        .set_data_type(frontend_dtype(descriptor.io_dtype));
    auto status = graph->build(handle, {fe::HeurMode_t::A});
    if (status.is_bad()) {
        plan->rejection_reason = status.get_message();
        plan->build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return plan;
    }
    int64_t workspace_bytes = 0;
    status = graph->get_workspace_size(workspace_bytes);
    if (status.is_bad() || workspace_bytes < 0) {
        plan->rejection_reason = status.is_bad()
            ? status.get_message() : "invalid cuDNN workspace size";
        plan->build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return plan;
    }
    if (workspace_bytes > 0) {
        const cudaError_t allocation = cudaMalloc(&plan->workspace, workspace_bytes);
        if (allocation != cudaSuccess) {
            plan->rejection_reason = std::string("cuDNN workspace allocation failed: ") +
                                     cudaGetErrorString(allocation);
            (void)cudaGetLastError();
            plan->build_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            return plan;
        }
    }
    plan->workspace_bytes = static_cast<size_t>(workspace_bytes);
    plan->graph = std::move(graph);
    plan->build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return plan;
}

}  // namespace

bool cudnn_legacy_conv_admitted(const CudnnConvDescriptor& descriptor,
                                std::string* reason) {
    if (!common_semantic_valid(descriptor, true, reason)) return false;
    if (descriptor.io_dtype != DType::F32 && descriptor.io_dtype != DType::BF16)
        return fail(reason, "legacy Conv dtype must be FP32 or BF16");
    if (descriptor.compute_dtype != DType::F32)
        return fail(reason, "legacy Conv compute dtype must be FP32");
    if (reason) reason->clear();
    return true;
}

bool cudnn_backend_conv_admitted(const CudnnConvDescriptor& descriptor,
                                 std::string* reason) {
    if (!common_semantic_valid(descriptor, false, reason)) return false;
    if (descriptor.io_dtype != DType::F32 && descriptor.io_dtype != DType::BF16)
        return fail(reason, "Backend Conv dtype must be FP32 or BF16");
    if (descriptor.compute_dtype != DType::F32)
        return fail(reason, "Backend Conv compute dtype must be FP32");
    if (descriptor.io_dtype == DType::BF16 && descriptor.compute_major < 8)
        return fail(reason, "Backend BF16 Conv requires compute capability 8.0+");
    if (descriptor.device < 0)
        return fail(reason, "invalid CUDA device");
    if (reason) reason->clear();
    return true;
}

struct CudnnConvPlanCache::Impl {
    explicit Impl(cudnnHandle_t value) : handle(value) {}
    cudnnHandle_t handle{};
    std::map<PlanKey, std::shared_ptr<Plan>> plans;
    uint64_t builds = 0;
    uint64_t hits = 0;
};

CudnnConvPlanCache::CudnnConvPlanCache(cudnnHandle_t handle)
    : impl_(std::make_unique<Impl>(handle)) {}
CudnnConvPlanCache::~CudnnConvPlanCache() = default;

CudnnConvExecution CudnnConvPlanCache::execute(
    const CudnnConvDescriptor& descriptor, const void* x,
    const void* weight, void* output) {
    const PlanKey key = plan_key(descriptor);
    auto found = impl_->plans.find(key);
    bool cache_hit = true;
    if (found == impl_->plans.end()) {
        cache_hit = false;
        if (impl_->plans.size() >= kMaximumCachedPlans)
            return {false, false, 0, 0.0,
                    "cuDNN Conv plan cache capacity reached"};
        found = impl_->plans.emplace(
            key, build_plan(impl_->handle, descriptor)).first;
        ++impl_->builds;
    } else {
        ++impl_->hits;
    }
    const std::shared_ptr<Plan>& plan = found->second;
    if (!plan->graph)
        return {false, cache_hit, plan->workspace_bytes,
                cache_hit ? 0.0 : plan->build_seconds,
                plan->rejection_reason};
    std::unordered_map<int64_t, void*> variant_pack = {
        {kXUid, const_cast<void*>(x)},
        {kWUid, const_cast<void*>(weight)},
        {kYUid, output}};
    auto status = plan->graph->execute(
        impl_->handle, variant_pack, plan->workspace);
    if (status.is_bad())
        return {false, cache_hit, plan->workspace_bytes,
                cache_hit ? 0.0 : plan->build_seconds, status.get_message()};
    return {true, cache_hit, plan->workspace_bytes,
            cache_hit ? 0.0 : plan->build_seconds, {}};
}

uint64_t CudnnConvPlanCache::build_count() const { return impl_->builds; }
uint64_t CudnnConvPlanCache::cache_hit_count() const { return impl_->hits; }
size_t CudnnConvPlanCache::cached_plan_count() const {
    return impl_->plans.size();
}

}  // namespace vrhino
