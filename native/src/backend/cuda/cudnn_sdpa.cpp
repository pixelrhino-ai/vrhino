#include "vrhino/backend/cudnn_sdpa.h"

#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vrhino {
namespace {
namespace fe = cudnn_frontend;

constexpr int64_t kQUid = 1;
constexpr int64_t kKUid = 2;
constexpr int64_t kVUid = 3;
constexpr int64_t kOUid = 4;
constexpr int64_t kBiasUid = 5;
constexpr int64_t kQueryLengthUid = 6;
constexpr int64_t kKeyValueLengthUid = 7;
constexpr size_t kMaximumCachedPlans = 128;

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

using PlanKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t,
                           uint32_t, bool, bool, bool, int,
                           std::array<int64_t, 4>, std::array<int64_t, 4>>;

PlanKey plan_key(const CudnnSdpaDescriptor& descriptor) {
    std::array<int64_t, 4> bias_dimensions{};
    std::array<int64_t, 4> bias_strides{};
    for (int index = 0; index < 4; ++index) {
        bias_dimensions[index] = descriptor.bias_dimensions[index];
        bias_strides[index] = descriptor.bias_strides[index];
    }
    return {descriptor.batch, descriptor.query_tokens, descriptor.key_tokens,
            descriptor.heads, descriptor.head_width,
            float_bits(descriptor.scale), descriptor.causal,
            descriptor.has_padding_mask, descriptor.has_additive_bias,
            descriptor.bias_rank,
            bias_dimensions, bias_strides};
}

struct Plan {
    std::shared_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::string rejection_reason;
    ~Plan() {
        if (workspace) cudaFree(workspace);
    }
};

std::shared_ptr<Plan> build_plan(cudnnHandle_t handle,
                                 const CudnnSdpaDescriptor& descriptor) {
    auto plan = std::make_shared<Plan>();
    if (descriptor.batch <= 0 || descriptor.query_tokens <= 0 ||
        descriptor.key_tokens <= 0 || descriptor.heads <= 0 ||
        descriptor.head_width <= 0 || descriptor.head_width % 8 != 0) {
        plan->rejection_reason = "unsupported dimensions or head alignment";
        return plan;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::vector<int64_t> q_dimensions = {
        descriptor.batch, descriptor.heads, descriptor.query_tokens,
        descriptor.head_width};
    const std::vector<int64_t> k_dimensions = {
        descriptor.batch, descriptor.heads, descriptor.key_tokens,
        descriptor.head_width};
    const std::vector<int64_t> q_strides = {
        descriptor.query_tokens * descriptor.heads * descriptor.head_width,
        descriptor.head_width,
        descriptor.heads * descriptor.head_width, 1};
    const std::vector<int64_t> k_strides = {
        descriptor.key_tokens * descriptor.heads * descriptor.head_width,
        descriptor.head_width,
        descriptor.heads * descriptor.head_width, 1};
    auto q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q").set_uid(kQUid)
                               .set_dim(q_dimensions).set_stride(q_strides));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K").set_uid(kKUid)
                               .set_dim(k_dimensions).set_stride(k_strides));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V").set_uid(kVUid)
                               .set_dim(k_dimensions).set_stride(k_strides));
    auto attributes = fe::graph::SDPA_attributes()
                          .set_name("vrhino_generic_sdpa")
                          .set_is_inference(true)
                          .set_attn_scale(descriptor.scale)
                          .set_causal_mask(descriptor.causal);
    if (descriptor.has_padding_mask) {
        const std::vector<int64_t> length_dimensions = {
            descriptor.batch, 1, 1, 1};
        const std::vector<int64_t> length_strides = {1, 1, 1, 1};
        auto query_lengths = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("query_lengths").set_uid(kQueryLengthUid)
                .set_dim(length_dimensions).set_stride(length_strides)
                .set_data_type(fe::DataType_t::INT32));
        auto key_value_lengths = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("key_value_lengths").set_uid(kKeyValueLengthUid)
                .set_dim(length_dimensions).set_stride(length_strides)
                .set_data_type(fe::DataType_t::INT32));
        attributes.set_padding_mask(true)
            .set_seq_len_q(query_lengths)
            .set_seq_len_kv(key_value_lengths);
    }
    if (descriptor.has_additive_bias) {
        std::vector<int64_t> dimensions(4), strides(4);
        for (int index = 0; index < 4; ++index) {
            dimensions[index] = descriptor.bias_dimensions[index];
            strides[index] = descriptor.bias_strides[index];
        }
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("additive_bias")
                                      .set_uid(kBiasUid)
                                      .set_dim(dimensions).set_stride(strides));
        attributes.set_bias(bias);
    }
    auto [output, statistics] = graph->sdpa(q, k, v, attributes);
    (void)statistics;
    output->set_output(true).set_uid(kOUid)
        .set_dim(q_dimensions).set_stride(q_strides)
        .set_data_type(fe::DataType_t::BFLOAT16);

    auto status = graph->build(handle, {fe::HeurMode_t::A});
    if (status.is_bad()) {
        plan->rejection_reason = status.get_message();
        return plan;
    }
    int64_t workspace_bytes = 0;
    status = graph->get_workspace_size(workspace_bytes);
    if (status.is_bad() || workspace_bytes < 0) {
        plan->rejection_reason = status.is_bad()
            ? status.get_message() : "invalid cuDNN workspace size";
        return plan;
    }
    if (workspace_bytes > 0) {
        const cudaError_t allocation = cudaMalloc(&plan->workspace, workspace_bytes);
        if (allocation != cudaSuccess) {
            plan->rejection_reason = std::string("cuDNN workspace allocation failed: ") +
                                     cudaGetErrorString(allocation);
            (void)cudaGetLastError();
            return plan;
        }
    }
    plan->workspace_bytes = static_cast<size_t>(workspace_bytes);
    plan->graph = std::move(graph);
    return plan;
}
}  // namespace

struct CudnnSdpaPlanCache::Impl {
    explicit Impl(cudnnHandle_t value) : handle(value) {}
    cudnnHandle_t handle{};
    std::map<PlanKey, std::shared_ptr<Plan>> plans;
    uint64_t builds = 0;
    uint64_t hits = 0;
};

CudnnSdpaPlanCache::CudnnSdpaPlanCache(cudnnHandle_t handle)
    : impl_(std::make_unique<Impl>(handle)) {}
CudnnSdpaPlanCache::~CudnnSdpaPlanCache() = default;

PrefixMaskCanonicalization canonicalize_boolean_prefix_mask(
    const Tensor& host_mask, int64_t batch, int64_t query_tokens,
    int64_t key_tokens) {
    PrefixMaskCanonicalization result;
    if (host_mask.device() != DeviceId::host() ||
        host_mask.dtype() != DType::Bool) {
        result.reason = "prefix recognition requires a host Boolean mask";
        return result;
    }
    if (batch <= 0 || query_tokens <= 0 || key_tokens <= 0 ||
        batch > INT32_MAX || query_tokens > INT32_MAX ||
        key_tokens > INT32_MAX ||
        host_mask.ndim() > 3) {
        result.reason = "invalid logical or mask rank";
        return result;
    }
    const std::array<int64_t, 3> logical = {batch, query_tokens, key_tokens};
    const int shift = 3 - static_cast<int>(host_mask.ndim());
    for (int index = 0; index < host_mask.ndim(); ++index) {
        const int64_t dimension = host_mask.dim(index);
        if (dimension != 1 && dimension != logical[index + shift]) {
            result.reason = "Boolean mask is not broadcastable to BQK";
            return result;
        }
    }
    const auto* values = host_mask.data_as<uint8_t>();
    const auto& shape = host_mask.shape();
    const auto& strides = host_mask.strides();
    const auto value_at = [&](int64_t b, int64_t q, int64_t k) {
        const std::array<int64_t, 3> coordinates = {b, q, k};
        int64_t offset = 0;
        for (int dimension = 0; dimension < 3; ++dimension) {
            const int input_dimension = dimension - shift;
            if (input_dimension >= 0 && shape[input_dimension] != 1)
                offset += coordinates[dimension] * strides[input_dimension];
        }
        return values[offset] != 0;
    };

    result.query_lengths.resize(static_cast<size_t>(batch));
    result.key_value_lengths.resize(static_cast<size_t>(batch));
    const int query_input_dimension = 1 - shift;
    const bool query_is_broadcast = query_input_dimension < 0 ||
        shape[static_cast<size_t>(query_input_dimension)] == 1;
    for (int64_t b = 0; b < batch; ++b) {
        if (query_is_broadcast) {
            int64_t key_value_length = 0;
            while (key_value_length < key_tokens &&
                   value_at(b, 0, key_value_length))
                ++key_value_length;
            if (key_value_length == 0) {
                result.reason =
                    "zero-length prefixes preserve no finite bounded-Attention output";
                return result;
            }
            for (int64_t k = 0; k < key_tokens; ++k) {
                if (value_at(b, 0, k) != (k < key_value_length)) {
                    result.reason =
                        "Boolean mask is not a rectangular prefix/padding pattern";
                    return result;
                }
            }
            result.query_lengths[static_cast<size_t>(b)] =
                static_cast<int32_t>(query_tokens);
            result.key_value_lengths[static_cast<size_t>(b)] =
                static_cast<int32_t>(key_value_length);
            continue;
        }
        int64_t query_length = 0;
        while (query_length < query_tokens) {
            bool any = false;
            for (int64_t k = 0; k < key_tokens; ++k)
                any = any || value_at(b, query_length, k);
            if (!any) break;
            ++query_length;
        }
        int64_t key_value_length = 0;
        if (query_length > 0)
            while (key_value_length < key_tokens &&
                   value_at(b, 0, key_value_length))
                ++key_value_length;
        if (query_length == 0 || key_value_length == 0) {
            result.reason = "zero-length prefixes preserve no finite bounded-Attention output";
            return result;
        }
        // The established bounded semantic produces no finite value for a
        // fully masked query row (zero softmax denominator).  cuDNN padding
        // instead defines storage outside seq_len_q independently.  Until the
        // shared Attention contract defines masked-query outputs, accepting a
        // shorter Q prefix would not be an exact backend substitution.
        if (query_length != query_tokens) {
            result.reason =
                "query-padding rows do not have equivalent bounded-Attention output semantics";
            return result;
        }
        for (int64_t q = 0; q < query_tokens; ++q) {
            for (int64_t k = 0; k < key_tokens; ++k) {
                const bool expected = q < query_length && k < key_value_length;
                if (value_at(b, q, k) != expected) {
                    result.reason = "Boolean mask is not a rectangular prefix/padding pattern";
                    return result;
                }
            }
        }
        result.query_lengths[static_cast<size_t>(b)] =
            static_cast<int32_t>(query_length);
        result.key_value_lengths[static_cast<size_t>(b)] =
            static_cast<int32_t>(key_value_length);
    }
    result.representable = true;
    return result;
}

CudnnSdpaExecution CudnnSdpaPlanCache::execute(
    const CudnnSdpaDescriptor& descriptor, const void* q, const void* k,
    const void* v, const void* additive_bias, const int32_t* query_lengths,
    const int32_t* key_value_lengths, void* output) {
    if (descriptor.has_padding_mask && (!query_lengths || !key_value_lengths))
        return {false, 0, "padding-mask execution requires Q/KV sequence lengths"};
    const PlanKey key = plan_key(descriptor);
    auto found = impl_->plans.find(key);
    if (found == impl_->plans.end()) {
        if (impl_->plans.size() >= kMaximumCachedPlans)
            return {false, 0, "cuDNN SDPA plan cache capacity reached"};
        found = impl_->plans.emplace(key, build_plan(impl_->handle, descriptor)).first;
        ++impl_->builds;
    } else {
        ++impl_->hits;
    }
    const std::shared_ptr<Plan>& plan = found->second;
    if (!plan->graph)
        return {false, plan->workspace_bytes, plan->rejection_reason};

    std::unordered_map<int64_t, void*> variant_pack = {
        {kQUid, const_cast<void*>(q)}, {kKUid, const_cast<void*>(k)},
        {kVUid, const_cast<void*>(v)}, {kOUid, output}};
    if (descriptor.has_additive_bias)
        variant_pack.emplace(kBiasUid, const_cast<void*>(additive_bias));
    if (descriptor.has_padding_mask) {
        variant_pack.emplace(kQueryLengthUid,
                             const_cast<int32_t*>(query_lengths));
        variant_pack.emplace(kKeyValueLengthUid,
                             const_cast<int32_t*>(key_value_lengths));
    }
    auto status = plan->graph->execute(
        impl_->handle, variant_pack, plan->workspace);
    if (status.is_bad())
        return {false, plan->workspace_bytes, status.get_message()};
    return {true, plan->workspace_bytes, {}};
}

uint64_t CudnnSdpaPlanCache::build_count() const { return impl_->builds; }
uint64_t CudnnSdpaPlanCache::cache_hit_count() const { return impl_->hits; }
size_t CudnnSdpaPlanCache::cached_plan_count() const { return impl_->plans.size(); }

}  // namespace vrhino
