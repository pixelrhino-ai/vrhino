#pragma once

#include <cmath>
#include "../runtime/neural_graph.h"
#include "vrhino/architecture.h"

// Private self-attention graph and canonical parameter binding for Wan production.
namespace vrhino::wan_internal {
inline constexpr const char* parameters[] = {
    "modulation", "self_attn.q.weight", "self_attn.q.bias",
    "self_attn.k.weight", "self_attn.k.bias", "self_attn.v.weight", "self_attn.v.bias",
    "self_attn.norm_q.weight", "self_attn.norm_k.weight", "self_attn.o.weight", "self_attn.o.bias"};
inline constexpr const char* checkpoints[] = {
    "input", "combined", "shift", "scale", "gate", "pre_attention_norm", "self_attention_input",
    "self_attention.q_linear", "self_attention.k_linear", "self_attention.v_linear",
    "self_attention.q_norm", "self_attention.k_norm", "self_attention.q_heads",
    "self_attention.k_heads", "self_attention.v_heads", "self_attention.q_positioned",
    "self_attention.k_positioned", "self_attention.attention_heads",
    "self_attention.attention_merged", "self_attention.output", "gated", "first_residual"};

inline neural_graph::Graph self_attention_graph(int64_t b, int64_t s, DType execution,
                                                DType hidden_dtype, bool diagnostics = false) {
    namespace ng = neural_graph;
    require(b == 1 && s > 0, "Wan bounded self-attention requires B=1 and positive S");
    require((execution == DType::F32 && hidden_dtype == DType::F32) ||
            (execution == DType::BF16 && (hidden_dtype == DType::F32 || hidden_dtype == DType::BF16)),
            "Unsupported Wan self-attention dtype contract");
    const std::vector<int64_t> hidden{b,s,1536}, part{b,1,1536}, heads{b,s,12,128};
    ng::Description d; d.schema = ng::schema_v1; d.execution_dtype = execution;
    d.values = {{hidden,hidden_dtype},{{b,6,1536}},{{1,s,1,128}},{{1,s,1,128}},
        {{1,6,1536}},{{1536,1536},execution},{{1536},execution},
        {{1536,1536},execution},{{1536},execution},{{1536,1536},execution},{{1536},execution},
        {{1536}},{{1536}},{{1536,1536},execution},{{1536},execution},{{1}}};
    d.inputs = {0,1,2,3}; d.parameters = {4,5,6,7,8,9,10,11,12,13,14}; d.constants = {{15,1.0f}};
    auto node = [&](ng::Primitive op, const std::vector<int64_t>& shape, DType dtype = DType::F32) {
        auto id = static_cast<ng::ValueId>(d.values.size());
        d.values.push_back({shape,dtype}); d.nodes.push_back({id,std::move(op)}); return id;
    };
    auto cast = [&](ng::ValueId x, DType dtype) {
        return d.values[x].dtype == dtype ? x : node(ng::Cast{x,dtype},d.values[x].shape,dtype);
    };
    auto combined = node(ng::Add{4,1},{b,6,1536});
    auto shift = node(ng::Slice{combined,1,0,1},part);
    auto scale = node(ng::Slice{combined,1,1,2},part);
    auto gate = node(ng::Slice{combined,1,2,3},part);
    auto norm = node(ng::LayerNorm{0,{},{},1e-6f},hidden,hidden_dtype);
    auto scale_one = node(ng::Add{15,scale},part);
    auto scaled = node(ng::Mul{cast(norm,DType::F32),scale_one},hidden);
    auto self_input = node(ng::Add{scaled,shift},hidden);
    auto operand = cast(self_input,execution);
    auto q = node(ng::Linear{operand,5,6,execution},hidden,execution);
    auto k = node(ng::Linear{operand,7,8,execution},hidden,execution);
    auto v = node(ng::Linear{operand,9,10,execution},hidden,execution);
    auto q_norm = node(ng::RmsNorm{q,11,2,1e-6f},hidden);
    auto k_norm = node(ng::RmsNorm{k,12,2,1e-6f},hidden);
    auto q_heads = node(ng::Reshape{q_norm,heads},heads);
    auto k_heads = node(ng::Reshape{k_norm,heads},heads);
    auto v_heads = node(ng::Reshape{v,heads},heads,execution);
    auto q_rope = node(ng::Rope{q_heads,2,3},heads);
    auto k_rope = node(ng::Rope{k_heads,2,3},heads);
    auto q_operand = cast(q_rope,execution), k_operand = cast(k_rope,execution);
    auto attention = node(ng::Attention{q_operand,k_operand,v_heads,1.0f/std::sqrt(128.0f)},heads,execution);
    auto merged = node(ng::Reshape{attention,hidden},hidden,execution);
    auto projection = node(ng::Linear{merged,13,14,execution},hidden,execution);
    auto gated = node(ng::Mul{cast(projection,DType::F32),gate},hidden);
    auto residual = node(ng::Add{cast(0,DType::F32),gated},hidden);
    d.outputs = diagnostics ? std::vector<ng::ValueId>{0,combined,shift,scale,gate,norm,self_input,
        q,k,v,q_norm,k_norm,q_heads,k_heads,v_heads,q_rope,k_rope,attention,merged,projection,gated,residual}
        : std::vector<ng::ValueId>{residual,combined};
    return ng::admit(d);
}

inline std::vector<Tensor> bind_parameters(Backend& backend, const neural_graph::Graph& graph,
                                            const WeightMap& weights) {
    const auto& d = graph.description();
    require(backend.execution_dtype() == d.execution_dtype, "Wan Graph context mismatch");
    // Validate the complete canonical binding before any cache materialization.
    for (size_t i = 0; i < std::size(parameters); ++i) {
        const auto& t = weights.at(parameters[i]);
        require(!t.is_quantized() && t.dtype() == DType::F32 && t.logical_dtype() == DType::F32 &&
                t.shape() == d.values[d.parameters[i]].shape, "Wan canonical parameter contract mismatch");
    }
    std::vector<Tensor> result;
    for (size_t i = 0; i < std::size(parameters); ++i)
        result.push_back(backend.copy_to_device(weights.at(parameters[i]),d.values[d.parameters[i]].dtype));
    return result;
}

inline std::vector<Tensor> self_attention(Backend& backend, const PrecisionPolicy& policy,
                                         const WeightMap& weights, const Tensor& hidden,
                                         const Tensor& modulation, const Tensor& cosine,
                                         const Tensor& sine, bool diagnostics = false) {
    require(policy.requested_dtype() == backend.execution_dtype(), "Wan Graph policy context mismatch");
    require(hidden.ndim() == 3, "Wan Graph hidden rank mismatch");
    auto graph = self_attention_graph(hidden.dim(0),hidden.dim(1),backend.execution_dtype(),hidden.dtype(),diagnostics);
    auto bound = bind_parameters(backend,graph,weights);
    return neural_graph::evaluate(backend,graph,{hidden,modulation,cosine,sine},bound);
}
}
