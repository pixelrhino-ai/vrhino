#pragma once

#include "vrhino/architecture.h"
#include "vrhino/tensor_util.h"

// Historical bounded F32 oracle. Never linked into production architecture.
namespace vrhino::ltx_test_reference {
inline Tensor projected_attention(Backend& backend, const PrecisionPolicy& policy,
                           const WeightMap& weights,
                           const std::string& prefix, const Tensor& query_states,
                           const Tensor& key_value_states, int heads,
                           const Tensor* rope_cos = nullptr,
                           const Tensor* rope_sin = nullptr,
                           const Tensor* mask = nullptr,
                           TensorBundle* trace = nullptr,
                           const std::string& trace_prefix = {}) {
    Tensor q = backend.linear(query_states, weights.at(prefix + ".to_q.weight"),
                              &weights.at(prefix + ".to_q.bias"));
    Tensor k = backend.linear(key_value_states, weights.at(prefix + ".to_k.weight"),
                              &weights.at(prefix + ".to_k.bias"));
    Tensor v = backend.linear(key_value_states, weights.at(prefix + ".to_v.weight"),
                              &weights.at(prefix + ".to_v.bias"));
    if (trace) {
        (*trace)[trace_prefix + ".q_projection"] = q;
        (*trace)[trace_prefix + ".k_projection"] = k;
        (*trace)[trace_prefix + ".v_projection"] = v;
    }
    q = attention_norm(backend, policy, q,
                       &weights.at(prefix + ".q_norm.weight"), 1e-5f);
    k = attention_norm(backend, policy, k,
                       &weights.at(prefix + ".k_norm.weight"), 1e-5f);
    if (trace) {
        (*trace)[trace_prefix + ".q_norm"] = q;
        (*trace)[trace_prefix + ".k_norm"] = k;
    }
    if (rope_cos) {
        q = operation_rope(backend, policy, q, *rope_cos, *rope_sin);
        k = operation_rope(backend, policy, k, *rope_cos, *rope_sin);
    }
    if (trace) {
        (*trace)[trace_prefix + ".q_positioned"] = q;
        (*trace)[trace_prefix + ".k_positioned"] = k;
    }
    q = split_heads(backend, q, heads); k = split_heads(backend, k, heads);
    v = split_heads(backend, v, heads);
    if (trace) {
        (*trace)[trace_prefix + ".q_heads"] = q;
        (*trace)[trace_prefix + ".k_heads"] = k;
        (*trace)[trace_prefix + ".v_heads"] = v;
    }
    Tensor attended = operation_attention(backend, policy, q, k, v, mask);
    if (trace) (*trace)[trace_prefix + ".attention_heads"] = attended;
    Tensor output = merge_heads(backend, attended);
    if (trace) (*trace)[trace_prefix + ".attention_merged"] = output;
    output = backend.linear(output, weights.at(prefix + ".to_out.0.weight"),
                            &weights.at(prefix + ".to_out.0.bias"));
    if (trace) (*trace)[trace_prefix + ".output_projection"] = output;
    return output;
}

inline std::vector<Tensor> block_modulation(Backend& backend, const WeightMap& weights,
                                      const Tensor& input, const Tensor& timestep,
                                      TensorBundle* trace) {
    const int64_t b = input.dim(0), width = input.dim(2), mod_tokens = timestep.dim(1);
    Tensor table = backend.reshape(weights.at("scale_shift_table"), {1, 1, 6, width});
    Tensor external = backend.reshape(timestep, {b, mod_tokens, 6, width});
    Tensor combined = backend.add(table, external);
    if (trace) (*trace)["block.0.modulation_table"] = combined;
    auto raw_parts = backend.split(combined, {1, 1, 1, 1, 1, 1}, 2);
    std::vector<Tensor> parts;
    for (Tensor& part : raw_parts) parts.push_back(backend.reshape(part, {b, mod_tokens, width}));
    if (trace) {
        (*trace)["block.0.shift"] = parts[0];
        (*trace)["block.0.scale"] = parts[1];
        (*trace)["block.0.gate"] = parts[2];
    }
    return parts;
}

inline Tensor self_attention_branch(Backend& backend, const PrecisionPolicy& policy,
                             const WeightMap& weights, const Tensor& input,
                             const std::vector<Tensor>& parts,
                             const Tensor& rope_cos, const Tensor& rope_sin,
                             TensorBundle* trace) {
    Tensor normalized = backend.rms_norm(input, nullptr, 1e-6f);
    if (trace) (*trace)["block.0.pre_attention_norm"] = normalized;
    Tensor self_input = modulate(backend, policy, normalized, parts[0], parts[1]);
    if (trace) (*trace)["block.0.self_attention_input"] = self_input;
    Tensor self_output = projected_attention(
        backend, policy, weights, "attn1", self_input, self_input, 32,
        &rope_cos, &rope_sin, nullptr, trace, "block.0.self_attention");
    Tensor hidden = gated_residual(backend, policy, input, self_output, parts[2]);
    if (trace) (*trace)["block.0.first_residual"] = hidden;
    return hidden;
}

inline Tensor self_attention_reference(Backend& backend, const WeightMap& weights,
                                             const Tensor& input, const Tensor& timestep,
                                             const Tensor& cosine, const Tensor& sine,
                                             TensorBundle* trace) {
    if (trace) (*trace)["block.0.input"] = input;
    auto parts = block_modulation(backend, weights, input, timestep, trace);
    return self_attention_branch(backend, PrecisionPolicy::fp32(), weights, input,
                                  parts, cosine, sine, trace);
}

}
