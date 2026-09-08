#pragma once

#include "../runtime/neural_graph.h"
#include "ltx_self_attention.h"

namespace vrhino::ltx_internal {
enum class SelfAttentionContract { F32, BF16F32Hidden, BF16Hidden };
SelfAttentionContract self_attention_contract(DType execution, DType hidden);
// Ordered block-local bindings; independent of checkpoint/model revision.
inline constexpr const char* self_attention_parameters[] = {
    "scale_shift_table", "attn1.to_q.weight", "attn1.to_q.bias",
    "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias",
    "attn1.q_norm.weight", "attn1.k_norm.weight",
    "attn1.to_out.0.weight", "attn1.to_out.0.bias"};
inline constexpr const char* self_attention_checkpoints[] = {
    "block.0.input", "block.0.modulation_table", "block.0.shift", "block.0.scale",
    "block.0.gate", "block.0.pre_attention_norm", "block.0.self_attention_input",
    "block.0.self_attention.q_projection", "block.0.self_attention.k_projection",
    "block.0.self_attention.v_projection", "block.0.self_attention.q_norm",
    "block.0.self_attention.k_norm", "block.0.self_attention.q_positioned",
    "block.0.self_attention.k_positioned", "block.0.self_attention.q_heads",
    "block.0.self_attention.k_heads", "block.0.self_attention.v_heads",
    "block.0.self_attention.attention_heads",
    "block.0.self_attention.attention_merged", "block.0.self_attention.output_projection",
    "block.0.first_residual"};
neural_graph::Graph self_attention_graph(int64_t batch, int64_t tokens,
                                         int64_t modulation_tokens, bool diagnostics = false,
                                         bool block_state = false,
                                         SelfAttentionContract contract = SelfAttentionContract::F32);
}
