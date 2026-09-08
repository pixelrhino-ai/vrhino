#pragma once

#include <cstddef>
#include <cstdint>

namespace vrhino::cuda_attention_config {

// Backend implementation parameters only.  They are deliberately independent
// of model identity, architecture identity, and sequence length.
inline constexpr int kWarpThreads = 32;
inline constexpr int kWarpsPerBlock = 8;
inline constexpr int kQueryTile = kWarpsPerBlock;
inline constexpr int kKeyTile = 32;
inline constexpr int kHeadTile = 128;
inline constexpr int kOutputsPerLane = kHeadTile / kWarpThreads;

// Phase 15R ordered FP32 default. The correct Phase 14 kernel remains available
// as a research baseline. This tile bounds global score/state scratch
// independently of the full sequence length and is not selected by model or
// shape identity.
inline constexpr int kOrderedCandidateKeyTile = 96;

// Generic two-pass BF16 Tensor-Op path.  The tile is an implementation
// parameter only: selection is based on dtype, layout, device capability and
// workspace, never model or architecture identity.
inline constexpr int kBf16TensorKeyTile = 96;
inline constexpr int64_t kBf16TensorMinimumScoreElements =
    64LL * 1024LL * 1024LL;

// Phase 20D generic steady-state crossover floor.  The calibration matrix
// covers score volumes from 8M through 128M+, varied Q/K ratios, head counts,
// head widths, and masked/unmasked descriptors.  Below the measured floor the
// established bounded implementation remains the fail-closed default.
inline constexpr int64_t kCudnnSdpaMinimumScoreElements =
    8LL * 1024LL * 1024LL;

inline constexpr std::size_t ordered_workspace_bytes(
    int64_t batch, int64_t query_tokens, int64_t heads, int64_t key_tile) {
    const std::size_t rows = static_cast<std::size_t>(batch * query_tokens * heads);
    return 2ULL * rows * static_cast<std::size_t>(key_tile) * sizeof(float) +
           2ULL * rows * sizeof(float);
}

inline constexpr std::size_t ordered_bf16_workspace_bytes(
    int64_t batch, int64_t query_tokens, int64_t heads, int64_t head_dimension,
    int64_t key_tile) {
    const std::size_t rows = static_cast<std::size_t>(batch * query_tokens * heads);
    // FP32 output accumulator plus FP32 views of the already-rounded BF16 Q/K
    // used by the numerically stable pedantic QK reduction. Q and K are equal
    // length in the current public self-Attention contract.
    return ordered_workspace_bytes(batch, query_tokens, heads, key_tile) +
           3ULL * rows * static_cast<std::size_t>(head_dimension) * sizeof(float);
}

inline constexpr std::size_t bf16_tensor_two_pass_workspace_bytes(
    int64_t batch, int64_t query_tokens, int64_t heads, int64_t head_dimension,
    int64_t key_tile) {
    const std::size_t rows = static_cast<std::size_t>(batch * query_tokens * heads);
    return rows * static_cast<std::size_t>(key_tile) *
               (sizeof(float) + sizeof(std::uint16_t)) +
           rows * static_cast<std::size_t>(head_dimension) * sizeof(float) +
           2ULL * rows * sizeof(float);
}

inline constexpr int64_t bf16_tensor_two_pass_dispatches_per_attention(
    int64_t batch, int64_t key_tokens, int64_t key_tile) {
    const int64_t tiles = (key_tokens + key_tile - 1) / key_tile;
    // initialize + pass1(QK per batch + stats) +
    // pass2(QK per batch + probabilities + PV per batch) + finalize.
    return 2 + tiles * (2 * batch + 2) + tiles * batch;
}

inline constexpr int64_t ordered_dispatches_per_attention(
    int64_t batch, int64_t key_tokens, int64_t key_tile) {
    const int64_t tiles = (key_tokens + key_tile - 1) / key_tile;
    return 2 + tiles * (batch + 2);
}

// K and V are staged as FP32 for both FP32 and BF16 execution.  Wider head
// dimensions use the same streaming kernel without the shared K/V cache.
inline constexpr std::size_t kCachedWorkspaceBytes =
    2ULL * kKeyTile * kHeadTile * sizeof(float);

inline constexpr std::size_t workspace_bytes(int64_t head_dimension) {
    return head_dimension <= kHeadTile ? kCachedWorkspaceBytes : 0;
}

}  // namespace vrhino::cuda_attention_config
