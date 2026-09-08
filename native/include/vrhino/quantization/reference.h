#pragma once

#include <cstdint>

#include "vrhino/tensor.h"

namespace vrhino {

inline constexpr const char* kPreconditionedSymmetricInt8V1 =
    "preconditioned_symmetric_int8.v1";
inline constexpr const char* kRegularBlockHadamardV1 =
    "regular_block_hadamard.v1";

// Materializes complete preconditioner blocks for a bounded range of output
// rows. The returned CPU tensor is FP32 [row_count, block_count * group_size].
// No other part of the stored tensor is read or expanded.
Tensor materialize_preconditioned_symmetric_int8_f32(
    const Tensor& stored_weight,
    int64_t first_output_row,
    int64_t output_row_count,
    int64_t first_preconditioner_block,
    int64_t preconditioner_block_count);

}  // namespace vrhino
