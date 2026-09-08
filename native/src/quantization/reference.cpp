#include "vrhino/quantization/reference.h"

#include <cmath>
#include <limits>
#include <vector>

#include "vrhino/error.h"

namespace vrhino {
namespace {

constexpr int kH4[4][4] = {
    { 1,  1,  1, -1},
    { 1,  1, -1,  1},
    { 1, -1,  1,  1},
    {-1,  1,  1,  1},
};

bool is_power_of_four(int64_t value) {
    if (value < 4) return false;
    while (value > 1) {
        if (value % 4 != 0) return false;
        value /= 4;
    }
    return true;
}

int hadamard_sign(int64_t row, int64_t column) {
    int result = 1;
    while (row != 0 || column != 0) {
        result *= kH4[row % 4][column % 4];
        row /= 4;
        column /= 4;
    }
    return result;
}

void checked_range(int64_t begin, int64_t count, int64_t bound,
                   const char* message) {
    require(begin >= 0 && count > 0 && begin <= bound && count <= bound - begin,
            message);
}

}  // namespace

Tensor materialize_preconditioned_symmetric_int8_f32(
        const Tensor& stored_weight,
        const int64_t first_output_row,
        const int64_t output_row_count,
        const int64_t first_preconditioner_block,
        const int64_t preconditioner_block_count) {
    require(stored_weight.defined() && stored_weight.device().is_host(),
            "Reference quantized materialization requires CPU storage");
    require(stored_weight.is_quantized(),
            "Bare signed-I8 storage has no executable quantization semantics");
    const QuantizationInfo& info = stored_weight.quantization();
    require(info.scheme == kPreconditionedSymmetricInt8V1 &&
                info.type == QuantType::INT8Symmetric &&
                stored_weight.dtype() == DType::I8,
            "Unsupported quantized tensor scheme for reference materialization");
    require(info.preconditioner == PreconditionerType::RegularBlockHadamardV1 &&
                info.dequantization ==
                    DequantizationSemantics::ScaleThenInversePreconditioner,
            "Unsupported executable preconditioner semantics");
    require(stored_weight.ndim() == 2 && info.axis == 1 &&
                info.preconditioner_axis == 1,
            "Preconditioned symmetric INT8 v1 requires a rank-2 input-axis transform");
    require(info.stored_shape == stored_weight.shape(),
            "Quantized stored/logical shape mismatch");
    const int64_t rows = stored_weight.dim(0);
    const int64_t columns = stored_weight.dim(1);
    const int64_t group = info.preconditioner_group_size;
    require(is_power_of_four(group) && columns % group == 0,
            "Incompatible regular block-Hadamard group semantics");
    require(info.group_size == columns && info.granularity == "per_channel",
            "Preconditioned symmetric INT8 v1 requires one scale per output row");
    require(info.scales.defined() && info.scales.device().is_host() &&
                info.scales.dtype() == DType::F32 &&
                info.scales.shape() == std::vector<int64_t>({rows, 1}),
            "Invalid per-output scale tensor");
    checked_range(first_output_row, output_row_count, rows,
                  "Output-row materialization range exceeds tensor");
    const int64_t blocks = columns / group;
    checked_range(first_preconditioner_block, preconditioner_block_count, blocks,
                  "Preconditioner-block materialization range exceeds tensor");
    require(preconditioner_block_count <=
                std::numeric_limits<int64_t>::max() / group,
            "Materialized column count overflow");
    const int64_t output_columns = preconditioner_block_count * group;
    Tensor output = Tensor::host({output_row_count, output_columns}, DType::F32);
    const auto* packed = stored_weight.data_as<const int8_t>();
    const auto* scales = info.scales.data_as<const float>();
    auto* destination = output.data_as<float>();
    const float normalization = 1.0f / std::sqrt(static_cast<float>(group));

    // The producer stores W_rot = W @ H^T. The regular H is symmetric and
    // orthonormal, so reconstruction is (q * scale) @ H. Preserve the simple
    // j-ascending FP32 reference accumulation order deliberately.
    for (int64_t output_row = 0; output_row < output_row_count; ++output_row) {
        const int64_t source_row = first_output_row + output_row;
        const float scale = scales[source_row];
        require(std::isfinite(scale) && scale > 0.0f,
                "Quantization scale must be finite and positive");
        for (int64_t block = 0; block < preconditioner_block_count; ++block) {
            const int64_t source_block = first_preconditioner_block + block;
            const int64_t source_base = source_row * columns + source_block * group;
            const int64_t destination_base = output_row * output_columns + block * group;
            for (int64_t column = 0; column < group; ++column) {
                float sum = 0.0f;
                for (int64_t rotated_column = 0; rotated_column < group;
                     ++rotated_column) {
                    sum += static_cast<float>(packed[source_base + rotated_column]) *
                           scale * static_cast<float>(
                               hadamard_sign(rotated_column, column));
                }
                destination[destination_base + column] = sum * normalization;
            }
        }
    }
    return output;
}

}  // namespace vrhino
