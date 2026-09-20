#include <cmath>
#include <iostream>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

vrhino::Tensor host(vrhino::Backend& backend, const vrhino::Tensor& value) {
    return value.device() == vrhino::Device::CPU ? value : backend.copy_to_host(value);
}

void require_finite(const vrhino::Tensor& value, const std::string& name) {
    vrhino::require(value.dtype() == vrhino::DType::F32,
                    name + " host check expects FP32");
    for (int64_t index = 0; index < value.numel(); ++index)
        vrhino::require(std::isfinite(value.data_as<float>()[index]),
                        name + " contains non-finite value");
}

void require_rope_product_boundaries(vrhino::CudaBackend& backend) {
    using namespace vrhino;
    // Analytic cancellation: (1+2^-23)*(1-2^-23) rounds to 1 in F32.
    // Separate product/add therefore gives zero; contraction gives -2^-46.
    const float u = std::ldexp(1.0f, -23);
    const Tensor x = backend.copy_to_device(
        host_f32({1, 1, 1, 2}, {1.0f + u, 1.0f}), DType::F32);
    const Tensor c = backend.copy_to_device(
        host_f32({1, 1, 1, 2}, {1.0f - u, 0.0f}), DType::F32);
    const Tensor s = backend.copy_to_device(
        host_f32({1, 1, 1, 2}, {1.0f, 0.0f}), DType::F32);
    const Tensor result = backend.copy_to_host(backend.rope_nd(x, c, s));
    require(result.dtype() == DType::F32 && result.data_as<float>()[0] == 0.0f,
            "RoPE contracted an F32 product across the addition boundary");

    // Cover broadcasting, pair orientation, tails and all input/frequency dtype
    // combinations. Volatile stores make the host oracle's rounding explicit.
    for (const int64_t width : {6, 64, 128}) {
        std::vector<float> values(3 * 2 * width), cosine(3 * width), sine(3 * width);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = std::sin(static_cast<float>(i + 1) * 0.173f);
        for (size_t i = 0; i < cosine.size(); ++i) {
            const float phase = static_cast<float>(i / 2 + 1) * 0.137f;
            cosine[i] = std::cos(phase);
            sine[i] = std::sin(phase);
        }
        for (const auto input_dtype : {DType::F32, DType::BF16}) {
            for (const auto frequency_dtype : {DType::F32, DType::BF16}) {
                const Tensor input = backend.copy_to_device(
                    host_f32({1, 3, 2, width}, values), input_dtype);
                const Tensor co = backend.copy_to_device(
                    host_f32({1, 3, 1, width}, cosine), frequency_dtype);
                const Tensor si = backend.copy_to_device(
                    host_f32({1, 3, 1, width}, sine), frequency_dtype);
                const Tensor ih = backend.copy_to_host(backend.cast(input, DType::F32));
                const Tensor ch = backend.copy_to_host(backend.cast(co, DType::F32));
                const Tensor sh = backend.copy_to_host(backend.cast(si, DType::F32));
                std::vector<float> expected(values.size());
                for (int64_t i = 0; i < input.numel(); ++i) {
                    const int64_t column = i % width;
                    const int64_t pair = i - column + (column ^ 1);
                    const int64_t frequency = (i / (2 * width)) * width + column;
                    const float rotated = (column & 1) ? ih.data_as<float>()[pair]
                                                      : -ih.data_as<float>()[pair];
                    volatile float first = ih.data_as<float>()[i] * ch.data_as<float>()[frequency];
                    volatile float second = rotated * sh.data_as<float>()[frequency];
                    expected[i] = first + second;
                }
                const Tensor actual = backend.rope_nd(input, co, si);
                require(actual.dtype() == input_dtype && actual.shape() == input.shape(),
                        "RoPE changed its storage dtype or shape");
                const Tensor ah = backend.copy_to_host(backend.cast(actual, DType::F32));
                const Tensor eh = backend.copy_to_host(backend.cast(backend.copy_to_device(
                    host_f32(input.shape(), expected), input_dtype), DType::F32));
                for (int64_t i = 0; i < actual.numel(); ++i)
                    require(ah.data_as<float>()[i] == eh.data_as<float>()[i],
                            "RoPE pair/broadcast/product-rounding contract mismatch");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        using namespace vrhino;
        CudaBackend backend;
        backend.set_execution_dtype(DType::BF16);
        require_rope_product_boundaries(backend);
        const PrecisionPolicy policy = PrecisionPolicy::unqualified_default(DType::BF16);

        // Non-tile-aligned widths deliberately exercise tail/layout handling.
        const Tensor x = host_f32({2, 3}, {
            0.101234f, -0.337891f, 0.906543f,
            -0.777123f, 0.250987f, 0.619876f,
        });
        const Tensor shift = host_f32({1, 3}, {0.00314159f, -0.00271828f, 0.00141421f});
        const Tensor scale = host_f32({1, 3}, {0.0078123f, -0.0065432f, 0.0054321f});
        const Tensor temporary = modulate(backend, policy, x, shift, scale);
        require(temporary.dtype() == DType::F32,
                "Modulation producer did not retain FP32 temporary");
        const Tensor temporary_host = host(backend, temporary);
        require_finite(temporary_host, "modulation temporary");

        const Tensor early_rounded = backend.cast(backend.cast(temporary, DType::BF16), DType::F32);
        const Tensor early_host = host(backend, early_rounded);
        bool rounding_observed = false;
        for (int64_t index = 0; index < temporary_host.numel(); ++index)
            rounding_observed |= temporary_host.data_as<float>()[index] !=
                                 early_host.data_as<float>()[index];
        require(rounding_observed,
                "Test vector did not distinguish FP32 temporary from early BF16 rounding");

        const Tensor weight = host_f32({5, 3}, {
             0.11f, -0.22f,  0.33f,
            -0.44f,  0.55f, -0.66f,
             0.77f, -0.88f,  0.99f,
             0.13f,  0.17f, -0.19f,
            -0.23f,  0.29f,  0.31f,
        });
        const Tensor output = operation_linear(
            backend, policy, PrecisionOperation::Linear,
            PrecisionSemantic::TemporaryCompute, temporary, weight);
        require(output.dtype() == DType::BF16,
                "Heavy Linear consumer did not apply BF16 operand/output contract");
        require(output.shape() == std::vector<int64_t>({2, 5}),
                "Operation-contract Linear tail shape mismatch");
        require_finite(host(backend, backend.cast(output, DType::F32)),
                       "contract Linear output");

        const PrecisionOperationContract linear = policy.operation_contract(
            PrecisionOperation::Linear, PrecisionSemantic::TemporaryCompute);
        const Tensor accumulator = backend.linear(
            backend.cast(temporary, linear.operand_dtype), weight, nullptr,
            linear.compute_dtype, DType::F32);
        require(accumulator.dtype() == DType::F32,
                "BF16 heavy Linear cannot expose FP32 accumulator output");
        require_finite(host(backend, accumulator), "Linear accumulator");

        const PrecisionOperationContract attention = policy.operation_contract(
            PrecisionOperation::Attention, PrecisionSemantic::TemporaryCompute);
        require(attention.reduction_semantics ==
                    PrecisionReductionSemantics::FP32StableOrderedOnline &&
                attention.temporary_dtype == DType::F32 &&
                attention.output_dtype == DType::BF16,
                "Attention reduction contract mismatch");

        const PrecisionOperationContract modulation = policy.operation_contract(
            PrecisionOperation::Modulation, PrecisionSemantic::TemporaryCompute);
        const PrecisionOperationContract rope = policy.operation_contract(
            PrecisionOperation::Rope, PrecisionSemantic::TemporaryCompute);
        require(modulation.compute_dtype == DType::F32 &&
                modulation.output_dtype == DType::F32 &&
                rope.compute_dtype == DType::F32 &&
                rope.output_dtype == DType::F32,
                "FP32 temporary operation contracts are incomplete");

        std::cout << "precision_operation_contract_tests=PASS\n"
                  << "modulation_temporary=fp32\n"
                  << "linear_operand=bf16\n"
                  << "linear_accumulator=fp32\n"
                  << "linear_output=bf16\n"
                  << "non_tile_aligned_linear=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "precision operation contract tests: " << error.what() << '\n';
        return 1;
    }
}
