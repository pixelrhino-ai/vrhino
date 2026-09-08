#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_attention_config.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

struct Metrics {
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double difference_square = 0.0;
    double reference_square = 0.0;
    double actual_square = 0.0;
    double dot = 0.0;
    int64_t failing = 0;
    int64_t first_failing = -1;
    int64_t non_finite = 0;
};

vrhino::Tensor host_fp32(vrhino::CudaBackend& backend, const vrhino::Tensor& value) {
    vrhino::Tensor fp32 = value.dtype() == vrhino::DType::F32
        ? value : backend.cast(value, vrhino::DType::F32);
    return fp32.device().is_host() ? fp32 : backend.copy_to_host(fp32);
}

Metrics compare(const vrhino::Tensor& reference, const vrhino::Tensor& actual,
                float atol, float rtol) {
    vrhino::require(reference.shape() == actual.shape() &&
                    reference.dtype() == vrhino::DType::F32 &&
                    actual.dtype() == vrhino::DType::F32,
                    "BF16 Attention comparison contract mismatch");
    Metrics result;
    for (int64_t index = 0; index < reference.numel(); ++index) {
        const double expected = reference.data_as<float>()[index];
        const double observed = actual.data_as<float>()[index];
        const double difference = std::abs(observed - expected);
        result.max_abs = std::max(result.max_abs, difference);
        result.sum_abs += difference;
        result.difference_square += difference * difference;
        result.reference_square += expected * expected;
        result.actual_square += observed * observed;
        result.dot += expected * observed;
        if (!std::isfinite(observed)) ++result.non_finite;
        if (difference > atol + rtol * std::abs(expected)) {
            if (result.first_failing < 0) result.first_failing = index;
            ++result.failing;
        }
    }
    return result;
}

}  // namespace

int main() {
    try {
        constexpr int64_t kBatch = 1;
        constexpr int64_t kQueries = 2;
        constexpr int64_t kTokens = 32760;
        constexpr int64_t kHeads = 1;
        constexpr int64_t kWidth = 128;
        constexpr float kAtol = 5e-3f;
        constexpr float kRtol = 5e-3f;
        const std::vector<int64_t> q_shape = {kBatch, kQueries, kHeads, kWidth};
        const std::vector<int64_t> kv_shape = {kBatch, kTokens, kHeads, kWidth};
        std::vector<float> q(static_cast<size_t>(vrhino::shape_numel(q_shape)));
        std::vector<float> k(static_cast<size_t>(vrhino::shape_numel(kv_shape)));
        std::vector<float> v(k.size());
        for (size_t index = 0; index < q.size(); ++index)
            q[index] = static_cast<float>(static_cast<int>((index * 13) % 61) - 30) / 128.0f;
        for (size_t index = 0; index < k.size(); ++index) {
            k[index] = static_cast<float>(static_cast<int>((index * 17) % 67) - 33) / 256.0f;
            v[index] = static_cast<float>(static_cast<int>((index * 29) % 79) - 39) / 64.0f;
        }

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        const vrhino::Tensor q_device = backend.copy_to_device(
            vrhino::host_f32(q_shape, q), vrhino::DType::BF16);
        const vrhino::Tensor k_device = backend.copy_to_device(
            vrhino::host_f32(kv_shape, k), vrhino::DType::BF16);
        const vrhino::Tensor v_device = backend.copy_to_device(
            vrhino::host_f32(kv_shape, v), vrhino::DType::BF16);

        setenv("VRHINO_CUDA_ATTENTION_VARIANT", "baseline", 1);
        const vrhino::Tensor baseline_device = backend.attention(
            q_device, k_device, v_device, nullptr, false, 0.0f);
        backend.synchronize();
        const vrhino::Tensor baseline = host_fp32(backend, baseline_device);

        setenv("VRHINO_CUDA_ATTENTION_VARIANT", "ordered", 1);
        const vrhino::Tensor ordered_first_device = backend.attention(
            q_device, k_device, v_device, nullptr, false, 0.0f);
        const vrhino::Tensor ordered_second_device = backend.attention(
            q_device, k_device, v_device, nullptr, false, 0.0f);
        backend.synchronize();
        const vrhino::Tensor ordered_first_bf16 = backend.copy_to_host(ordered_first_device);
        const vrhino::Tensor ordered_second_bf16 = backend.copy_to_host(ordered_second_device);
        const bool repeat_bit_exact = std::memcmp(
            ordered_first_bf16.data(), ordered_second_bf16.data(),
            ordered_first_bf16.bytes()) == 0;
        const vrhino::Tensor ordered = host_fp32(backend, ordered_first_device);
        unsetenv("VRHINO_CUDA_ATTENTION_VARIANT");

        const Metrics metrics = compare(baseline, ordered, kAtol, kRtol);
        vrhino::require(metrics.failing == 0,
                        "ordered BF16 long Attention failed fixed mixed gate");
        vrhino::require(metrics.non_finite == 0,
                        "ordered BF16 long Attention produced NaN/Inf");
        vrhino::require(repeat_bit_exact,
                        "ordered BF16 long Attention repeat was not bit exact");

        using namespace vrhino::cuda_attention_config;
        std::cout << std::setprecision(10)
                  << "status=PASS"
                  << "\nreference=phase7_streaming_bf16"
                  << "\ncandidate=ordered_tiled_bf16_qk_fp32_state"
                  << "\nq_tokens=" << kQueries
                  << "\nk_tokens=" << kTokens
                  << "\nheads=" << kHeads
                  << "\nhead_dimension=" << kWidth
                  << "\nkey_tile=" << kOrderedCandidateKeyTile
                  << "\ndynamic_shared_bytes=0"
                  << "\ntemporary_global_workspace_bytes="
                  << ordered_bf16_workspace_bytes(
                         kBatch, kQueries, kHeads, kWidth,
                         kOrderedCandidateKeyTile)
                  << "\nkernel_dispatches_per_attention="
                  << ordered_dispatches_per_attention(
                         kBatch, kTokens, kOrderedCandidateKeyTile)
                  << "\nmax_abs=" << metrics.max_abs
                  << "\nmean_abs=" << metrics.sum_abs / ordered.numel()
                  << "\nrelative_l2="
                  << std::sqrt(metrics.difference_square / metrics.reference_square)
                  << "\ncosine="
                  << metrics.dot / std::sqrt(metrics.actual_square * metrics.reference_square)
                  << "\nmixed_allclose=true"
                  << "\nfailing_element_count=" << metrics.failing
                  << "\nfirst_failing_index=" << metrics.first_failing
                  << "\nnan_inf_count=" << metrics.non_finite
                  << "\nrepeat_bit_exact=" << (repeat_bit_exact ? "true" : "false")
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase16r BF16 Attention: " << error.what() << '\n';
        return 1;
    }
}
