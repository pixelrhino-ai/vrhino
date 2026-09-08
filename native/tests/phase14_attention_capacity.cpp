#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "vrhino/backend/cuda_attention_config.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

int main() {
    try {
        using namespace vrhino::cuda_attention_config;
        constexpr int64_t kQueries = 2;
        constexpr int64_t kTokens = 32760;
        constexpr int64_t kHeads = 1;
        constexpr int64_t kWidth = 17;
        constexpr float kAtol = 5e-5f;
        constexpr float kRtol = 5e-5f;
        int default_shared = 0, optin_shared = 0;
        vrhino::require(cudaDeviceGetAttribute(&default_shared,
            cudaDevAttrMaxSharedMemoryPerBlock, 0) == cudaSuccess,
            "Cannot query default shared-memory capacity");
        vrhino::require(cudaDeviceGetAttribute(&optin_shared,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) == cudaSuccess,
            "Cannot query opt-in shared-memory capacity");
        std::vector<float> q(kQueries * kHeads * kWidth);
        std::vector<float> k(kTokens * kHeads * kWidth);
        std::vector<float> v(kTokens * kHeads * kWidth);
        for (size_t index = 0; index < q.size(); ++index)
            q[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 128.0f;
        for (size_t index = 0; index < k.size(); ++index) {
            k[index] = static_cast<float>(static_cast<int>((index * 17) % 67) - 33) / 256.0f;
            v[index] = static_cast<float>(static_cast<int>((index * 29) % 79) - 39) / 64.0f;
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(kWidth));
        std::vector<float> reference(q.size(), 0.0f);
        std::vector<float> scores(kTokens);
        for (int64_t query = 0; query < kQueries; ++query) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (int64_t key = 0; key < kTokens; ++key) {
                float score = 0.0f;
                for (int64_t dimension = 0; dimension < kWidth; ++dimension)
                    score += q[query * kWidth + dimension] *
                             k[key * kWidth + dimension];
                scores[key] = score * scale;
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int64_t dimension = 0; dimension < kWidth; ++dimension)
                for (int64_t key = 0; key < kTokens; ++key)
                    reference[query * kWidth + dimension] +=
                        scores[key] / denominator * v[key * kWidth + dimension];
        }

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        auto run = [&]() {
            vrhino::Tensor output = backend.attention(
                vrhino::host_f32({1, kQueries, kHeads, kWidth}, q),
                vrhino::host_f32({1, kTokens, kHeads, kWidth}, k),
                vrhino::host_f32({1, kTokens, kHeads, kWidth}, v),
                nullptr, false, 0.0f, nullptr, nullptr);
            return backend.copy_to_host(output);
        };
        const vrhino::Tensor first = run();
        const vrhino::Tensor second = run();

        double max_abs = 0.0, sum_abs = 0.0, reference_square = 0.0;
        double difference_square = 0.0, dot = 0.0, actual_square = 0.0;
        int64_t failing = 0, first_failing = -1, non_finite = 0;
        bool repeat_bit_exact = true;
        for (int64_t index = 0; index < first.numel(); ++index) {
            const float actual = first.data_as<float>()[index];
            const float expected = reference[index];
            const double difference = std::abs(static_cast<double>(actual) - expected);
            max_abs = std::max(max_abs, difference);
            sum_abs += difference;
            difference_square += difference * difference;
            reference_square += static_cast<double>(expected) * expected;
            actual_square += static_cast<double>(actual) * actual;
            dot += static_cast<double>(actual) * expected;
            if (!std::isfinite(actual)) ++non_finite;
            if (difference > kAtol + kRtol * std::abs(expected)) {
                if (first_failing < 0) first_failing = index;
                ++failing;
            }
            repeat_bit_exact = repeat_bit_exact &&
                std::memcmp(&actual, &second.data_as<float>()[index], sizeof(float)) == 0;
        }
        vrhino::require(failing == 0, "long-K Attention failed the fixed numerical gate");
        vrhino::require(non_finite == 0, "long-K Attention produced NaN/Inf");
        vrhino::require(repeat_bit_exact, "long-K Attention repeat was not bit exact");

        std::cout << std::setprecision(10)
                  << "status=PASS"
                  << "\nk_tokens=" << kTokens
                  << "\nq_tokens=" << kQueries
                  << "\nheads=" << kHeads
                  << "\nhead_dimension=" << kWidth
                  << "\nalgorithm=ordered_tiled_fp32"
                  << "\nkey_tile=" << kOrderedCandidateKeyTile
                  << "\ndynamic_shared_bytes=0"
                  << "\ntemporary_global_workspace_bytes="
                  << ordered_workspace_bytes(
                         1, kQueries, kHeads, kOrderedCandidateKeyTile)
                  << "\nkernel_dispatches_per_attention="
                  << ordered_dispatches_per_attention(
                         1, kTokens, kOrderedCandidateKeyTile)
                  << "\ndevice_default_shared_bytes=" << default_shared
                  << "\ndevice_optin_shared_bytes=" << optin_shared
                  << "\nmax_abs=" << max_abs
                  << "\nmean_abs=" << sum_abs / first.numel()
                  << "\nrelative_l2=" << std::sqrt(difference_square / reference_square)
                  << "\ncosine=" << dot / std::sqrt(actual_square * reference_square)
                  << "\nmixed_allclose=true"
                  << "\nfailing_element_count=" << failing
                  << "\nfirst_failing_index=" << first_failing
                  << "\nnan_inf_count=" << non_finite
                  << "\nrepeat_bit_exact=" << (repeat_bit_exact ? "true" : "false")
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "phase14 bounded attention: " << error.what() << '\n';
        return 1;
    }
}
