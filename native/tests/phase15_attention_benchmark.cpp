#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "vrhino/backend/cuda_attention_config.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor.h"

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 1 || argc == 2 || argc == 6,
                        "usage: vrhino-phase15-attention-benchmark [fp32|bf16] "
                        "[q_tokens k_tokens heads head_dimension]");
        const std::string dtype_name = argc == 2 ? argv[1] : "fp32";
        const std::string selected_dtype = argc == 6 ? argv[1] : dtype_name;
        vrhino::require(selected_dtype == "fp32" || selected_dtype == "bf16",
                        "Attention benchmark dtype must be fp32 or bf16");
        const vrhino::DType dtype = selected_dtype == "bf16"
            ? vrhino::DType::BF16 : vrhino::DType::F32;
        const int64_t q_tokens = argc == 6 ? std::strtoll(argv[2], nullptr, 10) : 32760;
        const int64_t k_tokens = argc == 6 ? std::strtoll(argv[3], nullptr, 10) : 32760;
        const int64_t heads = argc == 6 ? std::strtoll(argv[4], nullptr, 10) : 12;
        const int64_t width = argc == 6 ? std::strtoll(argv[5], nullptr, 10) : 128;
        vrhino::require(q_tokens > 0 && k_tokens > 0 && heads > 0 && width > 0,
                        "Attention benchmark dimensions must be positive");
        constexpr int kMeasuredRuns = 3;
        const std::vector<int64_t> q_shape = {1, q_tokens, heads, width};
        const std::vector<int64_t> kv_shape = {1, k_tokens, heads, width};
        vrhino::Tensor q = vrhino::Tensor::host(q_shape, vrhino::DType::F32);
        vrhino::Tensor k = vrhino::Tensor::host(kv_shape, vrhino::DType::F32);
        vrhino::Tensor v = vrhino::Tensor::host(kv_shape, vrhino::DType::F32);
        for (int64_t index = 0; index < q.numel(); ++index) {
            q.data_as<float>()[index] =
                static_cast<float>(index % 29 - 14) / 256.0f;
        }
        for (int64_t index = 0; index < k.numel(); ++index) {
            k.data_as<float>()[index] =
                static_cast<float>((index * 17) % 31 - 15) / 256.0f;
            v.data_as<float>()[index] =
                static_cast<float>((index * 23) % 37 - 18) / 64.0f;
        }

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(dtype);
        const vrhino::Tensor q_device = backend.copy_to_device(q, dtype);
        const vrhino::Tensor k_device = backend.copy_to_device(k, dtype);
        const vrhino::Tensor v_device = backend.copy_to_device(v, dtype);
        backend.enable_profiling(true);
        (void)backend.attention(q_device, k_device, v_device, nullptr, false, 0.0f);
        (void)backend.profile_stats();
        for (int run = 0; run < kMeasuredRuns; ++run)
            (void)backend.attention(q_device, k_device, v_device, nullptr, false, 0.0f);

        double milliseconds = 0.0;
        uint64_t calls = 0;
        double p50 = 0.0, p95 = 0.0;
        bool cudnn_sdpa = false;
        for (const auto& [name, stat] : backend.profile_stats()) {
            if (name.starts_with("attention.cudnn_sdpa")) cudnn_sdpa = true;
            // Nested route telemetry is diagnostic and must not be added to
            // the outer Attention critical-path event.
            if (name != "attention") continue;
            milliseconds += stat.device_milliseconds;
            calls += stat.calls;
            p50 = stat.p50_milliseconds;
            p95 = stat.p95_milliseconds;
        }
        vrhino::require(calls == kMeasuredRuns,
                        "production-shape Attention profile call count mismatch");
        const char* variant = std::getenv("VRHINO_CUDA_ATTENTION_VARIANT");
        const char* bf16_qk = std::getenv("VRHINO_CUDA_ATTENTION_BF16_QK");
        const bool tensor_two_pass = variant &&
            std::strcmp(variant, "tensor2pass") == 0;
        const bool baseline = variant
            ? std::strcmp(variant, "baseline") == 0
            : dtype == vrhino::DType::BF16 && !bf16_qk;
        const bool tensor_qk = dtype == vrhino::DType::BF16 && bf16_qk &&
                               std::strcmp(bf16_qk, "tensor") == 0;
        const int64_t rows = q_tokens * heads;
        const size_t ordered_state =
            vrhino::cuda_attention_config::ordered_workspace_bytes(
                1, q_tokens, heads,
                vrhino::cuda_attention_config::kOrderedCandidateKeyTile);
        const size_t global_workspace = tensor_two_pass
            ? vrhino::cuda_attention_config::bf16_tensor_two_pass_workspace_bytes(
                1, q_tokens, heads, width,
                vrhino::cuda_attention_config::kBf16TensorKeyTile)
            : baseline ? 0 : (
            dtype == vrhino::DType::BF16
                ? ordered_state + static_cast<size_t>(rows * width) * sizeof(float) *
                      (tensor_qk ? 1ULL : 3ULL)
                : ordered_state);
        std::cout << std::setprecision(10)
                  << "status=PASS"
                  << "\nq_tokens=" << q_tokens
                  << "\nk_tokens=" << k_tokens
                  << "\nheads=" << heads
                  << "\nhead_dimension=" << width
                  << "\nexecution_dtype=" << vrhino::dtype_name(dtype)
                  << "\nmeasured_calls=" << calls
                  << "\ntotal_device_ms=" << milliseconds
                  << "\nmean_device_ms=" << milliseconds / calls
                  << "\np50_device_ms=" << p50
                  << "\np95_device_ms=" << p95
                  << "\nalgorithm=" << (cudnn_sdpa ? "cudnn_sdpa" :
                      (tensor_two_pass ? "tensor_two_pass" :
                      (baseline ? "streaming" : "ordered_tiled"))
                      )
                      << '_' << selected_dtype
                  << (tensor_qk ? "_tensor_qk" : "")
                  << "\nkey_tile="
                  << vrhino::cuda_attention_config::kOrderedCandidateKeyTile
                  << "\ndynamic_shared_bytes=" << (baseline
                      ? vrhino::cuda_attention_config::workspace_bytes(width) : 0)
                  << "\ntemporary_global_workspace_bytes=" << global_workspace
                  << "\nkernel_dispatches_per_attention="
                  << (cudnn_sdpa ? 0 : tensor_two_pass
                      ? vrhino::cuda_attention_config::bf16_tensor_two_pass_dispatches_per_attention(
                            1, k_tokens,
                            vrhino::cuda_attention_config::kBf16TensorKeyTile)
                      : baseline ? 1
                      : vrhino::cuda_attention_config::ordered_dispatches_per_attention(
                            1, k_tokens,
                            vrhino::cuda_attention_config::kOrderedCandidateKeyTile))
                  << "\n";
    } catch (const std::exception& error) {
        std::cerr << "phase15 Attention benchmark: " << error.what() << '\n';
        return 1;
    }
}
