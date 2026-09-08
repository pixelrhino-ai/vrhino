#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {
struct Case {
    const char* name;
    int64_t queries;
    int64_t keys;
    int64_t heads;
    int64_t width;
    bool causal = false;
    bool additive_bias = false;
    bool boolean_mask = false;
};

vrhino::Tensor fp32_host(vrhino::CudaBackend& backend,
                         const vrhino::Tensor& tensor) {
    const vrhino::Tensor fp32 = tensor.dtype() == vrhino::DType::F32
        ? tensor : backend.cast(tensor, vrhino::DType::F32);
    return fp32.device().is_host() ? fp32 : backend.copy_to_host(fp32);
}

void run_case(const Case& spec) {
    using namespace vrhino;
    const std::vector<int64_t> q_shape = {1, spec.queries, spec.heads, spec.width};
    const std::vector<int64_t> k_shape = {1, spec.keys, spec.heads, spec.width};
    std::vector<float> q(static_cast<size_t>(shape_numel(q_shape)));
    std::vector<float> k(static_cast<size_t>(shape_numel(k_shape)));
    std::vector<float> v(k.size());
    for (size_t index = 0; index < q.size(); ++index)
        q[index] = static_cast<float>(static_cast<int>((index * 13) % 61) - 30) / 128.0f;
    for (size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int>((index * 17) % 67) - 33) / 256.0f;
        v[index] = static_cast<float>(static_cast<int>((index * 29) % 79) - 39) / 64.0f;
    }
    Tensor bias;
    Tensor mask;
    const Tensor* bias_pointer = nullptr;
    const Tensor* mask_pointer = nullptr;
    if (spec.additive_bias) {
        std::vector<float> values(static_cast<size_t>(spec.queries * spec.keys));
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 1024.0f;
        bias = host_f32({1, 1, spec.queries, spec.keys}, values);
        bias_pointer = &bias;
    }
    if (spec.boolean_mask) {
        std::vector<uint8_t> values(static_cast<size_t>(spec.queries * spec.keys), 1);
        for (size_t index = 0; index < values.size(); ++index)
            if (index % 11 == 0 && index % static_cast<size_t>(spec.keys) != 0)
                values[index] = 0;
        mask = host_bool({1, spec.queries, spec.keys}, values);
        mask_pointer = &mask;
    }

    CudaBackend backend;
    backend.set_execution_dtype(DType::BF16);
    const Tensor q_device = backend.copy_to_device(host_f32(q_shape, q), DType::BF16);
    const Tensor k_device = backend.copy_to_device(host_f32(k_shape, k), DType::BF16);
    const Tensor v_device = backend.copy_to_device(host_f32(k_shape, v), DType::BF16);

    setenv("VRHINO_CUDA_ATTENTION_VARIANT", "tensor2pass", 1);
    const Tensor reference = fp32_host(backend, backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f,
        bias_pointer));
    unsetenv("VRHINO_CUDA_ATTENTION_VARIANT");
    backend.enable_profiling(true);
    const Tensor first_device = backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f,
        bias_pointer);
    const Tensor second_device = backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f,
        bias_pointer);
    backend.synchronize();
    const auto profile = backend.profile_stats();
    const Tensor first_raw = backend.copy_to_host(first_device);
    const Tensor second_raw = backend.copy_to_host(second_device);
    const bool repeat_exact = std::memcmp(
        first_raw.data(), second_raw.data(), first_raw.bytes()) == 0;
    const Tensor actual = fp32_host(backend, first_device);

    double max_abs = 0.0, sum_abs = 0.0, difference_square = 0.0,
           reference_square = 0.0;
    int64_t failing = 0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double expected = reference.data_as<float>()[index];
        const double observed = actual.data_as<float>()[index];
        const double difference = std::abs(observed - expected);
        max_abs = std::max(max_abs, difference);
        sum_abs += difference;
        difference_square += difference * difference;
        reference_square += expected * expected;
        failing += difference > 5e-3 + 5e-3 * std::abs(expected);
        require(std::isfinite(observed), std::string(spec.name) + " NaN/Inf");
    }
    require(failing == 0, std::string(spec.name) + " mixed-allclose failure");
    require(repeat_exact, std::string(spec.name) + " repeat mismatch");
    bool sdpa_profile = false;
    for (const auto& [name, value] : profile) {
        (void)value;
        if (name.starts_with("attention.cudnn_sdpa")) sdpa_profile = true;
    }
    const long double score_elements = static_cast<long double>(spec.queries) *
        spec.keys * spec.heads;
    const bool expected_sdpa = !spec.boolean_mask &&
        score_elements >= 8.0L * 1024.0L * 1024.0L;
    require(sdpa_profile == expected_sdpa,
            std::string(spec.name) + " unexpected cuDNN selector result");
    std::cout << std::setprecision(10)
              << "case=" << spec.name
              << ",route=" << (sdpa_profile ? "cudnn_sdpa" : "fallback")
              << ",max_abs=" << max_abs
              << ",mean_abs=" << sum_abs / actual.numel()
              << ",relative_l2=" << std::sqrt(difference_square / reference_square)
              << ",failing=" << failing
              << ",repeat_exact=" << (repeat_exact ? "true" : "false") << '\n';
}
}  // namespace

int main() {
    try {
        run_case({"short", 17, 19, 2, 64});
        run_case({"medium", 512, 1024, 12, 128});
        run_case({"tail_causal_fallback", 128, 257, 4, 64, true});
        run_case({"causal_sdpa", 1024, 8192, 8, 64, true});
        run_case({"additive_bias_sdpa", 512, 16384, 8, 64, false, true});
        run_case({"boolean_mask_fallback", 64, 96, 4, 64, false, false, true});
        run_case({"stage_c_long", 32760, 32760, 12, 128});
        std::cout << "phase19c_cudnn_sdpa=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase19c cuDNN SDPA: " << error.what() << '\n';
        return 1;
    }
}
