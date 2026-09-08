#include <algorithm>
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
    bool mask_and_bias = false;
};

vrhino::Tensor host_fp32(vrhino::CudaBackend& backend, const vrhino::Tensor& value) {
    const vrhino::Tensor fp32 = value.dtype() == vrhino::DType::F32
        ? value : backend.cast(value, vrhino::DType::F32);
    return fp32.device().is_host() ? fp32 : backend.copy_to_host(fp32);
}

void run_case(const Case& spec) {
    using namespace vrhino;
    const std::vector<int64_t> q_shape = {1, spec.queries, spec.heads, spec.width};
    const std::vector<int64_t> kv_shape = {1, spec.keys, spec.heads, spec.width};
    std::vector<float> q(static_cast<size_t>(shape_numel(q_shape)));
    std::vector<float> k(static_cast<size_t>(shape_numel(kv_shape)));
    std::vector<float> v(k.size());
    for (size_t index = 0; index < q.size(); ++index)
        q[index] = static_cast<float>(static_cast<int>((index * 13) % 61) - 30) / 128.0f;
    for (size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int>((index * 17) % 67) - 33) / 256.0f;
        v[index] = static_cast<float>(static_cast<int>((index * 29) % 79) - 39) / 64.0f;
    }
    Tensor mask;
    Tensor bias;
    const Tensor* mask_pointer = nullptr;
    const Tensor* bias_pointer = nullptr;
    if (spec.mask_and_bias) {
        std::vector<uint8_t> mask_values(static_cast<size_t>(spec.queries * spec.keys), 1);
        for (size_t index = 0; index < mask_values.size(); ++index)
            if (index % 11 == 0 && index % static_cast<size_t>(spec.keys) != 0)
                mask_values[index] = 0;
        std::vector<float> bias_values(static_cast<size_t>(spec.queries * spec.keys));
        for (size_t index = 0; index < bias_values.size(); ++index)
            bias_values[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 1024.0f;
        mask = host_bool({1, spec.queries, spec.keys}, mask_values);
        bias = host_f32({1, 1, spec.queries, spec.keys}, bias_values);
        mask_pointer = &mask;
        bias_pointer = &bias;
    }

    CudaBackend backend;
    backend.set_execution_dtype(DType::BF16);
    const Tensor q_device = backend.copy_to_device(host_f32(q_shape, q), DType::BF16);
    const Tensor k_device = backend.copy_to_device(host_f32(kv_shape, k), DType::BF16);
    const Tensor v_device = backend.copy_to_device(host_f32(kv_shape, v), DType::BF16);
    setenv("VRHINO_CUDA_ATTENTION_VARIANT", "baseline", 1);
    const Tensor reference = host_fp32(backend, backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f, bias_pointer));
    setenv("VRHINO_CUDA_ATTENTION_VARIANT", "tensor2pass", 1);
    const Tensor first_device = backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f, bias_pointer);
    const Tensor second_device = backend.attention(
        q_device, k_device, v_device, mask_pointer, spec.causal, 0.0f, bias_pointer);
    backend.synchronize();
    const Tensor first_raw = backend.copy_to_host(first_device);
    const Tensor second_raw = backend.copy_to_host(second_device);
    const bool repeat_exact = std::memcmp(
        first_raw.data(), second_raw.data(), first_raw.bytes()) == 0;
    const Tensor actual = host_fp32(backend, first_device);
    unsetenv("VRHINO_CUDA_ATTENTION_VARIANT");

    double max_abs = 0.0, sum_abs = 0.0, diff_square = 0.0, ref_square = 0.0;
    int64_t failing = 0, non_finite = 0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double expected = reference.data_as<float>()[index];
        const double observed = actual.data_as<float>()[index];
        const double difference = std::abs(observed - expected);
        max_abs = std::max(max_abs, difference);
        sum_abs += difference;
        diff_square += difference * difference;
        ref_square += expected * expected;
        failing += difference > 5e-3 + 5e-3 * std::abs(expected);
        non_finite += !std::isfinite(observed);
    }
    require(failing == 0, std::string(spec.name) + " mixed-allclose failure");
    require(non_finite == 0, std::string(spec.name) + " NaN/Inf");
    require(repeat_exact, std::string(spec.name) + " repeat mismatch");
    std::cout << std::setprecision(10)
              << "case=" << spec.name
              << ",max_abs=" << max_abs
              << ",mean_abs=" << sum_abs / actual.numel()
              << ",relative_l2=" << std::sqrt(diff_square / ref_square)
              << ",failing=" << failing
              << ",repeat_exact=" << (repeat_exact ? "true" : "false") << '\n';
}

}  // namespace

int main() {
    try {
        run_case({"short", 17, 19, 2, 64});
        run_case({"medium", 512, 1024, 12, 128});
        run_case({"long_k", 16, 32760, 12, 128});
        run_case({"tail_mask_bias_causal", 128, 257, 4, 64, true, true});
        std::cout << "phase18c_bf16_attention=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase18c BF16 Attention: " << error.what() << '\n';
        return 1;
    }
}
