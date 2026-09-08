#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/backend/cudnn_sdpa.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

struct Case {
    const char* name;
    int64_t batch;
    int64_t queries;
    int64_t keys;
    int64_t heads;
    int64_t width;
    std::vector<int32_t> key_lengths;
    bool additive_bias = false;
    bool arbitrary = false;
    bool expect_sdpa = true;
    bool causal = false;
};

vrhino::Tensor as_fp32_host(vrhino::CudaBackend& backend,
                            const vrhino::Tensor& value) {
    const auto converted = value.dtype() == vrhino::DType::F32
        ? value : backend.cast(value, vrhino::DType::F32);
    return converted.device().is_host() ? converted : backend.copy_to_host(converted);
}

void run_case(const Case& spec) {
    using namespace vrhino;
    const std::vector<int64_t> q_shape = {
        spec.batch, spec.queries, spec.heads, spec.width};
    const std::vector<int64_t> k_shape = {
        spec.batch, spec.keys, spec.heads, spec.width};
    std::vector<float> q(static_cast<size_t>(shape_numel(q_shape)));
    std::vector<float> k(static_cast<size_t>(shape_numel(k_shape)));
    std::vector<float> v(k.size());
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = static_cast<float>(static_cast<int>((i * 13) % 61) - 30) / 128.0f;
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = static_cast<float>(static_cast<int>((i * 17) % 67) - 33) / 256.0f;
        v[i] = static_cast<float>(static_cast<int>((i * 29) % 79) - 39) / 64.0f;
    }
    std::vector<uint8_t> mask_values(
        static_cast<size_t>(spec.batch * spec.keys), 0);
    for (int64_t b = 0; b < spec.batch; ++b) {
        for (int64_t key = 0; key < spec.key_lengths.at(static_cast<size_t>(b)); ++key)
            mask_values[static_cast<size_t>(b * spec.keys + key)] = 1;
    }
    if (spec.arbitrary && spec.keys > 4) {
        mask_values[2] = 0;
        mask_values[3] = 1;
    }
    const Tensor mask = host_bool({spec.batch, 1, spec.keys}, mask_values);
    const PrefixMaskCanonicalization canonical = canonicalize_boolean_prefix_mask(
        mask, spec.batch, spec.queries, spec.keys);
    require(canonical.representable == !spec.arbitrary,
            std::string(spec.name) + " recognizer verdict mismatch");

    Tensor bias;
    const Tensor* bias_pointer = nullptr;
    if (spec.additive_bias) {
        std::vector<float> values(static_cast<size_t>(spec.queries * spec.keys));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 2048.0f;
        bias = host_f32({1, 1, spec.queries, spec.keys}, values);
        bias_pointer = &bias;
    }

    CudaBackend backend;
    backend.set_execution_dtype(DType::BF16);
    const Tensor q_device = backend.copy_to_device(host_f32(q_shape, q), DType::BF16);
    const Tensor k_device = backend.copy_to_device(host_f32(k_shape, k), DType::BF16);
    const Tensor v_device = backend.copy_to_device(host_f32(k_shape, v), DType::BF16);

    setenv("VRHINO_CUDA_ATTENTION_VARIANT", "baseline", 1);
    const Tensor reference = as_fp32_host(backend, backend.attention(
        q_device, k_device, v_device, &mask, spec.causal, 0.0f, bias_pointer));
    unsetenv("VRHINO_CUDA_ATTENTION_VARIANT");
    setenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION", "force", 1);
    backend.enable_profiling(true);
    const Tensor first_device = backend.attention(
        q_device, k_device, v_device, &mask, spec.causal, 0.0f, bias_pointer);
    const Tensor second_device = backend.attention(
        q_device, k_device, v_device, &mask, spec.causal, 0.0f, bias_pointer);
    const auto profile = backend.profile_stats();
    unsetenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION");
    const Tensor first_raw = backend.copy_to_host(first_device);
    const Tensor second_raw = backend.copy_to_host(second_device);
    require(std::memcmp(first_raw.data(), second_raw.data(), first_raw.bytes()) == 0,
            std::string(spec.name) + " deterministic repeat mismatch");
    const Tensor actual = as_fp32_host(backend, first_device);

    bool selected = false;
    for (const auto& [name, stat] : profile) {
        (void)stat;
        selected = selected || name.starts_with("attention.cudnn_sdpa");
    }
    require(selected == spec.expect_sdpa,
            std::string(spec.name) + " route mismatch");
    double max_abs = 0.0, sum_abs = 0.0, diff_sq = 0.0, ref_sq = 0.0;
    int64_t failures = 0;
    for (int64_t i = 0; i < actual.numel(); ++i) {
        const double expected = reference.data_as<float>()[i];
        const double observed = actual.data_as<float>()[i];
        require(std::isfinite(expected) && std::isfinite(observed),
                std::string(spec.name) + " NaN/Inf");
        const double difference = std::abs(observed - expected);
        max_abs = std::max(max_abs, difference);
        sum_abs += difference;
        diff_sq += difference * difference;
        ref_sq += expected * expected;
        failures += difference > 5e-3 + 5e-3 * std::abs(expected);
    }
    require(failures == 0, std::string(spec.name) + " mixed-allclose failure");
    std::cout << std::setprecision(10)
              << "case=" << spec.name
              << ",route=" << (selected ? "cudnn_sdpa" : "fallback")
              << ",max_abs=" << max_abs
              << ",mean_abs=" << sum_abs / actual.numel()
              << ",relative_l2=" << std::sqrt(diff_sq / ref_sq)
              << ",failing=" << failures << '\n';
}

void query_padding_is_fail_closed() {
    using namespace vrhino;
    constexpr int64_t batch = 2, queries = 5, keys = 7;
    std::vector<uint8_t> values(static_cast<size_t>(batch * queries * keys), 0);
    const int64_t query_lengths[batch] = {5, 3};
    const int64_t key_lengths[batch] = {4, 6};
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t q = 0; q < query_lengths[b]; ++q)
            for (int64_t k = 0; k < key_lengths[b]; ++k)
                values[static_cast<size_t>((b * queries + q) * keys + k)] = 1;
    const auto result = canonicalize_boolean_prefix_mask(
        host_bool({batch, queries, keys}, values), batch, queries, keys);
    require(!result.representable,
            "query-padding output semantics must remain fail-closed");
}

}  // namespace

int main() {
    try {
        run_case({"all_valid", 2, 31, 47, 4, 64, {47, 47}});
        run_case({"different_batch_lengths", 2, 97, 193, 4, 64, {193, 127}});
        run_case({"minimum_valid_length", 2, 33, 65, 2, 64, {1, 37}});
        run_case({"tail_with_additive_bias", 2, 35, 71, 2, 64,
                  {57, 63}, true});
        run_case({"arbitrary_pattern", 1, 32, 67, 2, 64, {51},
                  false, true, false});
        run_case({"causal_prefix", 2, 67, 67, 2, 64, {67, 53},
                  false, false, true, true});
        query_padding_is_fail_closed();
        std::cout << "phase20d_prefix_mask_sdpa=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20d prefix-mask SDPA: " << error.what() << '\n';
        return 1;
    }
}
