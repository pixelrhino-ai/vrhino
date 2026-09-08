#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"

namespace {

struct Case {
    const char* name;
    int64_t batch;
    int64_t queries;
    int64_t keys;
    int64_t heads;
    int64_t width;
    bool prefix_mask = false;
};

struct Timing {
    double cold_ms = 0.0;
    double warm_ms = 0.0;
    double device_ms = 0.0;
    size_t peak_bytes = 0;
};

Timing measure(const Case& spec, bool sdpa) {
    using namespace vrhino;
    CudaBackend backend;
    backend.set_execution_dtype(DType::BF16);
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
    const Tensor q_device = backend.copy_to_device(host_f32(q_shape, q), DType::BF16);
    const Tensor k_device = backend.copy_to_device(host_f32(k_shape, k), DType::BF16);
    const Tensor v_device = backend.copy_to_device(host_f32(k_shape, v), DType::BF16);
    Tensor mask;
    const Tensor* mask_pointer = nullptr;
    if (spec.prefix_mask) {
        std::vector<uint8_t> values(static_cast<size_t>(spec.batch * spec.keys), 0);
        for (int64_t b = 0; b < spec.batch; ++b) {
            const int64_t length = std::max<int64_t>(1, spec.keys - 7 - 11 * b);
            std::fill_n(values.begin() + b * spec.keys, length, uint8_t{1});
        }
        mask = host_bool({spec.batch, 1, spec.keys}, values);
        mask_pointer = &mask;
    }
    if (sdpa)
        setenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION", "force", 1);
    else
        setenv("VRHINO_CUDA_ATTENTION_VARIANT", "baseline", 1);
    const auto invoke = [&]() {
        const auto start = std::chrono::steady_clock::now();
        (void)backend.attention(q_device, k_device, v_device, mask_pointer,
                                false, 0.0f);
        backend.synchronize();
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };
    Timing result;
    result.cold_ms = invoke();
    std::vector<double> samples;
    for (int repeat = 0; repeat < 3; ++repeat) samples.push_back(invoke());
    std::sort(samples.begin(), samples.end());
    result.warm_ms = samples[1];
    backend.enable_profiling(true);
    (void)backend.attention(q_device, k_device, v_device, mask_pointer,
                            false, 0.0f);
    const auto profile = backend.profile_stats();
    for (const auto& [name, stat] : profile)
        if (name.starts_with("attention|") || name == "attention")
            result.device_ms += stat.device_milliseconds;
    result.peak_bytes = backend.peak_device_bytes();
    unsetenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION");
    unsetenv("VRHINO_CUDA_ATTENTION_VARIANT");
    return result;
}

void run(const Case& spec) {
    const Timing bounded = measure(spec, false);
    const Timing sdpa = measure(spec, true);
    const long double scores = static_cast<long double>(spec.batch) *
        spec.queries * spec.keys * spec.heads;
    std::cout << std::setprecision(10)
              << "case=" << spec.name
              << ",score_elements=" << static_cast<uint64_t>(scores)
              << ",heads=" << spec.heads
              << ",width=" << spec.width
              << ",q=" << spec.queries
              << ",k=" << spec.keys
              << ",mask=" << (spec.prefix_mask ? "prefix" : "none")
              << ",bounded_warm_ms=" << bounded.warm_ms
              << ",sdpa_warm_ms=" << sdpa.warm_ms
              << ",sdpa_over_bounded=" << sdpa.warm_ms / bounded.warm_ms
              << ",bounded_cold_ms=" << bounded.cold_ms
              << ",sdpa_cold_ms=" << sdpa.cold_ms
              << ",bounded_device_ms=" << bounded.device_ms
              << ",sdpa_device_ms=" << sdpa.device_ms
              << ",bounded_peak_bytes=" << bounded.peak_bytes
              << ",sdpa_peak_bytes=" << sdpa.peak_bytes << '\n';
}

}  // namespace

int main() {
    run({"8m", 1, 1024, 1024, 8, 64});
    run({"8m_prefix", 2, 1024, 128, 32, 64, true});
    run({"16m", 1, 2048, 1024, 8, 128});
    run({"24m", 1, 1536, 2048, 8, 64});
    run({"24m_prefix", 2, 3072, 128, 32, 64, true});
    run({"32m", 1, 2048, 2048, 8, 128});
    run({"ltx_43m_prefix", 2, 5280, 128, 32, 64, true});
    run({"48m", 1, 3072, 2048, 8, 64});
    run({"64m", 1, 4096, 2048, 8, 128});
    run({"96m", 1, 4096, 3072, 8, 64});
    run({"128m", 1, 4096, 4096, 8, 128});
    return 0;
}
