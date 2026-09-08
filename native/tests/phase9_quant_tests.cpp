#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

using vrhino::DType;
using vrhino::QuantType;
using vrhino::Tensor;

struct QuantWeight {
    std::vector<uint8_t> packed;
    Tensor tensor;
};

uint8_t fp8_code(int value) {
    // Exact E4M3FN encodings used by the deterministic test distribution.
    static const uint8_t codes[] = {0xc0, 0xbc, 0xb8, 0x00, 0x38, 0x3c, 0x40};
    return codes[value + 3];
}

QuantWeight make_weight(int64_t n, int64_t k, int64_t group_size, QuantType type) {
    const int64_t count = n * k;
    QuantWeight result;
    result.packed.assign(type == QuantType::INT4Symmetric ? (count + 1) / 2 : count, 0);
    for (int64_t index = 0; index < count; ++index) {
        const int value = static_cast<int>((index * 17 + index / std::max<int64_t>(1, k) * 3) % 7) - 3;
        if (type == QuantType::FP8E4M3FN) result.packed[index] = fp8_code(value);
        else if (type == QuantType::INT8Symmetric) result.packed[index] = static_cast<uint8_t>(static_cast<int8_t>(value));
        else {
            const uint8_t nibble = static_cast<uint8_t>(value) & 0xf;
            if (index & 1) result.packed[index / 2] |= static_cast<uint8_t>(nibble << 4);
            else result.packed[index / 2] = nibble;
        }
    }
    result.tensor = Tensor::borrowed(result.packed.data(), result.packed.size(), {n, k}, DType::U8);
    auto info = std::make_shared<vrhino::QuantizationInfo>();
    info->type = type;
    info->logical_dtype = DType::BF16;
    info->compute_dtype = DType::BF16;
    info->accumulation_dtype = DType::F32;
    info->group_size = group_size;
    info->axis = 1;
    info->symmetric = true;
    info->granularity = "per_group";
    info->zero_point_mode = "none";
    info->packing_layout = type == QuantType::FP8E4M3FN ? "byte_e4m3fn" :
        type == QuantType::INT8Symmetric ? "byte_twos_complement" :
        "nibble_low_first_twos_complement";
    const int64_t groups = (k + group_size - 1) / group_size;
    std::vector<float> scales(n * groups);
    for (int64_t row = 0; row < n; ++row) for (int64_t group = 0; group < groups; ++group)
        scales[row * groups + group] = 0.0078125f * static_cast<float>(1 + (row + group) % 5);
    info->scales = vrhino::host_f32({n, groups}, scales);
    result.tensor.set_quantization(info);
    return result;
}

Tensor make_activation(int64_t batch, int64_t m, int64_t k) {
    std::vector<float> values(batch * m * k);
    for (int64_t index = 0; index < static_cast<int64_t>(values.size()); ++index)
        values[index] = std::sin(static_cast<float>(index % 257) * 0.03125f) * 0.75f;
    return vrhino::host_f32(batch == 1 ? std::vector<int64_t>{m, k} :
                            std::vector<int64_t>{batch, m, k}, values);
}

struct ErrorMetrics { double maximum = 0.0; double mean = 0.0; double relative = 0.0; };

ErrorMetrics compare(vrhino::CudaBackend& true_backend, const Tensor& actual,
                     vrhino::CudaBackend& baseline_backend, const Tensor& expected) {
    Tensor actual_f32 = true_backend.cast(actual, DType::F32);
    Tensor expected_f32 = baseline_backend.cast(expected, DType::F32);
    Tensor ah = true_backend.copy_to_host(actual_f32);
    Tensor eh = baseline_backend.copy_to_host(expected_f32);
    ErrorMetrics metrics;
    double reference = 0.0;
    for (int64_t index = 0; index < ah.numel(); ++index) {
        const double difference = std::abs(static_cast<double>(ah.data_as<float>()[index]) -
                                           eh.data_as<float>()[index]);
        metrics.maximum = std::max(metrics.maximum, difference);
        metrics.mean += difference;
        reference += std::abs(static_cast<double>(eh.data_as<float>()[index]));
    }
    metrics.mean /= ah.numel();
    metrics.relative = metrics.mean / std::max(reference / ah.numel(), 1e-12);
    return metrics;
}

std::string quant_name(QuantType type) {
    if (type == QuantType::FP8E4M3FN) return "fp8";
    if (type == QuantType::INT8Symmetric) return "int8";
    return "int4";
}

void correctness() {
    struct Shape { int64_t batch, m, n, k, group; };
    const std::vector<Shape> shapes = {
        {1, 1, 1, 17, 32}, {2, 7, 19, 31, 32}, {1, 16, 16, 32, 32},
        {3, 23, 33, 65, 64}, {1, 129, 127, 130, 128}, {1, 257, 192, 256, 64},
    };
    double worst_maximum = 0.0, worst_mean = 0.0, worst_relative = 0.0;
    uint64_t true_calls = 0, fallback_calls = 0;
    for (QuantType type : {QuantType::FP8E4M3FN, QuantType::INT8Symmetric, QuantType::INT4Symmetric}) {
        for (const Shape& shape : shapes) {
            QuantWeight weight = make_weight(shape.n, shape.k, shape.group, type);
            Tensor activation = make_activation(shape.batch, shape.m, shape.k);
            std::vector<float> bias_values(shape.n);
            for (int64_t index = 0; index < shape.n; ++index) bias_values[index] = (index % 5 - 2) * 0.01f;
            Tensor bias = vrhino::host_f32({shape.n}, bias_values);
            vrhino::CudaBackend baseline, native;
            baseline.set_execution_dtype(DType::BF16);
            native.set_execution_dtype(DType::BF16);
            baseline.enable_true_quant_compute(false);
            Tensor expected = baseline.linear(activation, weight.tensor, &bias);
            Tensor actual = native.linear(activation, weight.tensor, &bias);
            baseline.synchronize(); native.synchronize();
            const ErrorMetrics metrics = compare(native, actual, baseline, expected);
            const auto stats = native.quant_compute_stats();
            const auto baseline_stats = baseline.quant_compute_stats();
            vrhino::require(stats.true_quant_calls == 1 && stats.fallback_calls == 0,
                            "true quant dispatch counter mismatch");
            vrhino::require(baseline_stats.true_quant_calls == 0 && baseline_stats.fallback_calls == 1,
                            "Phase 8 fallback counter mismatch");
            vrhino::require(metrics.maximum <= 0.125 && metrics.relative <= 0.035,
                            "quant kernel numerical regression exceeds Phase 9 gate");
            worst_maximum = std::max(worst_maximum, metrics.maximum);
            worst_mean = std::max(worst_mean, metrics.mean);
            worst_relative = std::max(worst_relative, metrics.relative);
            true_calls += stats.true_quant_calls;
            fallback_calls += stats.fallback_calls;
            std::cout << "CORRECTNESS format=" << quant_name(type)
                      << " batch=" << shape.batch << " m=" << shape.m << " n=" << shape.n
                      << " k=" << shape.k << " group=" << shape.group
                      << " max_abs=" << metrics.maximum << " mean_abs=" << metrics.mean
                      << " relative=" << metrics.relative << " dispatch=" << stats.last_dispatch << "\n";
        }
    }
    // Explicit unsupported-compute fallback audit.
    QuantWeight fallback_weight = make_weight(17, 33, 32, QuantType::INT8Symmetric);
    vrhino::CudaBackend fp32;
    fp32.set_execution_dtype(DType::F32);
    fp32.linear(make_activation(1, 3, 33), fallback_weight.tensor);
    fp32.synchronize();
    vrhino::require(fp32.quant_compute_stats().fallback_calls == 1,
                    "FP32 quant fallback was not observable");
    ++fallback_calls;
    std::cout << "SUMMARY cases=" << shapes.size() * 3 << " true_quant_calls=" << true_calls
              << " fallback_calls=" << fallback_calls << " worst_max_abs=" << worst_maximum
              << " worst_mean_abs=" << worst_mean << " worst_relative=" << worst_relative << "\n";
}

double timed(vrhino::CudaBackend& backend, const Tensor& activation, const Tensor& weight,
             int iterations) {
    backend.linear(activation, weight);
    backend.synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) backend.linear(activation, weight);
    backend.synchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
        iterations;
}

void benchmark() {
    const int64_t m = 1024, n = 2048, k = 2048;
    const int iterations = 12;
    Tensor activation = make_activation(1, m, k);
    for (QuantType type : {QuantType::FP8E4M3FN, QuantType::INT8Symmetric, QuantType::INT4Symmetric}) {
        const int64_t group = type == QuantType::INT4Symmetric ? 32 : 64;
        QuantWeight weight = make_weight(n, k, group, type);
        vrhino::CudaBackend baseline, native;
        baseline.set_execution_dtype(DType::BF16); native.set_execution_dtype(DType::BF16);
        baseline.enable_true_quant_compute(false);
        const double baseline_ms = timed(baseline, activation, weight.tensor, iterations);
        const double native_ms = timed(native, activation, weight.tensor, iterations);
        native.enable_profiling(true);
        native.linear(activation, weight.tensor); native.synchronize();
        const auto profiles = native.profile_stats();
        const auto stats = native.quant_compute_stats();
        double kernel_ms = 0.0;
        const auto found = profiles.find("quant_gemm." + quant_name(type));
        if (found != profiles.end()) kernel_ms = found->second.device_milliseconds;
        const double operations = 2.0 * m * n * k;
        const double tflops = kernel_ms > 0.0 ? operations / (kernel_ms * 1e9) : 0.0;
        std::cout << std::fixed << std::setprecision(6)
                  << "BENCHMARK format=" << quant_name(type) << " m=" << m << " n=" << n << " k=" << k
                  << " group=" << group << " baseline_ms=" << baseline_ms << " true_ms=" << native_ms
                  << " speedup=" << baseline_ms / native_ms << " kernel_ms=" << kernel_ms
                  << " effective_tflops=" << tflops << " true_quant_calls=" << stats.true_quant_calls
                  << " fallback_calls=" << stats.fallback_calls
                  << " cache_resident_bytes=" << native.weight_cache_resident_bytes()
                  << " peak_bytes=" << native.peak_device_bytes()
                  << " workspace_bytes=" << stats.workspace_bytes
                  << " scratch_peak_bytes=" << stats.scratch_peak_bytes
                  << " device_copy_bytes=" << stats.device_copy_bytes << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "correctness";
        if (mode == "correctness") correctness();
        else if (mode == "benchmark") benchmark();
        else vrhino::require(false, "usage: vrhino-phase9-quant-tests [correctness|benchmark]");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase9 quant tests: " << error.what() << "\n";
        return 1;
    }
}
