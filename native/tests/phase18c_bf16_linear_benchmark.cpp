#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

struct Result {
    vrhino::Tensor output;
    double milliseconds = 0.0;
    std::string profile_name;
};

Result run(const vrhino::Tensor& input, const vrhino::Tensor& weight,
           const vrhino::Tensor& bias, vrhino::DType dtype) {
    using namespace vrhino;
    CudaBackend backend;
    backend.set_execution_dtype(dtype);
    const Tensor input_device = backend.copy_to_device(input, dtype);
    const Tensor weight_device = backend.copy_to_device(weight, dtype);
    const Tensor bias_device = backend.copy_to_device(bias, dtype);
    backend.enable_profiling(true);
    (void)backend.linear(input_device, weight_device, &bias_device, dtype);
    (void)backend.profile_stats();
    Tensor final;
    constexpr int kRuns = 5;
    for (int index = 0; index < kRuns; ++index)
        final = backend.linear(input_device, weight_device, &bias_device, dtype);
    double total = 0.0;
    std::string profile_name;
    for (const auto& [name, stat] : backend.profile_stats()) {
        if (!name.starts_with("linear|")) continue;
        total += stat.device_milliseconds;
        profile_name = name;
    }
    require(!profile_name.empty(), "BF16 Linear benchmark profile missing");
    Tensor host = backend.copy_to_host(final);
    return {std::move(host), total / kRuns, std::move(profile_name)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        using namespace vrhino;
        require(argc == 4, "usage: vrhino-phase18c-bf16-linear-benchmark M N K");
        const int64_t m = std::strtoll(argv[1], nullptr, 10);
        const int64_t n = std::strtoll(argv[2], nullptr, 10);
        const int64_t k = std::strtoll(argv[3], nullptr, 10);
        require(m > 0 && n > 0 && k > 0, "BF16 Linear dimensions must be positive");
        std::vector<float> input_values(static_cast<size_t>(m * k));
        std::vector<float> weight_values(static_cast<size_t>(n * k));
        std::vector<float> bias_values(static_cast<size_t>(n));
        for (size_t index = 0; index < input_values.size(); ++index)
            input_values[index] = static_cast<float>(static_cast<int>((index * 13) % 53) - 26) / 128.0f;
        for (size_t index = 0; index < weight_values.size(); ++index)
            weight_values[index] = static_cast<float>(static_cast<int>((index * 17) % 59) - 29) / 256.0f;
        for (size_t index = 0; index < bias_values.size(); ++index)
            bias_values[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 512.0f;
        const Tensor input = host_f32({m, k}, input_values);
        const Tensor weight = host_f32({n, k}, weight_values);
        const Tensor bias = host_f32({n}, bias_values);
        const Result fp32 = run(input, weight, bias, DType::F32);
        const Result optimized = run(input, weight, bias, DType::BF16);
        require(optimized.output.dtype() == DType::BF16,
                "BF16 Linear output dtype mismatch");
        std::cout << std::setprecision(10)
                  << "status=PASS\nm=" << m << "\nn=" << n << "\nk=" << k
                  << "\noptimized_mean_device_ms=" << optimized.milliseconds
                  << "\nfp32_mean_device_ms=" << fp32.milliseconds
                  << "\noptimized_vs_fp32_ratio=" << optimized.milliseconds / fp32.milliseconds
                  << "\noptimized_profile=" << optimized.profile_name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "phase18c BF16 Linear benchmark: " << error.what() << '\n';
        return 1;
    }
}
