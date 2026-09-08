#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/memory.h"
#include "vrhino/tensor_util.h"

namespace {

using vrhino::DType;
using vrhino::QuantType;
using vrhino::Tensor;

struct Weight {
    std::vector<uint8_t> packed;
    Tensor tensor;
};

Weight weight(int64_t n, int64_t k, int seed) {
    Weight result;
    result.packed.resize(n * k);
    for (int64_t index = 0; index < n * k; ++index)
        result.packed[index] = static_cast<uint8_t>(static_cast<int8_t>((index * 13 + seed) % 15 - 7));
    result.tensor = Tensor::borrowed(result.packed.data(), result.packed.size(), {n, k}, DType::U8);
    auto info = std::make_shared<vrhino::QuantizationInfo>();
    info->type = QuantType::INT8Symmetric;
    info->logical_dtype = DType::BF16;
    info->compute_dtype = DType::BF16;
    info->accumulation_dtype = DType::F32;
    info->group_size = 64;
    info->axis = 1;
    info->symmetric = true;
    info->granularity = "per_group";
    info->zero_point_mode = "none";
    info->packing_layout = "byte_twos_complement";
    std::vector<float> scales(n * (k / 64));
    for (size_t index = 0; index < scales.size(); ++index) scales[index] = 0.01f * (1 + index % 3);
    info->scales = vrhino::host_f32({n, k / 64}, scales);
    result.tensor.set_quantization(info);
    return result;
}

Tensor activation(int64_t m, int64_t k) {
    std::vector<float> values(m * k);
    for (int64_t index = 0; index < m * k; ++index) values[index] = std::sin(index * 0.01f);
    return vrhino::host_f32({m, k}, values);
}

Tensor run(vrhino::CudaBackend& backend, const Tensor& x, const std::vector<Weight*>& weights) {
    Tensor output;
    for (int repeat = 0; repeat < 2; ++repeat)
        for (Weight* item : weights) output = backend.linear(x, item->tensor);
    backend.synchronize();
    return output;
}

std::vector<float> host_values(vrhino::CudaBackend& backend, const Tensor& tensor) {
    Tensor fp32 = backend.cast(tensor, DType::F32);
    Tensor host = backend.copy_to_host(fp32);
    return {host.data_as<float>(), host.data_as<float>() + host.numel()};
}

void check_close(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("memory output size mismatch");
    for (size_t index = 0; index < a.size(); ++index)
        if (a[index] != b[index]) throw std::runtime_error("memory runtime changed numerical output");
}

}  // namespace

int main() {
    try {
        constexpr size_t MiB = 1024 * 1024;
        Weight w1 = weight(1024, 1024, 1), w2 = weight(1024, 1024, 2),
               w3 = weight(1024, 1024, 3);
        std::vector<Weight*> weights = {&w1, &w2, &w3};
        Tensor x = activation(256, 1024);

        vrhino::MemoryRuntimeOptions high_options;
        high_options.enabled = true;
        high_options.prefetch = false;
        vrhino::CudaBackend high;
        high.set_execution_dtype(DType::BF16);
        high.configure_memory_runtime({64 * MiB, 8 * MiB, 128 * MiB, 8 * MiB, 8 * MiB}, high_options);
        high.set_vrm_mapped_bytes(4 * MiB);
        high.begin_memory_trace();
        Tensor expected = run(high, x, weights);
        std::vector<vrhino::MemoryAccess> trace = high.end_memory_trace();
        std::vector<float> expected_values = host_values(high, expected);

        vrhino::MemoryRuntimeOptions options;
        options.enabled = true;
        options.prefetch = true;
        options.host_staging = true;
        options.prefetch_lookahead = 1;
        options.eviction_policy = vrhino::EvictionPolicy::NextUse;
        vrhino::CudaBackend budgeted;
        budgeted.set_execution_dtype(DType::BF16);
        // 3 MiB weight capacity: two 1.06 MiB packed+scale entries fit, all
        // three do not. This forces generic eviction while retaining lookahead.
        budgeted.configure_memory_runtime({5 * MiB, 3 * MiB, 128 * MiB, 1 * MiB, 1 * MiB}, options);
        budgeted.set_vrm_mapped_bytes(4 * MiB);
        budgeted.set_memory_trace(std::move(trace));
        Tensor actual = run(budgeted, x, weights);
        check_close(expected_values, host_values(budgeted, actual));
        const auto stats = budgeted.memory_runtime_stats();
        if (stats.evictions == 0 || stats.prefetch_requests == 0 || stats.prefetch_hits == 0)
            throw std::runtime_error("eviction/prefetch path did not execute");
        if (stats.accounting.peak_device_resident_weight_bytes > 3 * MiB)
            throw std::runtime_error("GPU weight budget exceeded");
        if (stats.accounting.peak_host_staging_bytes > 3 * MiB)
            throw std::runtime_error("CPU pinned budget exceeded");
        if (stats.forced_syncs != 0)
            throw std::runtime_error("unexpected forced synchronization");
        std::cout << "phase10 CUDA memory correctness=pass"
                  << " evictions=" << stats.evictions
                  << " prefetch_requests=" << stats.prefetch_requests
                  << " prefetch_hits=" << stats.prefetch_hits
                  << " overlap_ratio=" << stats.overlap_ratio()
                  << " peak_gpu_weight_bytes=" << stats.accounting.peak_device_resident_weight_bytes
                  << " peak_pinned_bytes=" << stats.accounting.peak_host_staging_bytes << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase10 CUDA memory tests: " << error.what() << "\n";
        return 1;
    }
}
