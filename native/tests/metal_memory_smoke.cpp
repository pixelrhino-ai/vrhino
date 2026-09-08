#include <cmath>
#include <iostream>
#include <vector>

#include "vrhino/backend/metal_backend.h"
#include "vrhino/error.h"

int main() {
    try {
        vrhino::MetalBackend backend;
        const auto capabilities = backend.memory_capabilities();
        vrhino::require(capabilities.unified_memory && capabilities.host_visible_device_memory &&
                        capabilities.device_visible_host_memory && capabilities.device_preferred &&
                        !capabilities.explicit_transfer_required && capabilities.async_transfer_supported,
                        "unexpected Apple unified-memory capability mapping");

        constexpr size_t kib = 1024;
        vrhino::MemoryBudget budget{8 * kib, 2 * kib, 32 * kib, 1 * kib, 1 * kib};
        vrhino::MemoryRuntimeOptions options;
        options.enabled = true;
        backend.configure_memory_runtime(budget, options);
        backend.set_vrm_mapped_bytes(16 * kib);
        vrhino::require(backend.memory_runtime_enabled(), "Metal memory runtime was not enabled");
        vrhino::require(backend.weight_cache_capacity_bytes() == 6 * kib,
                        "logical device weight budget was not applied");

        std::vector<float> values(64);
        for (size_t index = 0; index < values.size(); ++index) values[index] = static_cast<float>(index);
        vrhino::Tensor borrowed = vrhino::Tensor::borrowed(values.data(), values.size() * sizeof(float),
                                                           {64}, vrhino::DType::F32);
        backend.begin_memory_trace();
        vrhino::Tensor resident = backend.copy_to_device(borrowed, vrhino::DType::F32);
        vrhino::Tensor reused = backend.copy_to_device(borrowed, vrhino::DType::F32);
        const auto trace = backend.end_memory_trace();
        vrhino::require(resident.data() == reused.data(), "unified-memory resident weight was not reused");
        vrhino::require(trace.size() == 2 && trace[0].tensor.data() == borrowed.data() &&
                        trace[0].target_dtype == vrhino::DType::F32,
                        "shared memory access trace was not recorded");

        vrhino::Tensor destination = backend.allocate_device({64}, vrhino::DType::F32);
        vrhino::TransferFence fence = backend.copy(borrowed, destination,
                                                   vrhino::Backend::CopyMode::Asynchronous);
        vrhino::require(fence.valid(), "asynchronous unified copy did not create a fence");
        backend.wait_fence(fence);
        vrhino::require(backend.query_fence(fence), "completed Metal fence did not query complete");
        backend.destroy_fence(fence);
        vrhino::Tensor copied = backend.copy_to_host(destination);
        for (size_t index = 0; index < values.size(); ++index)
            vrhino::require(std::abs(copied.data_as<float>()[index] - values[index]) < 1e-7f,
                            "unified-memory copy changed tensor contents");

        const auto stats = backend.memory_runtime_stats();
        vrhino::require(stats.cache_misses == 1 && stats.cache_hits == 1,
                        "resident cache transition accounting mismatch");
        vrhino::require(stats.accounting.device_resident_weight_bytes == values.size() * sizeof(float) &&
                        stats.accounting.device_resident_weight_bytes <= budget.device_weight_budget_bytes(),
                        "resident bytes exceeded the logical Metal budget");
        vrhino::require(stats.accounting.vrm_mapped_bytes == 16 * kib,
                        "VRM mapping accounting mismatch");
        std::cout << "unified_memory=true\nlogical_weight_budget_bytes="
                  << budget.device_weight_budget_bytes() << "\nresident_weight_bytes="
                  << stats.accounting.device_resident_weight_bytes << "\ncache_hits="
                  << stats.cache_hits << "\ncache_misses=" << stats.cache_misses
                  << "\ntrace_accesses=" << trace.size()
                  << "\nnative Metal memory smoke=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Metal memory smoke: " << error.what() << "\n";
        return 1;
    }
}
