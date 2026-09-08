#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/memory.h"
#include "vrhino/runtime.h"

namespace {

constexpr size_t GiB = 1024ULL * 1024 * 1024;

vrhino::MemoryBudget budget(size_t gib) {
    const size_t total = gib * GiB;
    return {total, 1 * GiB, 64 * GiB, total / 8, total / 16};
}

vrhino::MemoryBudget high_residency_budget() {
    // The 20 GiB stress point intentionally reserves 18.75% for workspace and
    // safety.  A separate 22 GiB reference budget leaves 19 GiB for weights so
    // Hunyuan BF16 can serve as a genuinely high-residency correctness oracle
    // on the 24 GiB Phase 10 device.
    return {22 * GiB, 1 * GiB, 64 * GiB, 2 * GiB, 1 * GiB};
}

struct Run {
    vrhino::RuntimeResult runtime;
    vrhino::MemoryRuntimeStats memory;
};

Run execute(vrhino::VrmModel& model, const vrhino::TensorBundle& input,
            const vrhino::MemoryBudget& memory_budget,
            const vrhino::MemoryRuntimeOptions& options,
            const std::vector<vrhino::MemoryAccess>* trace,
            std::vector<vrhino::MemoryAccess>* recorded,
            const std::filesystem::path& output) {
    vrhino::CudaBackend backend;
    backend.set_execution_dtype(vrhino::DType::BF16);
    backend.configure_memory_runtime(memory_budget, options);
    backend.set_vrm_mapped_bytes(model.file_size());
    if (trace) backend.set_memory_trace(*trace);
    if (recorded) backend.begin_memory_trace();
    auto architecture = vrhino::create_architecture(model);
    Run run;
    run.runtime = vrhino::NativeRuntime(backend).execute(*architecture, input);
    if (recorded) *recorded = backend.end_memory_trace();
    backend.synchronize();
    run.memory = backend.memory_runtime_stats();
    vrhino::write_bundle(output.string(), run.runtime.outputs);
    return run;
}

void report(const std::string& label, size_t budget_gib, const Run& run) {
    const auto& runtime = run.runtime;
    const auto& memory = run.memory;
    const double stall = std::max(0.0, memory.upload_seconds - memory.overlapped_upload_seconds);
    std::cout << std::fixed << std::setprecision(6)
              << "PHASE10_RUN label=" << label
              << ",budget_gib=" << budget_gib
              << ",runtime_s=" << runtime.execution_seconds
              << ",sampling_s=" << runtime.sampling_seconds
              << ",vae_s=" << runtime.decode_seconds
              << ",upload_bytes=" << memory.upload_bytes
              << ",upload_s=" << memory.upload_seconds
              << ",upload_gbps=" << memory.upload_bandwidth_gbps()
              << ",overlap_ratio=" << memory.overlap_ratio()
              << ",stall_estimate_s=" << stall
              << ",cache_hit_rate=" << memory.cache_hit_rate()
              << ",cache_hits=" << memory.cache_hits
              << ",cache_misses=" << memory.cache_misses
              << ",prefetch_hit_rate=" << memory.prefetch_hit_rate()
              << ",prefetch_requests=" << memory.prefetch_requests
              << ",prefetch_hits=" << memory.prefetch_hits
              << ",prefetch_dropped=" << memory.prefetch_dropped
              << ",evictions=" << memory.evictions
              << ",forced_syncs=" << memory.forced_syncs
              << ",stream_waits=" << memory.stream_waits
              << ",event_waits=" << memory.event_waits
              << ",peak_device_bytes=" << memory.accounting.peak_device_bytes
              << ",peak_device_weight_bytes=" << memory.accounting.peak_device_resident_weight_bytes
              << ",peak_device_activation_bytes=" << memory.accounting.peak_device_activation_bytes
              << ",peak_host_staging_bytes=" << memory.accounting.peak_host_staging_bytes
              << ",peak_host_total_bytes=" << memory.accounting.peak_host_total_bytes
              << ",vrm_mapped_bytes=" << memory.accounting.vrm_mapped_bytes
              << ",quant_packed_bytes=" << memory.accounting.quantized_packed_bytes
              << ",backend_repack_bytes=" << memory.accounting.backend_repack_bytes
              << ",temporary_peak_bytes=" << memory.accounting.peak_temporary_bytes
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc >= 5,
            "usage: vrhino-phase10-batch MODEL.vrm INPUT OUTPUT_DIR MODE [BUDGET_GIB ...]");
        const std::string mode = argv[4];
        vrhino::require(mode == "full" || mode == "no-prefetch" || mode == "no-pinned" ||
                        mode == "lru", "invalid Phase 10 mode");
        std::vector<size_t> budgets;
        for (int index = 5; index < argc; ++index) budgets.push_back(std::stoull(argv[index]));
        if (budgets.empty()) budgets = {4, 8, 12, 16, 20};
        std::filesystem::path output_root(argv[3]);
        std::filesystem::create_directories(output_root);
        vrhino::VrmModel model(argv[1], false);
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);

        vrhino::MemoryRuntimeOptions high_options;
        high_options.enabled = true;
        high_options.host_staging = true;
        high_options.prefetch = false;
        std::vector<vrhino::MemoryAccess> trace;
        Run high = execute(model, input, high_residency_budget(), high_options, nullptr, &trace,
                           output_root / "high.bundle");
        report("high", 22, high);
        std::cout << "PHASE10_TRACE accesses=" << trace.size() << "\n";

        for (size_t value : budgets) {
            vrhino::MemoryRuntimeOptions options;
            options.enabled = true;
            options.host_staging = mode != "no-pinned";
            options.prefetch = mode != "no-prefetch";
            options.prefetch_lookahead = 1;
            options.eviction_policy = mode == "lru" ? vrhino::EvictionPolicy::Lru :
                                                       vrhino::EvictionPolicy::NextUse;
            Run run = execute(model, input, budget(value), options, &trace, nullptr,
                              output_root / (mode + "-" + std::to_string(value) + ".bundle"));
            report(mode, value, run);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase10 batch: " << error.what() << "\n";
        return 1;
    }
}
