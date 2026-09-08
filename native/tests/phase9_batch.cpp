#include <iostream>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/runtime.h"

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 5,
                        "usage: vrhino-phase9-batch MODEL.vrm baseline|true INPUT OUTPUT");
        const std::string mode = argv[2];
        vrhino::require(mode == "baseline" || mode == "true", "invalid Phase 9 mode");
        vrhino::VrmModel model(argv[1], false);
        vrhino::TensorBundle input = vrhino::read_bundle(argv[3]);
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        backend.enable_true_quant_compute(mode == "true");
        backend.enable_profiling(true);
        auto architecture = vrhino::create_architecture(model);
        vrhino::RuntimeResult result = vrhino::NativeRuntime(backend).execute(*architecture, input);
        vrhino::write_bundle(argv[4], result.outputs);
        const vrhino::QuantComputeStats quant = backend.quant_compute_stats();
        std::cout << "mode=" << mode << "\n"
                  << "execution_seconds=" << result.execution_seconds << "\n"
                  << "sampling_seconds=" << result.sampling_seconds << "\n"
                  << "vae_seconds=" << result.decode_seconds << "\n"
                  << "weight_upload_seconds=" << result.upload_seconds << "\n"
                  << "weight_upload_bytes=" << result.upload_bytes << "\n"
                  << "peak_device_bytes=" << result.peak_device_bytes << "\n"
                  << "weight_cache_resident_bytes=" << backend.weight_cache_resident_bytes() << "\n"
                  << "weight_cache_hits=" << backend.weight_cache_hits() << "\n"
                  << "weight_cache_misses=" << backend.weight_cache_misses() << "\n"
                  << "true_quant_calls=" << quant.true_quant_calls << "\n"
                  << "fallback_calls=" << quant.fallback_calls << "\n"
                  << "fp8_calls=" << quant.fp8_calls << "\n"
                  << "int8_calls=" << quant.int8_calls << "\n"
                  << "int4_calls=" << quant.int4_calls << "\n"
                  << "workspace_bytes=" << quant.workspace_bytes << "\n"
                  << "quant_scratch_peak_bytes=" << quant.scratch_peak_bytes << "\n"
                  << "device_copy_bytes=" << quant.device_copy_bytes << "\n"
                  << "last_dispatch=" << quant.last_dispatch << "\n";
        for (const auto& [name, stat] : backend.profile_stats())
            std::cout << "profile=" << name << ",calls=" << stat.calls
                      << ",cuda_ms=" << stat.device_milliseconds << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase9 batch: " << error.what() << "\n";
        return 1;
    }
}
