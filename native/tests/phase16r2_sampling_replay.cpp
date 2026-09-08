#include <cstdlib>
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
            "usage: vrhino-phase16r2-sampling-replay MODEL.vrm INPUT.bundle "
            "float32|bfloat16 OUTPUT.bundle");
        const std::string dtype_name = argv[3];
        const vrhino::DType dtype = dtype_name == "float32"
            ? vrhino::DType::F32 : vrhino::DType::BF16;
        vrhino::require(dtype_name == "float32" || dtype_name == "bfloat16",
                        "sampling replay dtype must be float32 or bfloat16");

        vrhino::VrmModel model(argv[1]);
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(dtype);
        size_t device_budget_gib = 21;
        if (const char* budget = std::getenv("VRHINO_DEVICE_BUDGET_GIB"))
            device_budget_gib = static_cast<size_t>(std::strtoull(budget, nullptr, 10));
        vrhino::MemoryRuntimeOptions options;
        options.enabled = true;
        options.host_staging = false;
        options.prefetch = false;
        backend.configure_memory_runtime(
            vrhino::MemoryBudget{device_budget_gib << 30, 2ULL << 30,
                                 64ULL << 30, 2ULL << 30, 1ULL << 30}, options);
        backend.set_vrm_mapped_bytes(model.file_size());

        auto architecture = vrhino::create_architecture(model);
        vrhino::RuntimeResult result =
            vrhino::NativeRuntime(backend).execute(*architecture, input);
        backend.synchronize();
        vrhino::write_bundle(argv[4], result.outputs);
        std::cout << "status=PASS\nexecution_dtype=" << dtype_name
                  << "\nsampling_seconds=" << result.sampling_seconds
                  << "\ndecode_seconds=" << result.decode_seconds
                  << "\nexecution_seconds=" << result.execution_seconds
                  << "\npeak_device_bytes=" << backend.peak_device_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase16r2 sampling replay: " << error.what() << '\n';
        return 1;
    }
}
