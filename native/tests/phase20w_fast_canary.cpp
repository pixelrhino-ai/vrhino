#include <iostream>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/runtime.h"

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 4,
                        "usage: vrhino-phase20w-fast-canary MODEL INPUT OUTPUT");
        vrhino::VrmModel model(argv[1], false);
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        auto architecture = vrhino::create_architecture(model);
        vrhino::NativeRuntime runtime(backend);
        vrhino::RuntimeResult result = runtime.execute(*architecture, input);
        vrhino::require(result.component_execution.mode ==
                                vrhino::ComponentExecutionMode::Untiled &&
                            result.component_execution.tiling.graph_executions == 1,
                        "Default Component execution was not UNTILED");
        vrhino::write_bundle(argv[3], result.outputs);
        std::cout << "status=PASS\narchitecture=" << model.architecture_id()
                  << "\nexecutor_mode=" << vrhino::component_execution_mode_name(
                         result.component_execution.mode)
                  << "\ngraph_executions="
                  << result.component_execution.tiling.graph_executions
                  << "\nsampling_seconds=" << result.sampling_seconds
                  << "\ndecode_seconds=" << result.decode_seconds
                  << "\npeak_device_bytes=" << result.peak_device_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20w_fast_canary: " << error.what() << '\n';
        return 1;
    }
}
