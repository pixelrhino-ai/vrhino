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
        vrhino::require(argc >= 4 && argc % 2 == 0,
                        "usage: vrhino-phase8-batch MODEL.vrm INPUT OUTPUT [INPUT OUTPUT ...]");
        // Ephemeral research candidates are checksumed as they are written;
        // final VRMs are independently verified by the production loader
        // tests. Avoid hashing a 20+ GiB class-only candidate for every case.
        vrhino::VrmModel model(argv[1], false);
        for (int index = 2; index < argc; index += 2) {
            vrhino::TensorBundle input = vrhino::read_bundle(argv[index]);
            vrhino::CudaBackend backend;
            backend.set_execution_dtype(vrhino::DType::BF16);
            auto architecture = vrhino::create_architecture(model);
            vrhino::RuntimeResult result = vrhino::NativeRuntime(backend).execute(*architecture, input);
            vrhino::write_bundle(argv[index + 1], result.outputs);
            std::cout << "case=" << (index - 2) / 2 << ",input=" << argv[index]
                      << ",output=" << argv[index + 1]
                      << ",seconds=" << result.execution_seconds << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase8 batch: " << error.what() << "\n";
        return 1;
    }
}
