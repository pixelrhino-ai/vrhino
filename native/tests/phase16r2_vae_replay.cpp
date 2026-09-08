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
        vrhino::require(argc == 6,
            "usage: vrhino-phase16r2-vae-replay MODEL.vrm INPUT.bundle "
            "LATENT_KEY float32|bfloat16 OUTPUT.bundle");
        const std::string dtype_name = argv[4];
        const vrhino::DType dtype = dtype_name == "float32"
            ? vrhino::DType::F32 : vrhino::DType::BF16;
        vrhino::require(dtype_name == "float32" || dtype_name == "bfloat16",
                        "VAE replay dtype must be float32 or bfloat16");

        vrhino::VrmModel model(argv[1]);
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);
        const auto found = input.find(argv[3]);
        vrhino::require(found != input.end(), "VAE replay latent key is absent");

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(dtype);
        size_t device_budget_gib = 21;
        if (const char* budget = std::getenv("VRHINO_DEVICE_BUDGET_GIB"))
            device_budget_gib = static_cast<size_t>(std::strtoull(budget, nullptr, 10));
        backend.configure_memory_runtime(
            vrhino::MemoryBudget{device_budget_gib << 30, 2ULL << 30,
                                 64ULL << 30, 2ULL << 30, 1ULL << 30},
            vrhino::MemoryRuntimeOptions{true, false, false});
        backend.set_vrm_mapped_bytes(model.file_size());

        auto architecture = vrhino::create_architecture(model);
        const vrhino::PrecisionPolicy policy =
            vrhino::PrecisionPolicy::unqualified_default(dtype);
        vrhino::NativeRuntime runtime(backend, policy);
        const vrhino::Tensor video_device = runtime.decode_component(
            *architecture, found->second, vrhino::TensorBundle{});
        backend.synchronize();
        vrhino::TensorBundle output{
            {"latent", found->second},
            {"video", backend.copy_to_host(video_device)},
        };
        vrhino::write_bundle(argv[5], output);
        std::cout << "status=PASS\nexecution_dtype=" << dtype_name
                  << "\nlatent_key=" << argv[3]
                  << "\nvideo_dtype=" << vrhino::dtype_name(output.at("video").dtype())
                  << "\nexecutor_mode=" << vrhino::component_execution_mode_name(
                         runtime.component_execution_stats().mode)
                  << "\npeak_device_bytes=" << backend.peak_device_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase16r2 VAE replay: " << error.what() << '\n';
        return 1;
    }
}
