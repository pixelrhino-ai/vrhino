#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open precision policy");
    std::ostringstream output;
    output << stream.rdbuf();
    vrhino::require(stream.good() || stream.eof(), "Cannot read precision policy");
    return output.str();
}

bool finite_tensor(const vrhino::Tensor& tensor) {
    vrhino::require(tensor.device().is_host(), "Finite audit requires host tensor");
    if (tensor.dtype() == vrhino::DType::BF16) {
        for (int64_t index = 0; index < tensor.numel(); ++index) {
            const uint32_t bits = static_cast<uint32_t>(
                tensor.data_as<uint16_t>()[index]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value)) return false;
        }
        return true;
    }
    vrhino::require(tensor.dtype() == vrhino::DType::F32,
                    "Finite audit supports BF16/FP32 tensors");
    for (int64_t index = 0; index < tensor.numel(); ++index)
        if (!std::isfinite(tensor.data_as<float>()[index])) return false;
    return true;
}

bool exact_bundle(const vrhino::TensorBundle& left,
                  const vrhino::TensorBundle& right) {
    if (left.size() != right.size()) return false;
    for (const auto& [name, tensor] : left) {
        const auto found = right.find(name);
        if (found == right.end() || tensor.dtype() != found->second.dtype() ||
                tensor.shape() != found->second.shape() ||
                tensor.bytes() != found->second.bytes() ||
                std::memcmp(tensor.data(), found->second.data(), tensor.bytes()) != 0)
            return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 6,
            "usage: vrhino-hunyuan-sampling-short-chain POLICY.json MODEL.vrm "
            "INPUT.bundle RUN1.bundle RUN2.bundle");
        const vrhino::PrecisionPolicy policy = vrhino::PrecisionPolicy::from_json(
            vrhino::Json::parse(read_text(argv[1])));
        vrhino::VrmModel model(argv[2]);
        vrhino::TensorBundle input = vrhino::read_bundle(argv[3]);
        input.insert_or_assign("seed", vrhino::scalar_i64(5702));
        input.insert_or_assign("latent_shape",
            vrhino::host_i64({5}, {1, 16, 1, 4, 4}));
        input.insert_or_assign("sampling_steps", vrhino::scalar_i64(30));
        input.insert_or_assign("flow_shift", vrhino::scalar_f32(9.0f));
        input.insert_or_assign("embedded_guidance", vrhino::scalar_f32(6000.0f));

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
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
        backend.enable_weight_cache(true);

        auto architecture = vrhino::create_architecture(model);
        const vrhino::SamplingProgram production = architecture->create_program(input);
        vrhino::require(production.steps == 30 && production.sigmas.size() == 31,
                        "Production declaration did not produce 30-step schedule");
        auto denoiser = architecture->create_denoiser(backend, policy, input);

        vrhino::SamplingProgram prefix = production;
        prefix.steps = 3;
        prefix.sigmas.resize(4);
        prefix.model_timesteps.resize(3);
        prefix.update_deltas.resize(3);
        vrhino::SamplingRuntime runtime(backend, policy);
        const vrhino::SamplingResult first = runtime.run(*denoiser, prefix);
        const vrhino::SamplingResult second = runtime.run(*denoiser, prefix);
        backend.synchronize();

        int finite_predictions = 0;
        int finite_latents = 0;
        for (int step = 0; step < prefix.steps; ++step) {
            finite_predictions += finite_tensor(first.trace.at(
                "step." + std::to_string(step) + ".prediction.0"));
            finite_latents += finite_tensor(first.trace.at(
                "step." + std::to_string(step) + ".latent"));
        }
        const bool final_finite = finite_tensor(first.final_latent.device().is_host()
            ? first.final_latent : backend.copy_to_host(first.final_latent));
        const bool deterministic = exact_bundle(first.trace, second.trace);
        vrhino::require(finite_predictions == 3 && finite_latents == 3 && final_finite,
                        "Hunyuan short chain produced non-finite tensors");
        vrhino::require(deterministic, "Hunyuan short-chain repeat is not byte exact");
        vrhino::write_bundle(argv[4], first.trace);
        vrhino::write_bundle(argv[5], second.trace);

        std::cout << "status=PASS\n"
                  << "declared_steps=" << production.steps << '\n'
                  << "executed_prefix_steps=" << prefix.steps << '\n'
                  << "flow_shift=9\nembedded_guidance=6000\n"
                  << "timestep0=" << production.model_timesteps.at(0).data_as<float>()[0] << '\n'
                  << "timestep1=" << production.model_timesteps.at(1).data_as<float>()[0] << '\n'
                  << "timestep2=" << production.model_timesteps.at(2).data_as<float>()[0] << '\n'
                  << "finite_predictions=" << finite_predictions << "/3\n"
                  << "finite_latents=" << finite_latents << "/3\n"
                  << "nan_inf=0\ndeterministic_repeat=BYTE_EXACT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hunyuan_sampling_short_chain: " << error.what() << '\n';
        return 1;
    }
}
