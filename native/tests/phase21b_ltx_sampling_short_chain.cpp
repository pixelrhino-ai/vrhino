#include <cmath>
#include <cstring>
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
    vrhino::require(tensor.device().is_host(), "Finite audit requires a host tensor");
    if (tensor.dtype() == vrhino::DType::BF16) {
        for (int64_t index = 0; index < tensor.numel(); ++index) {
            const uint32_t bits = static_cast<uint32_t>(tensor.data_as<uint16_t>()[index]) << 16;
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
            "usage: vrhino-phase21b-ltx-sampling-short-chain POLICY MODEL INPUT RUN1 RUN2");
        const vrhino::PrecisionPolicy policy = vrhino::PrecisionPolicy::from_json(
            vrhino::Json::parse(read_text(argv[1])));
        vrhino::VrmModel model(argv[2]);
        vrhino::TensorBundle input = vrhino::read_bundle(argv[3]);
        input.insert_or_assign("sampling_steps", vrhino::scalar_i64(40));
        input.insert_or_assign("guidance_scale", vrhino::scalar_f32(3.0f));
        input.insert_or_assign("resolution_shift_min_tokens", vrhino::scalar_i64(1024));
        input.insert_or_assign("resolution_shift_max_tokens", vrhino::scalar_i64(4096));
        input.insert_or_assign("resolution_shift_min", vrhino::scalar_f32(0.95f));
        input.insert_or_assign("resolution_shift_max", vrhino::scalar_f32(2.05f));

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        backend.enable_weight_cache(true);
        auto architecture = vrhino::create_architecture(model);
        const vrhino::SamplingProgram production = architecture->create_program(input);
        vrhino::require(production.steps == 40 && production.sigmas.size() == 41 &&
                        production.guidance_coefficients == std::vector<float>({-2.0f, 3.0f}),
                        "LTX production declaration was not consumed by the architecture");
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
            finite_predictions += finite_tensor(first.trace.at(
                "step." + std::to_string(step) + ".prediction.1"));
            finite_latents += finite_tensor(first.trace.at(
                "step." + std::to_string(step) + ".latent"));
        }
        const vrhino::Tensor final = first.final_latent.device().is_host()
            ? first.final_latent : backend.copy_to_host(first.final_latent);
        vrhino::require(finite_predictions == 6 && finite_latents == 3 &&
                        finite_tensor(final), "LTX short chain produced non-finite tensors");
        vrhino::require(exact_bundle(first.trace, second.trace),
                        "LTX short-chain repeat is not byte exact");
        vrhino::write_bundle(argv[4], first.trace);
        vrhino::write_bundle(argv[5], second.trace);

        std::cout << "status=PASS\n"
                  << "declared_steps=40\nexecuted_prefix_steps=3\n"
                  << "guidance_scale=3\n"
                  << "timestep0=" << production.model_timesteps[0].data_as<float>()[0] << '\n'
                  << "timestep1=" << production.model_timesteps[1].data_as<float>()[0] << '\n'
                  << "timestep2=" << production.model_timesteps[2].data_as<float>()[0] << '\n'
                  << "finite_predictions=6/6\nfinite_latents=3/3\n"
                  << "nan_inf=0\ndeterministic_repeat=BYTE_EXACT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase21b_ltx_sampling_short_chain: " << error.what() << '\n';
        return 1;
    }
}
