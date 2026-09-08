#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/tensor_util.h"

namespace {

vrhino::TensorBundle declaration_input(int steps, float shift) {
    return {
        {"seed", vrhino::scalar_i64(5702)},
        {"sampling_steps", vrhino::scalar_i64(steps)},
        {"flow_shift", vrhino::scalar_f32(shift)},
        {"embedded_guidance", vrhino::scalar_f32(6000.0f)},
    };
}

float expected_sigma(int index, int steps, float shift) {
    const float u = 1.0f - static_cast<float>(index) / steps;
    return shift * u / (1.0f + (shift - 1.0f) * u);
}

void validate_schedule(const vrhino::SamplingProgram& program,
                       int steps, float shift) {
    vrhino::require(program.steps == steps, "Hunyuan sampling_steps binding mismatch");
    vrhino::require(program.scheduler == vrhino::SchedulerKind::Euler,
                    "Hunyuan solver is not generic Euler");
    vrhino::require(program.guidance_mode == vrhino::GuidanceMode::Linear &&
                    program.guidance_coefficients == std::vector<float>({1.0f}),
                    "Hunyuan one-branch guidance declaration mismatch");
    vrhino::require(program.sigmas.size() == static_cast<size_t>(steps + 1) &&
                    program.model_timesteps.size() == static_cast<size_t>(steps) &&
                    program.update_deltas.size() == static_cast<size_t>(steps),
                    "Hunyuan shifted-flow table size mismatch");
    for (int index = 0; index <= steps; ++index) {
        const float expected = expected_sigma(index, steps, shift);
        vrhino::require(std::abs(program.sigmas.at(static_cast<size_t>(index)) - expected) <= 1e-7f,
                        "Hunyuan sigma table mismatch");
        if (index == steps) continue;
        const vrhino::Tensor& timestep = program.model_timesteps.at(static_cast<size_t>(index));
        vrhino::require(timestep.device().is_host() && timestep.dtype() == vrhino::DType::F32 &&
                        timestep.shape().empty(), "Hunyuan timestep contract mismatch");
        vrhino::require(std::abs(timestep.data_as<float>()[0] - expected * 1000.0f) <= 1e-4f,
                        "Hunyuan timestep table mismatch");
        const float expected_delta = expected_sigma(index + 1, steps, shift) - expected;
        vrhino::require(std::abs(program.update_deltas.at(static_cast<size_t>(index)) -
                                 expected_delta) <= 1e-7f,
                        "Hunyuan Euler delta table mismatch");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 2,
            "usage: vrhino-hunyuan-sampling-declaration-tests MODEL.vrm");
        vrhino::VrmModel model(argv[1], false);
        vrhino::require(model.architecture_id() == "hunyuan_video",
                        "Sampling declaration test requires HunyuanVideo VRM");
        auto architecture = vrhino::create_architecture(model);

        const vrhino::TensorBundle legacy = declaration_input(3, 5.0f);
        const vrhino::SamplingProgram legacy_program = architecture->create_program(legacy);
        validate_schedule(legacy_program, 3, 5.0f);
        std::cout << "legacy_canary=PASS,steps=3,shift=5\n";

        std::cout << std::fixed << std::setprecision(9);
        int cases = 0;
        for (int steps : {3, 10, 30}) {
            for (float shift : {1.0f, 5.0f, 9.0f}) {
                const vrhino::SamplingProgram program =
                    architecture->create_program(declaration_input(steps, shift));
                validate_schedule(program, steps, shift);
                std::cout << "parameter_case=PASS,steps=" << steps
                          << ",shift=" << shift
                          << ",sigma0=" << program.sigmas.front()
                          << ",sigma_last=" << program.sigmas.back()
                          << ",timestep0="
                          << program.model_timesteps.front().data_as<float>()[0]
                          << ",delta0=" << program.update_deltas.front() << '\n';
                if (steps == 30 && shift == 9.0f) {
                    for (int index = 0; index < steps; ++index) {
                        std::cout << "official_row=" << index
                                  << ",sigma=" << program.sigmas.at(index)
                                  << ",timestep="
                                  << program.model_timesteps.at(index).data_as<float>()[0]
                                  << ",delta=" << program.update_deltas.at(index) << '\n';
                    }
                }
                ++cases;
            }
        }
        vrhino::require(cases == 9, "Hunyuan parameterization matrix count mismatch");
        std::cout << "parameterization_cases=9/9\nstatus=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hunyuan_sampling_declaration_tests: " << error.what() << '\n';
        return 1;
    }
}
