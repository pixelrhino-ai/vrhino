#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/tensor_util.h"

namespace {

vrhino::TensorBundle ltx_input(int64_t tokens) {
    return {
        {"seed", vrhino::scalar_i64(5703)},
        {"coordinates", vrhino::host_f32({1, 3, tokens},
            std::vector<float>(static_cast<size_t>(3 * tokens), 0.0f))},
    };
}

void add_declaration(vrhino::TensorBundle& input, int steps, float guidance,
                     int64_t min_tokens = 1024, int64_t max_tokens = 4096,
                     float min_shift = 0.95f, float max_shift = 2.05f) {
    input.insert_or_assign("sampling_steps", vrhino::scalar_i64(steps));
    input.insert_or_assign("guidance_scale", vrhino::scalar_f32(guidance));
    input.insert_or_assign("resolution_shift_min_tokens", vrhino::scalar_i64(min_tokens));
    input.insert_or_assign("resolution_shift_max_tokens", vrhino::scalar_i64(max_tokens));
    input.insert_or_assign("resolution_shift_min", vrhino::scalar_f32(min_shift));
    input.insert_or_assign("resolution_shift_max", vrhino::scalar_f32(max_shift));
}

bool same_tensor(const vrhino::Tensor& left, const vrhino::Tensor& right) {
    return left.dtype() == right.dtype() && left.shape() == right.shape() &&
           left.bytes() == right.bytes() &&
           std::memcmp(left.data(), right.data(), left.bytes()) == 0;
}

bool same_program(const vrhino::SamplingProgram& left,
                  const vrhino::SamplingProgram& right) {
    if (left.latent_shape != right.latent_shape || left.seed != right.seed ||
            left.steps != right.steps || left.guidance_mode != right.guidance_mode ||
            left.guidance_coefficients != right.guidance_coefficients ||
            left.scheduler != right.scheduler || left.sigmas != right.sigmas ||
            left.update_deltas != right.update_deltas ||
            left.subtract_prediction != right.subtract_prediction ||
            left.zero_is_frozen != right.zero_is_frozen ||
            left.model_timesteps.size() != right.model_timesteps.size())
        return false;
    for (size_t index = 0; index < left.model_timesteps.size(); ++index)
        if (!same_tensor(left.model_timesteps[index], right.model_timesteps[index]))
            return false;
    return true;
}

void validate_ltx(const std::string& model_path) {
    vrhino::VrmModel model(model_path, false);
    vrhino::require(model.architecture_id() == "ltx_v0_9_1",
                    "Phase 21B declaration test requires LTX v0.9.1");
    auto architecture = vrhino::create_architecture(model);

    const vrhino::TensorBundle implicit_input = ltx_input(4);
    vrhino::TensorBundle explicit_input = implicit_input;
    add_declaration(explicit_input, 3, 4.5f);
    const vrhino::SamplingProgram implicit = architecture->create_program(implicit_input);
    const vrhino::SamplingProgram explicit_canary = architecture->create_program(explicit_input);
    vrhino::require(same_program(implicit, explicit_canary),
                    "Explicit legacy LTX declaration changed the frozen canary program");
    std::cout << "legacy_canary=PASS,program_byte_exact=true\n";

    vrhino::TensorBundle production_input = ltx_input(5280);
    add_declaration(production_input, 40, 3.0f);
    const vrhino::SamplingProgram production =
        architecture->create_program(production_input);
    const vrhino::SamplingProgram repeated =
        architecture->create_program(production_input);
    vrhino::require(same_program(production, repeated),
                    "LTX production declaration is not deterministic");
    vrhino::require(production.steps == 40 && production.scheduler == vrhino::SchedulerKind::Euler &&
                    production.subtract_prediction &&
                    production.guidance_mode == vrhino::GuidanceMode::CFG &&
                    production.guidance_coefficients == std::vector<float>({-2.0f, 3.0f}),
                    "LTX production SamplingProgram metadata mismatch");
    vrhino::require(production.sigmas.size() == 41 &&
                    production.model_timesteps.size() == 40 &&
                    production.update_deltas.size() == 40,
                    "LTX production schedule table size mismatch");

    // Frozen Official v0.9.1 table produced by RectifiedFlowScheduler with
    // steps=40, SD3 shifting, and a [1,5280,128] sample. The architecture
    // declaration uses FP32 scalar parameters, so bounded FP32 differences from
    // Python-double interpolation are reported without changing Runtime tolerance.
    constexpr uint32_t official_sigma_bits[40] = {
        0x3f800000u, 0x3f7f72bcu, 0x3f7edeaeu, 0x3f7e435au, 0x3f7da032u,
        0x3f7cf499u, 0x3f7c3fe6u, 0x3f7b8159u, 0x3f7ab81du, 0x3f79e345u,
        0x3f7901cbu, 0x3f781283u, 0x3f771420u, 0x3f760529u, 0x3f74e3efu,
        0x3f73ae89u, 0x3f7262c9u, 0x3f70fe2au, 0x3f6f7dc5u, 0x3f6dde3au,
        0x3f6c1b96u, 0x3f6a3136u, 0x3f681997u, 0x3f65ce2bu, 0x3f63470eu,
        0x3f607aa7u, 0x3f5d5d38u, 0x3f59e029u, 0x3f55f134u, 0x3f51790bu,
        0x3f4c599fu, 0x3f466b5au, 0x3f3f7919u, 0x3f3739d9u, 0x3f2d4664u,
        0x3f2107f7u, 0x3f11994bu, 0x3efb1797u, 0x3ec4dd83u, 0x3e6eee08u,
    };
    float max_sigma_difference = 0.0f;
    float max_delta_difference = 0.0f;
    for (int index = 0; index < 40; ++index) {
        const float official = std::bit_cast<float>(official_sigma_bits[index]);
        const float actual = production.sigmas[static_cast<size_t>(index)];
        max_sigma_difference = std::max(max_sigma_difference, std::abs(actual - official));
        const vrhino::Tensor& timestep = production.model_timesteps[static_cast<size_t>(index)];
        vrhino::require(timestep.device().is_host() && timestep.dtype() == vrhino::DType::F32 &&
                        timestep.shape() == std::vector<int64_t>({1, 1}) &&
                        timestep.data_as<float>()[0] == actual,
                        "LTX model timestep does not match scheduler sigma");
        const float official_next = index + 1 == 40 ? 0.0f :
            std::bit_cast<float>(official_sigma_bits[index + 1]);
        const float official_delta = official - official_next;
        max_delta_difference = std::max(max_delta_difference,
            std::abs(production.update_deltas[static_cast<size_t>(index)] - official_delta));
    }
    vrhino::require(production.sigmas.back() == 0.0f &&
                    max_sigma_difference <= 3.0e-7f &&
                    max_delta_difference <= 3.0e-7f,
                    "LTX production schedule does not match frozen Official FP32 definition");

    vrhino::TensorBundle override_input = ltx_input(2000);
    add_declaration(override_input, 10, 2.0f, 1000, 3000, 0.5f, 1.5f);
    const vrhino::SamplingProgram override_program =
        architecture->create_program(override_input);
    const float expected_shift = std::exp(1.0f);
    const float expected_second_base = 0.9f;
    const float expected_second = expected_shift /
        (expected_shift + (1.0f / expected_second_base - 1.0f));
    vrhino::require(override_program.steps == 10 &&
                    override_program.guidance_coefficients == std::vector<float>({-1.0f, 2.0f}) &&
                    std::abs(override_program.sigmas[1] - expected_second) <= 1.0e-7f,
                    "LTX resolution-shift parameter binding mismatch");

    int fail_closed = 0;
    for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
        vrhino::TensorBundle invalid = ltx_input(4);
        if (invalid_case == 0) add_declaration(invalid, 0, 3.0f);
        if (invalid_case == 1) add_declaration(invalid, 40, 3.0f, 4096, 1024);
        if (invalid_case == 2)
            add_declaration(invalid, 40, 3.0f, 1024, 4096,
                            std::numeric_limits<float>::quiet_NaN(), 2.05f);
        try {
            (void)architecture->create_program(invalid);
        } catch (const vrhino::Error&) {
            ++fail_closed;
        }
    }
    vrhino::require(fail_closed == 3, "LTX declaration invalid cases did not fail closed");

    std::cout << "production_steps=40\n"
              << "production_cfg=3\n"
              << "production_tokens=5280\n"
              << "timestep_match=40/40\n"
              << "sigma_match=40/40\n"
              << "max_sigma_abs_difference=" << max_sigma_difference << '\n'
              << "max_delta_abs_difference=" << max_delta_difference << '\n'
              << "resolution_shift_override=PASS\n"
              << "deterministic_declaration=BYTE_EXACT_PASS\n"
              << "fail_closed_cases=3/3\n"
              << "status=PASS\n";
}

void add_snapshot(vrhino::TensorBundle& output, const std::string& prefix,
                  const vrhino::SamplingProgram& program) {
    output.emplace(prefix + ".steps", vrhino::scalar_i64(program.steps));
    output.emplace(prefix + ".seed", vrhino::scalar_i64(static_cast<int64_t>(program.seed)));
    output.emplace(prefix + ".latent_shape", vrhino::host_i64(
        {static_cast<int64_t>(program.latent_shape.size())}, program.latent_shape));
    output.emplace(prefix + ".guidance", vrhino::host_f32(
        {static_cast<int64_t>(program.guidance_coefficients.size())},
        program.guidance_coefficients));
    output.emplace(prefix + ".sigmas", vrhino::host_f32(
        {static_cast<int64_t>(program.sigmas.size())}, program.sigmas));
    output.emplace(prefix + ".deltas", vrhino::host_f32(
        {static_cast<int64_t>(program.update_deltas.size())}, program.update_deltas));
    output.emplace(prefix + ".subtract_prediction",
                   vrhino::scalar_i64(program.subtract_prediction ? 1 : 0));
    output.emplace(prefix + ".zero_is_frozen",
                   vrhino::scalar_i64(program.zero_is_frozen ? 1 : 0));
    for (int step = 0; step < program.steps; ++step)
        output.emplace(prefix + ".timestep." + std::to_string(step),
                       program.model_timestep_at(step));
}

void snapshot(int argc, char** argv) {
    vrhino::require(argc == 13,
        "usage: ... snapshot OUTPUT WAN_MODEL WAN_INPUT HUNYUAN_MODEL HUNYUAN_INPUT "
        "LTX_MODEL LTX_INPUT MOCHI_MODEL MOCHI_INPUT COG_MODEL COG_INPUT");
    vrhino::TensorBundle output;
    const std::vector<std::string> names = {"wan", "hunyuan", "ltx", "mochi", "cogvideox"};
    for (size_t index = 0; index < names.size(); ++index) {
        vrhino::VrmModel model(argv[3 + index * 2], false);
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[4 + index * 2]);
        auto architecture = vrhino::create_architecture(model);
        add_snapshot(output, names[index], architecture->create_program(input));
    }
    vrhino::write_bundle(argv[2], output);
    std::cout << "snapshot_tensors=" << output.size() << "\nstatus=PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc >= 2, "Phase 21B declaration test requires a mode");
        const std::string mode = argv[1];
        if (mode == "validate") {
            vrhino::require(argc == 3, "usage: ... validate LTX_MODEL");
            validate_ltx(argv[2]);
        } else if (mode == "snapshot") {
            snapshot(argc, argv);
        } else {
            throw vrhino::Error("Unsupported Phase 21B declaration test mode");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase21b_sampling_declaration_tests: " << error.what() << '\n';
        return 1;
    }
}
