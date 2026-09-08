#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/component_tiling.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open Phase 22E input: " + path);
    std::ostringstream output;
    output << stream.rdbuf();
    vrhino::require(stream.good() || stream.eof(),
                    "Cannot read Phase 22E input: " + path);
    return output.str();
}

vrhino::TensorBundle profile_input(const vrhino::Json& profile,
                                   const std::string& fixture_path) {
    vrhino::TensorBundle input = vrhino::read_bundle(fixture_path);
    input.insert_or_assign("seed", vrhino::scalar_i64(profile.at("seed").integer()));
    input.insert_or_assign("audit_trace", vrhino::scalar_i64(0));
    const vrhino::Json& static_inputs = profile.at("static_inputs");
    for (const auto& [name, descriptor] : static_inputs.object()) {
        std::vector<int64_t> shape;
        for (const vrhino::Json& value : descriptor.at("shape").array())
            shape.push_back(value.integer());
        if (descriptor.at("dtype").string() == "i64") {
            std::vector<int64_t> values;
            for (const vrhino::Json& value : descriptor.at("values").array())
                values.push_back(value.integer());
            input.insert_or_assign(name, vrhino::host_i64(shape, values));
        } else {
            vrhino::require(descriptor.at("dtype").string() == "f32",
                            "Unsupported Phase 22E profile static-input dtype");
            std::vector<float> values;
            for (const vrhino::Json& value : descriptor.at("values").array())
                values.push_back(static_cast<float>(value.number()));
            input.insert_or_assign(name, vrhino::host_f32(shape, values));
        }
    }
    return input;
}

bool exact_tensor(const vrhino::Tensor& left, const vrhino::Tensor& right) {
    return left.dtype() == right.dtype() && left.shape() == right.shape() &&
           left.bytes() == right.bytes() &&
           std::memcmp(left.data(), right.data(), left.bytes()) == 0;
}

bool exact_program(const vrhino::SamplingProgram& left,
                   const vrhino::SamplingProgram& right) {
    if (left.latent_shape != right.latent_shape || left.seed != right.seed ||
            left.steps != right.steps ||
            left.guidance_mode != right.guidance_mode ||
            left.guidance_coefficients != right.guidance_coefficients ||
            left.scheduler != right.scheduler || left.sigmas != right.sigmas ||
            left.update_deltas != right.update_deltas ||
            left.subtract_prediction != right.subtract_prediction ||
            left.zero_is_frozen != right.zero_is_frozen ||
            left.model_timesteps.size() != right.model_timesteps.size())
        return false;
    for (size_t index = 0; index < left.model_timesteps.size(); ++index)
        if (!exact_tensor(left.model_timesteps[index],
                          right.model_timesteps[index]))
            return false;
    return true;
}

std::vector<double> official_schedule(int steps, double threshold,
                                      int linear_steps) {
    std::vector<double> noise;
    noise.reserve(static_cast<size_t>(steps + 1));
    for (int index = 0; index < linear_steps; ++index)
        noise.push_back(index * threshold / linear_steps);
    const double difference = linear_steps - threshold * steps;
    const int quadratic_steps = steps - linear_steps;
    const double quadratic = difference /
        (linear_steps * quadratic_steps * quadratic_steps);
    const double linear = threshold / linear_steps -
        2.0 * difference / (quadratic_steps * quadratic_steps);
    const double constant = quadratic * linear_steps * linear_steps;
    for (int index = linear_steps; index < steps; ++index)
        noise.push_back(quadratic * index * index + linear * index + constant);
    noise.push_back(1.0);
    for (double& value : noise) value = 1.0 - value;
    return noise;
}

float bf16_to_float(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

bool finite_host(const vrhino::Tensor& tensor) {
    vrhino::require(tensor.device().is_host(),
                    "Phase 22E finite audit requires a host tensor");
    if (tensor.dtype() == vrhino::DType::BF16) {
        for (int64_t index = 0; index < tensor.numel(); ++index)
            if (!std::isfinite(
                    bf16_to_float(tensor.data_as<uint16_t>()[index])))
                return false;
        return true;
    }
    vrhino::require(tensor.dtype() == vrhino::DType::F32,
                    "Phase 22E finite audit supports FP32/BF16 tensors");
    for (int64_t index = 0; index < tensor.numel(); ++index)
        if (!std::isfinite(tensor.data_as<float>()[index])) return false;
    return true;
}

bool exact_bundle(const vrhino::TensorBundle& left,
                  const vrhino::TensorBundle& right) {
    if (left.size() != right.size()) return false;
    for (const auto& [name, tensor] : left) {
        const auto found = right.find(name);
        if (found == right.end() || !exact_tensor(tensor, found->second))
            return false;
    }
    return true;
}

void configure_backend(vrhino::CudaBackend& backend,
                       const vrhino::VrmModel& model,
                       bool weight_cache = true) {
    backend.set_execution_dtype(vrhino::DType::BF16);
    backend.enable_weight_cache(weight_cache);
    backend.configure_memory_runtime(
        vrhino::MemoryBudget{23ULL << 30, 2ULL << 30, 64ULL << 30,
                             8ULL << 30, 1ULL << 30},
        vrhino::MemoryRuntimeOptions{true, false, false});
    backend.set_vrm_mapped_bytes(model.file_size());
}

vrhino::PrecisionPolicy precision_policy(const vrhino::Json& profile) {
    return vrhino::PrecisionPolicy::from_json(vrhino::Json::parse(
        read_text(profile.at("precision_policy").string())));
}

void validate_declaration(const std::string& model_path,
                          const std::string& fixture_path,
                          const vrhino::Json& profile) {
    vrhino::VrmModel model(model_path, false);
    auto architecture = vrhino::create_architecture(model);
    vrhino::TensorBundle production_input =
        profile_input(profile, fixture_path);
    const vrhino::SamplingProgram production =
        architecture->create_program(production_input);
    const vrhino::SamplingProgram repeat =
        architecture->create_program(production_input);
    vrhino::require(exact_program(production, repeat),
                    "Mochi production declaration is not deterministic");
    vrhino::require(production.steps == 64 &&
                        production.guidance_mode == vrhino::GuidanceMode::CFG &&
                        production.guidance_coefficients ==
                            std::vector<float>({-5.0f, 6.0f}) &&
                        production.scheduler == vrhino::SchedulerKind::Euler &&
                        production.sigmas.size() == 65 &&
                        production.model_timesteps.size() == 64 &&
                        production.update_deltas.size() == 64,
                    "Mochi production SamplingProgram binding mismatch");

    const std::vector<double> official = official_schedule(64, 0.025, 32);
    int sigma_match = 0;
    int timestep_match = 0;
    int delta_match = 0;
    float max_sigma = 0.0f;
    float max_timestep = 0.0f;
    float max_delta = 0.0f;
    for (int index = 0; index <= 64; ++index) {
        const float difference = std::abs(
            production.sigmas[static_cast<size_t>(index)] -
            static_cast<float>(official[static_cast<size_t>(index)]));
        max_sigma = std::max(max_sigma, difference);
        sigma_match += difference <= 1.0e-7f;
        if (index == 64) continue;
        const vrhino::Tensor& timestep =
            production.model_timesteps[static_cast<size_t>(index)];
        vrhino::require(timestep.device().is_host() &&
                            timestep.dtype() == vrhino::DType::F32 &&
                            timestep.shape().empty(),
                        "Mochi runtime timestep tensor contract mismatch");
        const float expected_timestep = static_cast<float>(
            (1.0 - official[static_cast<size_t>(index)]) * 1000.0);
        const float timestep_difference = std::abs(
            timestep.data_as<float>()[0] - expected_timestep);
        max_timestep = std::max(max_timestep, timestep_difference);
        timestep_match += timestep_difference <= 1.0e-4f;
        const float expected_delta = static_cast<float>(
            official[static_cast<size_t>(index)] -
            official[static_cast<size_t>(index + 1)]);
        const float delta_difference = std::abs(
            production.update_deltas[static_cast<size_t>(index)] -
            expected_delta);
        max_delta = std::max(max_delta, delta_difference);
        delta_match += delta_difference <= 1.0e-7f;
    }
    vrhino::require(sigma_match == 65 && timestep_match == 64 &&
                        delta_match == 64,
                    "Mochi production runtime schedule differs from Official");

    vrhino::TensorBundle implicit_input = production_input;
    for (const char* name : {"sampling_steps", "guidance_scale",
                             "threshold_noise", "linear_steps"})
        implicit_input.erase(name);
    vrhino::TensorBundle explicit_canary = implicit_input;
    explicit_canary.insert_or_assign("sampling_steps", vrhino::scalar_i64(2));
    explicit_canary.insert_or_assign("guidance_scale", vrhino::scalar_f32(4.5f));
    explicit_canary.insert_or_assign("threshold_noise", vrhino::scalar_f32(0.025f));
    explicit_canary.insert_or_assign("linear_steps", vrhino::scalar_i64(1));
    vrhino::require(exact_program(
                        architecture->create_program(implicit_input),
                        architecture->create_program(explicit_canary)),
                    "Mochi legacy canary declaration changed");

    int fail_closed = 0;
    for (int invalid = 0; invalid < 4; ++invalid) {
        vrhino::TensorBundle input = production_input;
        if (invalid == 0)
            input.insert_or_assign("sampling_steps", vrhino::scalar_i64(1));
        if (invalid == 1)
            input.insert_or_assign("guidance_scale", vrhino::scalar_f32(
                std::numeric_limits<float>::quiet_NaN()));
        if (invalid == 2)
            input.insert_or_assign("threshold_noise", vrhino::scalar_f32(1.0f));
        if (invalid == 3)
            input.insert_or_assign("linear_steps", vrhino::scalar_i64(64));
        try {
            (void)architecture->create_program(input);
        } catch (const vrhino::Error&) {
            ++fail_closed;
        }
    }
    vrhino::require(fail_closed == 4,
                    "Mochi sampling declaration did not fail closed");

    const vrhino::ComponentExecutionConfig component =
        vrhino::component_execution_config_from_json(
            profile.at("component_execution"));
    vrhino::require(component.mode == vrhino::ComponentExecutionMode::Tiled &&
                        component.tiling.planner ==
                            vrhino::TilePlannerPolicy::RecursiveBisection &&
                        component.tiling.merge ==
                            vrhino::TileMergePolicy::RecursiveOverlapReplace &&
                        component.tiling.recursive.tile_counts ==
                            std::array<int64_t, 2>{2, 4},
                    "Mochi production Component execution binding mismatch");

    std::cout << "status=PASS\nproduction_steps=64\nproduction_cfg=6"
              << "\nguidance_coefficients=[-5,6]"
              << "\nsigma_match=" << sigma_match << "/65"
              << "\ntimestep_match=" << timestep_match << "/64"
              << "\neuler_delta_match=" << delta_match << "/64"
              << "\nmax_sigma_abs_difference=" << max_sigma
              << "\nmax_timestep_abs_difference=" << max_timestep
              << "\nmax_delta_abs_difference=" << max_delta
              << "\nlegacy_canary=BYTE_EXACT_PASS"
              << "\nfail_closed=4/4\ncomponent_mode=TILED"
              << "\nplanner=RECURSIVE_BISECTION"
              << "\nmerge=RECURSIVE_OVERLAP_REPLACE\n";
}

void validate_short_chain(const std::string& model_path,
                          const std::string& fixture_path,
                          const vrhino::Json& profile) {
    vrhino::VrmModel model(model_path, false);
    vrhino::TensorBundle input = profile_input(profile, fixture_path);
    auto architecture = vrhino::create_architecture(model);
    vrhino::SamplingProgram program = architecture->create_program(input);
    program.steps = 3;
    program.latent_shape = {1, 12, 1, 2, 2};
    program.sigmas.resize(4);
    program.model_timesteps.resize(3);
    program.update_deltas.resize(3);

    vrhino::CudaBackend backend;
    configure_backend(backend, model);
    const vrhino::PrecisionPolicy policy = precision_policy(profile);
    auto denoiser = architecture->create_denoiser(backend, policy, input);
    vrhino::SamplingRuntime runtime(backend, policy);
    const vrhino::SamplingResult first = runtime.run(*denoiser, program);
    const vrhino::SamplingResult second = runtime.run(*denoiser, program);
    backend.synchronize();
    vrhino::require(exact_bundle(first.trace, second.trace),
                    "Mochi production-prefix short chain is not deterministic");
    int finite = 0;
    for (const auto& [name, tensor] : first.trace) {
        (void)name;
        finite += finite_host(tensor);
    }
    vrhino::require(finite == static_cast<int>(first.trace.size()),
                    "Mochi production-prefix short chain is non-finite");
    std::cout << "status=PASS\ndeclared_steps=64\nexecuted_prefix_steps=3"
              << "\nproduction_cfg=6\nfinite_tensors=" << finite << '/'
              << first.trace.size()
              << "\nnan=0\ninf=0\ndeterministic=BYTE_EXACT_PASS\n";
}

void validate_production_step(const std::string& model_path,
                              const std::string& fixture_path,
                              const vrhino::Json& profile,
                              int64_t temporal_frames) {
    vrhino::VrmModel model(model_path, false);
    vrhino::TensorBundle input = profile_input(profile, fixture_path);
    if (temporal_frames != 28)
        input.insert_or_assign("latent_shape", vrhino::host_i64(
            {5}, {1, 12, temporal_frames, 60, 106}));
    auto architecture = vrhino::create_architecture(model);
    const vrhino::SamplingProgram program = architecture->create_program(input);

    vrhino::CudaBackend backend;
    configure_backend(backend, model, false);
    const vrhino::PrecisionPolicy policy = precision_policy(profile);
    auto denoiser = architecture->create_denoiser(backend, policy, input);
    vrhino::Tensor host_latent = vrhino::Tensor::host(
        program.latent_shape, vrhino::DType::BF16);
    std::memset(host_latent.data(), 0, host_latent.bytes());
    const vrhino::Tensor latent = backend.copy_to_device(
        host_latent, vrhino::DType::BF16);
    backend.enable_profiling(true);
    const auto started = std::chrono::steady_clock::now();
    std::vector<vrhino::Tensor> predictions;
    try {
        predictions = denoiser->evaluate(
            latent, program.model_timestep_at(0));
    } catch (...) {
        const vrhino::MemoryRuntimeStats memory = backend.memory_runtime_stats();
        std::cerr << "temporal_frames=" << temporal_frames
                  << "\npeak_device_bytes=" << backend.peak_device_bytes()
                  << "\nactive_activation_bytes="
                  << memory.accounting.device_activation_bytes
                  << "\npeak_activation_bytes="
                  << memory.accounting.peak_device_activation_bytes
                  << "\nresident_weight_bytes="
                  << memory.accounting.device_resident_weight_bytes
                  << "\ntemporary_pool_reuses="
                  << memory.temporary_pool_reuses << '\n';
        throw;
    }
    backend.synchronize();
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const std::vector<int64_t> expected_shape =
        {1, 12, temporal_frames, 60, 106};
    vrhino::require(predictions.size() == 2,
                    "Mochi production-shape denoiser did not return CFG branches");
    int finite_predictions = 0;
    for (const vrhino::Tensor& prediction : predictions) {
        const vrhino::Tensor host = backend.copy_to_host(prediction);
        vrhino::require(host.shape() == expected_shape,
                        "Mochi production-shape prediction shape mismatch");
        finite_predictions += finite_host(host);
    }
    vrhino::require(finite_predictions == 2,
                    "Mochi production-shape step is invalid");

    uint64_t attention = 0;
    uint64_t sdpa = 0;
    const auto profiles = backend.profile_stats();
    for (const auto& [name, stat] : profiles) {
        if (name == "attention") attention += stat.calls;
        if (name.rfind("attention.cudnn_sdpa|", 0) == 0)
            sdpa += stat.calls;
    }
    vrhino::require(attention == 49 && sdpa == 48,
                    "Mochi production SDPA admission telemetry mismatch");
    std::cout << "status=PASS\nlatent_shape="
              << "[1,12," << temporal_frames << ",60,106]"
              << "\nexecuted_steps=1\nattention_total=" << attention
              << "\ncudnn_sdpa_selected=" << sdpa
              << "\nfallback=" << attention - sdpa
              << "\nfallback_reason=score_volume_below_8m_threshold"
              << "\nfinite_predictions=2/2\nnan=0\ninf=0"
              << "\nwall_seconds=" << wall
              << "\npeak_device_bytes=" << backend.peak_device_bytes()
              << '\n';
}

vrhino::Tensor zero_latent(bool full_temporal) {
    vrhino::Tensor result = vrhino::Tensor::host(
        {1, 12, full_temporal ? 28 : 1, 60, 106}, vrhino::DType::BF16);
    std::memset(result.data(), 0, result.bytes());
    return result;
}

void require_plan(const std::vector<vrhino::ComponentTileRegion>& actual,
                  const std::vector<std::pair<int64_t, int64_t>>& expected,
                  const char* name) {
    vrhino::require(actual.size() == expected.size(),
                    std::string(name) + " planner count mismatch");
    for (size_t index = 0; index < expected.size(); ++index)
        vrhino::require(actual[index].start == expected[index].first &&
                            actual[index].stop == expected[index].second,
                        std::string(name) + " planner interval mismatch");
}

void validate_vae(const std::string& model_path, const vrhino::Json& profile,
                  bool full_temporal) {
    vrhino::VrmModel model(model_path, false);
    vrhino::CudaBackend backend;
    configure_backend(backend, model, false);
    auto architecture = vrhino::create_architecture(model);
    const vrhino::PrecisionPolicy policy = precision_policy(profile);
    const vrhino::ComponentExecutionConfig execution =
        vrhino::component_execution_config_from_json(
            profile.at("component_execution"));
    vrhino::NativeRuntime runtime(backend, policy, execution);
    const vrhino::Tensor latent = zero_latent(full_temporal);

    vrhino::Tensor first;
    const auto first_started = std::chrono::steady_clock::now();
    {
        vrhino::Tensor device = runtime.decode_component(
            *architecture, latent, {});
        backend.synchronize();
        first = backend.copy_to_host(device);
    }
    const double first_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - first_started).count();

    vrhino::Tensor second;
    const auto second_started = std::chrono::steady_clock::now();
    {
        vrhino::Tensor device = runtime.decode_component(
            *architecture, latent, {});
        backend.synchronize();
        second = backend.copy_to_host(device);
    }
    const double second_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - second_started).count();
    const std::vector<int64_t> expected_shape = full_temporal
        ? std::vector<int64_t>({1, 3, 163, 480, 848})
        : std::vector<int64_t>({1, 3, 1, 480, 848});
    vrhino::require(first.shape() == expected_shape &&
                        first.dtype() == vrhino::DType::BF16 &&
                        exact_tensor(first, second) && finite_host(first),
                    "Mochi production tiled VAE validation failed");
    const auto& stats = runtime.component_execution_stats();
    vrhino::require(stats.mode == vrhino::ComponentExecutionMode::Tiled &&
                        stats.tiling.temporal_tiles == 1 &&
                        stats.tiling.spatial_tiles_per_temporal == 8 &&
                        stats.tiling.graph_executions == 8,
                    "Mochi production tiled VAE telemetry mismatch");
    require_plan(stats.tiling.height_plan,
                 {{0, 34}, {26, 60}}, "height");
    require_plan(stats.tiling.width_plan,
                 {{0, 32}, {24, 57}, {49, 81}, {73, 106}}, "width");
    std::cout << "status=PASS\nexecutor_mode=TILED"
              << "\nplanner=RECURSIVE_BISECTION"
              << "\nmerge=RECURSIVE_OVERLAP_REPLACE"
              << "\ntemporal_tiles=1\nspatial_tiles=8"
              << "\ngraph_executions=8"
              << "\noutput_shape="
              << (full_temporal ? "[1,3,163,480,848]" : "[1,3,1,480,848]")
              << "\nfirst_wall_seconds=" << first_wall
              << "\nrepeat_wall_seconds=" << second_wall
              << "\npeak_device_bytes=" << backend.peak_device_bytes()
              << "\nfinite=PASS\nnan=0\ninf=0"
              << "\ndeterministic=BYTE_EXACT_PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool temporal_mode = argc >= 2 &&
            std::string(argv[1]) == "temporal-step";
        vrhino::require(argc == 5 || (temporal_mode && argc == 6),
            "usage: vrhino-phase22e-mochi-binding-tests "
            "{declaration|short-chain|production-step|spatial-step|temporal-step|vae|spatial-vae} "
            "MODEL FIXTURE PROFILE [TEMPORAL_FRAMES]");
        const std::string mode = argv[1];
        const vrhino::Json profile = vrhino::Json::parse(read_text(argv[4]));
        if (mode == "declaration")
            validate_declaration(argv[2], argv[3], profile);
        else if (mode == "short-chain")
            validate_short_chain(argv[2], argv[3], profile);
        else if (mode == "production-step")
            validate_production_step(argv[2], argv[3], profile, 28);
        else if (mode == "spatial-step")
            validate_production_step(argv[2], argv[3], profile, 1);
        else if (mode == "temporal-step")
            validate_production_step(
                argv[2], argv[3], profile, std::stoll(argv[5]));
        else if (mode == "vae")
            validate_vae(argv[2], profile, true);
        else if (mode == "spatial-vae")
            validate_vae(argv[2], profile, false);
        else
            throw vrhino::Error("Unknown Phase 22E validation mode");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase22e_mochi_binding_tests: " << error.what() << '\n';
        return 1;
    }
}
