#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/model_package.h"
#include "vrhino/precision.h"
#include "vrhino/sampling.h"
#include "vrhino/temporal_conditional_unet_2d.h"

namespace fs = std::filesystem;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Difference {
    double maximum = 0.0;
    double mean = 0.0;
    double cosine = 0.0;
    double reference_min = 0.0;
    double reference_max = 0.0;
    double native_min = 0.0;
    double native_max = 0.0;
    uint64_t nan = 0;
    uint64_t inf = 0;
};

Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& reference) {
    check(actual.device().is_host() && reference.device().is_host() &&
              actual.dtype() == vrhino::DType::F32 &&
              reference.dtype() == vrhino::DType::F32 &&
              actual.shape() == reference.shape(),
          "oracle comparison tensor contract mismatch");
    Difference result;
    const float* left = actual.data_as<float>();
    const float* right = reference.data_as<float>();
    result.reference_min = result.reference_max = right[0];
    result.native_min = result.native_max = left[0];
    long double error = 0.0, dot = 0.0, left_norm = 0.0, right_norm = 0.0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        result.nan += std::isnan(left[index]);
        result.inf += std::isinf(left[index]);
        result.reference_min = std::min<double>(result.reference_min, right[index]);
        result.reference_max = std::max<double>(result.reference_max, right[index]);
        result.native_min = std::min<double>(result.native_min, left[index]);
        result.native_max = std::max<double>(result.native_max, left[index]);
        const double difference = std::abs(
            static_cast<double>(left[index]) - right[index]);
        result.maximum = std::max(result.maximum, difference);
        error += difference;
        dot += static_cast<long double>(left[index]) * right[index];
        left_norm += static_cast<long double>(left[index]) * left[index];
        right_norm += static_cast<long double>(right[index]) * right[index];
    }
    result.mean = static_cast<double>(error / actual.numel());
    result.cosine = static_cast<double>(dot /
        std::sqrt(std::max<long double>(left_norm * right_norm, 1.0e-30L)));
    return result;
}

Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual, const fs::path& expected,
                   double maximum_gate, double cosine_gate) {
    const vrhino::Tensor host = actual.device().is_host()
        ? actual : backend.copy_to_host(actual);
    const Difference difference = compare(
        host, vrhino::test::read_npy_f16_as_f32(expected.string()));
    std::cout << std::setprecision(10) << name << " shape=";
    for (int64_t value : host.shape()) std::cout << value << 'x';
    std::cout << " ref=[" << difference.reference_min << ','
              << difference.reference_max << "] native=["
              << difference.native_min << ',' << difference.native_max
              << "] max_abs=" << difference.maximum
              << " mean_abs=" << difference.mean
              << " cosine=" << difference.cosine
              << " nan=" << difference.nan << " inf=" << difference.inf << '\n';
    check(difference.nan == 0 && difference.inf == 0 &&
              difference.maximum <= maximum_gate &&
              difference.cosine >= cosine_gate,
          name + " numerical gate failed");
    return difference;
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "cannot read Phase-1 qualification report");
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

struct Health {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double standard_deviation = 0.0;
    uint64_t nan = 0;
    uint64_t inf = 0;
};

Health health(const vrhino::Tensor& value) {
    check(value.device().is_host() && value.dtype() == vrhino::DType::F32 &&
              value.numel() > 0,
          "sampling health tensor contract mismatch");
    const float* data = value.data_as<float>();
    Health result;
    result.minimum = result.maximum = data[0];
    long double sum = 0.0;
    long double square_sum = 0.0;
    for (int64_t index = 0; index < value.numel(); ++index) {
        result.nan += std::isnan(data[index]);
        result.inf += std::isinf(data[index]);
        result.minimum = std::min<double>(result.minimum, data[index]);
        result.maximum = std::max<double>(result.maximum, data[index]);
        sum += data[index];
        square_sum += static_cast<long double>(data[index]) * data[index];
    }
    result.mean = static_cast<double>(sum / value.numel());
    const long double variance = square_sum / value.numel() -
        static_cast<long double>(result.mean) * result.mean;
    result.standard_deviation = std::sqrt(
        static_cast<double>(std::max<long double>(variance, 0.0)));
    return result;
}

void qualify_all_step_health(const std::string& prefix,
                             vrhino::CudaBackend& backend,
                             const vrhino::SamplingResult& result,
                             const vrhino::SamplingProgram& program,
                             const vrhino::Json& report_chunk) {
    const auto& steps = report_chunk.at("steps").array();
    check(steps.size() == 20 && report_chunk.at("frames").integer() ==
              program.latent_shape.at(2),
          prefix + " Phase-1 step-manifest contract mismatch");
    double worst_extreme = 0.0;
    double worst_mean = 0.0;
    double worst_standard_deviation = 0.0;
    for (int step = 0; step < 20; ++step) {
        const auto& expected = steps.at(static_cast<size_t>(step));
        const auto& expected_health = expected.at("health");
        check(expected.at("index").integer() == step &&
                  expected.at("epsilon_cfg_sha256").string().size() == 64 &&
                  expected.at("input_latent_sha256").string().size() == 64 &&
                  expected.at("output_latent_sha256").string().size() == 64,
              prefix + " Phase-1 step identity manifest is incomplete");
        const vrhino::Tensor& timestep = program.model_timestep_at(step);
        check(timestep.dtype() == vrhino::DType::I64 && timestep.numel() == 1 &&
                  timestep.data_as<int64_t>()[0] ==
                      expected.at("timestep").integer(),
              prefix + " DDIM timestep differs from Phase-1 manifest");
        const vrhino::Tensor& output =
            result.trace.at("step." + std::to_string(step) + ".latent");
        const Health actual = health(output.device().is_host()
            ? output : backend.copy_to_host(output));
        check(actual.nan == 0 && actual.inf == 0 &&
                  expected_health.at("nan").integer() == 0 &&
                  expected_health.at("inf").integer() == 0,
              prefix + " sampling step contains NaN/Inf");
        const double extreme_delta = std::max(
            std::abs(actual.minimum - expected_health.at("min").number()),
            std::abs(actual.maximum - expected_health.at("max").number()));
        const double mean_delta =
            std::abs(actual.mean - expected_health.at("mean").number());
        const double standard_deviation_delta = std::abs(
            actual.standard_deviation - expected_health.at("std").number());
        worst_extreme = std::max(worst_extreme, extreme_delta);
        worst_mean = std::max(worst_mean, mean_delta);
        worst_standard_deviation = std::max(
            worst_standard_deviation, standard_deviation_delta);
        check(extreme_delta <= 0.12 && mean_delta <= 0.01 &&
                  standard_deviation_delta <= 0.01,
              prefix + " all-step Phase-1 health gate failed");
        if (step > 0) {
            const vrhino::Tensor& input = result.trace.at(
                "step." + std::to_string(step) + ".input_latent");
            const vrhino::Tensor& previous = result.trace.at(
                "step." + std::to_string(step - 1) + ".latent");
            const Difference continuity = compare(input, previous);
            check(continuity.maximum == 0.0,
                  prefix + " sampling state chain is discontinuous");
        }
    }
    std::cout << prefix << " all_steps=20 worst_health_extreme_delta="
              << worst_extreme << " worst_health_mean_delta=" << worst_mean
              << " worst_health_std_delta=" << worst_standard_deviation
              << " state_chain_exact=true\n";
}

void qualify_all_step_tensors(const std::string& prefix,
                              vrhino::CudaBackend& backend,
                              const vrhino::SamplingResult& result,
                              const fs::path& diagnostic) {
    double worst_cfg_maximum = 0.0;
    double worst_cfg_mean = 0.0;
    double minimum_cfg_cosine = 1.0;
    double worst_output_maximum = 0.0;
    double worst_output_mean = 0.0;
    double minimum_output_cosine = 1.0;
    for (int step = 0; step < 20; ++step) {
        const std::string index = step < 10 ? "0" + std::to_string(step)
                                            : std::to_string(step);
        const std::string root = prefix + "_step" + index;
        const Difference cfg = qualify(root + ".all.cfg", backend,
            result.trace.at("step." + std::to_string(step) + ".guidance"),
            diagnostic / (root + "_epsilon_cfg.npy"), 0.18, 0.997);
        const Difference output = qualify(root + ".all.output", backend,
            result.trace.at("step." + std::to_string(step) + ".latent"),
            diagnostic / (root + "_output_latent.npy"), 0.2, 0.998);
        worst_cfg_maximum = std::max(worst_cfg_maximum, cfg.maximum);
        worst_cfg_mean = std::max(worst_cfg_mean, cfg.mean);
        minimum_cfg_cosine = std::min(minimum_cfg_cosine, cfg.cosine);
        worst_output_maximum = std::max(worst_output_maximum, output.maximum);
        worst_output_mean = std::max(worst_output_mean, output.mean);
        minimum_output_cosine = std::min(minimum_output_cosine, output.cosine);
    }
    std::cout << prefix << " all_step_tensor_comparisons=20"
              << " worst_cfg_max_abs=" << worst_cfg_maximum
              << " worst_cfg_mean_abs=" << worst_cfg_mean
              << " minimum_cfg_cosine=" << minimum_cfg_cosine
              << " worst_output_max_abs=" << worst_output_maximum
              << " worst_output_mean_abs=" << worst_output_mean
              << " minimum_output_cosine=" << minimum_output_cosine << '\n';
}

vrhino::Tensor zeros_like(const vrhino::Tensor& source) {
    vrhino::Tensor result = vrhino::Tensor::host(source.shape(), vrhino::DType::F32);
    std::fill(result.data_as<float>(),
              result.data_as<float>() + result.numel(), 0.0f);
    return result;
}

class OracleDenoiser final : public vrhino::Denoiser {
public:
    OracleDenoiser(vrhino::CudaBackend& backend,
                   vrhino::TemporalConditionalUNet2DComponentExecutor& executor,
                   const vrhino::Json& graph, const vrhino::Tensor& first_input,
                   const vrhino::Tensor& audio, const fs::path& boundaries,
                   bool capture_boundaries)
        : backend_(backend), executor_(executor), graph_(graph),
          boundaries_(boundaries), capture_boundaries_(capture_boundaries) {
        check(first_input.ndim() == 5 && first_input.dim(0) == 1 &&
                  first_input.dim(1) == 13,
              "oracle first UNet input mismatch");
        static_context_ = backend_.slice(first_input, 1, 4, 13);
        vrhino::Tensor audio_batch = backend_.reshape(audio,
            {1, audio.dim(0), audio.dim(1), audio.dim(2)});
        vrhino::Tensor uncond = backend_.copy_to_device(
            zeros_like(vrhino::Tensor::host(audio_batch.shape(), vrhino::DType::F32)),
            backend_.execution_dtype());
        audio_pair_ = backend_.concat({uncond, audio_batch}, 0);
    }

    std::vector<vrhino::Tensor> evaluate(
            const vrhino::Tensor& latent, const vrhino::Tensor& timestep) override {
        vrhino::Tensor assembled = backend_.concat({latent, static_context_}, 1);
        vrhino::Tensor pair = backend_.concat({assembled, assembled}, 0);
        vrhino::TemporalConditionalUNetObservation observation;
        if (capture_boundaries_ && calls_ == 0) {
            observation.capture = [&](const std::string& name,
                                      const vrhino::Tensor& tensor) {
                const double maximum_gate = name == "final_epsilon" ? 0.08 : 0.12;
                const double cosine_gate = name == "final_epsilon" ? 0.999 : 0.998;
                boundary_differences_[name] = qualify(name, backend_, tensor,
                    boundaries_ / (name + ".npy"), maximum_gate, cosine_gate);
            };
        }
        const auto result = executor_.execute(graph_, pair, timestep, audio_pair_,
            observation.capture ? &observation : nullptr);
        ++calls_;
        return backend_.split(result.epsilon, {1, 1}, 0);
    }

    int calls() const { return calls_; }
    const std::map<std::string, Difference>& boundary_differences() const {
        return boundary_differences_;
    }

private:
    vrhino::CudaBackend& backend_;
    vrhino::TemporalConditionalUNet2DComponentExecutor& executor_;
    const vrhino::Json& graph_;
    vrhino::Tensor static_context_;
    vrhino::Tensor audio_pair_;
    fs::path boundaries_;
    bool capture_boundaries_ = false;
    int calls_ = 0;
    std::map<std::string, Difference> boundary_differences_;
};

vrhino::SamplingProgram program_for(int64_t frames) {
    vrhino::SamplingProgram program;
    program.latent_shape = {1, 4, frames, 64, 64};
    program.seed = 1247;
    program.steps = 20;
    program.guidance_mode = vrhino::GuidanceMode::CFG;
    program.guidance_coefficients = {0.0f, 1.5f};
    program.contract.emplace(
        vrhino::PredictionContract{vrhino::PredictionSemantic::Epsilon},
        vrhino::SolverContract{vrhino::SolverSemantic::AffineFirstOrder, 1},
        vrhino::make_scaled_linear_ddim_schedule(
            1000, 0.00085f, 0.012f, 20, 1, false));
    return program;
}

vrhino::Tensor initial_latent(vrhino::CudaBackend& backend,
                              const vrhino::Tensor& first_input) {
    return backend.slice(first_input, 1, 0, 4);
}

void qualify_selected_steps(const std::string& prefix,
                            vrhino::CudaBackend& backend,
                            const vrhino::SamplingResult& result,
                            const fs::path& scheduler) {
    for (int step : {0, 9, 19}) {
        const std::string index = step < 10 ? "0" + std::to_string(step)
                                            : std::to_string(step);
        const std::string root = prefix + "_step" + index;
        qualify(root + ".input", backend,
            result.trace.at("step." + std::to_string(step) + ".input_latent"),
            scheduler / (root + "_input_latent.npy"), 0.12, 0.998);
        qualify(root + ".unconditional", backend,
            result.trace.at("step." + std::to_string(step) + ".prediction.0"),
            scheduler / (root + "_epsilon_uncond.npy"), 0.12, 0.998);
        qualify(root + ".conditional", backend,
            result.trace.at("step." + std::to_string(step) + ".prediction.1"),
            scheduler / (root + "_epsilon_cond.npy"), 0.12, 0.998);
        qualify(root + ".cfg", backend,
            result.trace.at("step." + std::to_string(step) + ".guidance"),
            scheduler / (root + "_epsilon_cfg.npy"), 0.18, 0.997);
        qualify(root + ".x0", backend,
            result.trace.at("step." + std::to_string(step) + ".predicted_x0"),
            scheduler / (root + "_predicted_x0.npy"), 0.4, 0.997);
        qualify(root + ".output", backend,
            result.trace.at("step." + std::to_string(step) + ".latent"),
            scheduler / (root + "_output_latent.npy"), 0.2, 0.998);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool bf16_profile = argc == 5 &&
        std::string(argv[3]) == "--bf16-profile";
    const bool bf16_reuse = argc == 5 &&
        std::string(argv[3]) == "--bf16-reuse";
    if (argc != 3 && argc != 4 && !bf16_profile && !bf16_reuse) {
        std::cerr << "usage: temporal_conditional_unet_oracle_tests "
                     "COMPONENT_VRM ORACLE_ROOT [ALL_STEP_REFERENCE]\n"
                     "   or: temporal_conditional_unet_oracle_tests "
                     "COMPONENT_VRM ORACLE_ROOT --bf16-profile POLICY.json\n"
                     "   or: temporal_conditional_unet_oracle_tests "
                     "COMPONENT_VRM ORACLE_ROOT --bf16-reuse POLICY.json\n";
        return 2;
    }
    try {
        check(vrhino::test::binary16_to_float(0x0001U) ==
                  std::ldexp(1.0f, -24) &&
                  vrhino::test::binary16_to_float(0x3c00U) == 1.0f,
              "binary16 oracle reader semantics mismatch");
        vrhino::VrmModel component(argv[1], true);
        check(component.architecture_id() == "temporal-unet" &&
                  component.graph().at("kind").string() ==
                      "temporal_conditional_unet_2d",
              "temporal UNet component identity mismatch");
        const vrhino::PrecisionPolicy policy = bf16_profile || bf16_reuse
            ? vrhino::PrecisionPolicy::from_json(
                  vrhino::Json::parse(read_text(argv[4])))
            : vrhino::PrecisionPolicy::fp32();
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(policy.requested_dtype());
        backend.enable_weight_cache(true);
        backend.enable_profiling(bf16_profile || bf16_reuse);
        vrhino::TemporalConditionalUNet2DComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())),
            policy);
        const fs::path root = argv[2];
        check(vrhino::product::sha256_file(root / "qualification-report.json") ==
                  "1507b0e27b37f27e21a2ec2b7cecd3442fa276adfad64fb9bccd413accbf8bd4",
              "Phase-1 qualification report identity drift");
        const vrhino::Json report = vrhino::Json::parse(
            read_text(root / "qualification-report.json"));
        const auto& report_chunks = report.at("chunks").array();
        check(report_chunks.size() == 2,
              "Phase-1 qualification report chunk count mismatch");
        const fs::path capture = root / "profile17/capture";
        const fs::path scheduler = capture / "scheduler";
        const auto input16 = vrhino::test::read_npy_f16_as_f32(
            (capture / "unet/chunk0_13ch_input.npy").string());
        const auto audio16 = vrhino::test::read_npy_f16_as_f32(
            (capture / "audio/chunk0_conditioning.npy").string());
        if (bf16_reuse) {
            struct ReuseResult {
                vrhino::Tensor final_latent;
                double seconds = 0.0;
                size_t upload_bytes = 0;
                double upload_seconds = 0.0;
                uint64_t cache_misses = 0;
                uint64_t cache_hits = 0;
                size_t resident_bytes = 0;
                double step0_seconds = 0.0;
                double steady_p50_seconds = 0.0;
                double sdpa_cold_excess_seconds = 0.0;
                double convolution_cold_excess_seconds = 0.0;
                double linear_cold_excess_seconds = 0.0;
            };
            auto execute_chunk = [&](const int64_t frames) {
                backend.set_execution_dtype(vrhino::DType::F32);
                const vrhino::Tensor input = frames == 16
                    ? input16 : backend.slice(input16, 2, 0, frames);
                const vrhino::Tensor audio = frames == 16
                    ? audio16 : backend.slice(audio16, 0, 0, frames);
                OracleDenoiser denoiser(
                    backend, executor, component.graph(), input, audio, {}, false);
                const vrhino::Tensor initial = initial_latent(backend, input);
                const size_t upload_before = backend.weight_upload_bytes();
                const double upload_seconds_before = backend.weight_upload_seconds();
                const uint64_t misses_before = backend.weight_cache_misses();
                const uint64_t hits_before = backend.weight_cache_hits();
                backend.set_execution_dtype(policy.requested_dtype());
                vrhino::SamplingRuntime runtime(backend, policy);
                backend.synchronize();
                const auto started = std::chrono::steady_clock::now();
                const vrhino::SamplingResult sampled =
                    runtime.run_with_external_initial_state_for_test(
                        denoiser, program_for(frames), initial);
                backend.synchronize();
                ReuseResult result;
                result.seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                result.final_latent = backend.copy_to_host(sampled.final_latent);
                result.upload_bytes =
                    backend.weight_upload_bytes() - upload_before;
                result.upload_seconds =
                    backend.weight_upload_seconds() - upload_seconds_before;
                result.cache_misses =
                    backend.weight_cache_misses() - misses_before;
                result.cache_hits = backend.weight_cache_hits() - hits_before;
                result.resident_bytes = backend.weight_cache_resident_bytes();
                const auto profile = backend.profile_stats();
                std::vector<double> steady_steps;
                for (int step = 0; step < 20; ++step) {
                    const auto& stat = profile.at(
                        "sampling.denoiser.step." + std::to_string(step));
                    const double seconds = stat.device_milliseconds / 1000.0;
                    if (step == 0) result.step0_seconds = seconds;
                    else steady_steps.push_back(seconds);
                }
                std::sort(steady_steps.begin(), steady_steps.end());
                result.steady_p50_seconds = steady_steps[9];
                for (const auto& [name, stat] : profile) {
                    const double excess = std::max(
                        0.0, stat.maximum_milliseconds - stat.p50_milliseconds) /
                        1000.0;
                    if (name.starts_with("attention.cudnn_sdpa|"))
                        result.sdpa_cold_excess_seconds += excess;
                    else if (name.starts_with("conv2d|"))
                        result.convolution_cold_excess_seconds += excess;
                    else if (name.starts_with("linear|"))
                        result.linear_cold_excess_seconds += excess;
                }
                return result;
            };

            std::vector<ReuseResult> chunks;
            for (const int64_t frames : {16, 16, 16, 2})
                chunks.push_back(execute_chunk(frames));
            check(compare(chunks[0].final_latent,
                          chunks[1].final_latent).maximum == 0.0 &&
                      compare(chunks[0].final_latent,
                              chunks[2].final_latent).maximum == 0.0,
                  "reused F=16 executor changed deterministic output");
            const Health tail = health(chunks[3].final_latent);
            check(tail.nan == 0 && tail.inf == 0 &&
                      chunks[3].final_latent.shape() ==
                          std::vector<int64_t>({1, 4, 2, 64, 64}),
                  "reused executor rejected or corrupted dynamic F=2 shape");
            check(chunks[0].cache_misses > 0 &&
                      chunks[1].cache_misses == 0 &&
                      chunks[2].cache_misses == 0 &&
                      chunks[3].cache_misses == 0,
                  "immutable parameter cache was reconstructed across chunks");
            check(chunks[0].resident_bytes > 0 &&
                      chunks[1].resident_bytes == chunks[0].resident_bytes &&
                      chunks[2].resident_bytes == chunks[0].resident_bytes &&
                      chunks[3].resident_bytes == chunks[0].resident_bytes,
                  "resident parameter cache lifetime changed across chunks");
            std::cout << std::setprecision(10)
                      << "BF16_REUSE pattern=16,16,16,2";
            for (size_t index = 0; index < chunks.size(); ++index) {
                std::cout << " chunk" << index
                          << "_seconds=" << chunks[index].seconds
                          << " chunk" << index
                          << "_upload_bytes=" << chunks[index].upload_bytes
                          << " chunk" << index
                          << "_upload_seconds=" << chunks[index].upload_seconds
                          << " chunk" << index
                          << "_cache_misses=" << chunks[index].cache_misses
                          << " chunk" << index
                          << "_cache_hits=" << chunks[index].cache_hits
                          << " chunk" << index
                          << "_resident_bytes=" << chunks[index].resident_bytes
                          << " chunk" << index
                          << "_step0_seconds=" << chunks[index].step0_seconds
                          << " chunk" << index
                          << "_steady_p50_seconds="
                          << chunks[index].steady_p50_seconds
                          << " chunk" << index
                          << "_sdpa_cold_excess_seconds="
                          << chunks[index].sdpa_cold_excess_seconds
                          << " chunk" << index
                          << "_conv_cold_excess_seconds="
                          << chunks[index].convolution_cold_excess_seconds
                          << " chunk" << index
                          << "_linear_cold_excess_seconds="
                          << chunks[index].linear_cold_excess_seconds;
            }
            std::cout << " repeated_f16_exact=true dynamic_f2=true\n";
            return 0;
        }
        if (bf16_profile) {
            backend.set_execution_dtype(vrhino::DType::F32);
            OracleDenoiser denoiser(backend, executor, component.graph(), input16,
                audio16, {}, false);
            const vrhino::Tensor initial = initial_latent(backend, input16);
            backend.set_execution_dtype(policy.requested_dtype());
            vrhino::SamplingRuntime runtime(backend, policy);
            const vrhino::SamplingProgram program = program_for(16);
            backend.synchronize();
            const auto started = std::chrono::steady_clock::now();
            const auto result = runtime.run_with_external_initial_state_for_test(
                denoiser, program, initial);
            backend.synchronize();
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const vrhino::Tensor actual = result.final_latent.device().is_host()
                ? result.final_latent : backend.copy_to_host(result.final_latent);
            const Difference difference = compare(actual,
                vrhino::test::read_npy_f16_as_f32(
                    (scheduler / "chunk0_final_latent_scaled.npy").string()));
            // This component-level Class-C gate compares twenty BF16 denoising
            // steps with the frozen F=16 FP32/Official-derived final latent.
            // The separate mean and cosine bounds protect the full tensor from
            // a sparse BF16 maximum while allowing the expected 7-bit-mantissa
            // rounding accumulated over the fixed schedule.
            check(difference.nan == 0 && difference.inf == 0 &&
                      difference.maximum <= 0.75 &&
                      difference.mean <= 0.004 &&
                      difference.cosine >= 0.9999,
                  "BF16 F=16 numerical qualification gate failed");
            std::cout << std::setprecision(10)
                      << "BF16_F16_total_seconds=" << seconds
                      << " BF16_F16_mean_step_seconds=" << seconds / 20.0
                      << " final_max_abs=" << difference.maximum
                      << " final_mean_abs=" << difference.mean
                      << " final_cosine=" << difference.cosine
                      << " weight_upload_bytes=" << backend.weight_upload_bytes()
                      << " weight_upload_seconds=" << backend.weight_upload_seconds()
                      << " weight_cache_resident_bytes="
                      << backend.weight_cache_resident_bytes()
                      << " peak_device_bytes=" << backend.peak_device_bytes()
                      << '\n';
            for (const auto& [name, stat] : backend.profile_stats())
                std::cout << "PROFILE name=" << name
                          << " calls=" << stat.calls
                          << " device_ms=" << stat.device_milliseconds
                          << " mean_ms=" << stat.mean_milliseconds
                          << " p50_ms=" << stat.p50_milliseconds
                          << " p95_ms=" << stat.p95_milliseconds
                          << " min_ms=" << stat.minimum_milliseconds
                          << " max_ms=" << stat.maximum_milliseconds << '\n';
            return 0;
        }
        OracleDenoiser denoiser16(backend, executor, component.graph(), input16,
            audio16, capture / "unet/boundaries", true);
        vrhino::SamplingRuntime runtime16(backend, vrhino::PrecisionPolicy::fp32());
        const vrhino::SamplingProgram program16 = program_for(16);
        backend.synchronize();
        const auto start16 = std::chrono::steady_clock::now();
        const auto result16 = runtime16.run_with_external_initial_state_for_test(
            denoiser16, program16, initial_latent(backend, input16));
        backend.synchronize();
        const double seconds16 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start16).count();
        check(denoiser16.calls() == 20 &&
                  denoiser16.boundary_differences().size() == 19,
              "F=16 temporal execution/observation count mismatch");
        qualify_selected_steps("chunk0", backend, result16, scheduler);
        qualify_all_step_health(
            "chunk0", backend, result16, program16, report_chunks.at(0));
        if (argc == 4)
            qualify_all_step_tensors("chunk0", backend, result16, argv[3]);
        qualify("chunk0.final", backend, result16.final_latent,
            scheduler / "chunk0_final_latent_scaled.npy", 0.2, 0.998);

        const auto input1 = vrhino::test::read_npy_f16_as_f32(
            (capture / "unet/chunk1_13ch_input.npy").string());
        const auto audio1 = vrhino::test::read_npy_f16_as_f32(
            (capture / "audio/chunk1_conditioning.npy").string());
        OracleDenoiser denoiser1(backend, executor, component.graph(), input1,
            audio1, {}, false);
        vrhino::SamplingRuntime runtime1(backend, vrhino::PrecisionPolicy::fp32());
        const vrhino::SamplingProgram program1 = program_for(1);
        backend.synchronize();
        const auto start1 = std::chrono::steady_clock::now();
        const auto result1 = runtime1.run_with_external_initial_state_for_test(
            denoiser1, program1, initial_latent(backend, input1));
        backend.synchronize();
        const double seconds1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start1).count();
        qualify_selected_steps("chunk1", backend, result1, scheduler);
        qualify_all_step_health(
            "chunk1", backend, result1, program1, report_chunks.at(1));
        if (argc == 4)
            qualify_all_step_tensors("chunk1", backend, result1, argv[3]);
        qualify("chunk1.final", backend, result1.final_latent,
            scheduler / "chunk1_final_latent_scaled.npy", 0.2, 0.998);

        OracleDenoiser independent1(backend, executor, component.graph(), input1,
            audio1, {}, false);
        vrhino::SamplingRuntime independent_runtime(
            backend, vrhino::PrecisionPolicy::fp32());
        const auto independent =
            independent_runtime.run_with_external_initial_state_for_test(
                independent1, program_for(1), initial_latent(backend, input1));
        const Difference independence = compare(
            backend.copy_to_host(result1.final_latent),
            backend.copy_to_host(independent.final_latent));
        check(independence.maximum == 0.0 && independence.nan == 0 &&
                  independence.inf == 0,
              "F=1 chunk retains hidden temporal state");
        std::cout << "F16_total_seconds=" << seconds16
                  << " F16_mean_step_seconds=" << seconds16 / 20.0
                  << " F1_total_seconds=" << seconds1
                  << " F1_mean_step_seconds=" << seconds1 / 20.0
                  << " peak_device_bytes=" << backend.peak_device_bytes()
                  << " chunk_independence_max_abs=" << independence.maximum
                  << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
