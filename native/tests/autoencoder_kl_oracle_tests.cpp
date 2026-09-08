#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "npy_fixture.h"
#include "vrhino/autoencoder_kl.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/loader.h"
#include "vrhino/tensor_util.h"

namespace fs = std::filesystem;

namespace {

struct Difference {
    double maximum_absolute = 0.0;
    double mean_absolute = 0.0;
    double maximum_relative = 0.0;
    double cosine = 0.0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
};

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& expected) {
    check(actual.device().is_host() && expected.device().is_host() &&
          actual.dtype() == vrhino::DType::F32 && expected.dtype() == vrhino::DType::F32 &&
          actual.shape() == expected.shape(), "oracle comparison contract mismatch");
    Difference result;
    long double absolute_sum = 0.0, dot = 0.0, left_norm = 0.0, right_norm = 0.0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double left = actual.data_as<float>()[index];
        const double right = expected.data_as<float>()[index];
        if (std::isnan(left)) ++result.nan_count;
        if (std::isinf(left)) ++result.inf_count;
        const double absolute = std::abs(left - right);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        absolute_sum += absolute;
        result.maximum_relative = std::max(result.maximum_relative,
            absolute / std::max(1.0e-8, std::abs(right)));
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    result.mean_absolute = static_cast<double>(absolute_sum / actual.numel());
    result.cosine = static_cast<double>(dot / std::sqrt(left_norm * right_norm));
    return result;
}

void report(const std::string& name, const Difference& value) {
    std::cout << std::setprecision(10) << name
              << " max_abs=" << value.maximum_absolute
              << " mean_abs=" << value.mean_absolute
              << " max_rel=" << value.maximum_relative
              << " cosine=" << value.cosine
              << " nan=" << value.nan_count
              << " inf=" << value.inf_count << '\n';
}

Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual, const fs::path& expected,
                   double maximum_absolute, double minimum_cosine) {
    const Difference difference = compare(backend.copy_to_host(actual),
        vrhino::test::read_npy_f32(expected.string()));
    report(name, difference);
    check(difference.nan_count == 0 && difference.inf_count == 0,
          name + " contains non-finite values");
    check(difference.maximum_absolute <= maximum_absolute &&
          difference.cosine >= minimum_cosine,
          name + " exceeds the FP32 oracle gate");
    return difference;
}

vrhino::Tensor fixed_epsilon(const vrhino::Tensor& mean,
                             const vrhino::Tensor& logvar,
                             const vrhino::Tensor& sample) {
    check(mean.shape() == logvar.shape() && mean.shape() == sample.shape(),
          "fixed epsilon source shape mismatch");
    vrhino::Tensor epsilon = vrhino::Tensor::host(mean.shape(), vrhino::DType::F32);
    for (int64_t index = 0; index < mean.numel(); ++index) {
        const float deviation = std::exp(0.5f * logvar.data_as<float>()[index]);
        epsilon.data_as<float>()[index] =
            (sample.data_as<float>()[index] - mean.data_as<float>()[index]) / deviation;
    }
    return epsilon;
}

vrhino::Tensor scale_host(const vrhino::Tensor& value, float scale) {
    vrhino::Tensor output = vrhino::Tensor::host(value.shape(), vrhino::DType::F32);
    for (int64_t index = 0; index < value.numel(); ++index)
        output.data_as<float>()[index] = value.data_as<float>()[index] * scale;
    return output;
}

void exact(const vrhino::Tensor& left, const vrhino::Tensor& right,
           const std::string& context) {
    check(left.shape() == right.shape() && left.dtype() == right.dtype(),
          context + " metadata mismatch");
    check(std::memcmp(left.data(), right.data(),
          static_cast<size_t>(left.numel()) * vrhino::dtype_size(left.dtype())) == 0,
          context + " was not bit-exact");
}

struct EncoderQualification {
    vrhino::AutoencoderKLEncoderResult result;
    vrhino::AutoencoderKLObservation observation;
    double milliseconds = 0.0;
};

EncoderQualification run_encoder(const std::string& branch,
        vrhino::CudaBackend& backend, vrhino::AutoencoderKLComponentExecutor& executor,
        const vrhino::Json& graph, const fs::path& phase1, const fs::path& internal) {
    const vrhino::Tensor input = vrhino::test::read_npy_f32(
        (phase1 / "vae" / ("frame_00." + branch + "_normalized.npy")).string());
    check(input.shape() == std::vector<int64_t>({1, 3, 256, 256}),
          branch + " normalized input shape drift");
    backend.synchronize();
    const auto started = std::chrono::steady_clock::now();
    EncoderQualification qualification;
    qualification.result = executor.encode(graph, input, &qualification.observation);
    backend.synchronize();
    qualification.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    qualify(branch + ".initial_conv", backend,
        qualification.observation.tensors.at("initial_conv"),
        internal / (branch + ".initial_conv.npy"), 2.0e-4, 0.999999);
    qualify(branch + ".down_block_0", backend,
        qualification.observation.tensors.at("down_block_0"),
        internal / (branch + ".down_block_0.npy"), 1.5e-2, 0.99999);
    qualify(branch + ".down_block_2", backend,
        qualification.observation.tensors.at("down_block_2"),
        internal / (branch + ".down_block_2.npy"), 1.0e-1, 0.9999);
    qualify(branch + ".mid_block", backend,
        qualification.observation.tensors.at("mid_block"),
        internal / (branch + ".mid_block.npy"), 8.5e-1, 0.99999);
    qualify(branch + ".pre_posterior", backend,
        qualification.observation.tensors.at("pre_posterior"),
        internal / (branch + ".pre_posterior.npy"), 3.0e-2, 0.99999);
    qualify(branch + ".posterior_mean", backend, qualification.result.posterior_mean,
        phase1 / "vae" / ("frame_00." + branch + "_posterior_mean.npy"),
        2.0e-3, 0.999999);
    qualify(branch + ".posterior_logvar", backend, qualification.result.posterior_logvar,
        phase1 / "vae" / ("frame_00." + branch + "_posterior_logvar.npy"),
        5.0e-4, 0.999999);
    return qualification;
}

void qualify_sample(const std::string& branch, vrhino::CudaBackend& backend,
        vrhino::AutoencoderKLComponentExecutor& executor, const vrhino::Json& graph,
        const vrhino::AutoencoderKLEncoderResult& encoded, const fs::path& phase1) {
    const fs::path base = phase1 / "vae";
    const vrhino::Tensor reference_mean = vrhino::test::read_npy_f32(
        (base / ("frame_00." + branch + "_posterior_mean.npy")).string());
    const vrhino::Tensor reference_logvar = vrhino::test::read_npy_f32(
        (base / ("frame_00." + branch + "_posterior_logvar.npy")).string());
    const vrhino::Tensor reference_sample = vrhino::test::read_npy_f32(
        (base / ("frame_00." + branch + "_sample_unscaled.npy")).string());
    const vrhino::Tensor epsilon = fixed_epsilon(
        reference_mean, reference_logvar, reference_sample);
    vrhino::RngState unused{11001, 0, "pytorch_compat.v1"};
    const auto sampled = executor.sample(
        graph, encoded.posterior_mean, encoded.posterior_logvar, unused, &epsilon);
    qualify(branch + ".sampled_latent", backend, sampled.sampled_latent,
        base / ("frame_00." + branch + "_sample_unscaled.npy"), 2.0e-3, 0.999999);
    qualify(branch + ".scaled_latent", backend, sampled.scaled_latent,
        base / ("frame_00." + branch + "_sample_scaled.npy"), 5.0e-4, 0.999999);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: autoencoder_kl_oracle_tests COMPONENT_VRM "
                     "PHASE1_TENSOR_ROOT ENCODER_ORACLE DECODER_ORACLE\n";
        return 2;
    }
    try {
        const fs::path phase1 = argv[2];
        const fs::path encoder_oracle = argv[3];
        const fs::path decoder_oracle = argv[4];
        vrhino::VrmModel component(argv[1], true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "autoencoder-kl",
              "AutoencoderKL component identity mismatch");
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        vrhino::AutoencoderKLComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())));

        EncoderQualification masked = run_encoder(
            "masked", backend, executor, component.graph(), phase1, encoder_oracle);
        EncoderQualification full = run_encoder(
            "full", backend, executor, component.graph(), phase1, encoder_oracle);
        qualify_sample("masked", backend, executor, component.graph(),
                       masked.result, phase1);
        qualify_sample("full", backend, executor, component.graph(),
                       full.result, phase1);

        vrhino::RngState first_rng{11001, 0, "pytorch_compat.v1"};
        vrhino::RngState repeated_rng{11001, 0, "pytorch_compat.v1"};
        vrhino::RngState different_rng{11002, 0, "pytorch_compat.v1"};
        const auto first_sample = executor.sample(component.graph(),
            full.result.posterior_mean, full.result.posterior_logvar, first_rng);
        const auto repeated_sample = executor.sample(component.graph(),
            full.result.posterior_mean, full.result.posterior_logvar, repeated_rng);
        const auto different_sample = executor.sample(component.graph(),
            full.result.posterior_mean, full.result.posterior_logvar, different_rng);
        const vrhino::Tensor first_host = backend.copy_to_host(first_sample.sampled_latent);
        const vrhino::Tensor repeated_host = backend.copy_to_host(repeated_sample.sampled_latent);
        const vrhino::Tensor different_host = backend.copy_to_host(different_sample.sampled_latent);
        exact(first_host, repeated_host, "component-local RNG repeat");
        check(std::memcmp(first_host.data(), different_host.data(),
              static_cast<size_t>(first_host.numel()) * sizeof(float)) != 0,
              "different RNG seed did not change posterior sample");

        const vrhino::Tensor decoder_unscaled = vrhino::test::read_npy_f32(
            (phase1 / "vae_decode/frame_00.predecode_unscaled.npy").string());
        const vrhino::Tensor decoder_scaled = scale_host(decoder_unscaled, 0.18215f);
        vrhino::AutoencoderKLObservation decoder_observation;
        backend.synchronize();
        const auto decode_started = std::chrono::steady_clock::now();
        const auto decoded = executor.decode(
            component.graph(), decoder_scaled, &decoder_observation);
        backend.synchronize();
        const double decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_started).count();
        qualify("decoder.predecode_unscaled", backend,
            decoder_observation.tensors.at("predecode_unscaled"),
            phase1 / "vae_decode/frame_00.predecode_unscaled.npy", 3.0e-6, 0.9999999);
        qualify("decoder.initial_conv", backend,
            decoder_observation.tensors.at("initial_conv"),
            phase1 / "vae_decoder/vae_decoder_initial_conv.npy", 5.0e-6, 0.9999999);
        qualify("decoder.mid_block", backend,
            decoder_observation.tensors.at("mid_block"),
            phase1 / "vae_decoder/vae_decoder_mid_block.npy", 3.0e-5, 0.9999999);
        qualify("decoder.upsample_stage", backend,
            decoder_observation.tensors.at("up_block_1"),
            phase1 / "vae_decoder/vae_decoder_upsample_stage.npy", 5.0e-4, 0.9999999);
        qualify("decoder.final_norm_activation", backend,
            decoder_observation.tensors.at("final_norm_activation"),
            decoder_oracle / "final_norm_activation.npy", 2.0e-2, 0.999999);
        qualify("decoder.raw", backend, decoded.decoded,
            phase1 / "vae_decode/frame_00.decoded_float.npy", 5.0e-6, 0.9999999);
        qualify("decoder.rgb_0_1", backend, decoded.rgb_0_1,
            phase1 / "vae_decode/frame_00.rgb_0_1.npy", 3.0e-6, 0.9999999);

        vrhino::RngState roundtrip_rng_a{77123, 0, "pytorch_compat.v1"};
        vrhino::RngState roundtrip_rng_b{77123, 0, "pytorch_compat.v1"};
        const auto roundtrip_sample_a = executor.sample(component.graph(),
            full.result.posterior_mean, full.result.posterior_logvar, roundtrip_rng_a);
        const auto roundtrip_sample_b = executor.sample(component.graph(),
            full.result.posterior_mean, full.result.posterior_logvar, roundtrip_rng_b);
        const auto roundtrip_a = executor.decode(
            component.graph(), roundtrip_sample_a.scaled_latent);
        const auto roundtrip_b = executor.decode(
            component.graph(), roundtrip_sample_b.scaled_latent);
        backend.synchronize();
        const vrhino::Tensor roundtrip_host_a = backend.copy_to_host(roundtrip_a.rgb_0_1);
        const vrhino::Tensor roundtrip_host_b = backend.copy_to_host(roundtrip_b.rgb_0_1);
        exact(roundtrip_host_a, roundtrip_host_b, "deterministic round trip");
        float minimum = 1.0f, maximum = 0.0f;
        for (int64_t index = 0; index < roundtrip_host_a.numel(); ++index) {
            const float value = roundtrip_host_a.data_as<float>()[index];
            check(std::isfinite(value), "round trip contains NaN/Inf");
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        check(minimum >= 0.0f && maximum <= 1.0f,
              "round-trip RGB range mismatch");

        const vrhino::Tensor full_input = vrhino::test::read_npy_f32(
            (phase1 / "vae/frame_00.full_normalized.npy").string());
        backend.synchronize();
        const auto component_started = std::chrono::steady_clock::now();
        const auto encode_started = component_started;
        const auto timed_encoded = executor.encode(component.graph(), full_input);
        backend.synchronize();
        const auto encode_finished = std::chrono::steady_clock::now();
        vrhino::RngState timed_rng{11001, 0, "pytorch_compat.v1"};
        const auto timed_sampled = executor.sample(component.graph(),
            timed_encoded.posterior_mean, timed_encoded.posterior_logvar, timed_rng);
        backend.synchronize();
        const auto sample_finished = std::chrono::steady_clock::now();
        (void)executor.decode(component.graph(), timed_sampled.scaled_latent);
        backend.synchronize();
        const auto component_finished = std::chrono::steady_clock::now();
        const double timed_encode_ms = std::chrono::duration<double, std::milli>(
            encode_finished - encode_started).count();
        const double timed_sample_ms = std::chrono::duration<double, std::milli>(
            sample_finished - encode_finished).count();
        const double timed_decode_ms = std::chrono::duration<double, std::milli>(
            component_finished - sample_finished).count();
        const double timed_total_ms = std::chrono::duration<double, std::milli>(
            component_finished - component_started).count();

        std::cout << "masked_encode_ms=" << masked.milliseconds << '\n'
                  << "full_encode_ms=" << full.milliseconds << '\n'
                  << "decode_ms=" << decode_ms << '\n'
                  << "warmed_encode_ms=" << timed_encode_ms << '\n'
                  << "warmed_reparameterization_ms=" << timed_sample_ms << '\n'
                  << "warmed_decode_ms=" << timed_decode_ms << '\n'
                  << "warmed_total_ms=" << timed_total_ms << '\n'
                  << "roundtrip_rgb_min=" << minimum << '\n'
                  << "roundtrip_rgb_max=" << maximum << '\n'
                  << "peak_device_bytes=" << backend.peak_device_bytes() << '\n'
                  << "AutoencoderKL Native FP32 oracle tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "AutoencoderKL Native FP32 oracle tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
