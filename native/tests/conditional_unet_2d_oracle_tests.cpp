#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "npy_fixture.h"
#include "vrhino/autoencoder_kl.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/conditional_unet_2d.h"
#include "vrhino/loader.h"

namespace fs = std::filesystem;

namespace {
struct Difference {
    double maximum_absolute = 0.0;
    double mean_absolute = 0.0;
    double maximum_relative = 0.0;
    double cosine = 0.0;
    double reference_minimum = std::numeric_limits<double>::infinity();
    double reference_maximum = -std::numeric_limits<double>::infinity();
    double native_minimum = std::numeric_limits<double>::infinity();
    double native_maximum = -std::numeric_limits<double>::infinity();
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
};

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& expected) {
    check(actual.device().is_host() && expected.device().is_host() &&
          actual.dtype() == vrhino::DType::F32 &&
          expected.dtype() == vrhino::DType::F32 &&
          actual.shape() == expected.shape(), "oracle comparison contract mismatch");
    Difference result;
    long double absolute_sum = 0.0, dot = 0.0, left_norm = 0.0, right_norm = 0.0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double left = actual.data_as<float>()[index];
        const double right = expected.data_as<float>()[index];
        if (std::isnan(left)) ++result.nan_count;
        if (std::isinf(left)) ++result.inf_count;
        result.native_minimum = std::min(result.native_minimum, left);
        result.native_maximum = std::max(result.native_maximum, left);
        result.reference_minimum = std::min(result.reference_minimum, right);
        result.reference_maximum = std::max(result.reference_maximum, right);
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
              << " ref_range=[" << value.reference_minimum << ','
              << value.reference_maximum << "] native_range=["
              << value.native_minimum << ',' << value.native_maximum << ']'
              << " max_abs=" << value.maximum_absolute
              << " mean_abs=" << value.mean_absolute
              << " max_rel=" << value.maximum_relative
              << " cosine=" << value.cosine
              << " nan=" << value.nan_count << " inf=" << value.inf_count << '\n';
}

Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual, const fs::path& expected,
                   double maximum_absolute, double minimum_cosine) {
    Difference result = compare(backend.copy_to_host(actual),
        vrhino::test::read_npy_f32(expected.string()));
    report(name, result);
    check(result.nan_count == 0 && result.inf_count == 0,
          name + " contains non-finite values");
    check(result.maximum_absolute <= maximum_absolute &&
          result.cosine >= minimum_cosine,
          name + " exceeds the FP32 oracle gate");
    return result;
}

std::string frame_name(int frame) {
    std::ostringstream stream;
    stream << "frame_" << std::setw(2) << std::setfill('0') << frame;
    return stream.str();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: conditional_unet_2d_oracle_tests UNET_VRM VAE_VRM "
                     "TENSOR_ROOT DEEP_PATH_ORACLE\n";
        return 2;
    }
    try {
        const fs::path root = argv[3];
        vrhino::VrmModel unet(argv[1], true);
        vrhino::VrmModel vae(argv[2], true);
        check(unet.profile_id() == "component" &&
              unet.architecture_id() == "cond-unet-2d",
              "conditional UNet component identity mismatch");
        check(vae.profile_id() == "component" &&
              vae.architecture_id() == "autoencoder-kl",
              "AutoencoderKL integration component identity mismatch");
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::ConditionalUNet2DComponentExecutor executor(
            backend, vrhino::WeightMap(unet.bindings(unet.graph())));
        vrhino::AutoencoderKLComponentExecutor vae_executor(
            backend, vrhino::WeightMap(vae.bindings(vae.graph())));
        vrhino::Tensor timestep = vrhino::Tensor::host({1}, vrhino::DType::I64);
        timestep.data_as<int64_t>()[0] = 0;

        const vrhino::Tensor latent = vrhino::test::read_npy_f32(
            (root / "vae/frame_00.unet_input_8ch.npy").string());
        const vrhino::Tensor conditioning = vrhino::test::read_npy_f32(
            (root / "audio/frame_00.position_encoded_conditioning.npy").string());
        vrhino::ConditionalUNet2DObservation observation;
        auto result = executor.execute(unet.graph(), latent, timestep,
                                       conditioning, &observation);
        backend.synchronize();

        qualify("input_conv", backend, observation.tensors.at("input_conv"),
                root / "unet/input_conv2d_output.npy", 1.0e-6, 0.9999999);
        qualify("timestep_embedding", backend,
                observation.tensors.at("timestep_embedding"),
                root / "unet/timestep_embedding.npy", 1.0e-6, 0.9999999);
        qualify("first_down_resnet", backend,
                observation.tensors.at("first_down_resnet"),
                root / "unet/first_down_resnet_output.npy", 2.0e-6, 0.9999999);
        qualify("first_self_attention", backend,
                observation.tensors.at("first_self_attention"),
                root / "unet/first_self_attention_output.npy", 3.0e-6, 0.9999999);
        qualify("first_audio_cross_attention", backend,
                observation.tensors.at("first_audio_cross_attention"),
                root / "unet/first_audio_cross_attention_output.npy", 1.0e-6, 0.9999999);
        qualify("first_downsample", backend,
                observation.tensors.at("first_downsample"),
                root / "unet/first_downsample_output.npy", 1.0e-5, 0.9999999);
        qualify("deep_down_path", backend,
                observation.tensors.at("deep_down_path"),
                fs::path(argv[4]) / "deep_down_path.npy", 1.0e-4, 0.9999999);
        qualify("mid_block", backend, observation.tensors.at("mid_block"),
                root / "unet/representative_mid_block.npy", 1.0e-4, 0.9999999);
        qualify("representative_up_block", backend,
                observation.tensors.at("representative_up_block"),
                root / "unet/representative_up_block.npy", 5.0e-5, 0.9999999);
        qualify("deep_up_path", backend,
                observation.tensors.at("deep_up_path"),
                fs::path(argv[4]) / "deep_up_path.npy", 1.0e-4, 0.9999999);
        qualify("final_normalization", backend,
                observation.tensors.at("final_normalization"),
                root / "unet/final_normalization.npy", 5.0e-5, 0.9999999);
        qualify("final_activation", backend,
                observation.tensors.at("final_activation"),
                root / "unet/final_activation.npy", 5.0e-5, 0.9999999);
        qualify("final_conv", backend, observation.tensors.at("final_conv"),
                root / "unet/final_conv2d.npy", 5.0e-5, 0.9999999);
        Difference frame0 = qualify("frame_00.output", backend,
            result.predicted_latent, root / "unet/frame_00.output.npy",
            5.0e-5, 0.9999999);

        backend.synchronize();
        const auto warm_start = std::chrono::steady_clock::now();
        result = executor.execute(unet.graph(), latent, timestep, conditioning);
        backend.synchronize();
        const double one_frame_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - warm_start).count();

        Difference worst = frame0;
        int worst_frame = 0;
        const auto eight_start = std::chrono::steady_clock::now();
        for (int frame = 0; frame < 8; ++frame) {
            const std::string name = frame_name(frame);
            const vrhino::Tensor frame_latent = vrhino::test::read_npy_f32(
                (root / "vae" / (name + ".unet_input_8ch.npy")).string());
            const vrhino::Tensor frame_conditioning = vrhino::test::read_npy_f32(
                (root / "audio" /
                 (name + ".position_encoded_conditioning.npy")).string());
            auto frame_result = executor.execute(unet.graph(), frame_latent,
                                                 timestep, frame_conditioning);
            Difference difference = qualify(name + ".output", backend,
                frame_result.predicted_latent,
                root / "unet" / (name + ".output.npy"), 5.0e-5, 0.9999999);
            if (difference.maximum_absolute > worst.maximum_absolute) {
                worst = difference;
                worst_frame = frame;
            }
        }
        backend.synchronize();
        const double eight_frame_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - eight_start).count();

        auto decoded = vae_executor.decode(vae.graph(), result.predicted_latent);
        Difference decoded_difference = qualify("native_unet_to_native_vae.rgb", backend,
            decoded.rgb_0_1, root / "vae_decode/frame_00.rgb_0_1.npy",
            5.0e-5, 0.9999999);
        check(decoded.rgb_0_1.shape() ==
              std::vector<int64_t>({1, 3, 256, 256}),
              "Native UNet to VAE output shape mismatch");

        std::cout << "Conditional UNet2D FP32 oracle qualification: PASS\n"
                  << "one_frame_ms=" << one_frame_ms << "\n"
                  << "eight_frame_ms=" << eight_frame_ms << "\n"
                  << "peak_device_bytes=" << backend.peak_device_bytes() << "\n"
                  << "weight_cache_resident_bytes="
                  << backend.weight_cache_resident_bytes() << "\n"
                  << "worst_frame=" << worst_frame << "\n"
                  << "worst_max_abs=" << worst.maximum_absolute << "\n"
                  << "worst_mean_abs=" << worst.mean_absolute << "\n"
                  << "minimum_cosine=" << worst.cosine << "\n"
                  << "decoded_max_abs=" << decoded_difference.maximum_absolute << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Conditional UNet2D FP32 oracle qualification: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
