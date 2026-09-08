#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/audio_conditioning.h"
#include "vrhino/audio_encoder.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"

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

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
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

void print_difference(const std::string& name, const Difference& value) {
    std::cout << std::setprecision(10) << name
              << " max_abs=" << value.maximum_absolute
              << " mean_abs=" << value.mean_absolute
              << " max_rel=" << value.maximum_relative
              << " cosine=" << value.cosine
              << " nan=" << value.nan_count
              << " inf=" << value.inf_count << '\n';
}

void require_fp32_tight(const std::string& name, const Difference& value,
                        double maximum_absolute, double minimum_cosine) {
    check(value.nan_count == 0 && value.inf_count == 0,
          name + " contains non-finite Native values");
    check(value.maximum_absolute <= maximum_absolute && value.cosine >= minimum_cosine,
          name + " exceeds FP32 qualification tolerance");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: whisper_tiny_oracle_tests COMPONENT_VRM PREPROCESSOR_JSON "
                     "WAVEFORM_NPY ORACLE_AUDIO_DIR FRAME_COUNT\n";
        return 2;
    }
    try {
        const int64_t frame_count = std::stoll(argv[5]);
        check(frame_count == 8, "frozen oracle frame count must be eight");
        vrhino::VrmModel component(argv[1], true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "audio-encoder",
              "component identity mismatch");
        const vrhino::Json preprocessor = vrhino::Json::parse(read_text(argv[2]));
        const vrhino::Tensor waveform = vrhino::test::read_npy_f32(argv[3]);
        const fs::path oracle = argv[4];
        const vrhino::Tensor log_mel = vrhino::whisper_log_mel_80(waveform, preprocessor);
        Difference mel_difference = compare(log_mel,
            vrhino::test::read_npy_f32((oracle / "whisper_log_mel.npy").string()));
        print_difference("log_mel", mel_difference);
        require_fp32_tight("log_mel", mel_difference, 2.0e-5, 0.999999999);

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        vrhino::AudioEncoderComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())));
        bool wrong_shape_rejected = false;
        try {
            (void)executor.execute(component.graph(),
                vrhino::Tensor::host({1, 79, 3000}, vrhino::DType::F32));
        } catch (const std::exception&) {
            wrong_shape_rejected = true;
        }
        check(wrong_shape_rejected,
              "AudioEncoder accepted wrong input channel shape");
        bool wrong_dtype_rejected = false;
        try {
            (void)executor.execute(component.graph(),
                vrhino::Tensor::host({1, 80, 3000}, vrhino::DType::F16));
        } catch (const std::exception&) {
            wrong_dtype_rejected = true;
        }
        check(wrong_dtype_rejected, "AudioEncoder accepted non-FP32 input");
        vrhino::AudioEncoderComponentResult encoded =
            executor.execute(component.graph(), log_mel);
        backend.synchronize();

        struct Boundary {
            const char* name;
            const char* file;
            double maximum_absolute;
        };
        const std::array<Boundary, 5> boundaries = {{
            {"frontend_hidden", "whisper_hidden_state_0.npy", 1.0e-5},
            {"encoder_block_1", "whisper_hidden_state_1.npy", 2.0e-5},
            {"encoder_block_2", "whisper_hidden_state_2.npy", 5.0e-5},
            {"encoder_block_3", "whisper_hidden_state_3.npy", 5.0e-3},
            {"encoder_block_4", "whisper_hidden_state_4.npy", 3.0e-3},
        }};
        std::vector<vrhino::Tensor> states;
        states.reserve(boundaries.size());
        for (const auto& [name, file, maximum_absolute] : boundaries) {
            const vrhino::Tensor host = backend.copy_to_host(encoded.outputs.at(name));
            const Difference difference = compare(host,
                vrhino::test::read_npy_f32((oracle / file).string()));
            print_difference(name, difference);
            require_fp32_tight(name, difference, maximum_absolute, 0.999999);
            states.push_back(encoded.outputs.at(name));
        }

        vrhino::Tensor stacked = vrhino::stack_audio_encoder_states(backend, states);
        Difference stacked_difference = compare(backend.copy_to_host(stacked),
            vrhino::test::read_npy_f32(
                (oracle / "whisper_stacked_five_states.npy").string()));
        print_difference("stacked_five_states", stacked_difference);
        require_fp32_tight("stacked_five_states", stacked_difference,
                           5.0e-3, 0.999999);

        vrhino::Tensor windowed = vrhino::frame_audio_feature_windows(
            backend, stacked, waveform.numel(), frame_count);
        Difference window_difference = compare(backend.copy_to_host(windowed),
            vrhino::test::read_npy_f32(
                (oracle / "frame_windowed_conditioning.npy").string()));
        print_difference("frame_windowed_conditioning", window_difference);
        require_fp32_tight("frame_windowed_conditioning", window_difference,
                           1.0e-4, 0.999999);

        vrhino::Tensor positioned = vrhino::add_sinusoidal_position_encoding(
            backend, windowed);
        double final_maximum = 0.0, final_mean = 0.0, final_cosine = 1.0;
        for (int64_t frame = 0; frame < frame_count; ++frame) {
            vrhino::Tensor one = backend.slice(positioned, 0, frame, frame + 1);
            const std::string file = "frame_" +
                std::string(frame < 10 ? "0" : "") + std::to_string(frame) +
                ".position_encoded_conditioning.npy";
            Difference difference = compare(backend.copy_to_host(one),
                vrhino::test::read_npy_f32((oracle / file).string()));
            print_difference("position_encoded_frame_" + std::to_string(frame), difference);
            require_fp32_tight("position_encoded_conditioning", difference,
                               1.0e-4, 0.999999);
            final_maximum = std::max(final_maximum, difference.maximum_absolute);
            final_mean += difference.mean_absolute;
            final_cosine = std::min(final_cosine, difference.cosine);
        }
        backend.synchronize();
        std::cout << "final_conditioning max_abs=" << final_maximum
                  << " mean_abs=" << final_mean / frame_count
                  << " minimum_frame_cosine=" << final_cosine << '\n'
                  << "peak_device_bytes=" << backend.peak_device_bytes() << '\n'
                  << "whisper tiny Native oracle tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "whisper tiny Native oracle tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
