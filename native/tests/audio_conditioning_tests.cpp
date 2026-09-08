#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "npy_fixture.h"
#include "vrhino/audio_conditioning.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"
#include "vrhino/tensor.h"

namespace fs = std::filesystem;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read fixture: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

double max_abs(const vrhino::Tensor& left, const vrhino::Tensor& right) {
    check(left.shape() == right.shape() && left.dtype() == vrhino::DType::F32 &&
          right.dtype() == vrhino::DType::F32, "comparison contract mismatch");
    double result = 0.0;
    for (int64_t index = 0; index < left.numel(); ++index)
        result = std::max(result, std::abs(static_cast<double>(left.data_as<float>()[index]) -
                                          right.data_as<float>()[index]));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: audio_conditioning_tests SOURCE_DIR WAVEFORM_NPY MEL_NPY OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1];
    const fs::path output = argv[4];
    std::error_code error;
    fs::remove(output, error);
    try {
        const vrhino::Json preprocessor = vrhino::Json::parse(
            read_text(source / "preprocessor_config.json"));
        const vrhino::Tensor waveform = vrhino::test::read_npy_f32(argv[2]);
        check(waveform.shape() == std::vector<int64_t>({5120}),
              "waveform contract drift");
        const vrhino::Tensor expected_mel = vrhino::test::read_npy_f32(argv[3]);
        const vrhino::Tensor actual_mel = vrhino::whisper_log_mel_80(
            waveform, preprocessor);
        const double mel_error = max_abs(actual_mel, expected_mel);
        check(actual_mel.shape() == std::vector<int64_t>({1, 80, 3000}),
              "log-mel output shape mismatch");
        check(mel_error <= 2.0e-5, "log-mel oracle mismatch: " +
              std::to_string(mel_error));

        bool waveform_shape_rejected = false;
        try {
            (void)vrhino::whisper_log_mel_80(
                vrhino::Tensor::host({1, 5120}, vrhino::DType::F32), preprocessor);
        } catch (const std::exception&) {
            waveform_shape_rejected = true;
        }
        check(waveform_shape_rejected, "wrong waveform shape was accepted");
        bool waveform_dtype_rejected = false;
        try {
            (void)vrhino::whisper_log_mel_80(
                vrhino::Tensor::host({5120}, vrhino::DType::F16), preprocessor);
        } catch (const std::exception&) {
            waveform_dtype_rejected = true;
        }
        check(waveform_dtype_rejected, "wrong waveform dtype was accepted");

        const vrhino::AudioFeatureWindowPlan window =
            vrhino::plan_audio_feature_windows(5120, 1500, 8);
        check(window.actual_feature_frames == 16 &&
              window.hidden_frames_per_video_frame == 2 &&
              window.left_padding == 4 && window.right_padding == 12 &&
              window.window_feature_frames == 10 &&
              window.starts == std::vector<int64_t>({0, 2, 4, 6, 8, 10, 12, 14}),
              "frozen frame-window indexing/padding contract drift");
        bool end_padding_rejected = false;
        try {
            (void)vrhino::plan_audio_feature_windows(5120, 1500, 13);
        } catch (const std::exception&) {
            end_padding_rejected = true;
        }
        check(end_padding_rejected,
              "frame window extending beyond end padding was accepted");

        const vrhino::Tensor position =
            vrhino::sinusoidal_position_encoding_f32(50, 384);
        check(position.shape() == std::vector<int64_t>({1, 50, 384}) &&
              position.data_as<float>()[0] == 0.0f &&
              position.data_as<float>()[1] == 1.0f &&
              std::abs(position.data_as<float>()[384] - std::sin(1.0f)) < 1.0e-7f &&
              std::abs(position.data_as<float>()[385] - std::cos(1.0f)) < 1.0e-7f,
              "sinusoidal position encoding contract drift");

        const auto converted =
            vrhino::product::convert_whisper_tiny_encoder_component(source, output);
        check(converted.source_checkpoint_bytes == 151095027ULL,
              "Whisper source byte count drift");
        check(converted.retained_tensor_count == 67 &&
              converted.retained_tensor_bytes == 32833536ULL,
              "Whisper retained encoder inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "audio-encoder",
              "Whisper component header identity mismatch");
        check(component.tensors().size() == 67,
              "Whisper component tensor count mismatch");
        check(component.graph().at("kind").string() == "audio_encoder_transformer" &&
              component.graph().at("blocks").array().size() == 4 &&
              component.graph().at("outputs").array().size() == 5,
              "Whisper component graph contract mismatch");
        check(component.graph().at("config").at("attention_heads").integer() == 6 &&
              component.graph().at("config").at("ffn_width").integer() == 1536,
              "Whisper graph topology drift");
        check(component.metadata().at("component").at("decoder_included").boolean() == false &&
              component.metadata().at("component").at("tokenizer_included").boolean() == false,
              "decoder/tokenizer unexpectedly retained");
        check(component.metadata().at("component").at("source_revision").string() ==
                  "169d4a4341b33bc18d8881c4b69c2e104e1cc0af" &&
              component.metadata().at("component").at("license").string() ==
                  "apache-2.0",
              "Whisper frozen source identity drift");

        const fs::path repeat = output.parent_path() / "repeat-whisper.vrm";
        fs::remove(repeat, error);
        const auto repeated =
            vrhino::product::convert_whisper_tiny_encoder_component(source, repeat);
        check(repeated.output_bytes == converted.output_bytes &&
              repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "Whisper component conversion is not deterministic");
        fs::remove(repeat, error);

        const fs::path cancelled = output.parent_path() / "cancelled-whisper.vrm";
        fs::remove(cancelled, error);
        bool cancellation_seen = false;
        try {
            vrhino::product::convert_whisper_tiny_encoder_component(
                source, cancelled, [] { return true; });
        } catch (const vrhino::product::ModelPackageError& failure) {
            cancellation_seen = failure.code() ==
                vrhino::product::ModelPackageErrorCode::Cancelled;
        }
        check(cancellation_seen && !fs::exists(cancelled),
              "cancelled conversion publication contract failed");

        const fs::path missing_root = output.parent_path() / "missing-source";
        fs::remove_all(missing_root, error);
        fs::create_directories(missing_root);
        bool missing_seen = false;
        try {
            vrhino::product::convert_whisper_tiny_encoder_component(
                missing_root, missing_root / "component.vrm");
        } catch (const vrhino::product::ModelPackageError& failure) {
            missing_seen = failure.code() ==
                vrhino::product::ModelPackageErrorCode::ArtifactMissing;
        }
        check(missing_seen, "missing frozen source did not fail closed");
        fs::remove_all(missing_root, error);

        const fs::path corrupt_root = output.parent_path() / "corrupt-source";
        fs::remove_all(corrupt_root, error);
        fs::create_directories(corrupt_root);
        fs::copy_file(source / "config.json", corrupt_root / "config.json");
        {
            std::fstream corrupt(corrupt_root / "config.json",
                                 std::ios::in | std::ios::out | std::ios::binary);
            check(static_cast<bool>(corrupt), "cannot create corrupt-source fixture");
            char byte = 0;
            corrupt.read(&byte, 1);
            byte ^= 1;
            corrupt.seekp(0);
            corrupt.write(&byte, 1);
        }
        bool corrupt_seen = false;
        try {
            vrhino::product::convert_whisper_tiny_encoder_component(
                corrupt_root, corrupt_root / "component.vrm");
        } catch (const vrhino::product::ModelPackageError& failure) {
            corrupt_seen = failure.code() ==
                vrhino::product::ModelPackageErrorCode::SourceIntegrityFailed;
        }
        check(corrupt_seen && !fs::exists(corrupt_root / "component.vrm"),
              "corrupt frozen source did not fail closed before publication");
        fs::remove_all(corrupt_root, error);
        std::cout << "audio conditioning CPU tests: PASS\n"
                  << "log_mel_max_abs=" << mel_error << "\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "audio conditioning CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
