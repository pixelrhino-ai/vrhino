#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "vrhino/loader.h"
#include "vrhino/product/converter.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

template <typename Operation>
void expect(product::ModelPackageErrorCode code, Operation&& operation) {
    try {
        operation();
        throw std::runtime_error("failure fixture succeeded");
    } catch (const product::ModelPackageError& error) {
        check(error.code() == code, "failure fixture error code mismatch");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: openai_whisper_converter_tests SOURCE_DIR "
                     "TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], tensor_map = argv[2], output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("openai-whisper-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error);
    fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        const auto first = product::convert_openai_whisper_tiny_encoder_component(
            source, tensor_map, output);
        check(first.source_checkpoint_bytes == 75572083 &&
              first.retained_tensor_bytes == 32833536 &&
              first.retained_tensor_count == 67,
              "OpenAI Whisper retained inventory mismatch");
        vrhino::VrmModel model(output.string(), true);
        check(model.profile_id() == "component" &&
              model.architecture_id() == "audio-encoder" &&
              model.tensors().size() == 67,
              "OpenAI Whisper component identity mismatch");
        const auto& graph = model.graph();
        check(graph.at("kind").string() == "audio_encoder_transformer" &&
              graph.at("config").at("encoder_blocks").integer() == 4 &&
              graph.at("blocks").array().size() == 4 &&
              graph.at("outputs").array().size() == 5 &&
              graph.at("outputs").array().at(4).at("source").string() ==
                  "block.3",
              "OpenAI Whisper generic graph contract mismatch");

        const fs::path repeat = scratch / "repeat.vrm";
        const auto second = product::convert_openai_whisper_tiny_encoder_component(
            source, tensor_map, repeat);
        check(first.output_sha256 == second.output_sha256 &&
              first.payload_blake2b128 == second.payload_blake2b128 &&
              fs::file_size(output) == fs::file_size(repeat),
              "OpenAI Whisper conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_openai_whisper_tiny_encoder_component(
                source, tensor_map, scratch / "cancel.vrm", [] { return true; });
        });
        expect(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_openai_whisper_tiny_encoder_component(
                scratch / "missing", tensor_map, scratch / "missing.vrm");
        });
        const fs::path corrupt = scratch / "corrupt";
        fs::create_directories(corrupt);
        fs::copy_file(source / "tiny.pt", corrupt / "tiny.pt");
        fs::resize_file(corrupt / "tiny.pt", 75572082);
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_openai_whisper_tiny_encoder_component(
                corrupt, tensor_map, scratch / "corrupt.vrm");
        });
        for (const auto& entry : fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-") ==
                      std::string::npos,
                  "OpenAI Whisper conversion left partial publication");
        fs::remove_all(scratch, error);
        std::cout << "OpenAI Whisper fixed conversion tests: PASS\n"
                  << "size=" << first.output_bytes << '\n'
                  << "sha256=" << first.output_sha256 << '\n'
                  << "blake2b128=" << first.payload_blake2b128 << '\n';
        return 0;
    } catch (const std::exception& failure) {
        fs::remove_all(scratch, error);
        std::cerr << "OpenAI Whisper fixed conversion tests: FAIL: "
                  << failure.what() << '\n';
        return 1;
    }
}
