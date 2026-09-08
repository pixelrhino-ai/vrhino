#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void expect_code(product::ModelPackageErrorCode code, Operation&& operation,
                 const std::string& context) {
    try { operation(); throw std::runtime_error(context + ": unexpectedly succeeded"); }
    catch (const product::ModelPackageError& failure) {
        check(failure.code() == code, context + ": wrong error code");
    }
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot read fixture");
    return std::string(std::istreambuf_iterator<char>(input), {});
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    check(static_cast<bool>(output), "cannot write fixture");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: conditional_unet_2d_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], mapping = argv[2], output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("conditional-unet-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error);
    fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        const auto converted = product::convert_musetalk_v15_unet_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 3400074924ULL &&
              converted.retained_tensor_count == 686 &&
              converted.retained_tensor_bytes == 3399791376ULL,
              "conditional UNet source inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "cond-unet-2d" &&
              component.tensors().size() == 686,
              "conditional UNet component identity mismatch");
        const auto& graph = component.graph();
        const auto& config = graph.at("config");
        check(graph.at("kind").string() == "conditional_unet_2d" &&
              graph.at("entry_point").string() == "execute" &&
              config.at("input_channels").integer() == 8 &&
              config.at("output_channels").integer() == 4 &&
              config.at("conditioning_width").integer() == 384 &&
              config.at("layers_per_block").integer() == 2 &&
              config.at("up_layers_per_block").integer() == 3 &&
              config.at("attention_heads").integer() == 8 &&
              config.at("block_out_channels").array().size() == 4 &&
              config.at("feed_forward").string() == "geglu",
              "conditional UNet frozen graph topology drift");
        check(component.metadata().at("component").at("source_revision").string() ==
                  "3ef28bc5cff08c90ad8178a25f1b570cd800170f" &&
              !component.metadata().at("component")
                   .at("production_python_dependency").boolean(),
              "conditional UNet source/runtime metadata drift");

        const fs::path repeat = scratch / "repeat.vrm";
        const auto repeated = product::convert_musetalk_v15_unet_component(
            source, mapping, repeat);
        check(repeated.output_bytes == converted.output_bytes &&
              repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "conditional UNet conversion is not deterministic");

        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_musetalk_v15_unet_component(
                source, mapping, scratch / "cancelled.vrm", [] { return true; });
        }, "conversion cancellation");
        check(!fs::exists(scratch / "cancelled.vrm"),
              "cancelled conditional UNet component was published");

        const fs::path missing = scratch / "missing";
        fs::create_directories(missing);
        expect_code(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_musetalk_v15_unet_component(
                missing, mapping, scratch / "missing.vrm");
        }, "missing source");

        const fs::path corrupt = scratch / "corrupt";
        fs::create_directories(corrupt);
        fs::copy_file(source / "musetalk.json", corrupt / "musetalk.json");
        {
            std::fstream file(corrupt / "musetalk.json",
                              std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            file.read(&byte, 1);
            byte ^= 1;
            file.seekp(0);
            file.write(&byte, 1);
        }
        expect_code(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_musetalk_v15_unet_component(
                corrupt, mapping, scratch / "corrupt.vrm");
        }, "source corruption");

        std::string wrong_dtype = read_text(mapping);
        const size_t dtype = wrong_dtype.find("\tF32\t");
        check(dtype != std::string::npos, "dtype fixture target missing");
        wrong_dtype.replace(dtype + 1, 3, "F16");
        const fs::path wrong_dtype_path = scratch / "wrong-dtype.tsv";
        write_text(wrong_dtype_path, wrong_dtype);
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_musetalk_v15_unet_component(
                source, wrong_dtype_path, scratch / "wrong-dtype.vrm");
        }, "wrong source dtype");

        std::string wrong_shape = read_text(mapping);
        const size_t shape = wrong_shape.find("320,8,3,3");
        check(shape != std::string::npos, "shape fixture target missing");
        wrong_shape.replace(shape, 9, "319,8,3,3");
        const fs::path wrong_shape_path = scratch / "wrong-shape.tsv";
        write_text(wrong_shape_path, wrong_shape);
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_musetalk_v15_unet_component(
                source, wrong_shape_path, scratch / "wrong-shape.vrm");
        }, "wrong source shape");

        fs::remove_all(scratch, error);
        std::cout << "Conditional UNet2D CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch, error);
        std::cerr << "Conditional UNet2D CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
