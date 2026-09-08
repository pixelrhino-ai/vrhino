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
    try {
        operation();
        throw std::runtime_error(context + ": unexpectedly succeeded");
    } catch (const product::ModelPackageError& failure) {
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
        std::cerr << "usage: autoencoder_kl_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1];
    const fs::path mapping = argv[2];
    const fs::path output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("autoencoder-kl-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error);
    fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        const auto converted = product::convert_sd_vae_ft_mse_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 334707217ULL,
              "source checkpoint byte identity drift");
        check(converted.retained_tensor_count == 248 &&
              converted.retained_tensor_bytes == 334615452ULL,
              "retained AutoencoderKL tensor inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "autoencoder-kl" &&
              component.tensors().size() == 248,
              "AutoencoderKL component identity mismatch");
        const auto& graph = component.graph();
        check(graph.at("kind").string() == "autoencoder_kl" &&
              graph.at("entry_points").array().size() == 2 &&
              graph.at("entry_points").array()[0].string() == "encode" &&
              graph.at("entry_points").array()[1].string() == "decode",
              "bounded encode/decode entry points drift");
        check(graph.at("config").at("block_out_channels").array().size() == 4 &&
              graph.at("config").at("layers_per_block").integer() == 2 &&
              graph.at("config").at("decoder_layers_per_block").integer() == 3 &&
              graph.at("config").at("latent_channels").integer() == 4 &&
              graph.at("config").at("norm_num_groups").integer() == 32 &&
              graph.at("config").at("mid_block_attention").boolean() &&
              graph.at("config").at("scaling_factor").number() == 0.18215,
              "AutoencoderKL frozen graph topology drift");
        check(component.metadata().at("component").at("source_revision").string() ==
                  "31f26fdeee1355a5c34592e401dd41e45d25a493" &&
              component.metadata().at("component").at("tensor_count").integer() == 248,
              "AutoencoderKL fixed source metadata drift");

        const fs::path repeat = scratch / "repeat.vrm";
        const auto repeated = product::convert_sd_vae_ft_mse_component(
            source, mapping, repeat);
        check(repeated.output_bytes == converted.output_bytes &&
              repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "AutoencoderKL conversion is not deterministic");

        const fs::path cancelled = scratch / "cancelled.vrm";
        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_sd_vae_ft_mse_component(
                source, mapping, cancelled, [] { return true; });
        }, "conversion cancellation");
        check(!fs::exists(cancelled), "cancelled component was published");

        const fs::path missing = scratch / "missing";
        fs::create_directories(missing);
        expect_code(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_sd_vae_ft_mse_component(
                missing, mapping, scratch / "missing.vrm");
        }, "missing source");

        const fs::path corrupt = scratch / "corrupt";
        fs::create_directories(corrupt);
        fs::copy_file(source / "config.json", corrupt / "config.json");
        {
            std::fstream file(corrupt / "config.json",
                              std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            file.read(&byte, 1);
            byte ^= 1;
            file.seekp(0);
            file.write(&byte, 1);
        }
        expect_code(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_sd_vae_ft_mse_component(
                corrupt, mapping, scratch / "corrupt.vrm");
        }, "source corruption");

        std::string wrong_dtype = read_text(mapping);
        const size_t dtype = wrong_dtype.find("\tF32\t");
        check(dtype != std::string::npos, "dtype fixture target missing");
        wrong_dtype.replace(dtype + 1, 3, "F16");
        const fs::path wrong_dtype_path = scratch / "wrong-dtype.tsv";
        write_text(wrong_dtype_path, wrong_dtype);
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_sd_vae_ft_mse_component(
                source, wrong_dtype_path, scratch / "wrong-dtype.vrm");
        }, "wrong source dtype");

        std::string wrong_shape = read_text(mapping);
        const size_t shape = wrong_shape.find("128,3,3,3");
        check(shape != std::string::npos, "shape fixture target missing");
        wrong_shape.replace(shape, 9, "127,3,3,3");
        const fs::path wrong_shape_path = scratch / "wrong-shape.tsv";
        write_text(wrong_shape_path, wrong_shape);
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_sd_vae_ft_mse_component(
                source, wrong_shape_path, scratch / "wrong-shape.vrm");
        }, "wrong source shape");

        fs::remove_all(scratch, error);
        std::cout << "AutoencoderKL CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch, error);
        std::cerr << "AutoencoderKL CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
