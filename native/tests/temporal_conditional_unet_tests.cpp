#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/pytorch_zip.h"

namespace fs = std::filesystem;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejects(Operation&& operation, const std::string& message) {
    bool failed = false;
    try { operation(); }
    catch (const vrhino::Error&) { failed = true; }
    catch (const vrhino::product::ModelPackageError&) { failed = true; }
    check(failed, message);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: temporal_conditional_unet_tests "
                     "SOURCE_DIR TENSOR_MAP OUTPUT_A OUTPUT_B\n";
        return 2;
    }
    try {
        const fs::path source = argv[1];
        const fs::path map = argv[2];
        const fs::path output_a = argv[3];
        const fs::path output_b = argv[4];
        const vrhino::product::PytorchZipArchive archive(
            source / "latentsync_unet.pt", 2048);
        check(archive.file_size() == 5072222488ULL &&
                  archive.entries().size() == 1250,
              "fixed PyTorch ZIP inventory mismatch");
        const auto pickle = archive.read("latentsync_unet/data.pkl", 1ULL << 20);
        const auto inventory = vrhino::product::inspect_restricted_tensor_pickle(
            pickle, {"torch._utils._rebuild_tensor_v2", "torch.FloatStorage",
                     "collections.OrderedDict"});
        check(inventory.globals.size() == 3 &&
                  inventory.unicode_strings.size() == 2495,
              "restricted pickle inventory mismatch");
        rejects([] {
            const std::vector<uint8_t> hostile = {
                0x80, 2, 'c', 'o', 's', '\n', 's', 'y', 's', 't', 'e', 'm',
                '\n', '.'};
            (void)vrhino::product::inspect_restricted_tensor_pickle(
                hostile, {"torch.FloatStorage"});
        }, "unsupported pickle global was accepted");
        rejects([] {
            const std::vector<uint8_t> unsupported = {0x80, 2, 0x93, '.'};
            (void)vrhino::product::inspect_restricted_tensor_pickle(
                unsupported, {});
        }, "unsupported pickle opcode was accepted");
        rejects([] {
            const std::vector<uint8_t> unsupported_protocol = {0x80, 5, '.'};
            (void)vrhino::product::inspect_restricted_tensor_pickle(
                unsupported_protocol, {});
        }, "unsupported pickle protocol was accepted");

        const fs::path corrupt_source = output_a.parent_path() /
            "corrupt-temporal-source";
        std::error_code error;
        fs::remove_all(corrupt_source, error);
        fs::create_directories(corrupt_source);
        {
            std::ofstream corrupt(corrupt_source / "latentsync_unet.pt",
                                  std::ios::binary);
            corrupt.put('\0');
        }
        rejects([&] {
            (void)vrhino::product::convert_temporal_conditional_unet_component(
                corrupt_source, map, corrupt_source / "component.vrm");
        }, "corrupt temporal UNet source identity was accepted");
        check(!fs::exists(corrupt_source / "component.vrm"),
              "corrupt conversion published a component");
        fs::remove_all(corrupt_source, error);

        fs::remove(output_a, error);
        fs::remove(output_b, error);
        const auto first =
            vrhino::product::convert_temporal_conditional_unet_component(
                source, map, output_a);
        const auto second =
            vrhino::product::convert_temporal_conditional_unet_component(
                source, map, output_b);
        check(first.retained_tensor_count == 1246 &&
                  first.retained_tensor_bytes == 5071778576ULL &&
                  first.output_bytes == second.output_bytes &&
                  first.output_sha256 == second.output_sha256 &&
                  first.payload_blake2b128 == second.payload_blake2b128,
              "temporal UNet deterministic conversion mismatch");
        vrhino::VrmModel component(output_a.string(), true);
        check(component.architecture_id() == "temporal-unet" &&
                  component.graph().at("kind").string() ==
                      "temporal_conditional_unet_2d" &&
                  component.graph().at("config").at("motion_module_count").integer() == 20 &&
                  component.tensors().size() == 1246,
              "temporal UNet component graph/topology mismatch");
        check(component.graph().at("inputs").at("latent").string() ==
                  "B,13,F,H,W" &&
              component.graph().at("outputs").at("epsilon").string() ==
                  "B,4,F,H,W",
              "temporal UNet raw schema mismatch");

        const fs::path cancelled = output_a.string() + ".cancelled";
        fs::remove(cancelled, error);
        rejects([&] {
            (void)vrhino::product::convert_temporal_conditional_unet_component(
                source, map, cancelled, [] { return true; });
        }, "temporal UNet conversion cancellation was ignored");
        check(!fs::exists(cancelled), "cancelled conversion published output");
        rejects([&] {
            (void)vrhino::product::convert_temporal_conditional_unet_component(
                source, map, output_a);
        }, "existing temporal UNet output was overwritten");

        std::cout << "checkpoint_entries=1250 tensors=1246 raw_bytes="
                  << first.retained_tensor_bytes << " component_bytes="
                  << first.output_bytes << " sha256=" << first.output_sha256
                  << " blake2b128=" << first.payload_blake2b128
                  << " first_conversion_seconds=" << first.conversion_seconds
                  << " second_conversion_seconds=" << second.conversion_seconds
                  << " largest_buffer_bytes="
                  << first.largest_temporary_buffer_bytes
                  << " deterministic=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
