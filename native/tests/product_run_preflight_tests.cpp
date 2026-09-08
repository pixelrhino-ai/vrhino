#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/product/run.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    require_test(static_cast<bool>(output), "cannot write test fixture: " + path.string());
}

template <typename Operation>
std::string expect_code(const product::ModelPackageErrorCode expected,
                        Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": operation unexpectedly succeeded");
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, context + ": wrong error code: " + error.what());
        return error.what();
    }
}

product::ResolvedRunnableModel model_with_minimum(
        const std::optional<uint64_t> minimum,
        const std::optional<uint64_t> recommended = std::nullopt) {
    product::ResolvedRunnableModel model;
    model.manifest.identity.name_space = "vrhino";
    model.manifest.identity.name = "preflight-fixture";
    model.manifest.identity.version = "1.0.0";
    model.manifest.default_preset = "default";
    std::ostringstream manifest;
    manifest << "{\"defaults\":{\"presets\":{\"default\":{\"profile_artifact\":"
                "\"profile\"}}},\"hardware\":{\"presets\":{\"default\":{"
                "\"minimum_vram_bytes\":";
    if (minimum) manifest << *minimum; else manifest << "null";
    manifest << ",\"recommended_vram_bytes\":";
    if (recommended) manifest << *recommended; else manifest << "null";
    manifest << ",\"minimum_compute_capability\":[8,0]}}}}";
    model.manifest.raw_json = manifest.str();
    return model;
}

bool has_preflight_probe(const fs::path& root) {
    if (!fs::exists(root)) return false;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root))
        if (entry.path().filename().string().starts_with(".vrhino-output-preflight-"))
            return true;
    return false;
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-product-run-preflight-tests-" + std::to_string(getpid()));
    std::error_code remove_error;
    fs::remove_all(root, remove_error);
    try {
        const product::ResolvedRunnableModel admitted_model =
            model_with_minimum(80 * kGiB, 96 * kGiB);
        product::HardwareSnapshot hardware;
        hardware.gpu_name = "fixture GPU";
        hardware.compute_major = 12;
        hardware.compute_minor = 0;
        hardware.total_vram_bytes = 100 * kGiB;
        hardware.available_vram_bytes = 90 * kGiB;
        const product::PreflightResult admitted = product::preflight_runnable_model(
            admitted_model, "default", hardware);
        require_test(admitted.status == product::PreflightStatus::Supported,
                     "available VRAM above required threshold was rejected");

        hardware.available_vram_bytes = 72 * kGiB + 440 * 1024 * 1024;
        const product::PreflightResult contended = product::preflight_runnable_model(
            admitted_model, "default", hardware);
        require_test(contended.status == product::PreflightStatus::InsufficientVram,
                     "available VRAM below required threshold was not rejected");
        require_test(contended.message.find("vrhino/preflight-fixture:1.0.0") !=
                         std::string::npos &&
                     contended.message.find("preset default") != std::string::npos &&
                     contended.message.find("80.00 GiB") != std::string::npos &&
                     contended.message.find("72.43 GiB") != std::string::npos &&
                     contended.message.find("100.00 GiB") != std::string::npos,
                     "insufficient VRAM message omitted identity or required/available/total");

        hardware.total_vram_bytes = 70 * kGiB;
        hardware.available_vram_bytes = 60 * kGiB;
        require_test(product::preflight_runnable_model(admitted_model, "default", hardware)
                         .status == product::PreflightStatus::InsufficientVram,
                     "required VRAM above total was not rejected");

        const product::ResolvedRunnableModel no_minimum =
            model_with_minimum(std::nullopt, 24 * kGiB);
        require_test(product::preflight_runnable_model(no_minimum, "default", hardware)
                         .status == product::PreflightStatus::SupportedWithWarning,
                     "unknown minimum VRAM did not preserve warning behavior");

        const fs::path output = root / "created-parent" / "nested" / "output.mp4";
        product::preflight_output_destination(output, false);
        require_test(fs::is_directory(output.parent_path()),
                     "creatable output parent was not created");
        require_test(!fs::exists(output) && !has_preflight_probe(root),
                     "output preflight published output or left a probe");

        const fs::path parent_file = root / "parent-file";
        write_text(parent_file, "not a directory");
        const std::string parent_error = expect_code(
            product::ModelPackageErrorCode::OutputInvalid, [&] {
                product::preflight_output_destination(parent_file / "output.mp4", false);
            }, "output parent is a file");
        require_test(parent_error.starts_with("OUTPUT_INVALID:"),
                     "output failure used a cache error");

        expect_code(product::ModelPackageErrorCode::OutputInvalid, [&] {
            product::preflight_output_destination(
                fs::path("/proc/1") / output.filename(), false);
        }, "unwritable output parent");
        expect_code(product::ModelPackageErrorCode::OutputInvalid, [&] {
            product::preflight_output_destination(
                fs::path("/proc") /
                    ("vrhino-preflight-" + std::to_string(getpid())) / "output.mp4",
                false);
        }, "special filesystem output parent");

        const fs::path existing = root / "existing.mp4";
        write_text(existing, "completed-user-output");
        expect_code(product::ModelPackageErrorCode::OutputExists, [&] {
            product::preflight_output_destination(existing, false);
        }, "existing output without overwrite");
        product::preflight_output_destination(existing, true);
        std::ifstream existing_input(existing, std::ios::binary);
        require_test(std::string(std::istreambuf_iterator<char>(existing_input), {}) ==
                         "completed-user-output",
                     "overwrite preflight changed existing output");
        require_test(!has_preflight_probe(root), "output preflight left garbage files");

        const fs::path encoder = root / "vrhino-ffmpeg";
        write_text(encoder, "#!/bin/sh\nexit 0\n");
        require_test(::chmod(encoder.c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0,
                     "cannot make encoder fixture executable");
        product::preflight_media_encoder(encoder);

        const std::string missing_encoder = expect_code(
            product::ModelPackageErrorCode::VideoEncodingFailed, [&] {
                product::preflight_media_encoder(root / "missing-encoder");
            }, "missing encoder");
        require_test(missing_encoder.find("incomplete or damaged") != std::string::npos &&
                         missing_encoder.find("inference did not start") != std::string::npos,
                     "missing encoder message is not actionable");

        const fs::path non_executable = root / "non-executable-encoder";
        write_text(non_executable, "not executable");
        require_test(::chmod(non_executable.c_str(), S_IRUSR | S_IWUSR) == 0,
                     "cannot set non-executable fixture mode");
        expect_code(product::ModelPackageErrorCode::VideoEncodingFailed, [&] {
            product::preflight_media_encoder(non_executable);
        }, "non-executable encoder");

        const fs::path unusable = root / "unusable-encoder";
        write_text(unusable, "#!/bin/sh\nexit 7\n");
        require_test(::chmod(unusable.c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0,
                     "cannot make unusable encoder fixture executable");
        expect_code(product::ModelPackageErrorCode::VideoEncodingFailed, [&] {
            product::preflight_media_encoder(unusable);
        }, "unlaunchable encoder");

        product::RunOptions run_options;
        run_options.output = parent_file / "never-start.mp4";
        run_options.encoder_path = encoder;
        int progress_events = 0;
        expect_code(product::ModelPackageErrorCode::OutputInvalid, [&] {
            (void)product::run_runnable_model({}, run_options,
                [&](const product::RunEvent&) { ++progress_events; });
        }, "run output preflight ordering");
        require_test(progress_events == 0,
                     "run emitted Runtime progress before output preflight failure");

        run_options.output = root / "encoder-ordering.mp4";
        run_options.encoder_path = root / "missing-encoder";
        expect_code(product::ModelPackageErrorCode::VideoEncodingFailed, [&] {
            (void)product::run_runnable_model({}, run_options,
                [&](const product::RunEvent&) { ++progress_events; });
        }, "run encoder preflight ordering");
        require_test(progress_events == 0,
                     "run emitted Runtime progress before encoder preflight failure");
        require_test(!fs::exists(run_options.output) && !has_preflight_probe(root),
                     "failed run preflight left output files");

        product::ResolvedRunnableModel lip_sync_model;
        lip_sync_model.manifest.product.family = "lip_sync";
        lip_sync_model.manifest.product.status = "technical_private";
        lip_sync_model.manifest.product.workflow_identity = "lip_sync_workflow_v1";
        product::RunOptions lip_sync_options;
        expect_code(product::ModelPackageErrorCode::InvalidInput, [&] {
            (void)product::run_runnable_model(lip_sync_model, lip_sync_options);
        }, "lip-sync missing video");
        lip_sync_options.video = root / "video.mp4";
        expect_code(product::ModelPackageErrorCode::InvalidInput, [&] {
            (void)product::run_runnable_model(lip_sync_model, lip_sync_options);
        }, "lip-sync missing audio");
        lip_sync_options.audio = root / "audio.wav";
        lip_sync_options.output = parent_file / "lip-sync.mp4";
        expect_code(product::ModelPackageErrorCode::OutputInvalid, [&] {
            (void)product::run_runnable_model(lip_sync_model, lip_sync_options);
        }, "lip-sync invalid output before component execution");

        product::ResolvedRunnableModel diffusion_model;
        diffusion_model.manifest.product.family = "lip_sync";
        diffusion_model.manifest.product.status = "technical_private";
        diffusion_model.manifest.product.workflow_identity =
            "lip_sync_diffusion_workflow_v1";
        product::RunOptions diffusion_options;
        expect_code(product::ModelPackageErrorCode::InvalidInput, [&] {
            (void)product::run_runnable_model(diffusion_model, diffusion_options);
        }, "lip-sync diffusion missing video");
        diffusion_options.video = root / "video.mp4";
        expect_code(product::ModelPackageErrorCode::InvalidInput, [&] {
            (void)product::run_runnable_model(diffusion_model, diffusion_options);
        }, "lip-sync diffusion missing audio");
        diffusion_options.audio = root / "audio.wav";
        diffusion_options.output = parent_file / "lip-sync-diffusion.mp4";
        expect_code(product::ModelPackageErrorCode::OutputInvalid, [&] {
            (void)product::run_runnable_model(
                diffusion_model, diffusion_options);
        }, "lip-sync diffusion invalid output before component execution");

        fs::remove_all(root, remove_error);
        std::cout << "product-run-preflight: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root, remove_error);
        std::cerr << "product-run-preflight: FAIL: " << error.what() << '\n';
        return 1;
    }
}
