#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

#include "vrhino/product/doctor.h"
#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr const char* kRevision = "0123456789abcdef0123456789abcdef01234567";

struct Fixture {
    std::string reference;
    fs::path package;
    std::string runtime_sha;
    std::string secondary_sha;
};

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    require_test(static_cast<bool>(output), "cannot write fixture: " + path.string());
}

std::string artifact_json(const std::string& id, const std::string& role,
                          const std::string& path, const uint64_t size,
                          const std::string& sha256) {
    std::ostringstream output;
    output << "{\"id\":\"" << id << "\",\"role\":\"" << role
           << "\",\"path\":\"" << path << "\",\"size\":" << size
           << ",\"sha256\":\"" << sha256 << "\",\"required\":true}";
    return output.str();
}

Fixture make_fixture(const fs::path& root, const fs::path& specs,
                     const std::string& name,
                     const std::optional<uint64_t> minimum_vram) {
    Fixture fixture;
    fixture.reference = "vrhino/" + name + ":1.0.0";
    fixture.package = root / (name + "-package");
    write_text(fixture.package / "model/model.vrm", "abc");
    write_text(fixture.package / "conditioning/secondary.bin", "data");
    write_text(fixture.package / "execution/default.json", "{}\n");
    write_text(fixture.package / "legal/LICENSE.txt", "license\n");
    fixture.runtime_sha = product::sha256_file(fixture.package / "model/model.vrm");
    fixture.secondary_sha = product::sha256_file(
        fixture.package / "conditioning/secondary.bin");
    const std::string profile_sha = product::sha256_file(
        fixture.package / "execution/default.json");
    const std::string license_sha = product::sha256_file(
        fixture.package / "legal/LICENSE.txt");

    std::ostringstream manifest;
    manifest
        << "{\n"
        << "\"schema_version\":1,\n"
        << "\"identity\":{\"namespace\":\"vrhino\",\"name\":\"" << name
        << "\",\"version\":\"1.0.0\",\"architecture\":\"test_arch\","
           "\"publisher\":\"VRhino\"},\n"
        << "\"compatibility\":{\"runtime_contract\":\"cuda-v1\","
           "\"vrm_schema\":{\"format_major\":0,\"format_minor\":1,"
           "\"metadata_schema\":1}},\n"
        << "\"artifacts\":["
        << artifact_json("runtime", "runtime.vrm", "model/model.vrm", 3,
                         fixture.runtime_sha) << ','
        << artifact_json("secondary", "conditioning.weights", "conditioning/secondary.bin",
                         4, fixture.secondary_sha) << ','
        << artifact_json("profile", "execution.profile", "execution/default.json", 3,
                         profile_sha) << ','
        << artifact_json("license", "legal.license", "legal/LICENSE.txt", 8,
                         license_sha) << "],\n"
        << "\"entrypoint\":{\"runtime_artifact\":\"runtime\",\"components\":[{"
           "\"id\":\"conditioning\",\"kind\":\"conditioning.text_encoder\","
           "\"artifacts\":[\"secondary\"]}],\"default_preset\":\"default\"},\n"
        << "\"defaults\":{\"default_preset\":\"default\",\"presets\":{"
           "\"default\":{\"profile_artifact\":\"profile\",\"inputs\":{}}}},\n"
        << "\"hardware\":{\"presets\":{\"default\":{\"minimum_vram_bytes\":";
    if (minimum_vram.has_value()) manifest << *minimum_vram; else manifest << "null";
    manifest
        << ",\"recommended_vram_bytes\":null,\"minimum_compute_capability\":null}}},\n"
        << "\"source\":{\"repository\":\"example/fixture\",\"revision\":\""
        << kRevision << "\",\"converter_version\":\"test\"},\n"
        << "\"license\":{\"identifier\":\"Apache-2.0\",\"artifact\":\"license\","
           "\"upstream_notice\":\"https://example.invalid/license\"}\n"
        << "}\n";
    write_text(fixture.package / product::kModelManifestName, manifest.str());

    const fs::path specification = specs / name;
    write_text(specification / product::kModelManifestName, manifest.str());
    write_text(specification / "pull-plan.json",
        "{\"schema_version\":1,\"model_reference\":\"" + fixture.reference +
        "\",\"distribution\":{\"kind\":\"source_backed\",\"source\":{"
        "\"provider\":\"huggingface\",\"repository\":\"example/fixture\","
        "\"revision\":\"" + kRevision + "\"},\"source_plan\":\"source-plan.json\","
        "\"converter\":\"native\"}}\n");
    write_text(specification / "source-plan.json",
        "{\"schema_version\":1,\"model_reference\":\"" + fixture.reference +
        "\",\"requested_source\":{\"provider\":\"huggingface\","
        "\"repository\":\"example/fixture\",\"revision\":\"" + kRevision +
        "\"},\"artifacts\":[{\"id\":\"runtime-source\","
        "\"role\":\"checkpoint.denoiser\",\"repository\":\"example/fixture\","
        "\"revision\":\"" + kRevision + "\",\"upstream_path\":\"model.vrm\","
        "\"local_path\":\"model.vrm\",\"size\":3,\"sha256\":\"" +
        fixture.runtime_sha + "\"}]}\n");
    return fixture;
}

product::HardwareSnapshot hardware(const uint64_t available = 90 * kGiB) {
    product::HardwareSnapshot result;
    result.gpu_name = "Fixture GPU";
    result.compute_major = 12;
    result.compute_minor = 0;
    result.total_vram_bytes = 100 * kGiB;
    result.available_vram_bytes = available;
    result.driver_version = 13020;
    result.runtime_version = 12080;
    return result;
}

product::DoctorOptions options(const fs::path& cache, const fs::path& specs,
                               const fs::path& encoder,
                               const std::optional<std::string>& model = std::nullopt) {
    product::DoctorOptions result;
    result.product_version = "v-test";
    result.git_head = "0123456789abcdef";
    result.cache_root = cache;
    result.converter_spec_root = specs;
    result.encoder_path = encoder;
    result.model_reference = model;
    result.hardware_override = hardware();
    return result;
}

void require_contains(const std::string& text, const std::string& expected,
                      const std::string& context) {
    require_test(text.find(expected) != std::string::npos,
                 context + " omitted: " + expected);
}

}  // namespace

int main() {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-product-doctor-tests-" + std::to_string(timestamp));
    std::error_code cleanup_error;
    try {
        fs::create_directories(root);
        const fs::path specs = root / "specs";
        const Fixture qualified = make_fixture(root, specs, "fixture", 80 * kGiB);
        const Fixture no_minimum = make_fixture(root, specs, "fixture-no-min", std::nullopt);
        const fs::path encoder = root / "vrhino-ffmpeg";
        write_text(encoder, "#!/bin/sh\nexit 0\n");
        require_test(::chmod(encoder.c_str(), 0700) == 0, "cannot chmod encoder fixture");

        const fs::path home = root / "home";
        fs::create_directories(home);
        require_test(::setenv("HOME", home.c_str(), 1) == 0, "cannot set HOME fixture");

        product::DoctorOptions global_options = options(home / ".vrhino", specs, encoder);
        global_options.cache_root_is_default = true;
        const product::DoctorReport global = product::run_doctor(global_options);
        require_test(global.overall == product::DoctorOverall::Ready &&
                     global.exit_code() == 0, "healthy global doctor was not READY");
        require_contains(global.text, "Root: ~/.vrhino", "home path normalization");
        require_contains(global.text, "Bundled encoder: OK", "healthy encoder");
        require_contains(global.text, "Filesystem:", "cache filesystem");
        require_contains(global.text, " bytes)", "cache free bytes");
        require_contains(global.text, "VRAM: 90.00 GiB available / 100.00 GiB total",
                         "device facts");
        require_test(!fs::exists(home / ".vrhino"),
                     "global doctor created the absent default cache");

        const fs::path healthy_cache = root / "healthy-cache";
        product::LocalModelCache healthy(healthy_cache);
        (void)healthy.install(qualified.package);
        (void)healthy.install(no_minimum.package);
        const product::DoctorReport installed = product::run_doctor(
            options(healthy_cache, specs, encoder, qualified.reference));
        require_test(installed.overall == product::DoctorOverall::Ready &&
                     installed.exit_code() == 0, "healthy installed model was not READY");
        require_contains(installed.text, "Installed: yes", "installed model");
        require_contains(installed.text, "Artifacts: 4/4", "artifact count");
        require_contains(installed.text, "Package health: OK", "package health");
        require_contains(installed.text, "Component conditioning: OK", "component health");
        require_contains(installed.text, "Status: PASS", "VRAM admission pass");

        const product::DoctorReport non_installed = product::run_doctor(
            options(root / "empty-cache", specs, encoder, qualified.reference));
        require_test(non_installed.overall == product::DoctorOverall::Failed &&
                     non_installed.exit_code() == 1, "non-installed model did not fail");
        require_contains(non_installed.text, "Installed: no", "non-installed model");
        require_test(!fs::exists(root / "empty-cache"),
                     "model doctor created the absent cache");

        product::DoctorOptions contended_options = options(
            healthy_cache, specs, encoder, qualified.reference);
        contended_options.hardware_override = hardware(70 * kGiB);
        const product::DoctorReport contended = product::run_doctor(contended_options);
        require_test(contended.overall == product::DoctorOverall::Failed,
                     "available-VRAM admission failure was missed");
        require_contains(contended.text, "Required available VRAM: 80.00 GiB",
                         "required VRAM");
        require_contains(contended.text, "Current available VRAM: 70.00 GiB",
                         "available VRAM");

        const product::DoctorReport unknown_threshold = product::run_doctor(
            options(healthy_cache, specs, encoder, no_minimum.reference));
        require_test(unknown_threshold.overall == product::DoctorOverall::Warning &&
                     unknown_threshold.exit_code() == 0,
                     "no-threshold model did not produce non-blocking WARNING");
        require_contains(unknown_threshold.text, "Status: NOT_APPLICABLE",
                         "no-threshold admission");

        const fs::path missing_runtime_cache = root / "missing-runtime-cache";
        product::LocalModelCache missing_runtime(missing_runtime_cache);
        (void)missing_runtime.install(qualified.package);
        fs::remove(missing_runtime.artifact_path(qualified.runtime_sha));
        const product::DoctorReport missing_runtime_report = product::run_doctor(
            options(missing_runtime_cache, specs, encoder, qualified.reference));
        require_test(missing_runtime_report.overall == product::DoctorOverall::Failed,
                     "missing runtime did not fail package health");
        require_contains(missing_runtime_report.text, "Runtime: MISSING", "missing runtime");
        require_contains(missing_runtime_report.text, "Missing artifact: runtime",
                         "missing runtime identity");
        require_contains(missing_runtime_report.text,
                         "Recommended action: reinstall or re-pull the model package",
                         "damaged package action");

        const fs::path missing_secondary_cache = root / "missing-secondary-cache";
        product::LocalModelCache missing_secondary(missing_secondary_cache);
        (void)missing_secondary.install(qualified.package);
        fs::remove(missing_secondary.artifact_path(qualified.secondary_sha));
        const product::DoctorReport missing_secondary_report = product::run_doctor(
            options(missing_secondary_cache, specs, encoder, qualified.reference));
        require_contains(missing_secondary_report.text, "Component conditioning: MISSING",
                         "missing secondary component");
        require_contains(missing_secondary_report.text, "Missing artifact: secondary",
                         "missing secondary identity");

        const fs::path invalid_secondary_cache = root / "invalid-secondary-cache";
        product::LocalModelCache invalid_secondary(invalid_secondary_cache);
        (void)invalid_secondary.install(qualified.package);
        const fs::path invalid_secondary_path = invalid_secondary.artifact_path(
            qualified.secondary_sha);
        fs::remove(invalid_secondary_path);
        fs::create_directory(invalid_secondary_path);
        const product::DoctorReport invalid_secondary_report = product::run_doctor(
            options(invalid_secondary_cache, specs, encoder, qualified.reference));
        require_contains(invalid_secondary_report.text, "Component conditioning: INVALID",
                         "invalid secondary component");
        require_contains(invalid_secondary_report.text, "Invalid artifact: secondary",
                         "invalid secondary identity");

        const product::DoctorReport missing_encoder = product::run_doctor(
            options(root / "cache-missing-encoder", specs, root / "missing-encoder"));
        require_test(missing_encoder.overall == product::DoctorOverall::Failed,
                     "missing encoder did not fail global health");
        require_contains(missing_encoder.text, "Bundled encoder: FAILED", "missing encoder");

        const fs::path non_executable = root / "non-executable-encoder";
        write_text(non_executable, "#!/bin/sh\nexit 0\n");
        require_test(::chmod(non_executable.c_str(), 0600) == 0,
                     "cannot chmod non-executable encoder fixture");
        const product::DoctorReport non_executable_report = product::run_doctor(
            options(root / "cache-non-executable", specs, non_executable));
        require_contains(non_executable_report.text, "Executable: no",
                         "non-executable encoder");

        const fs::path unwritable = root / "unwritable-cache";
        fs::create_directories(unwritable);
        require_test(::chmod(unwritable.c_str(), 0500) == 0,
                     "cannot chmod unwritable cache fixture");
        const product::DoctorReport unwritable_report = product::run_doctor(
            options(unwritable, specs, encoder));
        require_test(unwritable_report.overall == product::DoctorOverall::Failed,
                     "unwritable cache did not fail global health");
        require_contains(unwritable_report.text, "Writable: no", "unwritable cache");
        require_test(::chmod(unwritable.c_str(), 0700) == 0,
                     "cannot restore cache fixture mode");

        constexpr const char* token_secret = "doctor-hf-token-unique-sentinel";
        constexpr const char* proxy_password = "doctor-proxy-password-unique-sentinel";
        require_test(::setenv("HF_TOKEN", token_secret, 1) == 0, "cannot set token sentinel");
        const std::string proxy = std::string("https://user:") + proxy_password +
                                  "@example.invalid:443";
        require_test(::setenv("HTTPS_PROXY", proxy.c_str(), 1) == 0,
                     "cannot set proxy sentinel");
        require_test(::setenv("VRHINO_TEST_PROMPT", "private-prompt-sentinel", 1) == 0,
                     "cannot set prompt sentinel");
        const fs::path custom_secret_path = root / "private-user-mount-sentinel" / "cache";
        const product::DoctorReport privacy = product::run_doctor(
            options(custom_secret_path, specs, encoder));
        require_contains(privacy.text, "HF_TOKEN: set", "token state");
        require_contains(privacy.text, "HTTPS_PROXY: set", "proxy state");
        require_contains(privacy.text, "Root: <custom>", "custom cache path redaction");
        require_test(privacy.text.find(token_secret) == std::string::npos &&
                     privacy.text.find(proxy_password) == std::string::npos &&
                     privacy.text.find(proxy) == std::string::npos &&
                     privacy.text.find("private-prompt-sentinel") == std::string::npos &&
                     privacy.text.find("private-user-mount-sentinel") == std::string::npos,
                     "doctor leaked a secret, prompt, or custom path");

        fs::remove_all(root, cleanup_error);
        std::cout << "product doctor tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::permissions(root / "unwritable-cache", fs::perms::owner_all,
                        fs::perm_options::replace, cleanup_error);
        fs::remove_all(root, cleanup_error);
        std::cerr << "product doctor tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
