#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_file(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    require_test(static_cast<bool>(output), "failed to write test file: " + path.string());
}

std::string artifact_json(const std::string& id, const std::string& role,
                          const std::string& path, const uint64_t size,
                          const std::string& sha256) {
    std::ostringstream stream;
    stream << "{\"id\":\"" << id << "\",\"role\":\"" << role
           << "\",\"path\":\"" << path << "\",\"size\":" << size
           << ",\"sha256\":\"" << sha256 << "\",\"required\":true}";
    return stream.str();
}

fs::path make_package(const fs::path& parent, const std::string& version,
                      const int64_t schema = 1, const bool missing_profile = false,
                      const bool bad_runtime_hash = false) {
    const fs::path package = parent / ("package-" + version);
    fs::create_directories(package);
    write_file(package / "model/model.vrm", "abc");
    if (!missing_profile) write_file(package / "execution/default.json", "{}\n");
    write_file(package / "legal/LICENSE.txt", "test license\n");
    const std::string runtime_hash = bad_runtime_hash
                                         ? std::string(64, '0')
                                         : product::sha256_file(package / "model/model.vrm");
    const std::string profile_hash = missing_profile
                                         ? product::sha256_file(package / "model/model.vrm")
                                         : product::sha256_file(package / "execution/default.json");
    const std::string license_hash = product::sha256_file(package / "legal/LICENSE.txt");
    std::ostringstream manifest;
    manifest
        << "{\n"
        << "  \"schema_version\": " << schema << ",\n"
        << "  \"identity\": {\"namespace\":\"vrhino\",\"name\":\"fixture\","
           "\"version\":\"" << version
        << "\",\"architecture\":\"test_arch\",\"publisher\":\"VRhino\"},\n"
        << "  \"compatibility\": {\"runtime_contract\":\"cuda-v1\","
           "\"vrm_schema\":{\"format_major\":0,\"format_minor\":1,"
           "\"metadata_schema\":1}},\n"
        << "  \"artifacts\": [\n"
        << "    " << artifact_json("runtime", "runtime.vrm", "model/model.vrm", 3,
                                      runtime_hash) << ",\n"
        << "    " << artifact_json("profile", "execution.profile", "execution/default.json",
                                      missing_profile ? 3 : 3, profile_hash) << ",\n"
        << "    " << artifact_json("license", "legal.license", "legal/LICENSE.txt", 13,
                                      license_hash) << "\n"
        << "  ],\n"
        << "  \"entrypoint\": {\"runtime_artifact\":\"runtime\",\"components\":[],"
           "\"default_preset\":\"default\"},\n"
        << "  \"defaults\": {\"default_preset\":\"default\",\"presets\":{"
           "\"default\":{\"profile_artifact\":\"profile\",\"inputs\":{}}}},\n"
        << "  \"hardware\": {\"presets\":{\"default\":{"
           "\"minimum_vram_bytes\":null}}},\n"
        << "  \"source\": {\"repository\":\"example/fixture\","
           "\"revision\":\"0123456789abcdef\",\"converter_version\":\"test\"},\n"
        << "  \"license\": {\"identifier\":\"LicenseRef-Test\","
           "\"artifact\":\"license\",\"upstream_notice\":\"example.invalid\"}\n"
        << "}\n";
    write_file(package / product::kModelManifestName, manifest.str());
    return package;
}

template <typename Callable>
void expect_error(const product::ModelPackageErrorCode expected, Callable&& callable) {
    try {
        callable();
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, "unexpected model package error code");
        return;
    }
    throw std::runtime_error("expected ModelPackageError was not raised");
}

size_t count_blobs(const fs::path& root) {
    size_t count = 0;
    if (!fs::exists(root)) return 0;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()) ++count;
    }
    return count;
}

}  // namespace

int main() {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
                          ("vrhino-model-package-test-" + std::to_string(timestamp));
    try {
        const fs::path source = root / "source";
        const fs::path cache_root = root / "cache";
        product::LocalModelCache cache(cache_root);

        const fs::path configured_home = root / "configured-home";
        const fs::path process_home = root / "process-home";
        require_test(::setenv("VRHINO_HOME", configured_home.c_str(), 1) == 0,
                     "cannot set VRHINO_HOME fixture");
        require_test(::setenv("HOME", process_home.c_str(), 1) == 0,
                     "cannot set HOME fixture");
        require_test(product::cache_layout().root == fs::absolute(configured_home),
                     "VRHINO_HOME did not select the cache root");
        const fs::path explicit_home = root / "explicit-home";
        require_test(product::cache_layout(explicit_home).root ==
                         fs::absolute(explicit_home),
                     "explicit cache root did not override VRHINO_HOME");
        require_test(::unsetenv("VRHINO_HOME") == 0,
                     "cannot clear VRHINO_HOME fixture");
        require_test(product::cache_layout().root ==
                         fs::absolute(process_home / ".vrhino"),
                     "HOME/.vrhino is not the default cache root");

        const fs::path sha_fixture = root / "abc";
        write_file(sha_fixture, "abc");
        uint64_t hashed_bytes = 0;
        require_test(product::sha256_file(sha_fixture, [&](const uint64_t bytes) {
                         hashed_bytes += bytes;
                     }) ==
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                     "SHA256 implementation failed the abc test vector");
        require_test(hashed_bytes == 3, "SHA256 progress byte accounting is wrong");

        const auto first = cache.install(make_package(source, "1.0.0"));
        require_test(first.identity.reference() == "vrhino/fixture:1.0.0",
                     "unexpected installed identity");
        require_test(first.blobs_created == 3 && first.blobs_reused == 0,
                     "first install did not create three blobs");
        require_test((fs::status(first.manifest_path).permissions() & fs::perms::owner_write) ==
                         fs::perms::none,
                     "installed manifest is writable");
        const auto resolved = cache.resolve("vrhino/fixture:1.0.0", true);
        require_test(resolved.runtime_model_path.filename() == resolved.manifest.artifacts[0].sha256,
                     "runtime model was not resolved from CAS");
        require_test(resolved.artifacts.size() == 3, "resolver omitted required artifacts");

        const auto second = cache.install(make_package(source, "1.0.1"));
        require_test(second.blobs_created == 0 && second.blobs_reused == 3,
                     "second version did not reuse content-addressed blobs");
        require_test(cache.list().size() == 2, "version coexistence failed");
        require_test(count_blobs(cache.layout().blobs) == 3, "CAS contains duplicate blobs");

        expect_error(product::ModelPackageErrorCode::InstallFailed, [&] {
            cache.install(source / "package-1.0.0");
        });
        expect_error(product::ModelPackageErrorCode::ChecksumMismatch, [&] {
            cache.install(make_package(source, "bad-hash", 1, false, true));
        });
        expect_error(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            cache.install(make_package(source, "missing", 1, true));
        });
        expect_error(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            cache.install(make_package(source, "schema-two", 2));
        });
        require_test(cache.list().size() == 2,
                     "failed install published an incomplete package version");
        expect_error(product::ModelPackageErrorCode::ModelNotFound, [&] {
            (void)cache.resolve("vrhino/fixture:9.9.9");
        });

        cache.remove("vrhino/fixture:1.0.0");
        require_test(cache.list().size() == 1, "removing one version damaged coexistence");
        require_test(count_blobs(cache.layout().blobs) == 3,
                     "package removal unexpectedly removed shared blobs");
        cache.remove("vrhino/fixture:1.0.1");
        require_test(cache.list().empty(), "package references were not removed");
        require_test(count_blobs(cache.layout().blobs) == 3,
                     "v0 orphan retention policy was not preserved");

        const auto repair_install = cache.install(make_package(source, "repair"));
        const product::ResolvedRunnableModel repair_model =
            cache.resolve("vrhino/fixture:repair", true);
        const product::ArtifactDeclaration runtime_declaration =
            repair_model.artifacts.at("runtime").declaration;
        const fs::path runtime_blob = repair_model.artifacts.at("runtime").path;
        fs::permissions(runtime_blob, fs::perms::owner_write, fs::perm_options::add);
        write_file(runtime_blob, "abd");
        expect_error(product::ModelPackageErrorCode::ChecksumMismatch, [&] {
            (void)cache.resolve("vrhino/fixture:repair", true);
        });
        cache.discard_installed_package_for_repair("vrhino/fixture:repair");
        cache.discard_invalid_blob(runtime_declaration);
        require_test(!fs::exists(runtime_blob),
                     "corrupt component blob was not discarded for repair");
        const auto repaired = cache.install(source / "package-repair");
        require_test(repaired.blobs_created == 1 && repaired.blobs_reused == 2,
                     "normal install did not repair only the corrupt component");
        (void)cache.resolve("vrhino/fixture:repair", true);
        cache.remove("vrhino/fixture:repair");
        require_test(repair_install.identity.reference() == repaired.identity.reference(),
                     "repair changed package identity");

        fs::remove_all(root);
        std::cout << "model package/cache tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << "model package/cache tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
