#include "vrhino/product/component_package.h"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include "windows_component_archive.h"
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::product {
namespace fs = std::filesystem;
namespace {

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

const Json& object_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_object()) {
        fail(ModelPackageErrorCode::ComponentInvalid, key + " must be an object");
    }
    return *result;
}

const Json& array_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_array()) {
        fail(ModelPackageErrorCode::ComponentInvalid, key + " must be an array");
    }
    return *result;
}

std::string string_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_string() || result->string().empty()) {
        fail(ModelPackageErrorCode::ComponentInvalid, key + " must be a non-empty string");
    }
    return result->string();
}

int64_t integer_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_int()) {
        fail(ModelPackageErrorCode::ComponentInvalid, key + " must be an integer");
    }
    return result->integer();
}

bool bool_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_bool()) {
        fail(ModelPackageErrorCode::ComponentInvalid, key + " must be a boolean");
    }
    return result->boolean();
}

uint64_t unsigned_field(const Json& value, const std::string& key) {
    const int64_t result = integer_field(value, key);
    if (result < 0) fail(ModelPackageErrorCode::ComponentInvalid,
                         key + " must be non-negative");
    return static_cast<uint64_t>(result);
}

fs::path safe_relative_path(const std::string& value, const std::string& field) {
#ifdef _WIN32
    (void)field;
    return windows_component_archive::relative_path(value);
#else
    const fs::path path(value);
    if (path.empty() || path.is_absolute()) {
        fail(ModelPackageErrorCode::ComponentInvalid, field + " must be relative");
    }
    for (const fs::path& part : path) {
        if (part == ".." || part == ".") {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 field + " contains an unsafe path component");
        }
    }
    return path;
#endif
}

std::string read_file(const fs::path& path) {
#ifdef _WIN32
    return windows_component_archive::read_text(path);
#else
    std::ifstream input(path, std::ios::binary);
    if (!input) fail(ModelPackageErrorCode::ComponentInvalid,
                     "cannot read component manifest: " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    if (!input.eof() && input.fail()) fail(ModelPackageErrorCode::ComponentInvalid,
                                           "cannot read component manifest: " + path.string());
    return output.str();
#endif
}

#ifndef _WIN32
std::string run_capture(const std::vector<std::string>& arguments) {
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0) {
        fail(ModelPackageErrorCode::InstallFailed, "cannot create archive inspection pipe");
    }
    const pid_t child = ::fork();
    if (child < 0) {
        ::close(pipe_fds[0]);
        ::close(pipe_fds[1]);
        fail(ModelPackageErrorCode::InstallFailed, "cannot start archive tool");
    }
    if (child == 0) {
        ::close(pipe_fds[0]);
        ::dup2(pipe_fds[1], STDOUT_FILENO);
        ::dup2(pipe_fds[1], STDERR_FILENO);
        ::close(pipe_fds[1]);
        std::vector<char*> argv;
        for (const std::string& argument : arguments) {
            argv.push_back(const_cast<char*>(argument.c_str()));
        }
        argv.push_back(nullptr);
        ::execv(argv[0], argv.data());
        _exit(127);
    }
    ::close(pipe_fds[1]);
    std::string output;
    char buffer[4096];
    for (;;) {
        const ssize_t count = ::read(pipe_fds[0], buffer, sizeof(buffer));
        if (count > 0) output.append(buffer, static_cast<size_t>(count));
        else if (count == 0) break;
        else if (errno != EINTR) break;
    }
    ::close(pipe_fds[0]);
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(ModelPackageErrorCode::InstallFailed,
             "component archive tool failed: " + output);
    }
    return output;
}

void validate_archive_listing(const std::string& listing) {
    std::istringstream input(listing);
    std::string line;
    bool manifest_seen = false;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const fs::path path(line);
        if (path.is_absolute()) fail(ModelPackageErrorCode::ComponentInvalid,
                                     "component archive contains an absolute path");
        auto iterator = path.begin();
        if (iterator == path.end() || *iterator != "vrhino-media") {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component archive must have only the vrhino-media top-level directory");
        }
        for (const fs::path& part : path) {
            if (part == "..") fail(ModelPackageErrorCode::ComponentInvalid,
                                    "component archive contains path traversal");
        }
        if (path == fs::path("vrhino-media") / "vrhino-component.json") {
            manifest_seen = true;
        }
    }
    if (!manifest_seen) fail(ModelPackageErrorCode::ComponentInvalid,
                             "component archive has no vrhino-component.json");
}

#endif

fs::path destination_for(const fs::path& components, const PackageIdentity& identity) {
#ifdef _WIN32
    windows_component_archive::identity_segment(identity.name_space);
    windows_component_archive::identity_segment(identity.name);
    windows_component_archive::identity_segment(identity.version);
    return windows_cache::native_path(components /
        windows_component_archive::relative_path(identity.name_space) /
        windows_component_archive::relative_path(identity.name) /
        windows_component_archive::relative_path(identity.version));
#else
    return components / identity.name_space / identity.name / identity.version;
#endif
}

void validate_platform_and_contract(const ComponentPackageManifest& manifest) {
    if (manifest.schema_version != kComponentPackageSchemaVersion) {
        fail(ModelPackageErrorCode::ComponentVersionUnsupported,
             "unsupported component package schema");
    }
    if (manifest.contract_name != kMediaComponentContract ||
        manifest.contract_major != 1) {
        fail(ModelPackageErrorCode::ComponentVersionUnsupported,
             "unsupported media component contract");
    }
#ifdef _WIN32
    if (manifest.operating_system != "windows" || manifest.machine_architecture != "x86_64") {
        fail(ModelPackageErrorCode::ComponentVersionUnsupported, "component platform is not windows/x86_64");
    }
#else
    if (manifest.operating_system != "linux" ||
        manifest.machine_architecture != "x86_64") {
        fail(ModelPackageErrorCode::ComponentVersionUnsupported,
             "component platform is not linux/x86_64");
    }
#endif
}

ResolvedComponent validate_installed(const fs::path& root,
                                     const std::string& requested,
                                     const bool verify_hashes
#ifdef _WIN32
                                     , const WorkProgressCallback& progress = {}
#endif
                                     ) {
    const fs::path manifest_path = root / "vrhino-component.json";
    ComponentPackageManifest manifest = load_component_package_manifest(manifest_path);
    if (manifest.identity.reference() != requested) {
        fail(ModelPackageErrorCode::ComponentInvalid,
             "component identity does not match cache path");
    }
    for (const ComponentArtifact& artifact : manifest.artifacts) {
        const fs::path path = root / artifact.relative_path;
#ifdef _WIN32
        windows_cache::LocalSource object(path);
        if (static_cast<uint64_t>(object.before().standard.EndOfFile.QuadPart) != artifact.size)
            fail(ModelPackageErrorCode::ComponentInvalid, "component artifact size mismatch");
        if (verify_hashes && object.digest(progress) != artifact.sha256)
            fail(ModelPackageErrorCode::ChecksumMismatch, "component artifact SHA256 mismatch");
        if (artifact.executable && !windows_component_archive::executable(path))
            fail(ModelPackageErrorCode::ComponentInvalid, "component artifact is not executable");
        object.verify();
#else
        std::error_code error;
        if (!fs::is_regular_file(path, error) || error) {
            fail(ModelPackageErrorCode::ArtifactMissing,
                 "component artifact is missing: " + artifact.relative_path.string());
        }
        if (fs::file_size(path, error) != artifact.size || error) {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component artifact size mismatch: " + artifact.relative_path.string());
        }
        if (verify_hashes && sha256_file(path) != artifact.sha256) {
            fail(ModelPackageErrorCode::ChecksumMismatch,
                 "component artifact SHA256 mismatch: " + artifact.relative_path.string());
        }
        if (artifact.executable && ::access(path.c_str(), X_OK) != 0) {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component artifact is not executable: " + artifact.relative_path.string());
        }
#endif
    }
    const fs::path entrypoint = root / manifest.entrypoint;
#ifdef _WIN32
    if (!windows_component_archive::executable(entrypoint)) {
#else
    if (!fs::is_regular_file(entrypoint) || ::access(entrypoint.c_str(), X_OK) != 0) {
#endif
        fail(ModelPackageErrorCode::ArtifactMissing, "component entrypoint is unavailable");
    }
    return ResolvedComponent{std::move(manifest), root, entrypoint};
}

}  // namespace

ComponentPackageManifest load_component_package_manifest(const fs::path& path) {
    try {
        ComponentPackageManifest manifest;
        manifest.raw_json = read_file(path);
        const Json root = Json::parse(manifest.raw_json);
        if (!root.is_object()) fail(ModelPackageErrorCode::ComponentInvalid,
                                    "component manifest root must be an object");
        manifest.schema_version = integer_field(root, "schema_version");
        const Json& identity = object_field(root, "identity");
        manifest.identity.name_space = string_field(identity, "namespace");
        manifest.identity.name = string_field(identity, "name");
        manifest.identity.version = string_field(identity, "version");
        manifest.identity.publisher = string_field(identity, "publisher");
#ifdef _WIN32
        (void)parse_package_reference(manifest.identity.reference());
        windows_component_archive::identity_segment(manifest.identity.name_space);
        windows_component_archive::identity_segment(manifest.identity.name);
        windows_component_archive::identity_segment(manifest.identity.version);
#endif
        const Json& contract = object_field(root, "contract");
        manifest.contract_name = string_field(contract, "name");
        manifest.contract_major = integer_field(contract, "major");
        manifest.contract_minor = integer_field(contract, "minor");
        const Json& platform = object_field(root, "platform");
        manifest.operating_system = string_field(platform, "os");
        manifest.machine_architecture = string_field(platform, "architecture");
        manifest.entrypoint = safe_relative_path(string_field(root, "entrypoint"), "entrypoint");
#ifdef _WIN32
        const auto less = [](const fs::path& first, const fs::path& second) {
            const auto& a = first.native();
            const auto& b = second.native();
            return CompareStringOrdinal(a.data(), static_cast<int>(a.size()), b.data(),
                                        static_cast<int>(b.size()), TRUE) == CSTR_LESS_THAN;
        };
        std::set<fs::path, decltype(less)> paths(less);
#else
        std::set<fs::path> paths;
#endif
        for (const Json& value : array_field(root, "artifacts").array()) {
            if (!value.is_object()) fail(ModelPackageErrorCode::ComponentInvalid,
                                         "component artifact must be an object");
            ComponentArtifact artifact;
            artifact.relative_path = safe_relative_path(string_field(value, "path"), "artifact path");
            artifact.size = unsigned_field(value, "size");
            artifact.sha256 = string_field(value, "sha256");
            artifact.executable = bool_field(value, "executable");
            if (artifact.sha256.size() != 64 || !paths.insert(artifact.relative_path).second) {
                fail(ModelPackageErrorCode::ComponentInvalid,
                     "invalid or duplicate component artifact declaration");
            }
            manifest.artifacts.push_back(std::move(artifact));
        }
        if (manifest.artifacts.empty()) fail(ModelPackageErrorCode::ComponentInvalid,
                                             "component has no artifacts");
        const auto entrypoint = std::find_if(
            manifest.artifacts.begin(), manifest.artifacts.end(),
            [&manifest](const ComponentArtifact& artifact) {
                return artifact.relative_path == manifest.entrypoint && artifact.executable;
            });
        if (entrypoint == manifest.artifacts.end()) {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component entrypoint must be a declared executable artifact");
        }
        validate_platform_and_contract(manifest);
        return manifest;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::ComponentInvalid, error.what());
    }
}

LocalComponentCache::LocalComponentCache(fs::path root)
    : layout_(cache_layout(root)) {}

fs::path LocalComponentCache::components_root() const {
    return layout_.root / "components";
}

ResolvedComponent LocalComponentCache::resolve(const std::string& exact_reference,
                                               const bool verify_hashes) const {
    const PackageIdentity identity = parse_package_reference(exact_reference);
    const fs::path root = destination_for(components_root(), identity);
    if (!fs::exists(root)) fail(ModelPackageErrorCode::ComponentNotFound,
                                "media component is not installed: " + exact_reference);
    return validate_installed(root, exact_reference, verify_hashes);
}

ComponentInstallResult LocalComponentCache::install_archive(const fs::path& archive) {
#ifdef _WIN32
    return install_archive(archive, {});
#else
    if (!fs::is_regular_file(archive)) fail(ModelPackageErrorCode::ArtifactMissing,
                                            "component archive does not exist: " + archive.string());
    validate_archive_listing(run_capture({"/usr/bin/tar", "-tzf", archive.string()}));
    const fs::path staging = layout_.temporary /
        ("component-install-" + std::to_string(static_cast<long long>(::getpid())));
    std::error_code error;
    fs::remove_all(staging, error);
    fs::create_directories(staging, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create component staging directory: " + error.message());
    try {
        run_capture({"/usr/bin/tar", "-xzf", archive.string(), "--no-same-owner", "-C",
                     staging.string()});
        const fs::path extracted = staging / "vrhino-media";
        ComponentPackageManifest manifest = load_component_package_manifest(
            extracted / "vrhino-component.json");
        const std::string reference = manifest.identity.reference();
        (void)validate_installed(extracted, reference, true);
        const fs::path destination = destination_for(components_root(), manifest.identity);
        if (fs::exists(destination)) {
            (void)validate_installed(destination, reference, true);
            fs::remove_all(staging, error);
            return ComponentInstallResult{manifest.identity, destination, true};
        }
        fs::create_directories(destination.parent_path(), error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create component cache directory: " + error.message());
        fs::rename(extracted, destination, error);
        if (error) fail(ModelPackageErrorCode::InstallFailed,
                        "cannot atomically publish component: " + error.message());
        fs::remove_all(staging, error);
        return ComponentInstallResult{manifest.identity, destination, false};
    } catch (...) {
        fs::remove_all(staging, error);
        throw;
    }
#endif
}

#ifdef _WIN32
ComponentInstallResult LocalComponentCache::install_archive(
    const fs::path& archive, const std::function<bool()>& cancelled) {
    namespace wc = windows_cache;
    auto check_cancel = [&](uint64_t = 0) {
        if (cancelled && cancelled()) fail(ModelPackageErrorCode::Cancelled, "component extraction cancelled");
    };
    check_cancel();
    wc::LocalSource source(archive);
    const std::string digest = source.digest(check_cancel);
    wc::ensure_directory(layout_.temporary);
    // This archive-specific lock also owns recovery of its private interrupted
    // extraction. No general scavenger and no deletion of published components.
    wc::Lock lock(layout_.root / "locks" / ("component-extraction-" + digest + ".lock"));
    const fs::path staging = wc::native_path(layout_.temporary / ("component-extraction-" + digest));
    if (fs::exists(staging)) wc::remove_tree(staging);
    wc::ensure_directory(staging);
    try {
        windows_component_archive::extract(source, staging, cancelled);
        const fs::path extracted = staging / "vrhino-media";
        ComponentPackageManifest manifest = load_component_package_manifest(extracted / "vrhino-component.json");
        const std::string reference = manifest.identity.reference();
        (void)validate_installed(extracted, reference, true, check_cancel);
        source.verify();
        check_cancel();
        const fs::path destination = destination_for(components_root(), manifest.identity);
        wc::ensure_directory(destination.parent_path());
        // Reuse and publication share the established package-lock order at the
        // registry boundary; standalone competing installs use no-replace rename.
        if (fs::exists(destination)) {
            (void)validate_installed(destination, reference, true, check_cancel);
            wc::remove_tree(staging);
            return ComponentInstallResult{manifest.identity, destination, true};
        }
#ifdef VRHINO_WINDOWS_ARCHIVE_TESTING
        windows_component_archive::checkpoint(windows_component_archive::Point::BeforeCommit);
#endif
        source.verify();
        check_cancel();
        bool reused = false;
        try { wc::publish_directory(extracted, destination); }
        catch (...) {
            // A successful-but-uncertain rename or a competing winner is checked
            // in place. The final destination is never a rollback target.
            if (!fs::exists(destination)) throw;
            (void)validate_installed(destination, reference, true, check_cancel);
            reused = fs::exists(extracted);
        }
        wc::remove_tree(staging);
        return ComponentInstallResult{manifest.identity, destination, reused};
    } catch (...) {
        const auto failure = std::current_exception();
        wc::remove_tree(staging);
        std::rethrow_exception(failure);
    }
}
#endif

}  // namespace vrhino::product
