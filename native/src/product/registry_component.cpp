#include "registry_transport.h"
#include <ostream>
#include <utility>
namespace vrhino::product {
namespace fs = std::filesystem;
using namespace registry_detail;
ComponentRegistryClient::ComponentRegistryClient(LocalModelCache& blob_cache,
                                                 LocalComponentCache& component_cache,
                                                 RegistryOptions options)
    : blob_cache_(blob_cache), component_cache_(component_cache), options_(std::move(options)) {
    if (options_.base_url.empty()) {
        fail(ModelPackageErrorCode::RegistryUnavailable,
             "no component registry configured; use --component-registry, --registry, "
             "VRHINO_COMPONENT_REGISTRY, or VRHINO_REGISTRY");
    }
    options_.base_url = trim_trailing_slashes(options_.base_url);
    validate_remote_url(options_.base_url, options_);
}

ComponentPullResult ComponentRegistryClient::pull(const std::string& exact_reference,
                                                  std::ostream* progress_output) {
    const PackageIdentity requested = parse_package_reference(exact_reference);
    try {
        const ResolvedComponent installed = component_cache_.resolve(exact_reference, true);
        return ComponentPullResult{installed.manifest.identity, installed.root, 0, 0, 0, true};
    } catch (const ModelPackageError& error) {
        if (error.code() != ModelPackageErrorCode::ComponentNotFound) throw;
    }
    const fs::path lock_root = blob_cache_.layout().temporary / "locks";
    FileLock package_lock(lock_root / ("component-" + lock_safe_identity(requested) + ".lock"));
    try {
        const ResolvedComponent installed = component_cache_.resolve(exact_reference, true);
        return ComponentPullResult{installed.manifest.identity, installed.root, 0, 0, 0, true};
    } catch (const ModelPackageError& error) {
        if (error.code() != ModelPackageErrorCode::ComponentNotFound) throw;
    }

    const std::string descriptor_url = options_.base_url + "/v1/components/" +
        requested.name_space + "/" + requested.name + "/" + requested.version +
        "/index.json";
    const ComponentRegistryDescriptor descriptor = parse_component_registry_descriptor(
        http_get_text(descriptor_url, options_, ModelPackageErrorCode::RegistryUnavailable),
        options_);
    if (descriptor.identity != exact_reference) {
        fail(ModelPackageErrorCode::ComponentInvalid,
             "component registry identity does not match requested exact version");
    }
    ArtifactDeclaration archive_artifact;
    archive_artifact.id = "component-archive";
    archive_artifact.role = "component_archive";
    archive_artifact.relative_path = "component.tar.gz";
    archive_artifact.size = descriptor.package_size;
    archive_artifact.sha256 = descriptor.package_sha256;
    archive_artifact.required = true;

    ComponentPullResult result;
    FileLock artifact_lock(lock_root / ("artifact-" + archive_artifact.sha256 + ".lock"));
    fs::path archive;
    if (blob_cache_.contains_blob(archive_artifact, true)) {
        archive = blob_cache_.artifact_path(archive_artifact.sha256);
        result.reused_bytes = archive_artifact.size;
    } else {
        const fs::path partial = blob_cache_.layout().temporary / "downloads" /
                                 (archive_artifact.sha256 + ".partial");
        std::error_code error;
        const uint64_t partial_size = fs::exists(partial) ? fs::file_size(partial, error) : 0;
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect partial component archive: " + error.message());
        fs::create_directories(blob_cache_.layout().root, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create component cache root: " + error.message());
        const fs::space_info space = fs::space(blob_cache_.layout().root, error);
        const uint64_t remaining = partial_size <= archive_artifact.size
                                       ? archive_artifact.size - partial_size
                                       : archive_artifact.size;
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect component cache free space: " + error.message());
        if (space.available < remaining) {
            fail(ModelPackageErrorCode::InsufficientDiskSpace,
                 "media component needs " + std::to_string(remaining) +
                     " more bytes but cache filesystem has " + std::to_string(space.available));
        }
        const NativeDownloadResult download = download_file(
            descriptor.package_url, partial, archive_artifact.size, "media component",
            0, archive_artifact.size, options_, progress_output);
        result.downloaded_bytes = download.network_bytes;
        result.resumed_bytes = download.resumed_bytes;
#ifdef _WIN32
        archive = blob_cache_.admit_downloaded_blob(*download.staging, archive_artifact).path;
#else
        archive = blob_cache_.admit_downloaded_blob(partial, archive_artifact).path;
#endif
        if (options_.progress)
            options_.progress(archive_artifact.size, archive_artifact.size);
    }
#ifdef _WIN32
    const ComponentInstallResult installed = component_cache_.install_archive(archive, options_.cancellation_requested);
#else
    const ComponentInstallResult installed = component_cache_.install_archive(archive);
#endif
    result.identity = installed.identity;
    result.root = installed.root;
    result.already_installed = installed.already_installed;
    return result;
}

} // namespace vrhino::product
