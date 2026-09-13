// Included inside vrhino::product in model_package.cpp. Reuses the manifest
// parser and path policy; only the Windows filesystem mechanism differs.
namespace wc = windows_cache;

BlobAdmissionResult LocalModelCache::admit_downloaded_blob(
        const fs::path& path, const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress) {
    wc::StagedFile staging(path, true);
    return admit_downloaded_blob(staging, artifact, progress);
}

BlobAdmissionResult LocalModelCache::admit_downloaded_blob(
        wc::StagedFile& staging, const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress) {
    const auto destination = artifact_path(artifact.sha256);
    wc::ensure_directory(destination.parent_path());
    wc::Parents parent(destination.parent_path());
    wc::same_volume(staging.get(), parent.leaf());
    if (staging.size() != artifact.size)
        fail(ModelPackageErrorCode::DownloadFailed, "completed artifact size mismatch: " + artifact.id);
    try { staging.readonly(); staging.flush(); staging.validate(artifact.size, artifact.sha256, progress); }
    catch (...) { staging.discard(); throw; }
    if (contains_blob(artifact, true)) {
        staging.discard(); return {destination, false};
    }
    if (fs::exists(destination)) fail(ModelPackageErrorCode::CacheError, "existing CAS blob is invalid: " + artifact.sha256);
    try { staging.publish(destination); }
    catch (const ModelPackageError&) {
        if (staging.published()) throw;
        if (contains_blob(artifact, true)) { staging.discard(); return {destination, false}; }
        throw;
    }
    return {destination, true};
}

BlobAdmissionResult LocalModelCache::admit_local_blob(
        const fs::path& path, const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress) {
    const auto destination = artifact_path(artifact.sha256);
    wc::ensure_directory(destination.parent_path());
    auto source = std::make_unique<wc::LocalSource>(path, true);
    if (static_cast<uint64_t>(source->before().standard.EndOfFile.QuadPart) != artifact.size ||
        source->digest(progress) != artifact.sha256)
        fail(ModelPackageErrorCode::ChecksumMismatch, "local CAS source size/SHA256 mismatch: " + artifact.id);
    if (contains_blob(artifact, true, progress)) return {destination, false};
    const auto staging_path = wc::unique_path(destination.parent_path(), ".local-admission");

    if (!source->writable()) {
        // Approved Windows durability-preparation exception: retain the one
        // successful source open and its mutation guard throughout copying.
        // Never modify an attribute on the source or any of its aliases.
        wc::require_space(destination.parent_path(), artifact.size);
        wc::StagedFile staging(staging_path);
        staging.copy(*source);
        staging.readonly(); staging.flush();
        staging.validate(artifact.size, artifact.sha256, progress);
        source->verify();
        if (contains_blob(artifact, true)) { staging.discard(); return {destination, false}; }
        if (fs::exists(destination)) fail(ModelPackageErrorCode::CacheError, "existing CAS blob is invalid");
        staging.publish(destination);
        return {destination, true};
    }

    // Ordinary writable local admission retains its hardlink policy. The
    // source remains opened until the new link's identity has been checked.
    const auto source_identity = source->before();
    if (!CreateHardLinkW(wc::native_path(staging_path).c_str(), wc::native_path(path).c_str(), nullptr)) {
        // Existing local-admission policy, separate from the read-only exception:
        // Linux also copies only when the actual hardlink creation fails.
        source->verify();
        wc::require_space(destination.parent_path(), artifact.size);
        wc::StagedFile staging(staging_path);
        staging.copy(*source);
        staging.readonly(); staging.flush();
        staging.validate(artifact.size, artifact.sha256, progress);
        source->verify();
        staging.publish(destination);
        return {destination, true};
    }
    try {
        {
            wc::Handle linked(CreateFileW(wc::native_path(staging_path).c_str(), FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
            if (!linked || !wc::same_identity(source_identity, wc::snapshot(linked.get())))
                fail(ModelPackageErrorCode::ChecksumMismatch, "local hardlink source identity changed");
        }
        // The link creation is our own metadata mutation. Content is checked
        // again through the initial writable staging acquisition before commit.
        source.reset();
        wc::StagedFile staging(staging_path, true);
        if (!wc::same_identity(source_identity, wc::snapshot(staging.get())))
            fail(ModelPackageErrorCode::ChecksumMismatch, "local staging identity changed");
        return admit_downloaded_blob(staging, artifact, progress);
    } catch (...) { wc::remove(staging_path); throw; }
}

InstallResult LocalModelCache::publish_manifest(const fs::path& manifest_path) {
    return publish_manifest(manifest_path, {});
}

InstallResult LocalModelCache::publish_manifest(const fs::path& manifest_path, const std::string& expected_sha256) {
    wc::LocalSource manifest_source(manifest_path);
    const auto manifest = load_model_package_manifest(manifest_path);
    if (!expected_sha256.empty() && manifest_source.digest() != expected_sha256)
        fail(ModelPackageErrorCode::ChecksumMismatch, "manifest changed before publication");
    manifest_source.verify();
    const auto target_manifest = package_manifest_path(layout_, manifest.identity);
    const auto target = target_manifest.parent_path();
    if (fs::exists(target)) fail(ModelPackageErrorCode::InstallFailed, "immutable package version is already installed");
    wc::ensure_directory(target.parent_path());
    wc::ensure_directory(layout_.temporary);
    InstallResult result; result.identity = manifest.identity; result.manifest_path = target_manifest;
    for (const auto& artifact : manifest.artifacts) {
        if (!contains_blob(artifact)) {
            if (artifact.required) fail(ModelPackageErrorCode::ArtifactMissing, "required CAS artifact is missing or invalid: " + artifact.id);
            continue;
        }
        ++result.blobs_reused; result.bytes_reused += artifact.size;
    }
    const auto staging = wc::unique_path(layout_.temporary, "install");
    wc::ensure_directory(staging);
    try {
        wc::write_text(staging / kModelManifestName, manifest.raw_json, true);
        wc::publish_directory(staging, target);
    } catch (...) { if (fs::exists(staging)) wc::remove_tree(staging); throw; }
    return result;
}

InstallResult LocalModelCache::install(const fs::path& package_directory) {
    std::error_code error;
    const auto root = fs::canonical(package_directory, error);
    if (error || !fs::is_directory(root)) fail(ModelPackageErrorCode::PackageInvalid, "package source is not a directory");
    wc::Parents source_parent(root);
    const auto source_manifest = root / kModelManifestName;
    wc::LocalSource manifest_source(source_manifest);
    const auto manifest = load_model_package_manifest(source_manifest);
    const auto manifest_digest = manifest_source.digest();
    const auto target_manifest = package_manifest_path(layout_, manifest.identity);
    if (fs::exists(target_manifest.parent_path())) fail(ModelPackageErrorCode::InstallFailed, "immutable package version is already installed");
    InstallResult result; result.identity = manifest.identity; result.manifest_path = target_manifest;
    for (const auto& artifact : manifest.artifacts) {
        const auto path = root / artifact.relative_path;
        if (!fs::is_regular_file(fs::symlink_status(path, error)) || error) {
            if (artifact.required) fail(ModelPackageErrorCode::ArtifactMissing, "required artifact is missing: " + artifact.id);
            error.clear(); continue;
        }
        const auto canonical = fs::canonical(path, error);
        if (error || !path_is_within(canonical, root)) fail(ModelPackageErrorCode::PackageInvalid, "artifact escapes package directory");
        wc::LocalSource source(path);
        if (static_cast<uint64_t>(source.before().standard.EndOfFile.QuadPart) != artifact.size)
            fail(ModelPackageErrorCode::PackageInvalid, "artifact size mismatch: " + artifact.id);
        const auto destination = artifact_path(artifact.sha256);
        wc::ensure_directory(destination.parent_path());
        if (fs::exists(destination)) {
            if (!contains_blob(artifact)) fail(ModelPackageErrorCode::CacheError, "existing CAS blob is invalid");
            if (source.digest() != artifact.sha256) fail(ModelPackageErrorCode::ChecksumMismatch, "artifact SHA256 mismatch");
            ++result.blobs_reused; result.bytes_reused += artifact.size; continue;
        }
        // Package installation already copies artifacts on Linux.
        wc::require_space(destination.parent_path(), artifact.size);
        wc::StagedFile staging(wc::unique_path(destination.parent_path(), ".incoming"));
        staging.copy(source);
        staging.readonly(); staging.flush(); staging.validate(artifact.size, artifact.sha256);
        source.verify();
        staging.publish(destination);
        ++result.blobs_created; result.bytes_created += artifact.size;
    }
    manifest_source.verify();
    publish_manifest(source_manifest, manifest_digest);
    return result;
}
