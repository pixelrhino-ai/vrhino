#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/product/input_schema.h"

namespace vrhino::product {
#ifdef _WIN32
namespace windows_cache { class StagedFile; }
#endif

using WorkProgressCallback = std::function<void(uint64_t)>;

inline constexpr int64_t kModelPackageSchemaVersion = 1;
inline constexpr const char* kCudaRuntimeContract = "cuda-v1";
inline constexpr int64_t kVrmFormatMajor = 0;
inline constexpr int64_t kVrmFormatMinor = 1;
inline constexpr int64_t kVrmMetadataSchema = 1;
inline constexpr const char* kModelManifestName = "vrhino-model.json";

enum class ModelPackageErrorCode {
    ModelNotFound,
    PackageInvalid,
    PackageVersionUnsupported,
    ArtifactMissing,
    ChecksumMismatch,
    InstallFailed,
    CacheError,
    RegistryUnavailable,
    DownloadFailed,
    DownloadResumeFailed,
    NetworkError,
    InsufficientDiskSpace,
    UnsupportedGpu,
    InsufficientVram,
    DriverIncompatible,
    InvalidInput,
    RuntimeError,
    OutOfMemory,
    OutputExists,
    OutputInvalid,
    VideoEncodingFailed,
    Cancelled,
    ComponentNotFound,
    ComponentInvalid,
    ComponentVersionUnsupported,
    SourceInvalid,
    SourceRevisionRequired,
    SourceNotFound,
    SourceDownloadFailed,
    SourceDownloadResumeFailed,
    SourceIntegrityFailed,
    SourceDiskFull,
    PullPlanNotFound,
};

const char* model_package_error_code_name(ModelPackageErrorCode code);

class ModelPackageError : public std::runtime_error {
public:
    ModelPackageError(ModelPackageErrorCode code, const std::string& message);
    ModelPackageErrorCode code() const noexcept { return code_; }

private:
    ModelPackageErrorCode code_;
};

struct PackageIdentity {
    std::string name_space;
    std::string name;
    std::string version;
    std::string architecture;
    std::string publisher;

    std::string reference() const;
};

struct ArtifactDeclaration {
    std::string id;
    std::string role;
    std::filesystem::path relative_path;
    uint64_t size = 0;
    std::string sha256;
    bool required = true;
};

struct ComponentDeclaration {
    std::string id;
    std::string role;
    std::string kind;
    std::vector<std::string> artifact_ids;
};

struct ProductDeclaration {
    std::string family = "text_to_video";
    std::string status = "public_supported";
    std::string workflow_identity;
    std::string workflow_artifact_id;
    std::string execution_artifact_id;
    std::vector<std::string> required_inputs;
    std::string public_distribution = "supported";
    std::optional<ProductInputSchema> input_schema;
    std::optional<ProductFrozenProfile> frozen_profile;
};

struct ModelPackageManifest {
    int64_t schema_version = 0;
    PackageIdentity identity;
    std::string runtime_contract;
    int64_t vrm_format_major = -1;
    int64_t vrm_format_minor = -1;
    int64_t vrm_metadata_schema = -1;
    ProductDeclaration product;
    std::vector<ArtifactDeclaration> artifacts;
    std::string runtime_artifact_id;
    std::vector<ComponentDeclaration> components;
    std::string default_preset;
    std::string license_identifier;
    std::string license_artifact_id;
    std::string source_repository;
    std::string source_revision;
    std::optional<uint64_t> minimum_vram_bytes;
    std::optional<uint64_t> recommended_vram_bytes;
    std::string raw_json;

    uint64_t logical_size() const;
};

struct ResolvedArtifact {
    ArtifactDeclaration declaration;
    std::filesystem::path path;
};

struct ResolvedRunnableModel {
    ModelPackageManifest manifest;
    std::filesystem::path manifest_path;
    std::filesystem::path runtime_model_path;
    std::map<std::string, ResolvedArtifact> artifacts;
};

struct CacheLayout {
    std::filesystem::path root;
    std::filesystem::path models;
    std::filesystem::path blobs;
    std::filesystem::path temporary;
};

struct InstallResult {
    PackageIdentity identity;
    std::filesystem::path manifest_path;
    size_t blobs_created = 0;
    size_t blobs_reused = 0;
    uint64_t bytes_created = 0;
    uint64_t bytes_reused = 0;
};

struct InstalledPackage {
    PackageIdentity identity;
    std::filesystem::path manifest_path;
};

struct BlobAdmissionResult {
    std::filesystem::path path;
    bool created = false;
};

CacheLayout cache_layout(const std::filesystem::path& explicit_root = {});
PackageIdentity parse_package_reference(const std::string& reference);
ModelPackageManifest load_model_package_manifest(const std::filesystem::path& path);
std::string sha256_file(const std::filesystem::path& path,
                        const WorkProgressCallback& progress = {});
std::string sha256_file_descriptor(int descriptor, uint64_t size,
                                   const WorkProgressCallback& progress = {});
std::string copy_file_and_sha256(const std::filesystem::path& source,
                                 const std::filesystem::path& destination,
                                 const WorkProgressCallback& progress = {});

class LocalModelCache {
public:
    explicit LocalModelCache(std::filesystem::path root = {});

    const CacheLayout& layout() const noexcept { return layout_; }
    InstallResult install(const std::filesystem::path& package_directory);
    InstallResult publish_manifest(const std::filesystem::path& manifest_path);
    bool contains_blob(const ArtifactDeclaration& artifact,
                       bool verify_hash = false,
                       const WorkProgressCallback& progress = {}) const;
    std::filesystem::path artifact_path(const std::string& sha256) const;
    BlobAdmissionResult admit_downloaded_blob(
        const std::filesystem::path& completed_download,
        const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress = {});
    BlobAdmissionResult admit_local_blob(
        const std::filesystem::path& verified_local_file,
        const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress = {});
#ifdef _WIN32
    InstallResult publish_manifest(const std::filesystem::path& manifest_path,
                                   const std::string& expected_sha256);
    BlobAdmissionResult admit_downloaded_blob(
        windows_cache::StagedFile& staging,
        const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress = {});
#endif
    void discard_installed_package_for_repair(const std::string& reference);
    void discard_invalid_blob(const ArtifactDeclaration& artifact);
    ResolvedRunnableModel resolve(const std::string& reference,
                                  bool verify_hashes = false) const;
    std::vector<InstalledPackage> list() const;
    void remove(const std::string& reference);

private:
    CacheLayout layout_;
};

}  // namespace vrhino::product
