#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "vrhino/product/model_package.h"
#include "vrhino/product/registry.h"

namespace vrhino::product {

inline constexpr int64_t kSourcePlanSchemaVersion = 1;

struct SourceReference {
    std::string provider;
    std::string repository;
    std::string revision;

    std::string canonical() const;
};

SourceReference parse_source_reference(const std::string& reference);

struct SourceArtifactPlan {
    std::string id;
    std::string role;
    std::string provider = "huggingface";
    std::string repository;
    std::string revision;
    std::string fixed_url;
    std::filesystem::path upstream_path;
    std::filesystem::path local_path;
    uint64_t size = 0;
    std::string sha256;
};

struct SourceArtifactPlanDocument {
    int64_t schema_version = 0;
    std::string model_reference;
    SourceReference requested_source;
    std::vector<SourceArtifactPlan> artifacts;
    std::filesystem::path document_path;
    std::string raw_json;
};

SourceArtifactPlanDocument load_source_artifact_plan(
    const std::string& model_reference,
    const std::filesystem::path& converter_spec_root);

struct SourceCacheLayout {
    std::filesystem::path root;
    std::filesystem::path blobs;
    std::filesystem::path trees;
    std::filesystem::path temporary;
};

struct AcquiredSourceArtifact {
    SourceArtifactPlan declaration;
    std::filesystem::path blob_path;
    bool downloaded = false;
};

enum class HuggingFaceEndpointKind {
    Official,
    Mirror,
};

struct HuggingFaceEndpoint {
    HuggingFaceEndpointKind kind = HuggingFaceEndpointKind::Official;
    std::string base_url;
};

struct AcquisitionOptions {
    RegistryOptions network;
    HuggingFaceEndpoint huggingface_official{
        HuggingFaceEndpointKind::Official, "https://huggingface.co"};
    HuggingFaceEndpoint huggingface_mirror{
        HuggingFaceEndpointKind::Mirror, "https://hf-mirror.com"};
    bool automatic_huggingface_fallback = true;
    long official_availability_connect_timeout_seconds = 10;
    long official_availability_low_speed_timeout_seconds = 15;
    std::string huggingface_token;
    std::optional<uint64_t> available_space_override;
};

struct AcquisitionResult {
    std::string model_reference;
    SourceReference source;
    std::filesystem::path materialized_directory;
    std::vector<AcquiredSourceArtifact> artifacts;
    uint64_t downloaded_bytes = 0;
    uint64_t reused_bytes = 0;
    uint64_t resumed_bytes = 0;
    uint64_t temporary_disk_peak_bytes = 0;
    double acquisition_seconds = 0.0;
    std::optional<HuggingFaceEndpointKind> transport_endpoint;
};

struct SourceCleanupResult {
    uint64_t reclaimed_bytes = 0;
    size_t files_removed = 0;
};

class LocalSourceCache {
public:
    explicit LocalSourceCache(std::filesystem::path root,
                              std::filesystem::path temporary_root = {});

    const SourceCacheLayout& layout() const noexcept { return layout_; }

    AcquisitionResult acquire(
        const SourceReference& requested,
        const SourceArtifactPlanDocument& plan,
        const AcquisitionOptions& options = {},
        std::ostream* progress_output = nullptr);

    SourceCleanupResult reclaim_after_install(
        const AcquisitionResult& acquisition);

private:
    SourceCacheLayout layout_;
    std::filesystem::path legacy_temporary_;
};

}  // namespace vrhino::product
