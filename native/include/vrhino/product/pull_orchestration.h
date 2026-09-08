#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>

#include "vrhino/product/converter.h"
#include "vrhino/product/registry.h"
#include "vrhino/product/source_acquisition.h"

namespace vrhino::product {

inline constexpr int64_t kPullPlanSchemaVersion = 1;

enum class PullDistributionKind {
    InstalledPackage,
    RegistryPackage,
    SourceBacked,
    MultiComponentSourceBacked,
    PrivateMultiComponentSourceBacked,
};

struct PullDistributionPlan {
    int64_t schema_version = 0;
    std::string model_reference;
    PullDistributionKind kind = PullDistributionKind::SourceBacked;
    SourceReference source;
    std::filesystem::path source_plan;
    std::string converter;
    std::filesystem::path document_path;
};

std::optional<PullDistributionPlan> find_pull_distribution_plan(
    const std::string& model_reference,
    const std::filesystem::path& converter_spec_root);

struct UnifiedPullOptions {
    std::filesystem::path converter_spec_root;
    RegistryOptions registry;
    std::string huggingface_token;
    std::function<bool()> cancellation_requested;
    std::function<void(uint64_t, uint64_t)> conversion_progress;
    std::function<void(uint64_t, uint64_t)> finalization_progress;
    std::optional<uint64_t> source_available_space_override;
    std::optional<uint64_t> conversion_available_space_override;
};

struct UnifiedPullResult {
    PackageIdentity identity;
    std::filesystem::path manifest_path;
    PullDistributionKind distribution = PullDistributionKind::InstalledPackage;
    std::optional<SourceReference> source;
    uint64_t source_downloaded_bytes = 0;
    uint64_t source_reused_bytes = 0;
    uint64_t source_resumed_bytes = 0;
    uint64_t source_reclaimed_bytes = 0;
    uint64_t registry_downloaded_bytes = 0;
    uint64_t registry_reused_bytes = 0;
    bool already_installed = false;
    bool converter_invoked = false;
    bool conversion_performed = false;
    double resolution_seconds = 0.0;
    double acquisition_seconds = 0.0;
    double conversion_seconds = 0.0;
    double finalization_seconds = 0.0;
    double total_seconds = 0.0;
    std::string runtime_vrm_sha256;
    std::string source_cleanup_warning;
};

UnifiedPullResult pull_runnable_model(
    const std::string& exact_reference,
    LocalModelCache& cache,
    const UnifiedPullOptions& options = {},
    std::ostream* progress_output = nullptr);

}  // namespace vrhino::product
