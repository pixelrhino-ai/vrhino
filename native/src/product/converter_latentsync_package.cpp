#include "vrhino/product/converter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>

#include <unistd.h>

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

constexpr const char* kPrivateReference = "private/latentsync-1.6:1.0.0";
constexpr const char* kPublicReference = "vrhino/latentsync-1.6:1.0.0";
constexpr uint64_t kMappingCount = 67 + 248 + 1246 + 74 + 395;
constexpr uint64_t kSourceCheckpointBytes =
    75572083ULL + 334707217ULL + 5072222488ULL + 229746ULL + 406878486ULL;

[[noreturn]] void fail(const ModelPackageErrorCode code,
                       const std::string& message) {
    throw ModelPackageError(code, message);
}

const ArtifactDeclaration& artifact(const ModelPackageManifest& manifest,
                                    const std::string& id) {
    const auto found = std::find_if(
        manifest.artifacts.begin(), manifest.artifacts.end(),
        [&](const ArtifactDeclaration& value) { return value.id == id; });
    if (found == manifest.artifacts.end())
        fail(ModelPackageErrorCode::PackageInvalid,
             "LatentSync package descriptor is missing artifact: " + id);
    return *found;
}

fs::path unique_staging(const fs::path& root, const std::string& name) {
    static std::atomic<uint64_t> sequence{0};
    return root / (name + "-" + std::to_string(::getpid()) + "-" +
                   std::to_string(std::chrono::steady_clock::now()
                                      .time_since_epoch().count()) + "-" +
                   std::to_string(sequence.fetch_add(1)));
}

uint64_t checked_add(const uint64_t left, const uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        fail(ModelPackageErrorCode::PackageInvalid,
             "LatentSync package byte count overflow");
    return left + right;
}

}  // namespace

ImportResult import_latentsync_16_package(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options, const bool public_package) {
    const auto conversion_started = std::chrono::steady_clock::now();
    if (!fs::is_directory(source_directory))
        fail(ModelPackageErrorCode::SourceInvalid,
             "LatentSync multi-component source tree is missing");
    const fs::path spec_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root() : fs::canonical(options.converter_spec_root);
    const fs::path package_spec = spec_root /
        (public_package ? "public_latentsync_16" : "private_latentsync_16");
    const fs::path manifest_path = options.package_manifest.empty()
        ? package_spec / kModelManifestName
        : fs::canonical(options.package_manifest);
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const std::string expected_reference = options.expected_model_reference.empty()
        ? public_package ? kPublicReference : kPrivateReference
        : options.expected_model_reference;
    if (manifest.identity.reference() != expected_reference ||
        manifest.product.family != "lip_sync" ||
        manifest.product.status !=
            (public_package ? "public_supported" : "technical_private") ||
        manifest.product.public_distribution !=
            (public_package ? "mode_c_local_conversion" :
                              "pending_distribution_gate") ||
        manifest.product.workflow_identity != "lip_sync_diffusion_workflow_v1")
        fail(ModelPackageErrorCode::PackageInvalid,
             "LatentSync package descriptor identity/status drift");

    uint64_t total_output = 0;
    uint64_t missing_output = 0;
    uint64_t largest_missing = 0;
    for (const ComponentDeclaration& component : manifest.components) {
        if (component.artifact_ids.size() != 1)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "neural component must reference exactly one VRM artifact");
        const ArtifactDeclaration& declaration =
            artifact(manifest, component.artifact_ids.front());
        total_output = checked_add(total_output, declaration.size);
        if (!cache.contains_blob(declaration, true)) {
            missing_output = checked_add(missing_output, declaration.size);
            largest_missing = std::max(largest_missing, declaration.size);
        }
    }
    const uint64_t available = options.available_space_override.value_or(
        fs::space(cache.layout().root).available);
    require_conversion_disk_space(
        checked_add(missing_output, largest_missing), available,
        64ULL * 1024 * 1024, cache.layout().root);

    const fs::path staging = unique_staging(
        cache.layout().temporary, "latentsync-conversion");
    std::error_code error;
    fs::create_directories(staging, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot create conversion staging: " + error.message());

    const uint64_t total_work = checked_add(total_output, kSourceCheckpointBytes);
    uint64_t copied = 0;
    uint64_t reused = 0;
    uint64_t completed = 0;
    uint64_t largest_buffer = 0;
    bool conversion_performed = false;
    auto progress = [&](const uint64_t bytes) {
        completed = checked_add(completed, bytes);
        if (options.progress)
            options.progress(total_work == 0 ? 0 :
                                 std::min(completed, total_work - 1),
                             total_work);
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Package conversion interrupted; verified sources retained.");
    };
    auto convert = [&](const std::string& artifact_id, const auto& operation) {
        const ArtifactDeclaration& declaration = artifact(manifest, artifact_id);
        if (cache.contains_blob(declaration, true)) {
            reused = checked_add(reused, declaration.size);
            return;
        }
        cache.discard_invalid_blob(declaration);
        const fs::path output = staging / (artifact_id + ".vrm");
        const FrozenComponentConversionResult converted = operation(output, progress);
        if (converted.output_bytes != declaration.size ||
            converted.output_sha256 != declaration.sha256)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "converted component identity drift: " + artifact_id);
        largest_buffer = std::max(
            largest_buffer, converted.largest_temporary_buffer_bytes);
        const BlobAdmissionResult admitted =
            cache.admit_downloaded_blob(output, declaration);
        if (admitted.created) copied = checked_add(copied, declaration.size);
        else reused = checked_add(reused, declaration.size);
        conversion_performed = true;
    };

    try {
        if (options.progress) options.progress(0, total_work);
        convert("audio-encoder", [&](const fs::path& output,
                                      const WorkProgressCallback& work) {
            return convert_openai_whisper_tiny_encoder_component(
                source_directory / "whisper",
                spec_root / "latentsync_16_whisper/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        convert("autoencoder", [&](const fs::path& output,
                                    const WorkProgressCallback& work) {
            return convert_sd_vae_ft_mse_component(
                source_directory / "vae", spec_root / "sd_vae_ft_mse/tensor-map.tsv",
                output, options.cancellation_requested, work);
        });
        convert("temporal-editor", [&](const fs::path& output,
                                        const WorkProgressCallback& work) {
            return convert_temporal_conditional_unet_component(
                source_directory / "temporal_unet",
                spec_root / "latentsync_16_temporal_unet/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        convert("face-detector", [&](const fs::path& output,
                                      const WorkProgressCallback& work) {
            return convert_blazeface_short_range_component(
                source_directory / "blazeface",
                spec_root / "blazeface_short_range/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        convert("pose-estimator", [&](const fs::path& output,
                                       const WorkProgressCallback& work) {
            return convert_dwpose_keypoint_component(
                source_directory / "dwpose",
                spec_root / "dwpose_keypoint/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        if (options.progress) options.progress(total_work, total_work);
        const auto conversion_finished = std::chrono::steady_clock::now();

        const auto finalization_started = std::chrono::steady_clock::now();
        std::vector<std::pair<std::string, fs::path>> static_artifacts = {
            {"whisper-preprocessor", source_directory / "whisper/preprocessor_config.json"},
            {"mel-filters", source_directory / "whisper/mel_filters.npz"},
            {"unet-config", source_directory / "temporal_unet/stage2_512.yaml"},
            {"scheduler-config", source_directory / "temporal_unet/scheduler_config.json"},
            {"fixed-mask", source_directory / "workflow/mask.png"},
            {"workflow-config", spec_root / "latentsync_16_workflow/workflow.json"},
            {"execution-profile", package_spec / "execution.json"},
            {"precision-policy", package_spec /
                "policy/bf16-heavy-consumer-fp32-state-v1.json"},
        };
        if (public_package) {
            static_artifacts.insert(static_artifacts.end(), {
                {"third-party-notices", package_spec / "THIRD_PARTY_NOTICES.txt"},
                {"license-openrail-plus", package_spec /
                    "licenses/CreativeML-Open-RAIL++-M.txt"},
                {"license-apache", package_spec / "licenses/Apache-2.0.txt"},
                {"license-mit", package_spec / "licenses/MIT.txt"},
            });
        } else {
            static_artifacts.emplace_back(
                "private-notice", package_spec / "PRIVATE-NOTICE.txt");
        }
        uint64_t final_total = 0;
        for (const auto& item : static_artifacts)
            final_total = checked_add(final_total, artifact(manifest, item.first).size);
        uint64_t final_completed = 0;
        if (options.finalization_progress)
            options.finalization_progress(0, final_total);
        for (const auto& [id, path] : static_artifacts) {
            if (options.cancellation_requested && options.cancellation_requested())
                fail(ModelPackageErrorCode::Cancelled,
                     "Package finalization interrupted before publication.");
            const ArtifactDeclaration& declaration = artifact(manifest, id);
            cache.discard_invalid_blob(declaration);
            const BlobAdmissionResult admitted = cache.admit_local_blob(path, declaration);
            if (admitted.created) copied = checked_add(copied, declaration.size);
            else reused = checked_add(reused, declaration.size);
            final_completed = checked_add(final_completed, declaration.size);
            if (options.finalization_progress)
                options.finalization_progress(final_completed, final_total);
        }
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Package finalization interrupted before publication.");
        const InstallResult installation = cache.publish_manifest(manifest_path);
        const auto finalization_finished = std::chrono::steady_clock::now();

        fs::remove_all(staging, error);
        ImportResult result;
        result.installation = installation;
        result.source_checkpoint_bytes = kSourceCheckpointBytes;
        result.output_vrm_bytes = total_output;
        result.temporary_disk_peak_bytes = largest_missing;
        result.largest_temporary_buffer_bytes = largest_buffer;
        result.mapping_count = kMappingCount;
        result.copied_artifact_bytes = copied;
        result.reused_artifact_bytes = reused;
        result.conversion_performed = conversion_performed;
        result.conversion_seconds = std::chrono::duration<double>(
            conversion_finished - conversion_started).count();
        result.finalization_seconds = std::chrono::duration<double>(
            finalization_finished - finalization_started).count();
        result.runtime_vrm_sha256 = artifact(manifest, "workflow-config").sha256;
        return result;
    } catch (...) {
        fs::remove_all(staging, error);
        throw;
    }
}

ImportResult import_private_latentsync_16(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options) {
    return import_latentsync_16_package(
        source_directory, cache, options, false);
}

ImportResult import_public_latentsync_16(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options) {
    return import_latentsync_16_package(
        source_directory, cache, options, true);
}

}  // namespace vrhino::product
