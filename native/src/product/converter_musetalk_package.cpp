#include "vrhino/product/converter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>

#ifdef _WIN32
#include "windows_converter_io.h"
#else
#include <unistd.h>
#endif

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

constexpr const char* kV1Reference = "private/musetalk-v1.5:1.0.0";
constexpr const char* kV2Reference = "private/musetalk-v1.5:2.0.0";
constexpr const char* kPublicReference = "vrhino/musetalk-v1.5:1.0.0";
constexpr uint64_t kV1MappingCount = 67 + 248 + 686 + 65 + 395 + 148;
constexpr uint64_t kV2MappingCount = 67 + 248 + 686 + 74 + 395 + 181;
constexpr uint64_t kV1SourceCheckpointBytes =
    3400074924ULL + 151095027ULL + 334707217ULL + 89843225ULL +
    406878486ULL + 53289463ULL;
constexpr uint64_t kV2SourceCheckpointBytes =
    3400074924ULL + 151095027ULL + 334707217ULL + 229746ULL +
    406878486ULL + 16371837ULL;

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
             "private package descriptor is missing artifact: " + id);
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
             "private package byte count overflow");
    return left + right;
}

}  // namespace

ImportResult import_private_musetalk_v15_impl(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options, const bool successor,
        const bool public_package = false) {
    const auto conversion_started = std::chrono::steady_clock::now();
    if (!fs::is_directory(source_directory))
        fail(ModelPackageErrorCode::SourceInvalid,
             "multi-component source tree is missing");
    const fs::path spec_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root() : fs::canonical(options.converter_spec_root);
    const fs::path package_spec = spec_root /
        (public_package ? "public_musetalk_v15" :
         successor ? "private_musetalk_v15_v2" : "private_musetalk_v15");
    const fs::path manifest_path = options.package_manifest.empty()
        ? package_spec / kModelManifestName
        : fs::canonical(options.package_manifest);
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const std::string expected_reference = options.expected_model_reference.empty()
        ? public_package ? kPublicReference : successor ? kV2Reference : kV1Reference
        : options.expected_model_reference;
    const std::string expected_status = public_package
        ? "public_supported" : "technical_private";
    if (manifest.identity.reference() != expected_reference ||
        manifest.product.family != "lip_sync" ||
        manifest.product.status != expected_status)
        fail(ModelPackageErrorCode::PackageInvalid,
             "MuseTalk package descriptor identity/status drift");

    uint64_t total_output = 0;
    for (const ComponentDeclaration& component : manifest.components) {
        if (component.artifact_ids.size() != 1)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "neural component must reference exactly one VRM artifact");
        total_output = checked_add(
            total_output, artifact(manifest, component.artifact_ids.front()).size);
    }
    const uint64_t available = options.available_space_override.value_or(
        fs::space(cache.layout().root).available);
    uint64_t missing_output = 0;
    for (const ComponentDeclaration& component : manifest.components) {
        const ArtifactDeclaration& declaration =
            artifact(manifest, component.artifact_ids.front());
        if (!cache.contains_blob(declaration, true))
            missing_output = checked_add(missing_output, declaration.size);
    }
    require_conversion_disk_space(missing_output, available,
                                  64ULL * 1024 * 1024, cache.layout().root);

    const fs::path staging = unique_staging(
        cache.layout().temporary, "musetalk-conversion");
    std::error_code error;
    fs::create_directories(staging, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot create conversion staging: " + error.message());

    const uint64_t source_checkpoint_bytes = successor
        ? kV2SourceCheckpointBytes : kV1SourceCheckpointBytes;
    const uint64_t mapping_count = successor ? kV2MappingCount : kV1MappingCount;
    const uint64_t total_work = checked_add(total_output, source_checkpoint_bytes);
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
        if (!admitted.created)
            reused = checked_add(reused, declaration.size);
        else
            copied = checked_add(copied, declaration.size);
        conversion_performed = true;
    };

    try {
        if (options.progress) options.progress(0, total_work);
        convert("audio-encoder", [&](const fs::path& output, const WorkProgressCallback& work) {
            return convert_whisper_tiny_encoder_component(
                source_directory / "whisper", output,
                options.cancellation_requested, work);
        });
        convert("autoencoder", [&](const fs::path& output, const WorkProgressCallback& work) {
            return convert_sd_vae_ft_mse_component(
                source_directory / "vae",
                spec_root / "sd_vae_ft_mse/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        convert("latent-editor", [&](const fs::path& output, const WorkProgressCallback& work) {
            return convert_musetalk_v15_unet_component(
                source_directory / "unet",
                spec_root / "musetalk_v15_unet/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        if (successor) {
            convert("face-detector", [&](const fs::path& output,
                                          const WorkProgressCallback& work) {
                return convert_blazeface_short_range_component(
                    source_directory / "blazeface",
                    spec_root / "blazeface_short_range/tensor-map.tsv", output,
                    options.cancellation_requested, work);
            });
        } else {
            convert("face-detector", [&](const fs::path& output,
                                          const WorkProgressCallback& work) {
                return convert_s3fd_face_detector_component(
                    source_directory / "s3fd",
                    spec_root / "s3fd_face_detector/tensor-map.tsv", output,
                    options.cancellation_requested, work);
            });
        }
        convert("pose-estimator", [&](const fs::path& output, const WorkProgressCallback& work) {
            return convert_dwpose_keypoint_component(
                source_directory / "dwpose",
                spec_root / "dwpose_keypoint/tensor-map.tsv", output,
                options.cancellation_requested, work);
        });
        if (successor) {
            convert("semantic-segmenter", [&](const fs::path& output,
                                               const WorkProgressCallback& work) {
                return convert_selfie_multiclass_component(
                    source_directory / "selfie",
                    spec_root / "selfie_multiclass_256x256/tensor-map.tsv", output,
                    options.cancellation_requested, work);
            });
        } else {
            convert("semantic-segmenter", [&](const fs::path& output,
                                               const WorkProgressCallback& work) {
                return convert_bisenet_face_parser_component(
                    source_directory / "bisenet",
                    spec_root / "bisenet_face_parser/tensor-map.tsv", output,
                    options.cancellation_requested, work);
            });
        }
        if (options.progress) options.progress(total_work, total_work);
        const auto conversion_finished = std::chrono::steady_clock::now();

        const auto finalization_started = std::chrono::steady_clock::now();
        std::vector<std::pair<std::string, fs::path>> static_artifacts = {
            {"whisper-preprocessor", source_directory / "whisper/preprocessor_config.json"},
            {"workflow-config", spec_root /
                (successor ? "musetalk_v15_workflow_v2/workflow.json"
                           : "musetalk_v15_workflow/workflow.json")},
            {"execution-profile", package_spec / "execution.json"},
        };
        if (public_package) {
            static_artifacts.insert(static_artifacts.end(), {
                {"third-party-notices", package_spec / "THIRD_PARTY_NOTICES.txt"},
                {"license-openrail", package_spec / "licenses/CreativeML-OpenRAIL-M.txt"},
                {"license-apache", package_spec / "licenses/Apache-2.0.txt"},
                {"license-mit", package_spec / "licenses/MIT.txt"},
            });
        } else {
            static_artifacts.push_back(
                {"private-notice", package_spec / "PRIVATE-NOTICE.txt"});
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
        result.source_checkpoint_bytes = source_checkpoint_bytes;
        result.output_vrm_bytes = total_output;
        result.temporary_disk_peak_bytes = 3400098560ULL;
        result.largest_temporary_buffer_bytes = largest_buffer;
        result.mapping_count = mapping_count;
        result.copied_artifact_bytes = copied;
        result.reused_artifact_bytes = reused;
        result.conversion_performed = conversion_performed;
        result.conversion_seconds = std::chrono::duration<double>(
            conversion_finished - conversion_started).count();
        result.finalization_seconds = std::chrono::duration<double>(
            finalization_finished - finalization_started).count();
        result.runtime_vrm_sha256 =
            artifact(manifest, "workflow-config").sha256;
        return result;
    } catch (...) {
        fs::remove_all(staging, error);
        throw;
    }
}

ImportResult import_private_musetalk_v15(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options) {
    return import_private_musetalk_v15_impl(
        source_directory, cache, options, false);
}

ImportResult import_private_musetalk_v15_successor(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options) {
    return import_private_musetalk_v15_impl(
        source_directory, cache, options, true);
}

ImportResult import_public_musetalk_v15(
        const fs::path& source_directory, LocalModelCache& cache,
        const ImportOptions& options) {
    return import_private_musetalk_v15_impl(
        source_directory, cache, options, true, true);
}

}  // namespace vrhino::product
