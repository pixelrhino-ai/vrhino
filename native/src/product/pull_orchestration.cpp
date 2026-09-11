#include "vrhino/product/pull_orchestration.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#ifdef _WIN32
#include "vrhino/product/windows_cache.h"
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
        fail(ModelPackageErrorCode::PackageInvalid,
             "cannot open pull distribution plan: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

const Json& object_field(const Json& object, const std::string& name) {
    const Json* value = object.find(name);
    if (value == nullptr || !value->is_object())
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan field must be an object: " + name);
    return *value;
}

std::string string_field(const Json& object, const std::string& name) {
    const Json* value = object.find(name);
    if (value == nullptr || !value->is_string() || value->string().empty())
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan field must be a non-empty string: " + name);
    return value->string();
}

void validate_relative_path(const fs::path& path) {
    if (path.empty() || path.is_absolute())
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan source_plan must be a relative path");
    for (const fs::path& component : path) {
        const std::string value = component.string();
        if (value.empty() || value == "." || value == "..")
            fail(ModelPackageErrorCode::PackageInvalid,
                 "pull plan source_plan contains an unsafe component");
    }
}

PullDistributionPlan parse_pull_plan(const fs::path& path) {
    try {
        const Json root = Json::parse(read_text(path));
        if (!root.is_object())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "pull distribution plan root must be an object");
        const Json* schema = root.find("schema_version");
        if (schema == nullptr || !schema->is_int() ||
            schema->integer() != kPullPlanSchemaVersion)
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "unsupported pull distribution plan schema");
        PullDistributionPlan result;
        result.schema_version = schema->integer();
        result.model_reference = string_field(root, "model_reference");
        (void)parse_package_reference(result.model_reference);
        const Json& distribution = object_field(root, "distribution");
        const std::string kind = string_field(distribution, "kind");
        if (kind != "source_backed" &&
            kind != "multi_component_source_backed" &&
            kind != "private_multi_component_source_backed")
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "unsupported declarative pull distribution kind: " + kind);
        if (kind == "source_backed")
            result.kind = PullDistributionKind::SourceBacked;
        else if (kind == "multi_component_source_backed")
            result.kind = PullDistributionKind::MultiComponentSourceBacked;
        else
            result.kind = PullDistributionKind::PrivateMultiComponentSourceBacked;
        const Json& source = object_field(distribution, "source");
        const std::string provider = string_field(source, "provider");
        if (provider != "huggingface")
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "unsupported declarative source provider: " + provider);
        result.source = parse_source_reference(
            "hf://" + string_field(source, "repository") + "@" +
            string_field(source, "revision"));
        result.source_plan = string_field(distribution, "source_plan");
        validate_relative_path(result.source_plan);
        result.converter = string_field(distribution, "converter");
        if (result.converter != "native")
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "unsupported converter capability: " + result.converter);
        result.document_path = path;
        return result;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "malformed pull distribution plan: " + std::string(error.what()));
    }
}

#ifdef _WIN32
using FileLock = windows_cache::Lock;
#else
class FileLock {
public:
    explicit FileLock(const fs::path& path) {
        std::error_code error;
        fs::create_directories(path.parent_path(), error);
        if (error)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot create pull lock directory: " + error.message());
        descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (descriptor_ < 0 || ::flock(descriptor_, LOCK_EX) != 0) {
            if (descriptor_ >= 0) ::close(descriptor_);
            fail(ModelPackageErrorCode::CacheError,
                 "cannot acquire pull lock: " + path.string());
        }
    }
    ~FileLock() {
        if (descriptor_ >= 0) {
            ::flock(descriptor_, LOCK_UN);
            ::close(descriptor_);
        }
    }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
private:
    int descriptor_ = -1;
};
#endif

std::string lock_identity(const PackageIdentity& identity) {
    return identity.name_space + "-" + identity.name + "-" + identity.version;
}

std::string runtime_sha(const ModelPackageManifest& manifest) {
    const std::string primary = manifest.runtime_artifact_id.empty()
        ? manifest.product.workflow_artifact_id : manifest.runtime_artifact_id;
    for (const ArtifactDeclaration& artifact : manifest.artifacts)
        if (artifact.id == primary) return artifact.sha256;
    fail(ModelPackageErrorCode::PackageInvalid,
         "installed package has no declared executable artifact");
}

uint64_t checked_disk_add(const uint64_t left, const uint64_t right) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull disk-space requirement overflows uint64");
    return left + right;
}

bool regular_file_with_size(const fs::path& path, const uint64_t expected_size) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) return false;
    return fs::file_size(path, error) == expected_size && !error;
}

uint64_t partial_remaining(const fs::path& path, const uint64_t total) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) return total;
    const uint64_t size = fs::file_size(path, error);
    return !error && size <= total ? total - size : total;
}

void require_source_backed_pull_disk_space(
        const PackageIdentity& requested,
        const PullDistributionPlan& plan,
        const SourceArtifactPlanDocument& source_plan,
        const LocalSourceCache& source_cache,
        LocalModelCache& cache,
        const UnifiedPullOptions& options) {
    const fs::path manifest_path = plan.document_path.parent_path() /
                                   "vrhino-model.json";
    if (!fs::is_regular_file(manifest_path)) return;

    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    if (manifest.identity.reference() != requested.reference() ||
        manifest.source_repository != plan.source.repository ||
        manifest.source_revision != plan.source.revision)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan and package manifest identity/source mismatch");

    std::set<std::string> source_hashes;
    uint64_t required = 0;
    uint64_t largest_conversion_staging = 0;
    for (const SourceArtifactPlan& artifact : source_plan.artifacts) {
        source_hashes.insert(artifact.sha256);
        const fs::path blob = source_cache.layout().blobs /
                              artifact.sha256.substr(0, 2) / artifact.sha256;
        if (regular_file_with_size(blob, artifact.size)) continue;
        const fs::path partial = cache.layout().temporary / "downloads" /
                                 (artifact.sha256 + ".partial");
        const fs::path legacy_partial = source_cache.layout().root / "tmp" /
                                        "downloads" /
                                        (artifact.sha256 + ".partial");
        uint64_t remaining = partial_remaining(partial, artifact.size);
        if (remaining == artifact.size && partial != legacy_partial)
            remaining = partial_remaining(legacy_partial, artifact.size);
        required = checked_disk_add(required, remaining);
    }
    for (const ArtifactDeclaration& artifact : manifest.artifacts) {
        if (source_hashes.contains(artifact.sha256) ||
            cache.contains_blob(artifact, false))
            continue;
        required = checked_disk_add(required, artifact.size);
        if (artifact.role == "component.vrm")
            largest_conversion_staging = std::max(
                largest_conversion_staging, artifact.size);
    }
    // Fixed converters publish one component at a time. In addition to final
    // CAS/package bytes, reserve the largest remaining component for atomic
    // conversion staging; this remains correct as verified work accumulates.
    required = checked_disk_add(required, largest_conversion_staging);

    std::error_code error;
    fs::create_directories(cache.layout().root, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot create local cache root: " + error.message());
    const uint64_t available = options.conversion_available_space_override.value_or(
        fs::space(cache.layout().root, error).available);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot query local cache disk space: " + error.message());
    require_conversion_disk_space(required, available,
                                  64ULL * 1024 * 1024, cache.layout().root);
}

std::optional<UnifiedPullResult> installed_result(
        const std::string& reference, LocalModelCache& cache,
        const std::chrono::steady_clock::time_point started,
        bool* invalid = nullptr) {
    try {
        const ResolvedRunnableModel installed = cache.resolve(reference, true);
        UnifiedPullResult result;
        result.identity = installed.manifest.identity;
        result.manifest_path = installed.manifest_path;
        result.already_installed = true;
        result.runtime_vrm_sha256 = runtime_sha(installed.manifest);
        result.total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    } catch (const ModelPackageError& error) {
        if (error.code() == ModelPackageErrorCode::ModelNotFound)
            return std::nullopt;
        if (error.code() == ModelPackageErrorCode::PackageInvalid ||
            error.code() == ModelPackageErrorCode::ArtifactMissing ||
            error.code() == ModelPackageErrorCode::ChecksumMismatch ||
            error.code() == ModelPackageErrorCode::CacheError) {
            if (invalid != nullptr) *invalid = true;
            return std::nullopt;
        }
        throw;
    }
}

}  // namespace

std::optional<PullDistributionPlan> find_pull_distribution_plan(
        const std::string& model_reference, const fs::path& converter_spec_root) {
    (void)parse_package_reference(model_reference);
    if (!fs::is_directory(converter_spec_root))
        fail(ModelPackageErrorCode::PullPlanNotFound,
             "installed pull specification root is missing");
    std::error_code error;
    std::optional<PullDistributionPlan> match;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(
             converter_spec_root, fs::directory_options::skip_permission_denied, error)) {
        if (error)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot scan installed pull plans: " + error.message());
        if (!entry.is_regular_file() || entry.path().filename() != "pull-plan.json")
            continue;
        PullDistributionPlan candidate = parse_pull_plan(entry.path());
        if (candidate.model_reference != model_reference) continue;
        if (match.has_value())
            fail(ModelPackageErrorCode::PackageInvalid,
                 "multiple pull plans match exact model reference: " + model_reference);
        match = std::move(candidate);
    }
    return match;
}

UnifiedPullResult pull_runnable_model(
        const std::string& exact_reference, LocalModelCache& cache,
        const UnifiedPullOptions& options, std::ostream* progress_output) {
    const auto started = std::chrono::steady_clock::now();
    const PackageIdentity requested = parse_package_reference(exact_reference);
    bool installed_invalid = false;
    if (const auto installed = installed_result(
            exact_reference, cache, started, &installed_invalid))
        return *installed;

    FileLock lock(cache.layout().temporary / "locks" /
                  ("unified-pull-" + lock_identity(requested) + ".lock"));
    if (const auto installed = installed_result(
            exact_reference, cache, started, &installed_invalid))
        return *installed;

    if (progress_output != nullptr)
        *progress_output << "Resolving " << exact_reference << '\n';
    const fs::path specification_root = options.converter_spec_root.empty()
        ? discover_converter_spec_root()
        : fs::canonical(options.converter_spec_root);
    const std::optional<PullDistributionPlan> plan =
        find_pull_distribution_plan(exact_reference, specification_root);
    if (!plan.has_value()) {
        if (installed_invalid)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "installed package is corrupt and no fixed repair plan is available");
        if (options.registry.base_url.empty())
            fail(ModelPackageErrorCode::PullPlanNotFound,
                 "no installed source-backed pull plan and no registry configured for: " +
                     exact_reference);
        RegistryClient registry(cache, options.registry);
        const PullResult pulled = registry.pull(exact_reference, progress_output);
        UnifiedPullResult result;
        result.identity = pulled.identity;
        result.manifest_path = pulled.manifest_path;
        result.distribution = PullDistributionKind::RegistryPackage;
        result.registry_downloaded_bytes = pulled.downloaded_bytes;
        result.registry_reused_bytes = pulled.reused_bytes;
        result.already_installed = pulled.already_installed;
        result.runtime_vrm_sha256 = runtime_sha(
            cache.resolve(exact_reference, false).manifest);
        result.total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }

    if (installed_invalid) {
        if (plan->kind != PullDistributionKind::MultiComponentSourceBacked &&
            plan->kind != PullDistributionKind::PrivateMultiComponentSourceBacked)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "installed package is corrupt; remove it before re-pulling");
        const fs::path descriptor_path =
            plan->document_path.parent_path() / "vrhino-model.json";
        const ModelPackageManifest descriptor =
            load_model_package_manifest(descriptor_path);
        if (descriptor.identity.reference() != exact_reference)
            fail(ModelPackageErrorCode::PackageInvalid,
                 "multi-component repair descriptor identity mismatch");
        cache.discard_installed_package_for_repair(exact_reference);
        for (const ArtifactDeclaration& artifact : descriptor.artifacts)
            cache.discard_invalid_blob(artifact);
    }

    const fs::path declared_source_plan = fs::weakly_canonical(
        plan->document_path.parent_path() / plan->source_plan);
    const SourceArtifactPlanDocument source_plan = load_source_artifact_plan(
        exact_reference, plan->document_path.parent_path());
    if (fs::weakly_canonical(source_plan.document_path) != declared_source_plan)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan did not resolve its declared source plan");
    if (source_plan.requested_source.provider != plan->source.provider ||
        source_plan.requested_source.repository != plan->source.repository ||
        source_plan.requested_source.revision != plan->source.revision)
        fail(ModelPackageErrorCode::PackageInvalid,
             "pull plan source does not match source artifact plan");

    LocalSourceCache source_cache(cache.layout().root / "sources",
                                  cache.layout().temporary);
    require_source_backed_pull_disk_space(requested, *plan, source_plan,
                                          source_cache, cache, options);
    if (progress_output != nullptr) *progress_output << "Acquiring source\n";
    const auto acquisition_started = std::chrono::steady_clock::now();
    AcquisitionOptions acquisition_options;
    acquisition_options.network = options.registry;
    acquisition_options.network.cancellation_requested =
        options.cancellation_requested;
    acquisition_options.huggingface_token = options.huggingface_token;
    acquisition_options.available_space_override =
        options.source_available_space_override;
    const AcquisitionResult acquired = source_cache.acquire(
        plan->source, source_plan, acquisition_options, progress_output);

    if (options.cancellation_requested && options.cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "Pull interrupted before conversion. Downloaded source retained.");
    ImportOptions import_options;
    import_options.converter_spec_root = specification_root;
    import_options.package_manifest =
        plan->document_path.parent_path() / kModelManifestName;
    import_options.expected_model_reference = exact_reference;
    import_options.cancellation_requested = options.cancellation_requested;
    import_options.progress = options.conversion_progress;
    import_options.finalization_progress = options.finalization_progress;
    import_options.available_space_override =
        options.conversion_available_space_override;
    const ImportResult imported = import_local_model(
        exact_reference, acquired.materialized_directory, cache, import_options);
    const auto installed = std::chrono::steady_clock::now();
    (void)cache.resolve(exact_reference, false);

    SourceCleanupResult cleanup;
    std::string cleanup_warning;
    try {
        cleanup = source_cache.reclaim_after_install(acquired);
    } catch (const ModelPackageError& error) {
        cleanup_warning = error.what();
    } catch (const std::exception& error) {
        cleanup_warning = error.what();
    }
    if (progress_output != nullptr) {
        *progress_output << "Installed " << exact_reference << '\n';
        if (!cleanup_warning.empty())
            *progress_output << "Warning: source cache cleanup was not completed: "
                             << cleanup_warning << '\n';
    }

    UnifiedPullResult result;
    result.identity = imported.installation.identity;
    result.manifest_path = imported.installation.manifest_path;
    result.distribution = plan->kind;
    result.source = acquired.source;
    result.source_downloaded_bytes = acquired.downloaded_bytes;
    result.source_reused_bytes = acquired.reused_bytes;
    result.source_resumed_bytes = acquired.resumed_bytes;
    result.source_reclaimed_bytes = cleanup.reclaimed_bytes;
    result.converter_invoked = true;
    result.conversion_performed = imported.conversion_performed;
    result.resolution_seconds = std::chrono::duration<double>(
        acquisition_started - started).count();
    result.acquisition_seconds = acquired.acquisition_seconds;
    result.conversion_seconds = imported.conversion_seconds;
    result.finalization_seconds = imported.finalization_seconds +
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - installed).count();
    result.runtime_vrm_sha256 = imported.runtime_vrm_sha256;
    result.source_cleanup_warning = cleanup_warning;
    result.total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace vrhino::product
