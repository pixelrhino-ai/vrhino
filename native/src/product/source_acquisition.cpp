#include "vrhino/product/source_acquisition.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#ifdef _WIN32
#include "vrhino/product/windows_cache.h"
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
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

uint64_t checked_add_bytes(const uint64_t left, const uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        fail(ModelPackageErrorCode::CacheError,
             "source cleanup byte count overflows uint64");
    return left + right;
}

void cache_directories(const fs::path& path, std::error_code& error) {
#ifdef _WIN32
    windows_cache::ensure_directory(path); error.clear();
#else
    fs::create_directories(path, error);
#endif
}
bool cache_remove(const fs::path& path, std::error_code& error) {
#ifdef _WIN32
    const bool existed = fs::exists(path);
    windows_cache::remove(path); error.clear(); return existed;
#else
    return fs::remove(path, error);
#endif
}
void cache_remove_tree(const fs::path& path, std::error_code& error) {
#ifdef _WIN32
    windows_cache::remove_tree(path); error.clear();
#else
    fs::remove_all(path, error);
#endif
}
void cache_link(const fs::path& source, const fs::path& target, std::error_code& error) {
#ifdef _WIN32
    windows_cache::required_link(source, target); error.clear();
#else
    fs::create_hard_link(source, target, error);
#endif
}
bool cache_equivalent(const fs::path& source, const fs::path& target, std::error_code& error) {
#ifdef _WIN32
    error.clear(); return windows_cache::equivalent(source, target);
#else
    return fs::equivalent(source, target, error);
#endif
}
uint64_t cache_link_count(const fs::path& path, std::error_code& error) {
#ifdef _WIN32
    error.clear(); return windows_cache::link_count(path);
#else
    return fs::hard_link_count(path, error);
#endif
}

bool safe_segment(const std::string& value) {
    if (value.empty() || value == "." || value == "..") return false;
    return std::all_of(value.begin(), value.end(), [](const unsigned char character) {
        return std::isalnum(character) || character == '-' || character == '_' ||
               character == '.';
    });
}

bool fixed_commit(const std::string& value) {
    return value.size() == 40 &&
        std::all_of(value.begin(), value.end(), [](const unsigned char character) {
            return std::isxdigit(character) != 0;
        });
}

void validate_repository(const std::string& repository) {
    const size_t slash = repository.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 == repository.size() ||
        repository.find('/', slash + 1) != std::string::npos ||
        !safe_segment(repository.substr(0, slash)) ||
        !safe_segment(repository.substr(slash + 1))) {
        fail(ModelPackageErrorCode::SourceInvalid,
             "Hugging Face repository must be exactly NAMESPACE/NAME");
    }
}

void validate_fixed_https_url(const std::string& url) {
    constexpr std::string_view prefix = "https://";
    if (!url.starts_with(prefix) || url.size() <= prefix.size() ||
        url.find_first_of("\r\n\t ") != std::string::npos ||
        url.find('#') != std::string::npos) {
        fail(ModelPackageErrorCode::SourceInvalid,
             "fixed source URL must be a bounded HTTPS URL");
    }
    const size_t authority_end = url.find('/', prefix.size());
    const std::string authority = url.substr(
        prefix.size(), authority_end == std::string::npos
            ? std::string::npos : authority_end - prefix.size());
    if (authority.empty() || authority.find('@') != std::string::npos ||
        authority.front() == '.' || authority.back() == '.')
        fail(ModelPackageErrorCode::SourceInvalid,
             "fixed source URL authority is invalid");
}

void validate_relative_path(const fs::path& value, const std::string& context) {
    if (value.empty() || value.is_absolute())
        fail(ModelPackageErrorCode::SourceInvalid, context + " must be relative");
    for (const fs::path& component : value) {
        const std::string segment = component.string();
        if (!safe_segment(segment))
            fail(ModelPackageErrorCode::SourceInvalid,
                 context + " contains an unsafe path component");
    }
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
        fail(ModelPackageErrorCode::SourceInvalid,
             "cannot open source artifact plan: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

const Json& object_field(const Json& object, const std::string& name) {
    const Json* value = object.find(name);
    if (value == nullptr || !value->is_object())
        fail(ModelPackageErrorCode::SourceInvalid,
             "source plan field must be an object: " + name);
    return *value;
}

std::string string_field(const Json& object, const std::string& name) {
    const Json* value = object.find(name);
    if (value == nullptr || !value->is_string() || value->string().empty())
        fail(ModelPackageErrorCode::SourceInvalid,
             "source plan field must be a non-empty string: " + name);
    return value->string();
}

uint64_t unsigned_field(const Json& object, const std::string& name) {
    const Json* value = object.find(name);
    if (value == nullptr || !value->is_int() || value->integer() < 0)
        fail(ModelPackageErrorCode::SourceInvalid,
             "source plan field must be a non-negative integer: " + name);
    return static_cast<uint64_t>(value->integer());
}

SourceArtifactPlanDocument parse_plan(const fs::path& path) {
    SourceArtifactPlanDocument plan;
    plan.document_path = path;
    plan.raw_json = read_text(path);
    try {
        const Json root = Json::parse(plan.raw_json);
        if (!root.is_object()) fail(ModelPackageErrorCode::SourceInvalid,
                                    "source plan root must be an object");
        const Json* schema = root.find("schema_version");
        if (schema == nullptr || !schema->is_int())
            fail(ModelPackageErrorCode::SourceInvalid,
                 "source plan schema_version must be an integer");
        plan.schema_version = schema->integer();
        if (plan.schema_version != kSourcePlanSchemaVersion)
            fail(ModelPackageErrorCode::SourceInvalid,
                 "unsupported source artifact plan schema");
        plan.model_reference = string_field(root, "model_reference");
        const Json& requested = object_field(root, "requested_source");
        plan.requested_source.provider = string_field(requested, "provider");
        plan.requested_source.repository = string_field(requested, "repository");
        plan.requested_source.revision = string_field(requested, "revision");
        if (plan.requested_source.provider != "huggingface")
            fail(ModelPackageErrorCode::SourceInvalid,
                 "Phase 26C source plan provider must be huggingface");
        validate_repository(plan.requested_source.repository);
        if (!fixed_commit(plan.requested_source.revision))
            fail(ModelPackageErrorCode::SourceRevisionRequired,
                 "source plan must pin a 40-hex immutable revision");
        const Json* artifacts = root.find("artifacts");
        if (artifacts == nullptr || !artifacts->is_array() || artifacts->array().empty())
            fail(ModelPackageErrorCode::SourceInvalid,
                 "source plan artifacts must be a non-empty array");
        std::set<std::string> ids;
        std::set<fs::path> local_paths;
        for (const Json& value : artifacts->array()) {
            if (!value.is_object())
                fail(ModelPackageErrorCode::SourceInvalid,
                     "source plan artifact must be an object");
            SourceArtifactPlan artifact;
            artifact.id = string_field(value, "id");
            artifact.role = string_field(value, "role");
            if (const Json* provider = value.find("provider"); provider != nullptr) {
                if (!provider->is_string() || provider->string().empty())
                    fail(ModelPackageErrorCode::SourceInvalid,
                         "source artifact provider must be a non-empty string");
                artifact.provider = provider->string();
            }
            if (artifact.provider == "huggingface") {
                artifact.repository = string_field(value, "repository");
                artifact.revision = string_field(value, "revision");
                artifact.upstream_path = string_field(value, "upstream_path");
                validate_repository(artifact.repository);
                if (!fixed_commit(artifact.revision))
                    fail(ModelPackageErrorCode::SourceRevisionRequired,
                         "every Hugging Face source artifact must pin a 40-hex revision");
                validate_relative_path(artifact.upstream_path, "upstream_path");
            } else if (artifact.provider == "fixed_https") {
                artifact.fixed_url = string_field(value, "url");
                validate_fixed_https_url(artifact.fixed_url);
            } else {
                fail(ModelPackageErrorCode::SourceInvalid,
                     "unsupported fixed source provider: " + artifact.provider);
            }
            artifact.local_path = string_field(value, "local_path");
            artifact.size = unsigned_field(value, "size");
            artifact.sha256 = string_field(value, "sha256");
            validate_relative_path(artifact.local_path, "local_path");
            if (artifact.size == 0 || artifact.sha256.size() != 64 ||
                !std::all_of(artifact.sha256.begin(), artifact.sha256.end(),
                             [](const unsigned char c) { return std::isxdigit(c) != 0; }))
                fail(ModelPackageErrorCode::SourceInvalid,
                     "source artifact size/SHA256 declaration is invalid: " + artifact.id);
            if (!ids.insert(artifact.id).second ||
                !local_paths.insert(artifact.local_path).second)
                fail(ModelPackageErrorCode::SourceInvalid,
                     "duplicate source artifact id/local path: " + artifact.id);
            plan.artifacts.push_back(std::move(artifact));
        }
        return plan;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::SourceInvalid,
             "malformed source artifact plan: " + std::string(error.what()));
    }
}

#ifdef _WIN32
using FileLock = windows_cache::Lock;
constexpr auto LOCK_SH = windows_cache::LockMode::Shared;
constexpr auto LOCK_EX = windows_cache::LockMode::Exclusive;
#else
class FileLock {
public:
    explicit FileLock(const fs::path& path, const int operation = LOCK_EX) {
        std::error_code error;
        cache_directories(path.parent_path(), error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create source lock directory: " + error.message());
        descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (descriptor_ < 0 || ::flock(descriptor_, operation) != 0) {
            if (descriptor_ >= 0) ::close(descriptor_);
            fail(ModelPackageErrorCode::CacheError,
                 "cannot acquire source-cache lock: " + path.string());
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

std::string trim_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}

std::string hf_url(const std::string& base, const SourceArtifactPlan& artifact) {
    return trim_slashes(base) + "/" + artifact.repository + "/resolve/" +
           artifact.revision + "/" + artifact.upstream_path.generic_string();
}

fs::path blob_path(const SourceCacheLayout& layout, const std::string& sha256) {
    return layout.blobs / sha256.substr(0, 2) / sha256;
}

bool verified_file(const fs::path& path, const uint64_t size, const std::string& sha256,
                   const std::function<bool()>& cancellation_requested = {},
                   const WorkProgressCallback& work_progress = {}) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status) ||
        fs::file_size(path, error) != size || error) return false;
    const WorkProgressCallback progress = cancellation_requested || work_progress
        ? WorkProgressCallback([&](const uint64_t bytes) {
              if (work_progress) work_progress(bytes);
              if (cancellation_requested && cancellation_requested())
                  fail(ModelPackageErrorCode::Cancelled,
                       "Download interrupted. Partial download preserved for resume.");
          })
        : WorkProgressCallback{};
    return sha256_file(path, progress) == sha256;
}

std::string safe_identity(const PackageIdentity& identity) {
    return identity.name_space + "-" + identity.name + "-" + identity.version;
}

fs::path tree_path(const SourceCacheLayout& layout, const SourceReference& source,
                   const PackageIdentity& model) {
    const size_t slash = source.repository.find('/');
    return layout.trees / source.repository.substr(0, slash) /
           source.repository.substr(slash + 1) / source.revision /
           model.name_space / model.name / model.version;
}

fs::path unique_staging(const fs::path& root) {
#ifdef _WIN32
    return windows_cache::unique_path(root, "source-tree");
#else
    return root / ("source-tree-" + std::to_string(getpid()) + "-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
#endif
}

void remove_tree(const fs::path& path) {
    std::error_code error;
    cache_remove_tree(path, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot remove invalid source tree: " + error.message());
}

bool verified_tree(const fs::path& tree, const SourceArtifactPlanDocument& plan,
                   const std::function<bool()>& cancellation_requested = {},
                   const std::function<void(uint64_t, uint64_t)>& progress = {}) {
    if (!fs::is_directory(tree)) return false;
    uint64_t total = 0;
    for (const SourceArtifactPlan& artifact : plan.artifacts)
        total = checked_add_bytes(total, artifact.size);
    uint64_t completed = 0;
    if (progress) progress(0, total);
    for (const SourceArtifactPlan& artifact : plan.artifacts) {
        const WorkProgressCallback work = progress
            ? WorkProgressCallback([&](const uint64_t bytes) {
                  completed = checked_add_bytes(completed, bytes);
                  progress(std::min(completed, total), total);
              })
            : WorkProgressCallback{};
        if (!verified_file(tree / artifact.local_path, artifact.size, artifact.sha256,
                           cancellation_requested, work))
            return false;
    }
    return true;
}

bool verified_linked_tree(const fs::path& tree, const SourceCacheLayout& layout,
                          const SourceArtifactPlanDocument& plan) {
    if (!fs::is_directory(tree)) return false;
    std::error_code error;
    for (const SourceArtifactPlan& artifact : plan.artifacts) {
        const fs::path target = tree / artifact.local_path;
        const fs::path blob = blob_path(layout, artifact.sha256);
        if (!fs::is_regular_file(target, error) || error ||
            fs::file_size(target, error) != artifact.size || error ||
            !cache_equivalent(target, blob, error) || error)
            return false;
    }
    return fs::is_regular_file(tree / "vrhino-source-plan.json", error) && !error;
}

void materialize_tree(const fs::path& destination,
                      const SourceCacheLayout& layout,
                      const SourceArtifactPlanDocument& plan) {
    const fs::path staging = unique_staging(layout.temporary / "materialize");
    std::error_code error;
    cache_directories(staging, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create source materialization staging: " + error.message());
    try {
        for (const SourceArtifactPlan& artifact : plan.artifacts) {
            const fs::path target = staging / artifact.local_path;
            cache_directories(target.parent_path(), error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot create source tree directory: " + error.message());
            cache_link(blob_path(layout, artifact.sha256), target, error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot hard-link source CAS object: " + error.message());
        }
#ifdef _WIN32
        windows_cache::write_text(staging / "vrhino-source-plan.json", plan.raw_json);
#else
        {
            std::ofstream output(staging / "vrhino-source-plan.json",
                                 std::ios::binary | std::ios::trunc);
            output << plan.raw_json;
            output.close();
            if (!output) fail(ModelPackageErrorCode::CacheError,
                              "cannot write materialized source plan");
        }
#endif
        cache_directories(destination.parent_path(), error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create source tree parent: " + error.message());
#ifdef _WIN32
        windows_cache::publish_directory(staging, destination);
#else
        fs::rename(staging, destination, error);
#endif
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot atomically publish source tree: " + error.message());
    } catch (...) {
        cache_remove_tree(staging, error);
        throw;
    }
}

ModelPackageErrorCode map_download_error(const ModelPackageError& error) {
    if (error.code() == ModelPackageErrorCode::Cancelled)
        return ModelPackageErrorCode::Cancelled;
    if (const auto* transport = dynamic_cast<const NativeDownloadError*>(&error);
        transport != nullptr) {
        if (transport->failure_class() ==
            NativeDownloadFailureClass::ResumeProtocol)
            return ModelPackageErrorCode::SourceDownloadResumeFailed;
        if (transport->failure_class() == NativeDownloadFailureClass::HttpStatus &&
            transport->http_status() == 404)
            return ModelPackageErrorCode::SourceNotFound;
        return ModelPackageErrorCode::SourceDownloadFailed;
    }
    if (error.code() == ModelPackageErrorCode::DownloadResumeFailed)
        return ModelPackageErrorCode::SourceDownloadResumeFailed;
    return ModelPackageErrorCode::SourceDownloadFailed;
}

const char* endpoint_name(const HuggingFaceEndpointKind kind) {
    return kind == HuggingFaceEndpointKind::Official ? "official" : "mirror";
}

const char* failure_name(const NativeDownloadFailureClass failure) {
    switch (failure) {
        case NativeDownloadFailureClass::DnsResolution: return "DNS resolution";
        case NativeDownloadFailureClass::Connection: return "connection";
        case NativeDownloadFailureClass::Timeout: return "timeout";
        case NativeDownloadFailureClass::TlsHandshake: return "TLS handshake";
        case NativeDownloadFailureClass::InterruptedTransfer:
            return "interrupted transfer";
        case NativeDownloadFailureClass::HttpStatus: return "HTTP status";
        case NativeDownloadFailureClass::ResumeProtocol: return "resume protocol";
        case NativeDownloadFailureClass::Other: return "transport";
    }
    return "transport";
}

std::string failure_description(const NativeDownloadError& error) {
    if (error.failure_class() == NativeDownloadFailureClass::HttpStatus)
        return "HTTP " + std::to_string(error.http_status());
    return failure_name(error.failure_class());
}

bool fallback_worthy(const NativeDownloadError& error) {
    switch (error.failure_class()) {
        case NativeDownloadFailureClass::DnsResolution:
        case NativeDownloadFailureClass::Connection:
        case NativeDownloadFailureClass::Timeout:
        case NativeDownloadFailureClass::TlsHandshake:
        case NativeDownloadFailureClass::InterruptedTransfer:
            return true;
        case NativeDownloadFailureClass::HttpStatus:
            return error.http_status() == 429 || error.http_status() == 500 ||
                   error.http_status() == 502 || error.http_status() == 503 ||
                   error.http_status() == 504;
        case NativeDownloadFailureClass::ResumeProtocol:
        case NativeDownloadFailureClass::Other:
            return false;
    }
    return false;
}

RegistryOptions official_selection_options(const AcquisitionOptions& options) {
    RegistryOptions network = options.network;
    network.retry_count = 0;
    network.connect_timeout_seconds = std::min(
        network.connect_timeout_seconds,
        options.official_availability_connect_timeout_seconds);
    network.low_speed_timeout_seconds = std::min(
        network.low_speed_timeout_seconds,
        options.official_availability_low_speed_timeout_seconds);
    return network;
}

NativeDownloadResult download_from_endpoint(
        const HuggingFaceEndpoint& endpoint,
        const SourceArtifactPlan& artifact,
        const fs::path& partial,
        const uint64_t aggregate_complete,
        const uint64_t aggregate_total,
        const AcquisitionOptions& options,
        std::ostream* progress_output) {
    const bool official = endpoint.kind == HuggingFaceEndpointKind::Official;
    const RegistryOptions network = official && options.automatic_huggingface_fallback
        ? official_selection_options(options) : options.network;
    return download_native_artifact(
        hf_url(endpoint.base_url, artifact), partial, artifact.size, artifact.id,
        aggregate_complete, aggregate_total, network, progress_output,
        official ? options.huggingface_token : std::string{});
}

NativeDownloadResult download_fixed_https(
        const SourceArtifactPlan& artifact,
        const fs::path& partial,
        const uint64_t aggregate_complete,
        const uint64_t aggregate_total,
        const AcquisitionOptions& options,
        std::ostream* progress_output) {
    return download_native_artifact(
        artifact.fixed_url, partial, artifact.size, artifact.id,
        aggregate_complete, aggregate_total, options.network, progress_output);
}

}  // namespace

std::string SourceReference::canonical() const {
    return "hf://" + repository + "@" + revision;
}

SourceReference parse_source_reference(const std::string& reference) {
    constexpr const char* prefix = "hf://";
    if (!reference.starts_with(prefix))
        fail(ModelPackageErrorCode::SourceInvalid,
             "remote source must use hf://REPOSITORY@REVISION");
    const std::string body = reference.substr(std::char_traits<char>::length(prefix));
    const size_t separator = body.rfind('@');
    if (separator == std::string::npos || separator + 1 == body.size())
        fail(ModelPackageErrorCode::SourceRevisionRequired,
             "Hugging Face source requires an explicit immutable revision");
    SourceReference result{"huggingface", body.substr(0, separator),
                           body.substr(separator + 1)};
    validate_repository(result.repository);
    if (!fixed_commit(result.revision))
        fail(ModelPackageErrorCode::SourceRevisionRequired,
             "Hugging Face revision must be an explicit 40-hex commit");
    return result;
}

SourceArtifactPlanDocument load_source_artifact_plan(
        const std::string& model_reference,
        const fs::path& converter_spec_root) {
    std::error_code error;
    if (!fs::is_directory(converter_spec_root))
        fail(ModelPackageErrorCode::SourceInvalid,
             "converter specification root is missing");
    std::optional<SourceArtifactPlanDocument> match;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(
             converter_spec_root, fs::directory_options::skip_permission_denied, error)) {
        if (error) fail(ModelPackageErrorCode::SourceInvalid,
                        "cannot scan converter source plans: " + error.message());
        if (!entry.is_regular_file() || entry.path().filename() != "source-plan.json") continue;
        SourceArtifactPlanDocument candidate = parse_plan(entry.path());
        if (candidate.model_reference != model_reference) continue;
        if (match.has_value())
            fail(ModelPackageErrorCode::SourceInvalid,
                 "multiple source plans match model reference: " + model_reference);
        match = std::move(candidate);
    }
    if (!match.has_value())
        fail(ModelPackageErrorCode::SourceNotFound,
             "no installed source plan for model: " + model_reference);
    return std::move(*match);
}

LocalSourceCache::LocalSourceCache(fs::path root, fs::path temporary_root) {
    if (root.empty())
        fail(ModelPackageErrorCode::SourceInvalid,
             "source cache root must be explicit");
    layout_.root = std::move(root);
    layout_.blobs = layout_.root / "blobs" / "sha256";
    layout_.trees = layout_.root / "trees";
    legacy_temporary_ = layout_.root / "tmp";
    layout_.temporary = temporary_root.empty()
        ? legacy_temporary_ : std::move(temporary_root);
#ifdef _WIN32
    windows_cache::ensure_directory(layout_.root);
    windows_cache::ensure_directory(layout_.temporary);
    windows_cache::Parents root_parent(layout_.root), temp_parent(layout_.temporary);
    windows_cache::same_volume(root_parent.leaf(), temp_parent.leaf());
#endif
}

AcquisitionResult LocalSourceCache::acquire(
        const SourceReference& requested,
        const SourceArtifactPlanDocument& plan,
        const AcquisitionOptions& options,
        std::ostream* progress_output) {
    const auto started = std::chrono::steady_clock::now();
    if (requested.provider != plan.requested_source.provider ||
        requested.repository != plan.requested_source.repository ||
        requested.revision != plan.requested_source.revision)
        fail(ModelPackageErrorCode::SourceInvalid,
             "requested source does not match the model's immutable source plan");
    if (options.huggingface_official.kind != HuggingFaceEndpointKind::Official ||
        options.huggingface_mirror.kind != HuggingFaceEndpointKind::Mirror ||
        options.huggingface_official.base_url.empty() ||
        options.huggingface_mirror.base_url.empty() ||
        options.network.retry_count < 0 || options.network.maximum_redirects < 0 ||
        options.network.connect_timeout_seconds <= 0 ||
        options.network.low_speed_timeout_seconds <= 0 ||
        options.official_availability_connect_timeout_seconds <= 0 ||
        options.official_availability_low_speed_timeout_seconds <= 0)
        fail(ModelPackageErrorCode::SourceInvalid,
             "invalid Hugging Face endpoint selection options");
    const PackageIdentity model = parse_package_reference(plan.model_reference);
    const fs::path destination = tree_path(layout_, requested, model);
    const fs::path lock_root = layout_.temporary / "locks";
    FileLock cache_lock(lock_root / "source-cache.lock", LOCK_SH);
    FileLock plan_lock(lock_root / ("plan-" + safe_identity(model) + ".lock"));

    AcquisitionResult result;
    result.model_reference = plan.model_reference;
    result.source = requested;
    result.materialized_directory = destination;
    HuggingFaceEndpoint selected_endpoint = options.huggingface_official;
    if (verified_tree(destination, plan, options.network.cancellation_requested,
                      options.network.progress)) {
        for (const SourceArtifactPlan& artifact : plan.artifacts) {
            result.artifacts.push_back({artifact, blob_path(layout_, artifact.sha256), false});
            result.reused_bytes += artifact.size;
        }
        result.acquisition_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }
    if (fs::exists(destination)) remove_tree(destination);

    uint64_t aggregate_total = 0;
    for (const SourceArtifactPlan& artifact : plan.artifacts)
        aggregate_total += artifact.size;
    uint64_t aggregate_complete = 0;
    for (const SourceArtifactPlan& artifact : plan.artifacts) {
        if (options.network.cancellation_requested &&
            options.network.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Download interrupted. Partial download preserved for resume.");
        const fs::path blob = blob_path(layout_, artifact.sha256);
        FileLock artifact_lock(lock_root / ("artifact-" + artifact.sha256 + ".lock"));
        uint64_t verified_blob_bytes = 0;
        const WorkProgressCallback blob_progress = options.network.progress
            ? WorkProgressCallback([&](const uint64_t bytes) {
                  verified_blob_bytes = checked_add_bytes(verified_blob_bytes, bytes);
                  options.network.progress(
                      std::min(aggregate_total,
                               aggregate_complete + verified_blob_bytes),
                      aggregate_total);
              })
            : WorkProgressCallback{};
        if (fs::exists(blob) && !verified_file(
                blob, artifact.size, artifact.sha256,
                options.network.cancellation_requested, blob_progress)) {
            std::error_code error;
            cache_remove(blob, error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot remove corrupt source CAS object: " + error.message());
        }
        bool downloaded = false;
        if (!fs::exists(blob)) {
            const fs::path partial = layout_.temporary / "downloads" /
                                     (artifact.sha256 + ".partial");
            const fs::path legacy_partial = legacy_temporary_ / "downloads" /
                                            (artifact.sha256 + ".partial");
            std::error_code error;
            if (partial != legacy_partial && !fs::exists(partial) &&
                fs::exists(legacy_partial)) {
                cache_directories(partial.parent_path(), error);
                if (error) fail(ModelPackageErrorCode::CacheError,
                                "cannot create download directory: " + error.message());
#ifdef _WIN32
                windows_cache::migrate_partial(legacy_partial, partial);
#else
                fs::rename(legacy_partial, partial, error);
#endif
                if (error) fail(ModelPackageErrorCode::CacheError,
                                "cannot migrate legacy source partial: " + error.message());
            }
            cache_directories(layout_.root, error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot create source cache root: " + error.message());
            const uint64_t partial_size = fs::exists(partial)
                ? fs::file_size(partial, error) : 0;
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot inspect source partial: " + error.message());
            const uint64_t remaining = partial_size <= artifact.size
                ? artifact.size - partial_size : artifact.size;
            const uint64_t available = options.available_space_override.value_or(
                fs::space(layout_.root, error).available);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot inspect source cache free space: " + error.message());
            if (available < remaining)
                fail(ModelPackageErrorCode::SourceDiskFull,
                     "cache root " + layout_.root.parent_path().string() + " has " +
                     std::to_string(available) + " bytes available; source artifact " +
                     artifact.id + " requires " + std::to_string(remaining) + " bytes");
            result.temporary_disk_peak_bytes = std::max(
                result.temporary_disk_peak_bytes, artifact.size);
#ifdef _WIN32
            std::shared_ptr<windows_cache::StagedFile> staging;
#endif
            bool valid = false;
            for (int integrity_attempt = 0; integrity_attempt < 2 && !valid;
                 ++integrity_attempt) {
                try {
                    NativeDownloadResult transfer;
                    if (artifact.provider == "fixed_https") {
                        transfer = download_fixed_https(
                            artifact, partial, aggregate_complete,
                            aggregate_total, options, progress_output);
                    } else try {
                        transfer = download_from_endpoint(
                            selected_endpoint, artifact, partial,
                            aggregate_complete, aggregate_total, options,
                            progress_output);
                    } catch (const NativeDownloadError& official_error) {
                        result.downloaded_bytes += official_error.network_bytes();
                        if (!options.automatic_huggingface_fallback ||
                            selected_endpoint.kind !=
                                HuggingFaceEndpointKind::Official ||
                            !fallback_worthy(official_error))
                            throw;
                        selected_endpoint = options.huggingface_mirror;
                        try {
                            transfer = download_from_endpoint(
                                selected_endpoint, artifact, partial,
                                aggregate_complete, aggregate_total, options,
                                progress_output);
                        } catch (const NativeDownloadError& mirror_error) {
                            result.downloaded_bytes += mirror_error.network_bytes();
                            fail(map_download_error(mirror_error),
                                 std::string("Hugging Face source unavailable through ") +
                                     endpoint_name(
                                         HuggingFaceEndpointKind::Official) +
                                     " (" + failure_description(official_error) +
                                     ") and " + endpoint_name(
                                         HuggingFaceEndpointKind::Mirror) +
                                     " (" + failure_description(mirror_error) + ")");
                        }
                    }
#ifdef _WIN32
                    staging = transfer.staging;
#endif
                    result.downloaded_bytes += transfer.network_bytes;
                    result.resumed_bytes += transfer.resumed_bytes;
                    if (artifact.provider == "huggingface")
                        result.transport_endpoint = selected_endpoint.kind;
                } catch (const ModelPackageError& download_error) {
                    if (download_error.code() == ModelPackageErrorCode::Cancelled) throw;
                    if (download_error.code() ==
                            ModelPackageErrorCode::SourceDownloadFailed ||
                        download_error.code() ==
                            ModelPackageErrorCode::SourceDownloadResumeFailed ||
                        download_error.code() == ModelPackageErrorCode::SourceNotFound)
                        throw;
                    fail(map_download_error(download_error), download_error.what());
                }
#ifdef _WIN32
                try {
                    staging->flush(); staging->validate(artifact.size, artifact.sha256);
                    valid = true;
                } catch (const ModelPackageError& validation_error) {
                    if (validation_error.code() != ModelPackageErrorCode::ChecksumMismatch) throw;
                    valid = false;
                }
#else
                valid = verified_file(partial, artifact.size, artifact.sha256,
                                      options.network.cancellation_requested);
#endif
                if (!valid) {
#ifdef _WIN32
                    staging->discard(); staging.reset();
#else
                    cache_remove(partial, error);
#endif
                    if (error) fail(ModelPackageErrorCode::CacheError,
                                    "cannot remove corrupt source partial: " + error.message());
                }
            }
            if (!valid)
                fail(ModelPackageErrorCode::SourceIntegrityFailed,
                     "downloaded source artifact failed SHA256: " + artifact.id);
            cache_directories(blob.parent_path(), error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot create source CAS directory: " + error.message());
#ifdef _WIN32
            staging->publish(blob);
#else
            fs::rename(partial, blob, error);
#endif
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot atomically publish source CAS object: " + error.message());
            downloaded = true;
        } else {
            result.reused_bytes += artifact.size;
        }
        result.artifacts.push_back({artifact, blob, downloaded});
        aggregate_complete += artifact.size;
        if (options.network.progress)
            options.network.progress(aggregate_complete, aggregate_total);
    }
    if (options.network.cancellation_requested &&
        options.network.cancellation_requested())
        fail(ModelPackageErrorCode::Cancelled,
             "Download interrupted. Partial download preserved for resume.");
    materialize_tree(destination, layout_, plan);
    if (!verified_linked_tree(destination, layout_, plan)) {
        remove_tree(destination);
        fail(ModelPackageErrorCode::SourceIntegrityFailed,
             "materialized source tree failed integrity validation");
    }
    result.acquisition_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

SourceCleanupResult LocalSourceCache::reclaim_after_install(
        const AcquisitionResult& acquisition) {
    const PackageIdentity model = parse_package_reference(acquisition.model_reference);
    const fs::path expected_tree = tree_path(layout_, acquisition.source, model);
    if (fs::weakly_canonical(acquisition.materialized_directory) !=
        fs::weakly_canonical(expected_tree))
        fail(ModelPackageErrorCode::CacheError,
             "refusing to clean an unexpected source tree");
    for (const AcquiredSourceArtifact& artifact : acquisition.artifacts)
        if (artifact.blob_path != blob_path(layout_, artifact.declaration.sha256))
            fail(ModelPackageErrorCode::CacheError,
                 "refusing to clean an unexpected source blob path");

    const fs::path lock_root = layout_.temporary / "locks";
    FileLock cache_lock(lock_root / "source-cache.lock", LOCK_EX);
    SourceCleanupResult result;
    std::error_code error;
    if (fs::exists(expected_tree)) {
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(
                 expected_tree, fs::directory_options::skip_permission_denied, error)) {
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot inspect source tree for cleanup: " + error.message());
            if (!entry.is_regular_file(error)) {
                if (error) fail(ModelPackageErrorCode::CacheError,
                                "cannot inspect source tree entry: " + error.message());
                continue;
            }
            ++result.files_removed;
            const uint64_t links = cache_link_count(entry.path(), error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot inspect source tree link count: " + error.message());
            if (links == 1) result.reclaimed_bytes = checked_add_bytes(
                result.reclaimed_bytes, fs::file_size(entry.path()));
        }
        cache_remove_tree(expected_tree, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot remove completed source tree: " + error.message());
    }

    for (const AcquiredSourceArtifact& artifact : acquisition.artifacts) {
        FileLock artifact_lock(lock_root /
            ("artifact-" + artifact.declaration.sha256 + ".lock"));
        const std::array<fs::path, 2> partials = {
            layout_.temporary / "downloads" /
                (artifact.declaration.sha256 + ".partial"),
            legacy_temporary_ / "downloads" /
                (artifact.declaration.sha256 + ".partial"),
        };
        for (const fs::path& partial : partials) {
            if (!fs::is_regular_file(partial, error) || error) {
                error.clear();
                continue;
            }
            const uint64_t links = cache_link_count(partial, error);
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot inspect stale partial link count: " + error.message());
            if (links == 1) result.reclaimed_bytes = checked_add_bytes(
                result.reclaimed_bytes, fs::file_size(partial));
            if (!cache_remove(partial, error) || error)
                fail(ModelPackageErrorCode::CacheError,
                     "cannot remove stale completed partial: " + error.message());
            ++result.files_removed;
        }

        const fs::path blob = blob_path(layout_, artifact.declaration.sha256);
        if (!fs::is_regular_file(blob, error) || error) {
            error.clear();
            continue;
        }
        const uint64_t links = cache_link_count(blob, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect source blob link count: " + error.message());
        if (links != 1) continue;
        result.reclaimed_bytes = checked_add_bytes(
            result.reclaimed_bytes, fs::file_size(blob));
        if (!cache_remove(blob, error) || error)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot remove unshared source blob: " + error.message());
        ++result.files_removed;
#ifdef _WIN32
        if (fs::is_empty(blob.parent_path(), error) && !error) windows_cache::remove(blob.parent_path());
#else
        cache_remove(blob.parent_path(), error);
#endif
        error.clear();
    }
    return result;
}

}  // namespace vrhino::product
