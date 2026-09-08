#include "vrhino/product/model_package.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::product {
namespace fs = std::filesystem;

namespace {

constexpr uint64_t kMaximumManifestBytes = 8U * 1024U * 1024U;
std::atomic<uint64_t> g_staging_sequence = 0;

[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

std::string read_text_file(const fs::path& path) {
    std::error_code error;
    const uint64_t size = fs::file_size(path, error);
    if (error) fail(ModelPackageErrorCode::PackageInvalid,
                    "cannot stat manifest: " + path.string());
    if (size > kMaximumManifestBytes) {
        fail(ModelPackageErrorCode::PackageInvalid, "manifest exceeds 8 MiB");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) fail(ModelPackageErrorCode::PackageInvalid,
                     "cannot open manifest: " + path.string());
    std::ostringstream stream;
    stream << input.rdbuf();
    if (!input.eof() && input.fail()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "cannot read manifest: " + path.string());
    }
    return stream.str();
}

const Json& object_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_object()) {
        fail(ModelPackageErrorCode::PackageInvalid, key + " must be an object");
    }
    return *result;
}

const Json& array_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_array()) {
        fail(ModelPackageErrorCode::PackageInvalid, key + " must be an array");
    }
    return *result;
}

std::string string_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_string() || result->string().empty()) {
        fail(ModelPackageErrorCode::PackageInvalid, key + " must be a non-empty string");
    }
    return result->string();
}

int64_t integer_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_int()) {
        fail(ModelPackageErrorCode::PackageInvalid, key + " must be an integer");
    }
    return result->integer();
}

bool bool_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_bool()) {
        fail(ModelPackageErrorCode::PackageInvalid, key + " must be a bool");
    }
    return result->boolean();
}

bool safe_identifier(const std::string& value) {
    if (value.empty() || value == "." || value == "..") return false;
    return std::all_of(value.begin(), value.end(), [](const unsigned char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= '0' && character <= '9') || character == '.' ||
               character == '_' || character == '-';
    });
}

bool safe_role(const std::string& value) {
    if (value.empty()) return false;
    return std::all_of(value.begin(), value.end(), [](const unsigned char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= '0' && character <= '9') || character == '.' ||
               character == '_' || character == '-';
    });
}

bool valid_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](const unsigned char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

void validate_relative_path(const fs::path& path) {
    if (path.empty() || path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "artifact path must be relative: " + path.string());
    }
    for (const auto& component : path) {
        if (component == "." || component == ".." || component.empty()) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "artifact path is not normalized: " + path.string());
        }
    }
    if (path.lexically_normal() != path) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "artifact path is not normalized: " + path.string());
    }
}

bool path_is_within(const fs::path& child, const fs::path& parent) {
    auto child_iterator = child.begin();
    auto parent_iterator = parent.begin();
    for (; parent_iterator != parent.end(); ++parent_iterator, ++child_iterator) {
        if (child_iterator == child.end() || *child_iterator != *parent_iterator) return false;
    }
    return true;
}

fs::path package_manifest_path(const CacheLayout& layout, const PackageIdentity& identity) {
    return layout.models / identity.name_space / identity.name / identity.version /
           kModelManifestName;
}

fs::path cache_blob_path(const CacheLayout& layout, const std::string& sha256) {
    return layout.blobs / "sha256" / sha256.substr(0, 2) / sha256;
}

fs::path unique_staging_path(const fs::path& parent, const std::string& prefix) {
    const auto clock = std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t sequence = g_staging_sequence.fetch_add(1);
    return parent / (prefix + "-" + std::to_string(::getpid()) + "-" +
                     std::to_string(clock) + "-" + std::to_string(sequence));
}

void sync_file(const fs::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) fail(ModelPackageErrorCode::CacheError,
                             "cannot open file for sync: " + path.string());
    const int result = ::fsync(descriptor);
    ::close(descriptor);
    if (result != 0) fail(ModelPackageErrorCode::CacheError,
                          "cannot sync file: " + path.string());
}

void sync_directory(const fs::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) fail(ModelPackageErrorCode::CacheError,
                             "cannot open directory for sync: " + path.string());
    const int result = ::fsync(descriptor);
    ::close(descriptor);
    if (result != 0) fail(ModelPackageErrorCode::CacheError,
                          "cannot sync directory: " + path.string());
}

void create_cache_directories(const fs::path& path) {
    std::error_code error;
    fs::create_directories(path, error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create cache directory: " + path.string() + ": " + error.message());
}

void write_manifest(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) fail(ModelPackageErrorCode::CacheError,
                      "cannot create installed manifest: " + path.string());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.flush();
    if (!output) fail(ModelPackageErrorCode::CacheError,
                      "cannot write installed manifest: " + path.string());
    output.close();
    sync_file(path);
}

void ensure_reference_exists(const std::set<std::string>& artifact_ids,
                             const std::string& artifact_id,
                             const std::string& context) {
    if (!artifact_ids.contains(artifact_id)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             context + " references unknown artifact: " + artifact_id);
    }
}

void ensure_required_reference(const std::map<std::string, bool>& artifact_required,
                               const std::string& artifact_id,
                               const std::string& context) {
    const auto found = artifact_required.find(artifact_id);
    if (found == artifact_required.end() || !found->second) {
        fail(ModelPackageErrorCode::PackageInvalid,
             context + " must reference a required artifact: " + artifact_id);
    }
}

void validate_compatibility(const ModelPackageManifest& manifest) {
    if (manifest.schema_version != kModelPackageSchemaVersion) {
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "unsupported model package schema: " + std::to_string(manifest.schema_version));
    }
    if (manifest.runtime_contract != kCudaRuntimeContract) {
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "unsupported Runtime contract: " + manifest.runtime_contract);
    }
    if (manifest.vrm_format_major != kVrmFormatMajor ||
        manifest.vrm_format_minor != kVrmFormatMinor ||
        manifest.vrm_metadata_schema != kVrmMetadataSchema) {
        fail(ModelPackageErrorCode::PackageVersionUnsupported,
             "unsupported .vrm compatibility declaration");
    }
}

}  // namespace

const char* model_package_error_code_name(const ModelPackageErrorCode code) {
    switch (code) {
        case ModelPackageErrorCode::ModelNotFound: return "MODEL_NOT_FOUND";
        case ModelPackageErrorCode::PackageInvalid: return "PACKAGE_INVALID";
        case ModelPackageErrorCode::PackageVersionUnsupported:
            return "PACKAGE_VERSION_UNSUPPORTED";
        case ModelPackageErrorCode::ArtifactMissing: return "ARTIFACT_MISSING";
        case ModelPackageErrorCode::ChecksumMismatch: return "CHECKSUM_MISMATCH";
        case ModelPackageErrorCode::InstallFailed: return "INSTALL_FAILED";
        case ModelPackageErrorCode::CacheError: return "CACHE_ERROR";
        case ModelPackageErrorCode::RegistryUnavailable: return "REGISTRY_UNAVAILABLE";
        case ModelPackageErrorCode::DownloadFailed: return "DOWNLOAD_FAILED";
        case ModelPackageErrorCode::DownloadResumeFailed: return "DOWNLOAD_RESUME_FAILED";
        case ModelPackageErrorCode::NetworkError: return "NETWORK_ERROR";
        case ModelPackageErrorCode::InsufficientDiskSpace: return "INSUFFICIENT_DISK_SPACE";
        case ModelPackageErrorCode::UnsupportedGpu: return "UNSUPPORTED_GPU";
        case ModelPackageErrorCode::InsufficientVram: return "INSUFFICIENT_VRAM";
        case ModelPackageErrorCode::DriverIncompatible: return "DRIVER_INCOMPATIBLE";
        case ModelPackageErrorCode::InvalidInput: return "INVALID_INPUT";
        case ModelPackageErrorCode::RuntimeError: return "RUNTIME_ERROR";
        case ModelPackageErrorCode::OutOfMemory: return "OUT_OF_MEMORY";
        case ModelPackageErrorCode::OutputExists: return "OUTPUT_EXISTS";
        case ModelPackageErrorCode::OutputInvalid: return "OUTPUT_INVALID";
        case ModelPackageErrorCode::VideoEncodingFailed: return "VIDEO_ENCODING_FAILED";
        case ModelPackageErrorCode::Cancelled: return "CANCELLED";
        case ModelPackageErrorCode::ComponentNotFound: return "COMPONENT_NOT_FOUND";
        case ModelPackageErrorCode::ComponentInvalid: return "COMPONENT_INVALID";
        case ModelPackageErrorCode::ComponentVersionUnsupported:
            return "COMPONENT_VERSION_UNSUPPORTED";
        case ModelPackageErrorCode::SourceInvalid: return "SOURCE_INVALID";
        case ModelPackageErrorCode::SourceRevisionRequired:
            return "SOURCE_REVISION_REQUIRED";
        case ModelPackageErrorCode::SourceNotFound: return "SOURCE_NOT_FOUND";
        case ModelPackageErrorCode::SourceDownloadFailed:
            return "SOURCE_DOWNLOAD_FAILED";
        case ModelPackageErrorCode::SourceDownloadResumeFailed:
            return "SOURCE_DOWNLOAD_RESUME_FAILED";
        case ModelPackageErrorCode::SourceIntegrityFailed:
            return "SOURCE_INTEGRITY_FAILED";
        case ModelPackageErrorCode::SourceDiskFull: return "SOURCE_DISK_FULL";
        case ModelPackageErrorCode::PullPlanNotFound: return "PULL_PLAN_NOT_FOUND";
    }
    return "CACHE_ERROR";
}

ModelPackageError::ModelPackageError(const ModelPackageErrorCode code,
                                     const std::string& message)
    : std::runtime_error(std::string(model_package_error_code_name(code)) + ": " + message),
      code_(code) {}

std::string PackageIdentity::reference() const {
    return name_space + "/" + name + ":" + version;
}

uint64_t ModelPackageManifest::logical_size() const {
    uint64_t result = 0;
    for (const ArtifactDeclaration& artifact : artifacts) {
        if (artifact.size > std::numeric_limits<uint64_t>::max() - result) {
            fail(ModelPackageErrorCode::PackageInvalid, "logical package size overflows uint64");
        }
        result += artifact.size;
    }
    return result;
}

CacheLayout cache_layout(const fs::path& explicit_root) {
    fs::path root = explicit_root;
    if (root.empty()) {
        if (const char* configured = std::getenv("VRHINO_HOME");
            configured != nullptr && *configured != '\0') {
            root = configured;
        } else if (const char* user_home = std::getenv("HOME");
                   user_home != nullptr && *user_home != '\0') {
            root = fs::path(user_home) / ".vrhino";
        } else {
            fail(ModelPackageErrorCode::CacheError,
                 "neither an explicit cache root, VRHINO_HOME, nor HOME is available");
        }
    }
    root = fs::absolute(root).lexically_normal();
    return CacheLayout{root, root / "models", root / "blobs", root / "tmp"};
}

PackageIdentity parse_package_reference(const std::string& reference) {
    const size_t slash = reference.find('/');
    const size_t colon = reference.rfind(':');
    if (slash == std::string::npos || colon == std::string::npos || slash == 0 ||
        colon <= slash + 1 || colon + 1 >= reference.size() ||
        reference.find('/', slash + 1) != std::string::npos) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package reference must be namespace/name:version");
    }
    PackageIdentity result;
    result.name_space = reference.substr(0, slash);
    result.name = reference.substr(slash + 1, colon - slash - 1);
    result.version = reference.substr(colon + 1);
    if (!safe_identifier(result.name_space) || !safe_identifier(result.name) ||
        !safe_identifier(result.version)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package reference contains an unsafe identifier");
    }
    return result;
}

ModelPackageManifest load_model_package_manifest(const fs::path& path) {
    ModelPackageManifest manifest;
    manifest.raw_json = read_text_file(path);
    try {
        const Json root = Json::parse(manifest.raw_json);
        if (!root.is_object()) fail(ModelPackageErrorCode::PackageInvalid,
                                    "manifest root must be an object");
        manifest.schema_version = integer_field(root, "schema_version");

        const Json& identity = object_field(root, "identity");
        manifest.identity.name_space = string_field(identity, "namespace");
        manifest.identity.name = string_field(identity, "name");
        manifest.identity.version = string_field(identity, "version");
        manifest.identity.architecture = string_field(identity, "architecture");
        manifest.identity.publisher = string_field(identity, "publisher");
        if (!safe_identifier(manifest.identity.name_space) ||
            !safe_identifier(manifest.identity.name) ||
            !safe_identifier(manifest.identity.version) ||
            !safe_identifier(manifest.identity.architecture)) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "manifest identity contains an unsafe identifier");
        }

        const Json& compatibility = object_field(root, "compatibility");
        manifest.runtime_contract = string_field(compatibility, "runtime_contract");
        const Json& vrm_schema = object_field(compatibility, "vrm_schema");
        manifest.vrm_format_major = integer_field(vrm_schema, "format_major");
        manifest.vrm_format_minor = integer_field(vrm_schema, "format_minor");
        manifest.vrm_metadata_schema = integer_field(vrm_schema, "metadata_schema");

        std::set<std::string> artifact_ids;
        std::set<fs::path> artifact_paths;
        std::map<std::string, bool> artifact_required;
        for (const Json& value : array_field(root, "artifacts").array()) {
            if (!value.is_object()) fail(ModelPackageErrorCode::PackageInvalid,
                                         "artifact declaration must be an object");
            ArtifactDeclaration artifact;
            artifact.id = string_field(value, "id");
            artifact.role = string_field(value, "role");
            artifact.relative_path = string_field(value, "path");
            const int64_t size = integer_field(value, "size");
            if (size < 0) fail(ModelPackageErrorCode::PackageInvalid,
                               "artifact size cannot be negative");
            artifact.size = static_cast<uint64_t>(size);
            artifact.sha256 = string_field(value, "sha256");
            artifact.required = bool_field(value, "required");
            if (!safe_identifier(artifact.id) || !safe_role(artifact.role)) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "artifact id or role is invalid: " + artifact.id);
            }
            validate_relative_path(artifact.relative_path);
            if (!valid_sha256(artifact.sha256)) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "artifact SHA256 must be 64 lowercase hex characters: " + artifact.id);
            }
            if (!artifact_ids.insert(artifact.id).second ||
                !artifact_paths.insert(artifact.relative_path).second) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "artifact id and path must be unique: " + artifact.id);
            }
            artifact_required.emplace(artifact.id, artifact.required);
            manifest.artifacts.push_back(std::move(artifact));
        }
        if (manifest.artifacts.empty()) fail(ModelPackageErrorCode::PackageInvalid,
                                             "package has no artifacts");

        if (const Json* product = root.find("product"); product != nullptr) {
            if (!product->is_object())
                fail(ModelPackageErrorCode::PackageInvalid, "product must be an object");
            const std::set<std::string> product_fields = {
                "execution_artifact", "family", "frozen_profile", "input_schema",
                "public_distribution", "required_inputs", "status",
                "workflow_artifact", "workflow_identity",
            };
            for (const auto& [name, ignored] : product->object()) {
                (void)ignored;
                if (!product_fields.contains(name))
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "product contains unsupported semantic field: " + name);
            }
            manifest.product.family = string_field(*product, "family");
            manifest.product.status = string_field(*product, "status");
            manifest.product.public_distribution =
                string_field(*product, "public_distribution");
            if (!safe_role(manifest.product.family) ||
                !safe_role(manifest.product.status) ||
                !safe_role(manifest.product.public_distribution))
                fail(ModelPackageErrorCode::PackageInvalid,
                     "product declaration contains an unsafe identifier");

            if (manifest.product.family != "text_to_video") {
                manifest.product.workflow_identity =
                    string_field(*product, "workflow_identity");
                manifest.product.workflow_artifact_id =
                    string_field(*product, "workflow_artifact");
                manifest.product.execution_artifact_id =
                    string_field(*product, "execution_artifact");
                if (!safe_role(manifest.product.workflow_identity))
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "product workflow identity is unsafe");
                ensure_reference_exists(artifact_ids,
                                        manifest.product.workflow_artifact_id,
                                        "product workflow");
                ensure_required_reference(artifact_required,
                                          manifest.product.workflow_artifact_id,
                                          "product workflow");
                ensure_reference_exists(artifact_ids,
                                        manifest.product.execution_artifact_id,
                                        "product execution");
                ensure_required_reference(artifact_required,
                                          manifest.product.execution_artifact_id,
                                          "product execution");
            }

            const Json* input_schema = product->find("input_schema");
            const Json* frozen_profile = product->find("frozen_profile");
            if ((input_schema == nullptr) != (frozen_profile == nullptr))
                fail(ModelPackageErrorCode::PackageInvalid,
                     "product input_schema and frozen_profile must appear together");
            if (input_schema != nullptr) {
                if (product->find("required_inputs") != nullptr)
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "schema-backed product must not duplicate required_inputs");
                manifest.product.input_schema =
                    parse_product_input_schema(*input_schema);
                manifest.product.frozen_profile =
                    parse_product_frozen_profile(*frozen_profile);
                validate_product_contract_for_family(
                    manifest.product.family,
                    manifest.product.workflow_identity,
                    *manifest.product.input_schema,
                    *manifest.product.frozen_profile);
                for (const ProductFieldDeclaration& input :
                     manifest.product.input_schema->inputs)
                    if (input.required)
                        manifest.product.required_inputs.push_back(input.name);
            } else {
                std::set<std::string> input_names;
                for (const Json& input :
                     array_field(*product, "required_inputs").array()) {
                    if (!input.is_string() || !safe_role(input.string()) ||
                        !input_names.insert(input.string()).second)
                        fail(ModelPackageErrorCode::PackageInvalid,
                             "product required_inputs must contain unique safe strings");
                    manifest.product.required_inputs.push_back(input.string());
                }
                if (manifest.product.required_inputs.empty())
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "product required_inputs cannot be empty");
            }
        }

        const Json& entrypoint = object_field(root, "entrypoint");
        if (const Json* runtime = entrypoint.find("runtime_artifact"); runtime != nullptr) {
            if (!runtime->is_string() || runtime->string().empty())
                fail(ModelPackageErrorCode::PackageInvalid,
                     "entrypoint runtime_artifact must be a non-empty string");
            manifest.runtime_artifact_id = runtime->string();
            ensure_reference_exists(artifact_ids, manifest.runtime_artifact_id, "entrypoint");
            ensure_required_reference(artifact_required, manifest.runtime_artifact_id, "entrypoint");
        } else if (manifest.product.family == "text_to_video") {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "text_to_video entrypoint requires runtime_artifact");
        }
        if (manifest.product.family != "text_to_video") {
            const std::string workflow_artifact = string_field(entrypoint, "workflow_artifact");
            if (workflow_artifact != manifest.product.workflow_artifact_id)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "entrypoint and product name different workflow artifacts");
        }
        const std::string entrypoint_preset = string_field(entrypoint, "default_preset");
        std::set<std::string> component_ids;
        std::set<std::string> component_roles;
        for (const Json& value : array_field(entrypoint, "components").array()) {
            if (!value.is_object()) fail(ModelPackageErrorCode::PackageInvalid,
                                         "component declaration must be an object");
            ComponentDeclaration component;
            component.id = string_field(value, "id");
            if (const Json* role = value.find("role"); role != nullptr) {
                if (!role->is_string() || role->string().empty())
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "component role must be a non-empty string");
                component.role = role->string();
            }
            component.kind = string_field(value, "kind");
            if (!safe_identifier(component.id) || !safe_role(component.kind) ||
                (!component.role.empty() && !safe_role(component.role))) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "component id or kind is invalid: " + component.id);
            }
            if (!component_ids.insert(component.id).second) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "component id must be unique: " + component.id);
            }
            if (!component.role.empty() &&
                !component_roles.insert(component.role).second)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "component role must be unique: " + component.role);
            for (const Json& artifact : array_field(value, "artifacts").array()) {
                if (!artifact.is_string() || artifact.string().empty()) {
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "component artifact reference must be a string");
                }
                ensure_reference_exists(artifact_ids, artifact.string(),
                                        "component " + component.id);
                ensure_required_reference(artifact_required, artifact.string(),
                                          "component " + component.id);
                component.artifact_ids.push_back(artifact.string());
            }
            manifest.components.push_back(std::move(component));
        }
        if (manifest.product.family == "lip_sync") {
            std::map<std::string, std::set<std::string>> required_components;
            if (manifest.product.workflow_identity == "lip_sync_workflow_v1") {
                required_components = {
                    {"audio_encoder", {"audio_encoder_transformer"}},
                    {"image_autoencoder", {"autoencoder_kl"}},
                    {"neural_edit", {"conditional_unet_2d"}},
                    {"face_detector", {"vision_detector_multiscale",
                                       "vision_detector_dense_anchors"}},
                    {"pose_estimator", {"pose_estimator_2d"}},
                    {"semantic_segmenter", {"semantic_segmenter_2d"}},
                };
            } else if (manifest.product.workflow_identity ==
                       "lip_sync_diffusion_workflow_v1") {
                required_components = {
                    {"audio_encoder", {"audio_encoder_transformer"}},
                    {"image_autoencoder", {"autoencoder_kl"}},
                    {"neural_edit", {"temporal_conditional_unet_2d"}},
                    {"face_detector", {"vision_detector_dense_anchors"}},
                    {"pose_estimator", {"pose_estimator_2d"}},
                };
            } else {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "lip_sync product workflow family is unsupported");
            }
            if (manifest.components.size() != required_components.size())
                fail(ModelPackageErrorCode::PackageInvalid,
                     "lip_sync product typed component count mismatch");
            for (const auto& [required_role, required_kinds] : required_components) {
                const auto found = std::find_if(
                    manifest.components.begin(), manifest.components.end(),
                    [&](const ComponentDeclaration& component) {
                        return component.role == required_role;
                    });
                if (found == manifest.components.end() ||
                    !required_kinds.contains(found->kind) ||
                    found->artifact_ids.size() != 1)
                    fail(ModelPackageErrorCode::PackageInvalid,
                         "lip_sync component role contract mismatch: " + required_role);
            }
            const std::set<std::string> required_inputs(
                manifest.product.required_inputs.begin(),
                manifest.product.required_inputs.end());
            const std::set<std::string> expected_inputs =
                manifest.product.input_schema.has_value()
                    ? std::set<std::string>{"audio", "video"}
                    : std::set<std::string>{"audio", "output", "video"};
            if (required_inputs != expected_inputs)
                fail(ModelPackageErrorCode::PackageInvalid,
                     "lip_sync product required input contract mismatch");
        }

        const Json& defaults = object_field(root, "defaults");
        manifest.default_preset = string_field(defaults, "default_preset");
        if (entrypoint_preset != manifest.default_preset) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "entrypoint and defaults name different default presets");
        }
        const Json& presets = object_field(defaults, "presets");
        const Json* default_value = presets.find(manifest.default_preset);
        if (default_value == nullptr || !default_value->is_object()) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "default_preset does not name an object in defaults.presets");
        }
        const std::string profile_artifact = string_field(*default_value, "profile_artifact");
        ensure_reference_exists(artifact_ids, profile_artifact, "default preset");
        ensure_required_reference(artifact_required, profile_artifact, "default preset");

        const Json& hardware = object_field(root, "hardware");
        const Json& hardware_presets = object_field(hardware, "presets");
        const Json* hardware_default = hardware_presets.find(manifest.default_preset);
        if (hardware_default == nullptr || !hardware_default->is_object()) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "hardware metadata is missing for the default preset");
        }
        const auto parse_optional_bytes = [&](const std::string& key) -> std::optional<uint64_t> {
            const Json* value = hardware_default->find(key);
            if (value == nullptr || value->is_null()) return std::nullopt;
            if (!value->is_int() || value->integer() < 0) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "hardware " + key + " must be null or a non-negative integer");
            }
            return static_cast<uint64_t>(value->integer());
        };
        manifest.minimum_vram_bytes = parse_optional_bytes("minimum_vram_bytes");
        manifest.recommended_vram_bytes = parse_optional_bytes("recommended_vram_bytes");

        const Json& source = object_field(root, "source");
        manifest.source_repository = string_field(source, "repository");
        manifest.source_revision = string_field(source, "revision");
        (void)string_field(source, "converter_version");

        const Json& license = object_field(root, "license");
        manifest.license_identifier = string_field(license, "identifier");
        manifest.license_artifact_id = string_field(license, "artifact");
        (void)string_field(license, "upstream_notice");
        ensure_reference_exists(artifact_ids, manifest.license_artifact_id, "license");
        ensure_required_reference(artifact_required, manifest.license_artifact_id, "license");

        validate_compatibility(manifest);
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::PackageInvalid, error.what());
    }
    return manifest;
}

LocalModelCache::LocalModelCache(fs::path root) : layout_(cache_layout(root)) {}

fs::path LocalModelCache::artifact_path(const std::string& sha256) const {
    if (!valid_sha256(sha256)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "artifact SHA256 must be 64 lowercase hex characters");
    }
    return cache_blob_path(layout_, sha256);
}

bool LocalModelCache::contains_blob(const ArtifactDeclaration& artifact,
                                    const bool verify_hash,
                                    const WorkProgressCallback& progress) const {
    const fs::path path = artifact_path(artifact.sha256);
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) return false;
    const uint64_t size = fs::file_size(path, error);
    if (error || size != artifact.size) return false;
    return !verify_hash || sha256_file(path, progress) == artifact.sha256;
}

BlobAdmissionResult LocalModelCache::admit_downloaded_blob(
    const fs::path& completed_download,
    const ArtifactDeclaration& artifact,
    const WorkProgressCallback& progress) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(completed_download, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) {
        fail(ModelPackageErrorCode::ArtifactMissing,
             "completed download is missing or not a regular file: " + artifact.id);
    }
    const uint64_t size = fs::file_size(completed_download, error);
    if (error || size != artifact.size) {
        fail(ModelPackageErrorCode::DownloadFailed,
             "completed artifact size mismatch: " + artifact.id);
    }
    if (sha256_file(completed_download, progress) != artifact.sha256) {
        fs::remove(completed_download, error);
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "downloaded artifact SHA256 mismatch: " + artifact.id);
    }

    const fs::path destination = artifact_path(artifact.sha256);
    create_cache_directories(destination.parent_path());
    if (contains_blob(artifact, true)) {
        fs::remove(completed_download, error);
        return BlobAdmissionResult{destination, false};
    }
    if (fs::exists(destination)) {
        fail(ModelPackageErrorCode::CacheError,
             "existing CAS blob is invalid: " + artifact.sha256);
    }

    fs::permissions(completed_download,
                    fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                    fs::perm_options::replace, error);
    if (error) {
        fail(ModelPackageErrorCode::CacheError,
             "cannot make downloaded CAS blob immutable: " + error.message());
    }
    sync_file(completed_download);
    fs::rename(completed_download, destination, error);
    if (error) {
        if (contains_blob(artifact, true)) {
            fs::remove(completed_download, error);
            return BlobAdmissionResult{destination, false};
        }
        fail(ModelPackageErrorCode::CacheError,
             "cannot publish downloaded CAS blob: " + error.message());
    }
    sync_directory(destination.parent_path());
    return BlobAdmissionResult{destination, true};
}

BlobAdmissionResult LocalModelCache::admit_local_blob(
        const fs::path& verified_local_file,
        const ArtifactDeclaration& artifact,
        const WorkProgressCallback& progress) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(verified_local_file, error);
    if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) {
        fail(ModelPackageErrorCode::ArtifactMissing,
             "local CAS source is missing or not a regular file: " + artifact.id);
    }
    if (fs::file_size(verified_local_file, error) != artifact.size || error ||
        sha256_file(verified_local_file, progress) != artifact.sha256) {
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "local CAS source size/SHA256 mismatch: " + artifact.id);
    }
    if (contains_blob(artifact, true, progress))
        return BlobAdmissionResult{artifact_path(artifact.sha256), false};

    create_cache_directories(layout_.temporary / "local-admission");
    const fs::path staging = unique_staging_path(
        layout_.temporary / "local-admission", artifact.sha256 + ".partial");
    fs::create_hard_link(verified_local_file, staging, error);
    if (error) {
        error.clear();
        // The progress contract counts the validation and CAS-admission hash
        // passes. A cross-filesystem staging copy is an implementation fallback,
        // not an additional determinate work unit.
        const std::string copied = copy_file_and_sha256(
            verified_local_file, staging);
        if (copied != artifact.sha256) {
            fs::remove(staging, error);
            fail(ModelPackageErrorCode::ChecksumMismatch,
                 "local CAS source changed during fallback copy: " + artifact.id);
        }
    }
    try {
        return admit_downloaded_blob(staging, artifact, progress);
    } catch (...) {
        fs::remove(staging, error);
        throw;
    }
}

InstallResult LocalModelCache::publish_manifest(const fs::path& manifest_path) {
    std::error_code error;
    const fs::file_status source_status = fs::symlink_status(manifest_path, error);
    if (error || fs::is_symlink(source_status) || !fs::is_regular_file(source_status)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package manifest is missing, a symlink, or not a regular file");
    }
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    const fs::path target_manifest = package_manifest_path(layout_, manifest.identity);
    const fs::path target_directory = target_manifest.parent_path();
    if (fs::exists(target_directory)) {
        fail(ModelPackageErrorCode::InstallFailed,
             "immutable package version is already installed: " + manifest.identity.reference());
    }

    create_cache_directories(layout_.models);
    create_cache_directories(layout_.blobs / "sha256");
    create_cache_directories(layout_.temporary);
    InstallResult result;
    result.identity = manifest.identity;
    result.manifest_path = target_manifest;
    for (const ArtifactDeclaration& artifact : manifest.artifacts) {
        if (!contains_blob(artifact, false)) {
            if (artifact.required) {
                fail(ModelPackageErrorCode::ArtifactMissing,
                     "required CAS artifact is missing or invalid: " + artifact.id);
            }
            continue;
        }
        ++result.blobs_reused;
        result.bytes_reused += artifact.size;
    }

    const fs::path staging = unique_staging_path(layout_.temporary, "install");
    create_cache_directories(staging);
    try {
        const fs::path staged_manifest = staging / kModelManifestName;
        write_manifest(staged_manifest, manifest.raw_json);
        fs::permissions(staged_manifest,
                        fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                        fs::perm_options::replace, error);
        if (error) {
            fail(ModelPackageErrorCode::CacheError,
                 "cannot make installed manifest immutable");
        }
        sync_file(staged_manifest);
        sync_directory(staging);
        create_cache_directories(target_directory.parent_path());
        fs::rename(staging, target_directory, error);
        if (error) {
            fail(ModelPackageErrorCode::InstallFailed,
                 "cannot atomically install package: " + error.message());
        }
        sync_directory(target_directory.parent_path());
    } catch (...) {
        fs::remove_all(staging, error);
        throw;
    }
    return result;
}

InstallResult LocalModelCache::install(const fs::path& package_directory) {
    std::error_code error;
    const fs::path package_root = fs::canonical(package_directory, error);
    if (error || !fs::is_directory(package_root)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package source is not a directory: " + package_directory.string());
    }
    const fs::path source_manifest = package_root / kModelManifestName;
    const fs::file_status manifest_status = fs::symlink_status(source_manifest, error);
    if (error || fs::is_symlink(manifest_status) || !fs::is_regular_file(manifest_status)) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package manifest is missing, a symlink, or not a regular file");
    }
    ModelPackageManifest manifest = load_model_package_manifest(source_manifest);
    const fs::path target_manifest = package_manifest_path(layout_, manifest.identity);
    const fs::path target_directory = target_manifest.parent_path();
    if (fs::exists(target_directory)) {
        fail(ModelPackageErrorCode::InstallFailed,
             "immutable package version is already installed: " + manifest.identity.reference());
    }

    create_cache_directories(layout_.models);
    create_cache_directories(layout_.blobs / "sha256");
    create_cache_directories(layout_.temporary);
    const fs::path staging = unique_staging_path(layout_.temporary, "install");
    create_cache_directories(staging);

    InstallResult result;
    result.identity = manifest.identity;
    result.manifest_path = target_manifest;
    try {
        for (const ArtifactDeclaration& artifact : manifest.artifacts) {
            const fs::path source_path = package_root / artifact.relative_path;
            const fs::file_status status = fs::symlink_status(source_path, error);
            if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) {
                if (artifact.required) {
                    fail(ModelPackageErrorCode::ArtifactMissing,
                         "required artifact is missing or not a regular file: " + artifact.id);
                }
                continue;
            }
            const fs::path canonical_source = fs::canonical(source_path, error);
            if (error || !path_is_within(canonical_source, package_root)) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "artifact escapes package directory: " + artifact.id);
            }
            const uint64_t actual_size = fs::file_size(source_path, error);
            if (error || actual_size != artifact.size) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "artifact size mismatch: " + artifact.id);
            }

            const fs::path destination = cache_blob_path(layout_, artifact.sha256);
            create_cache_directories(destination.parent_path());
            if (fs::exists(destination)) {
                if (fs::is_symlink(fs::symlink_status(destination)) ||
                    !fs::is_regular_file(destination) || fs::file_size(destination) != artifact.size) {
                    fail(ModelPackageErrorCode::CacheError,
                         "existing CAS blob is invalid: " + artifact.sha256);
                }
                if (sha256_file(source_path) != artifact.sha256) {
                    fail(ModelPackageErrorCode::ChecksumMismatch,
                         "artifact SHA256 mismatch: " + artifact.id);
                }
                ++result.blobs_reused;
                result.bytes_reused += artifact.size;
                continue;
            }

            const fs::path staged_blob = unique_staging_path(destination.parent_path(), ".incoming");
            try {
                const std::string copied_hash = copy_file_and_sha256(source_path, staged_blob);
                if (copied_hash != artifact.sha256) {
                    fs::remove(staged_blob, error);
                    fail(ModelPackageErrorCode::ChecksumMismatch,
                         "artifact SHA256 mismatch: " + artifact.id);
                }
                sync_file(staged_blob);
                fs::permissions(staged_blob,
                                fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                                fs::perm_options::replace, error);
                if (error) fail(ModelPackageErrorCode::CacheError,
                                "cannot make CAS blob immutable: " + staged_blob.string());
                fs::rename(staged_blob, destination, error);
                if (error) {
                    fs::remove(staged_blob);
                    fail(ModelPackageErrorCode::CacheError,
                         "cannot publish CAS blob: " + destination.string() + ": " + error.message());
                }
                sync_directory(destination.parent_path());
            } catch (...) {
                fs::remove(staged_blob, error);
                throw;
            }
            ++result.blobs_created;
            result.bytes_created += artifact.size;
        }

        const fs::path staged_manifest = staging / kModelManifestName;
        write_manifest(staged_manifest, manifest.raw_json);
        fs::permissions(staged_manifest,
                        fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
                        fs::perm_options::replace, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot make installed manifest immutable");
        sync_file(staged_manifest);
        sync_directory(staging);
        create_cache_directories(target_directory.parent_path());
        fs::rename(staging, target_directory, error);
        if (error) {
            fail(ModelPackageErrorCode::InstallFailed,
                 "cannot atomically install package: " + error.message());
        }
        sync_directory(target_directory.parent_path());
    } catch (...) {
        fs::remove_all(staging, error);
        throw;
    }
    return result;
}

ResolvedRunnableModel LocalModelCache::resolve(const std::string& reference,
                                               const bool verify_hashes) const {
    const PackageIdentity requested = parse_package_reference(reference);
    const fs::path manifest_path = package_manifest_path(layout_, requested);
    const fs::file_status installed_status = fs::symlink_status(manifest_path);
    if (fs::is_symlink(installed_status) || !fs::is_regular_file(installed_status)) {
        fail(ModelPackageErrorCode::ModelNotFound, "package is not installed: " + reference);
    }
    ResolvedRunnableModel result;
    result.manifest = load_model_package_manifest(manifest_path);
    result.manifest_path = manifest_path;
    if (result.manifest.identity.name_space != requested.name_space ||
        result.manifest.identity.name != requested.name ||
        result.manifest.identity.version != requested.version) {
        fail(ModelPackageErrorCode::CacheError,
             "installed manifest identity does not match its cache path");
    }
    for (const ArtifactDeclaration& artifact : result.manifest.artifacts) {
        const fs::path path = cache_blob_path(layout_, artifact.sha256);
        std::error_code error;
        const fs::file_status status = fs::symlink_status(path, error);
        if (error || fs::is_symlink(status) || !fs::is_regular_file(status)) {
            if (artifact.required) {
                fail(ModelPackageErrorCode::ArtifactMissing,
                     "installed artifact is missing: " + artifact.id);
            }
            continue;
        }
        if (fs::file_size(path, error) != artifact.size || error) {
            fail(ModelPackageErrorCode::ArtifactMissing,
                 "installed artifact size mismatch: " + artifact.id);
        }
        if (verify_hashes && sha256_file(path) != artifact.sha256) {
            fail(ModelPackageErrorCode::ChecksumMismatch,
                 "installed artifact SHA256 mismatch: " + artifact.id);
        }
        result.artifacts.emplace(artifact.id, ResolvedArtifact{artifact, path});
    }
    if (!result.manifest.runtime_artifact_id.empty()) {
        const auto runtime = result.artifacts.find(result.manifest.runtime_artifact_id);
        if (runtime == result.artifacts.end()) {
            fail(ModelPackageErrorCode::ArtifactMissing,
                 "runtime artifact is not installed: " + result.manifest.runtime_artifact_id);
        }
        result.runtime_model_path = runtime->second.path;
    }
    return result;
}

void LocalModelCache::discard_installed_package_for_repair(
        const std::string& reference) {
    const PackageIdentity requested = parse_package_reference(reference);
    const fs::path manifest_path = package_manifest_path(layout_, requested);
    const fs::path version_directory = manifest_path.parent_path();
    if (!fs::exists(version_directory)) return;
    for (const fs::directory_entry& entry : fs::directory_iterator(version_directory)) {
        if (entry.path().filename() != kModelManifestName || entry.is_symlink() ||
            !entry.is_regular_file())
            fail(ModelPackageErrorCode::CacheError,
                 "refusing to repair package directory containing unexpected data");
    }
    std::error_code error;
    fs::remove_all(version_directory, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot discard invalid installed package: " + error.message());
}

void LocalModelCache::discard_invalid_blob(const ArtifactDeclaration& artifact) {
    if (contains_blob(artifact, true)) return;
    const fs::path path = artifact_path(artifact.sha256);
    std::error_code error;
    if (!fs::exists(path, error)) return;
    if (error || fs::is_symlink(fs::symlink_status(path, error)) ||
        !fs::is_regular_file(path, error))
        fail(ModelPackageErrorCode::CacheError,
             "refusing to discard non-regular CAS path: " + artifact.sha256);
    fs::remove(path, error);
    if (error)
        fail(ModelPackageErrorCode::CacheError,
             "cannot discard invalid CAS blob: " + error.message());
}

std::vector<InstalledPackage> LocalModelCache::list() const {
    std::vector<InstalledPackage> result;
    if (!fs::exists(layout_.models)) return result;
    std::error_code error;
    for (const fs::directory_entry& namespace_entry : fs::directory_iterator(layout_.models, error)) {
        if (error) fail(ModelPackageErrorCode::CacheError, "cannot scan model cache");
        if (!namespace_entry.is_directory() || namespace_entry.is_symlink()) continue;
        for (const fs::directory_entry& name_entry : fs::directory_iterator(namespace_entry.path())) {
            if (!name_entry.is_directory() || name_entry.is_symlink()) continue;
            for (const fs::directory_entry& version_entry : fs::directory_iterator(name_entry.path())) {
                if (!version_entry.is_directory() || version_entry.is_symlink()) continue;
                const fs::path manifest_path = version_entry.path() / kModelManifestName;
                if (!fs::is_regular_file(manifest_path)) continue;
                const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
                result.push_back(InstalledPackage{manifest.identity, manifest_path});
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const InstalledPackage& left,
                                               const InstalledPackage& right) {
        return left.identity.reference() < right.identity.reference();
    });
    return result;
}

void LocalModelCache::remove(const std::string& reference) {
    const PackageIdentity requested = parse_package_reference(reference);
    const fs::path manifest_path = package_manifest_path(layout_, requested);
    const fs::path version_directory = manifest_path.parent_path();
    const fs::file_status installed_status = fs::symlink_status(manifest_path);
    if (fs::is_symlink(installed_status) || !fs::is_regular_file(installed_status)) {
        fail(ModelPackageErrorCode::ModelNotFound, "package is not installed: " + reference);
    }
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_path);
    if (manifest.identity.name_space != requested.name_space ||
        manifest.identity.name != requested.name || manifest.identity.version != requested.version) {
        fail(ModelPackageErrorCode::CacheError,
             "installed manifest identity does not match its cache path");
    }
    size_t entries = 0;
    for (const auto& entry : fs::directory_iterator(version_directory)) {
        ++entries;
        if (entry.path().filename() != kModelManifestName || entry.is_symlink() ||
            !entry.is_regular_file()) {
            fail(ModelPackageErrorCode::CacheError,
                 "refusing to remove package directory containing unexpected data");
        }
    }
    if (entries != 1) fail(ModelPackageErrorCode::CacheError,
                           "installed package directory is malformed");
    std::error_code error;
    fs::remove(manifest_path, error);
    if (error || !fs::remove(version_directory, error)) {
        fail(ModelPackageErrorCode::CacheError,
             "cannot remove installed package reference: " + reference);
    }
    fs::remove(version_directory.parent_path(), error);
    fs::remove(version_directory.parent_path().parent_path(), error);
    sync_directory(layout_.models);
}

}  // namespace vrhino::product
