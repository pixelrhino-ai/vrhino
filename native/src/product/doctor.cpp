#include "vrhino/product/doctor.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string_view>
#include <vector>

#include <gnu/libc-version.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/vfs.h>
#include <unistd.h>

#include "vrhino/product/pull_orchestration.h"

namespace vrhino::product {
namespace {

namespace fs = std::filesystem;

struct SystemFacts {
    std::string operating_system = "UNKNOWN";
    std::string kernel = "UNKNOWN";
    std::string architecture = "UNKNOWN";
    std::string glibc = "UNKNOWN";
    std::string nvidia_driver = "UNKNOWN";
};

struct CacheFacts {
    std::string display_root;
    std::string filesystem = "UNKNOWN";
    std::optional<uint64_t> available_bytes;
    bool writable = false;
    bool usable = false;
};

enum class ArtifactState {
    Ok,
    Missing,
    Invalid,
};

void escalate(DoctorOverall& current, const DoctorOverall next) {
    if (next == DoctorOverall::Failed ||
        (next == DoctorOverall::Warning && current == DoctorOverall::Ready))
        current = next;
}

std::string format_bytes(const uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    long double value = static_cast<long double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0L && unit + 1 < std::size(units)) {
        value /= 1024.0L;
        ++unit;
    }
    std::ostringstream output;
    output << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
           << value << ' ' << units[unit];
    return output.str();
}

bool environment_set(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0';
}

std::string environment_state(const char* name) {
    return environment_set(name) ? "set" : "not set";
}

bool path_is_within(const fs::path& child, const fs::path& parent) {
    auto child_iterator = child.begin();
    for (auto parent_iterator = parent.begin(); parent_iterator != parent.end();
         ++parent_iterator, ++child_iterator) {
        if (child_iterator == child.end() || *child_iterator != *parent_iterator)
            return false;
    }
    return true;
}

std::string display_cache_root(const fs::path& input, const bool is_default) {
    const fs::path root = fs::absolute(input).lexically_normal();
    if (!is_default) return "<custom>";
    const char* home_value = std::getenv("HOME");
    if (home_value != nullptr && *home_value != '\0') {
        const fs::path home = fs::absolute(home_value).lexically_normal();
        if (path_is_within(root, home)) {
            const fs::path relative = root.lexically_relative(home);
            return relative.empty() || relative == "."
                ? "~" : (fs::path("~") / relative).string();
        }
    }
    return "<custom>";
}

fs::path nearest_existing_path(fs::path path) {
    std::error_code error;
    while (!path.empty()) {
        if (fs::exists(path, error) && !error) return path;
        error.clear();
        const fs::path parent = path.parent_path();
        if (parent == path) break;
        path = parent;
    }
    return {};
}

bool has_write_permission(const fs::file_status& status) {
    const fs::perms write = fs::perms::owner_write | fs::perms::group_write |
                            fs::perms::others_write;
    return (status.permissions() & write) != fs::perms::none;
}

std::string filesystem_name(const long type) {
    switch (static_cast<unsigned long>(type)) {
        case 0xEF53: return "ext";
        case 0x58465342: return "xfs";
        case 0x794C7630: return "overlay";
        case 0x01021994: return "tmpfs";
        case 0x6969: return "nfs";
        case 0x9123683E: return "btrfs";
        case 0x2FC12FC1: return "zfs";
        default: return "UNKNOWN";
    }
}

CacheFacts inspect_cache(const fs::path& requested, const bool is_default) {
    CacheFacts result;
    const fs::path root = fs::absolute(requested).lexically_normal();
    result.display_root = display_cache_root(root, is_default);
    const fs::path existing = nearest_existing_path(root);
    if (existing.empty()) return result;

    std::error_code error;
    if (fs::exists(root, error) && !error && !fs::is_directory(root, error)) return result;
    error.clear();
    const fs::file_status status = fs::status(existing, error);
    if (error || !fs::is_directory(status)) return result;
    result.writable = has_write_permission(status) &&
                      ::access(existing.c_str(), W_OK | X_OK) == 0;

    const fs::space_info space = fs::space(existing, error);
    if (!error && space.available != std::numeric_limits<uintmax_t>::max())
        result.available_bytes = static_cast<uint64_t>(space.available);

    struct statfs filesystem {};
    if (::statfs(existing.c_str(), &filesystem) == 0)
        result.filesystem = filesystem_name(filesystem.f_type);
    result.usable = result.available_bytes.has_value() && result.writable;
    return result;
}

std::string read_small_file(const fs::path& path, const size_t maximum = 64 * 1024) {
    std::error_code error;
    const uintmax_t size = fs::file_size(path, error);
    if (error || size > maximum) return {};
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::string unquote(std::string value) {
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);
    return value;
}

std::string os_release_name() {
    const std::string document = read_small_file("/etc/os-release");
    std::istringstream lines(document);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.starts_with("PRETTY_NAME="))
            return unquote(line.substr(std::string("PRETTY_NAME=").size()));
    }
    return "UNKNOWN";
}

bool dotted_version(const std::string_view token) {
    size_t dots = 0;
    if (token.empty()) return false;
    for (const unsigned char character : token) {
        if (character == '.') ++dots;
        else if (!std::isdigit(character)) return false;
    }
    return dots >= 2;
}

std::string nvidia_driver_version() {
    std::istringstream tokens(read_small_file("/proc/driver/nvidia/version", 4096));
    std::string token;
    while (tokens >> token)
        if (dotted_version(token)) return token;
    return "UNKNOWN";
}

SystemFacts inspect_system() {
    SystemFacts result;
    result.operating_system = os_release_name();
    struct utsname information {};
    if (::uname(&information) == 0) {
        result.kernel = std::string(information.sysname) + " " + information.release;
        result.architecture = information.machine;
    }
    if (const char* version = ::gnu_get_libc_version(); version != nullptr && *version != '\0')
        result.glibc = version;
    result.nvidia_driver = nvidia_driver_version();
    return result;
}

std::string encoder_display_path(const fs::path& path) {
    if (path.empty()) return "UNKNOWN";
    const std::string filename = path.filename().string();
    return filename.empty() ? "<installation>/encoder"
                            : "<installation>/" + filename;
}

const char* artifact_state_name(const ArtifactState state) {
    switch (state) {
        case ArtifactState::Ok: return "OK";
        case ArtifactState::Missing: return "MISSING";
        case ArtifactState::Invalid: return "INVALID";
    }
    return "INVALID";
}

int artifact_state_rank(const ArtifactState state) {
    switch (state) {
        case ArtifactState::Ok: return 0;
        case ArtifactState::Missing: return 1;
        case ArtifactState::Invalid: return 2;
    }
    return 2;
}

ArtifactState combine_state(const ArtifactState left, const ArtifactState right) {
    return artifact_state_rank(right) > artifact_state_rank(left) ? right : left;
}

bool equivalent_artifact_contract(const ModelPackageManifest& left,
                                  const ModelPackageManifest& right) {
    if (left.identity.reference() != right.identity.reference() ||
        left.runtime_artifact_id != right.runtime_artifact_id ||
        left.product.family != right.product.family ||
        left.product.status != right.product.status ||
        left.product.workflow_identity != right.product.workflow_identity ||
        left.product.workflow_artifact_id != right.product.workflow_artifact_id ||
        left.product.execution_artifact_id != right.product.execution_artifact_id ||
        left.product.required_inputs != right.product.required_inputs ||
        left.product.public_distribution != right.product.public_distribution ||
        left.source_repository != right.source_repository ||
        left.source_revision != right.source_revision ||
        left.license_identifier != right.license_identifier ||
        left.artifacts.size() != right.artifacts.size() ||
        left.components.size() != right.components.size())
        return false;
    for (size_t index = 0; index < left.artifacts.size(); ++index) {
        const ArtifactDeclaration& a = left.artifacts[index];
        const ArtifactDeclaration& b = right.artifacts[index];
        if (a.id != b.id || a.role != b.role || a.size != b.size ||
            a.sha256 != b.sha256 || a.required != b.required)
            return false;
    }
    for (size_t index = 0; index < left.components.size(); ++index) {
        const ComponentDeclaration& a = left.components[index];
        const ComponentDeclaration& b = right.components[index];
        if (a.id != b.id || a.role != b.role || a.kind != b.kind ||
            a.artifact_ids != b.artifact_ids)
            return false;
    }
    return true;
}

std::optional<ModelPackageManifest> descriptor_manifest(
        const std::string& reference, const fs::path& specification_root,
        std::optional<PullDistributionPlan>& pull_plan,
        std::optional<SourceArtifactPlanDocument>& source_plan) {
    if (specification_root.empty()) return std::nullopt;
    try {
        pull_plan = find_pull_distribution_plan(reference, specification_root);
        if (!pull_plan.has_value()) return std::nullopt;
        source_plan = load_source_artifact_plan(
            reference, pull_plan->document_path.parent_path());
        return load_model_package_manifest(
            pull_plan->document_path.parent_path() / kModelManifestName);
    } catch (const ModelPackageError&) {
        return std::nullopt;
    }
}

uint64_t source_logical_bytes(const SourceArtifactPlanDocument& plan) {
    uint64_t total = 0;
    for (const SourceArtifactPlan& artifact : plan.artifacts) {
        if (artifact.size > std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
        total += artifact.size;
    }
    return total;
}

}  // namespace

const char* doctor_overall_name(const DoctorOverall overall) {
    switch (overall) {
        case DoctorOverall::Ready: return "READY";
        case DoctorOverall::Warning: return "WARNING";
        case DoctorOverall::Failed: return "FAILED";
    }
    return "FAILED";
}

DoctorReport run_doctor(const DoctorOptions& options) {
    DoctorReport report;
    std::ostringstream output;
    output << "VRhino Doctor\n\n"
           << "VRhino\n"
           << "Version: " << (options.product_version.empty() ? "UNKNOWN" : options.product_version)
           << '\n'
           << "Git HEAD: " << (options.git_head.empty() ? "UNKNOWN" : options.git_head) << '\n'
           << "CUDA contract: " << options.cuda_contract << "\n\n";

    const SystemFacts system = inspect_system();
    output << "System\n"
           << "OS: " << system.operating_system << '\n'
           << "Kernel: " << system.kernel << '\n'
           << "Architecture: " << system.architecture << '\n'
           << "glibc: " << system.glibc << "\n\n";

    std::optional<HardwareSnapshot> hardware;
    std::string hardware_failure;
    if (options.hardware_override.has_value()) {
        hardware = options.hardware_override;
    } else {
        try {
            hardware = query_hardware();
        } catch (const ModelPackageError& error) {
            hardware_failure = model_package_error_code_name(error.code());
            escalate(report.overall, DoctorOverall::Failed);
        }
    }
    output << "Device\n";
    if (hardware.has_value()) {
        output << "GPU: " << hardware->gpu_name << '\n'
               << "Compute capability: " << hardware->compute_major << '.'
               << hardware->compute_minor << '\n'
               << "NVIDIA driver: " << system.nvidia_driver << '\n'
               << "VRAM: " << format_bytes(hardware->available_vram_bytes)
               << " available / " << format_bytes(hardware->total_vram_bytes) << " total\n"
               << "CUDA driver API: " << hardware->driver_version << '\n'
               << "CUDA runtime API: " << hardware->runtime_version << '\n';
    } else {
        output << "Status: FAILED (" << (hardware_failure.empty() ? "UNKNOWN" : hardware_failure)
               << ")\n"
               << "NVIDIA driver: " << system.nvidia_driver << '\n';
    }
    output << '\n';

    const CacheLayout layout = cache_layout(options.cache_root);
    const CacheFacts cache_facts = inspect_cache(layout.root, options.cache_root_is_default);
    output << "Cache\n"
           << "Root: " << cache_facts.display_root << '\n'
           << "Filesystem: " << cache_facts.filesystem << '\n'
           << "Free: " << (cache_facts.available_bytes.has_value()
                                 ? format_bytes(*cache_facts.available_bytes) + " (" +
                                       std::to_string(*cache_facts.available_bytes) + " bytes)"
                                 : "UNKNOWN") << '\n'
           << "Writable: " << (cache_facts.writable ? "yes" : "no") << "\n\n";
    if (!cache_facts.usable) escalate(report.overall, DoctorOverall::Failed);

    output << "Network configuration\n"
           << "Hugging Face primary transport: enabled\n"
           << "Mirror fallback: enabled\n"
           << "Mirror: hf-mirror.com (third party)\n"
           << "HF_TOKEN: " << environment_state("HF_TOKEN") << '\n'
           << "HTTPS_PROXY: " << environment_state("HTTPS_PROXY") << '\n'
           << "NO_PROXY: " << environment_state("NO_PROXY") << "\n\n";

    bool encoder_ready = false;
    std::string encoder_problem;
    std::error_code encoder_error;
    const fs::file_status encoder_status = fs::status(options.encoder_path, encoder_error);
    const bool encoder_exists = !encoder_error && fs::is_regular_file(encoder_status);
    const bool encoder_executable = encoder_exists &&
                                    ::access(options.encoder_path.c_str(), X_OK) == 0;
    if (options.encoder_resolution_error.has_value()) {
        encoder_problem = *options.encoder_resolution_error;
    } else {
        try {
            preflight_media_encoder(options.encoder_path);
            encoder_ready = true;
        } catch (const ModelPackageError&) {
            encoder_problem = !encoder_exists ? "bundled encoder is missing"
                : !encoder_executable ? "bundled encoder is not executable"
                                      : "bundled encoder readiness check failed";
        }
    }
    output << "Media\n"
           << "Bundled encoder: " << (encoder_ready ? "OK" : "FAILED") << '\n'
           << "Path: " << encoder_display_path(options.encoder_path) << '\n'
           << "Exists: " << (encoder_exists ? "yes" : "no") << '\n'
           << "Executable: " << (encoder_executable ? "yes" : "no") << '\n'
           << "Readiness: " << (encoder_ready ? "PASS" : "FAIL") << '\n';
    if (!encoder_ready) {
        output << "Reason: " << (encoder_problem.empty() ? "bundled encoder is unavailable"
                                                        : encoder_problem) << '\n';
        escalate(report.overall, DoctorOverall::Failed);
    }

    if (options.model_reference.has_value()) {
        const std::string& reference = *options.model_reference;
        LocalModelCache cache(layout.root);
        const PackageIdentity identity = parse_package_reference(reference);
        std::optional<PullDistributionPlan> pull_plan;
        std::optional<SourceArtifactPlanDocument> source_plan;
        std::optional<ModelPackageManifest> descriptor = descriptor_manifest(
            reference, options.converter_spec_root, pull_plan, source_plan);
        const fs::path installed_path = layout.models / identity.name_space / identity.name /
                                        identity.version / kModelManifestName;
        std::error_code installed_error;
        const fs::file_status installed_status = fs::symlink_status(installed_path, installed_error);
        const bool installed_entry = !installed_error && fs::exists(installed_status);
        const bool installed_regular = installed_entry && !fs::is_symlink(installed_status) &&
                                       fs::is_regular_file(installed_status);
        std::optional<ModelPackageManifest> installed_manifest;
        bool manifest_valid = installed_regular;
        if (installed_regular) {
            try {
                installed_manifest = load_model_package_manifest(installed_path);
            } catch (const ModelPackageError&) {
                manifest_valid = false;
            }
        } else if (installed_entry) {
            manifest_valid = false;
        }
        const ModelPackageManifest* manifest = installed_manifest.has_value()
            ? &*installed_manifest : descriptor.has_value() ? &*descriptor : nullptr;
        if (descriptor.has_value() && installed_manifest.has_value() &&
            !equivalent_artifact_contract(*descriptor, *installed_manifest))
            manifest_valid = false;

        std::map<std::string, ArtifactState> artifact_states;
        size_t resolved_count = 0;
        bool required_failure = false;
        if (manifest != nullptr && installed_entry) {
            for (const ArtifactDeclaration& artifact : manifest->artifacts) {
                const fs::path path = cache.artifact_path(artifact.sha256);
                std::error_code error;
                const fs::file_status status = fs::symlink_status(path, error);
                ArtifactState state = ArtifactState::Ok;
                if (error || !fs::exists(status)) state = ArtifactState::Missing;
                else if (fs::is_symlink(status) || !fs::is_regular_file(status))
                    state = ArtifactState::Invalid;
                else if (fs::file_size(path, error) != artifact.size || error)
                    state = ArtifactState::Invalid;
                else if (manifest->product.family == "lip_sync" &&
                         sha256_file(path) != artifact.sha256)
                    state = ArtifactState::Invalid;
                if (state == ArtifactState::Ok) ++resolved_count;
                else if (artifact.required) required_failure = true;
                artifact_states.emplace(artifact.id, state);
            }
        }

        bool resolver_ok = false;
        if (installed_entry && manifest_valid && !required_failure) {
            try {
                (void)cache.resolve(reference, false);
                resolver_ok = true;
            } catch (const ModelPackageError&) {
                resolver_ok = false;
            }
        }
        const bool package_healthy = installed_entry && manifest_valid &&
                                     !required_failure && resolver_ok;

        output << "\nModel\n"
               << "Identity: " << reference << '\n'
               << "Product family: "
               << (manifest != nullptr ? manifest->product.family : "UNKNOWN") << '\n'
               << "Product status: "
               << (manifest != nullptr ? manifest->product.status : "UNKNOWN") << '\n'
               << "Public distribution: "
               << (manifest != nullptr ? manifest->product.public_distribution
                                       : "UNKNOWN") << '\n'
               << "Source transport: "
               << (pull_plan.has_value() ? "Hugging Face" : "UNKNOWN") << '\n'
               << "Repository: " << (pull_plan.has_value() ? pull_plan->source.repository
                                                              : manifest != nullptr
                                                                  ? manifest->source_repository
                                                                  : "UNKNOWN") << '\n'
               << "Fixed revision: " << (pull_plan.has_value() ? pull_plan->source.revision
                                                                  : manifest != nullptr
                                                                      ? manifest->source_revision
                                                                      : "UNKNOWN") << '\n'
               << "License: " << (manifest != nullptr ? manifest->license_identifier
                                                        : "UNKNOWN") << '\n'
               << "Installed: " << (installed_entry ? "yes" : "no") << '\n'
               << "Artifacts: " << resolved_count << '/'
               << (manifest != nullptr ? manifest->artifacts.size() : 0) << '\n';
        if (source_plan.has_value())
            output << "Source plan: " << source_plan->artifacts.size() << " artifacts, "
                   << format_bytes(source_logical_bytes(*source_plan)) << '\n';
        output << "Package health: " << (package_healthy ? "OK" : "FAILED") << '\n';
        if (!manifest_valid && installed_entry) output << "Manifest: INVALID\n";
        if (manifest != nullptr && installed_entry) {
            if (!manifest->runtime_artifact_id.empty()) {
                const auto runtime = artifact_states.find(manifest->runtime_artifact_id);
                output << "Runtime: "
                       << (runtime == artifact_states.end() ? "MISSING"
                                                            : artifact_state_name(runtime->second))
                       << '\n';
            } else {
                output << "Workflow: " << manifest->product.workflow_identity << '\n';
            }
            for (const ComponentDeclaration& component : manifest->components) {
                ArtifactState component_state = ArtifactState::Ok;
                for (const std::string& artifact_id : component.artifact_ids) {
                    const auto state = artifact_states.find(artifact_id);
                    component_state = combine_state(component_state,
                        state == artifact_states.end() ? ArtifactState::Missing : state->second);
                }
                output << "Component " << component.id << ": "
                       << artifact_state_name(component_state) << '\n';
            }
            for (const ArtifactDeclaration& artifact : manifest->artifacts) {
                const auto state = artifact_states.find(artifact.id);
                if (state != artifact_states.end() && state->second != ArtifactState::Ok)
                    output << (state->second == ArtifactState::Missing
                                   ? "Missing artifact: " : "Invalid artifact: ")
                           << artifact.id << '\n';
            }
        }
        if (!package_healthy) {
            output << "Recommended action: reinstall or re-pull the model package\n";
            escalate(report.overall, DoctorOverall::Failed);
        }

        output << "\nAdmission\n";
        if (manifest == nullptr) {
            output << "Preset: UNKNOWN\n"
                   << "Required available VRAM: UNKNOWN\n"
                   << "Current available VRAM: "
                   << (hardware.has_value() ? format_bytes(hardware->available_vram_bytes)
                                            : "UNKNOWN") << '\n'
                   << "Status: FAIL\n";
            escalate(report.overall, DoctorOverall::Failed);
        } else {
            const std::string preset = manifest->default_preset;
            output << "Preset: " << preset << '\n';
            if (!hardware.has_value()) {
                output << "Required available VRAM: "
                       << (manifest->minimum_vram_bytes.has_value()
                               ? format_bytes(*manifest->minimum_vram_bytes) : "not declared")
                       << '\n'
                       << "Current available VRAM: UNKNOWN\n"
                       << "Status: FAIL\n";
                escalate(report.overall, DoctorOverall::Failed);
            } else {
                ResolvedRunnableModel preflight_model;
                preflight_model.manifest = *manifest;
                const PreflightResult admission = preflight_runnable_model(
                    preflight_model, preset, *hardware);
                output << "Required available VRAM: "
                       << (admission.minimum_vram_bytes.has_value()
                               ? format_bytes(*admission.minimum_vram_bytes) : "not declared")
                       << '\n'
                       << "Current available VRAM: "
                       << format_bytes(hardware->available_vram_bytes) << '\n';
                if (!admission.minimum_vram_bytes.has_value()) {
                    output << "Status: NOT_APPLICABLE (no declared threshold)\n";
                    escalate(report.overall, DoctorOverall::Warning);
                } else if (admission.status == PreflightStatus::Supported) {
                    output << "Status: PASS\n";
                } else if (admission.status == PreflightStatus::SupportedWithWarning) {
                    output << "Status: WARNING\n"
                           << "Concern: " << admission.message << '\n';
                    escalate(report.overall, DoctorOverall::Warning);
                } else {
                    output << "Status: FAIL\n";
                    escalate(report.overall, DoctorOverall::Failed);
                }
            }
        }
        if (source_plan.has_value())
            output << "Relevant source acquisition size: "
                   << format_bytes(source_logical_bytes(*source_plan)) << '\n';
        output << "Cache free: "
               << (cache_facts.available_bytes.has_value()
                       ? format_bytes(*cache_facts.available_bytes) + " (" +
                             std::to_string(*cache_facts.available_bytes) + " bytes)"
                       : "UNKNOWN") << '\n';
    }

    output << "\nOverall\n" << doctor_overall_name(report.overall) << '\n'
           << "Meaning: preflight environment checks only; inference success is not guaranteed\n";
    report.text = output.str();
    return report;
}

}  // namespace vrhino::product
