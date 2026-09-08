#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "vrhino/api/server.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/info.h"
#include "vrhino/product/version.h"
#include "vrhino/product/component_package.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/pull_orchestration.h"
#include "vrhino/product/progress.h"
#include "vrhino/product/registry.h"
#include "vrhino/product/source_acquisition.h"
#if VRHINO_PRODUCT_RUN_ENABLED
#include "vrhino/product/doctor.h"
#include "vrhino/product/run.h"
#include "vrhino/product/run_session.h"
#endif

namespace product = vrhino::product;

namespace {

[[noreturn]] void usage() {
    std::cerr
        << "Usage:\n"
        << "  vrhino [OPTIONS] pull NAMESPACE/NAME:VERSION\n"
        << "  vrhino [OPTIONS] import NAMESPACE/NAME:VERSION SOURCE-DIRECTORY\n"
        << "  vrhino [OPTIONS] list\n"
        << "  vrhino [OPTIONS] info NAMESPACE/NAME:VERSION [--json]\n"
        << "  vrhino [OPTIONS] doctor [NAMESPACE/NAME:VERSION]\n"
        << "  vrhino [OPTIONS] rm NAMESPACE/NAME:VERSION\n"
        << "  vrhino [OPTIONS] component install MEDIA-PACKAGE.tar.gz\n"
        << "  vrhino [OPTIONS] serve [--host HOST] [--port PORT]\n"
        << "  vrhino [OPTIONS] run NAMESPACE/NAME:VERSION --prompt TEXT [RUN OPTIONS]\n"
        << "  vrhino [OPTIONS] run NAMESPACE/NAME:VERSION --video PATH --audio PATH [RUN OPTIONS]\n\n"
        << "  vrhino --version\n"
        << "  vrhino device\n\n"
        << "Options:\n"
        << "  --cache-root PATH   Override VRHINO_HOME/~/.vrhino\n"
        << "  --registry URL      Override VRHINO_REGISTRY\n"
        << "  --component-registry URL  Override VRHINO_COMPONENT_REGISTRY\n"
        << "  --converter-spec-root PATH  Override installed converter specifications\n"
        << "  --ca-file PATH      Additional/private HTTPS CA bundle\n"
        << "  --allow-http        Development only: permit loopback HTTP registry\n";
#if VRHINO_PRODUCT_RUN_ENABLED
    std::cerr
        << "\nRun options:\n"
        << "  --prompt TEXT       Required prompt\n"
        << "  --video PATH        Required video for lip-sync products\n"
        << "  --audio PATH        Required driving audio for lip-sync products\n"
        << "  --output PATH       Output MP4 (default: output.mp4)\n"
        << "  --preset NAME       Installed package preset (default: package default)\n"
        << "  --seed N            Deterministic seed override\n"
        << "  --overwrite         Replace an existing output\n"
        << "  --debug             Show product debug details\n";
#endif
    std::exit(2);
}

std::string format_bytes(const uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream output;
    output << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << units[unit];
    return output.str();
}

std::string optional_bytes(const std::optional<uint64_t>& value) {
    return value.has_value() ? format_bytes(*value) : "unknown";
}

struct CliOptions {
    std::filesystem::path cache_root;
    std::filesystem::path converter_spec_root;
    product::RegistryOptions registry;
    product::RegistryOptions component_registry;
};

volatile std::sig_atomic_t cancellation_requested = 0;
volatile std::sig_atomic_t serve_stop_requested = 0;

void cancel_handler(int) { cancellation_requested = 1; }
void serve_stop_handler(int) { serve_stop_requested = 1; }

vrhino::api::ServerConfig parse_serve_config(
    const std::vector<std::string>& arguments,
    const std::filesystem::path& cache_root) {
    if (arguments.empty() || arguments.front() != "serve") usage();
    vrhino::api::ServerConfig result;
    result.cache_root = cache_root;
    bool host_seen = false;
    bool port_seen = false;
    for (size_t index = 1; index < arguments.size(); ++index) {
        const std::string& option = arguments[index];
        if ((option != "--host" && option != "--port") ||
            index + 1 >= arguments.size()) {
            usage();
        }
        const std::string value = arguments[++index];
        if (option == "--host") {
            if (host_seen || value.empty()) usage();
            host_seen = true;
            result.host = value;
            continue;
        }
        if (port_seen || value.empty() ||
            !std::all_of(value.begin(), value.end(), [](const unsigned char byte) {
                return std::isdigit(byte);
            })) {
            usage();
        }
        port_seen = true;
        size_t consumed = 0;
        uint64_t port = 0;
        try {
            port = std::stoull(value, &consumed);
        } catch (...) {
            usage();
        }
        if (consumed != value.size() ||
            !vrhino::api::valid_server_port(static_cast<uint32_t>(
                std::min<uint64_t>(port, std::numeric_limits<uint32_t>::max())))) {
            usage();
        }
        result.port = static_cast<uint16_t>(port);
    }
    return result;
}

#if VRHINO_PRODUCT_RUN_ENABLED
product::RunOptions parse_run_options(const std::vector<std::string>& arguments) {
    if (arguments.size() < 2 || arguments[0] != "run") usage();
    product::RunOptions result;
    result.model_reference = arguments[1];
    for (size_t index = 2; index < arguments.size(); ++index) {
        const std::string& option = arguments[index];
        if (option == "--overwrite") { result.overwrite = true; continue; }
        if (option == "--debug") { result.debug = true; continue; }
        if (option != "--prompt" && option != "--video" && option != "--audio" &&
            option != "--output" && option != "--preset" &&
            option != "--seed" && option != "--encoder") usage();
        if (index + 1 >= arguments.size()) usage();
        const std::string value = arguments[++index];
        if (option == "--prompt") result.prompt = value;
        else if (option == "--video") result.video = value;
        else if (option == "--audio") result.audio = value;
        else if (option == "--output") result.output = value;
        else if (option == "--preset") result.preset = value;
        else if (option == "--encoder") result.encoder_path = value;
        else {
            if (value.empty() ||
                !std::all_of(value.begin(), value.end(),
                             [](const unsigned char byte) {
                                 return std::isdigit(byte);
                             }))
                usage();
            size_t consumed = 0;
            try { result.seed = std::stoull(value, &consumed); }
            catch (...) { usage(); }
            if (consumed != value.size()) usage();
        }
    }
    result.cancellation_requested = [] { return cancellation_requested != 0; };
    return result;
}

const char* preflight_status_name(const product::PreflightStatus status) {
    switch (status) {
        case product::PreflightStatus::Supported: return "SUPPORTED";
        case product::PreflightStatus::SupportedWithWarning:
            return "SUPPORTED_WITH_WARNING";
        case product::PreflightStatus::InsufficientVram: return "INSUFFICIENT_VRAM";
        case product::PreflightStatus::UnsupportedGpu: return "UNSUPPORTED_GPU";
        case product::PreflightStatus::DriverIncompatible: return "DRIVER_INCOMPATIBLE";
    }
    return "UNSUPPORTED_GPU";
}
#endif

CliOptions consume_options(std::vector<std::string>& arguments) {
    CliOptions options;
    if (const char* registry = std::getenv("VRHINO_REGISTRY");
        registry != nullptr && *registry != '\0') {
        options.registry.base_url = registry;
    }
    if (const char* registry = std::getenv("VRHINO_COMPONENT_REGISTRY");
        registry != nullptr && *registry != '\0') {
        options.component_registry.base_url = registry;
    }
    for (size_t index = 0; index < arguments.size();) {
        if (arguments[index] == "--cache-root" ||
            arguments[index] == "--converter-spec-root" ||
            arguments[index] == "--registry" ||
            arguments[index] == "--component-registry" || arguments[index] == "--ca-file") {
            if (index + 1 >= arguments.size()) usage();
            const std::string option = arguments[index];
            const std::string value = arguments[index + 1];
            if (option == "--cache-root") options.cache_root = value;
            if (option == "--converter-spec-root") options.converter_spec_root = value;
            if (option == "--registry") options.registry.base_url = value;
            if (option == "--component-registry") options.component_registry.base_url = value;
            if (option == "--ca-file") {
                options.registry.ca_file = value;
                options.component_registry.ca_file = value;
            }
            arguments.erase(arguments.begin() + static_cast<std::ptrdiff_t>(index),
                            arguments.begin() + static_cast<std::ptrdiff_t>(index + 2));
            continue;
        }
        if (arguments[index] == "--allow-http") {
            options.registry.allow_development_http = true;
            options.component_registry.allow_development_http = true;
            arguments.erase(arguments.begin() + static_cast<std::ptrdiff_t>(index));
            continue;
        }
        ++index;
    }
    if (options.component_registry.base_url.empty()) {
        options.component_registry.base_url = options.registry.base_url;
    }
    return options;
}

void print_info(const product::ResolvedRunnableModel& model) {
    const product::ModelPackageManifest& manifest = model.manifest;
    std::cout << "Identity: " << manifest.identity.reference() << '\n'
              << "Architecture: " << manifest.identity.architecture << '\n'
              << "Publisher: " << manifest.identity.publisher << '\n'
              << "Product family: " << manifest.product.family << '\n'
              << "Product status: " << manifest.product.status << '\n'
              << "Public distribution: " << manifest.product.public_distribution << '\n'
              << "Workflow: " << (manifest.product.workflow_identity.empty()
                    ? "not applicable" : manifest.product.workflow_identity) << '\n'
              << "Package schema: " << manifest.schema_version << '\n'
              << "CUDA contract: " << manifest.runtime_contract << '\n'
              << "VRM: " << manifest.vrm_format_major << '.' << manifest.vrm_format_minor
              << " / schema" << manifest.vrm_metadata_schema << '\n'
              << "Default preset: " << manifest.default_preset << '\n'
              << "Minimum VRAM: " << optional_bytes(manifest.minimum_vram_bytes) << '\n'
              << "Recommended VRAM: " << optional_bytes(manifest.recommended_vram_bytes) << '\n'
              << "Source: " << manifest.source_repository << '@' << manifest.source_revision << '\n'
              << "License: " << manifest.license_identifier << '\n'
              << "Logical size: " << format_bytes(manifest.logical_size()) << " ("
              << manifest.logical_size() << " bytes)\n"
              << "Artifacts: " << model.artifacts.size() << '/' << manifest.artifacts.size()
              << " present in local CAS\n";
    if (!manifest.product.required_inputs.empty()) {
        std::cout << "Required inputs:";
        for (const std::string& input : manifest.product.required_inputs)
            std::cout << ' ' << input;
        std::cout << '\n';
    }
    for (const product::ComponentDeclaration& component : manifest.components)
        std::cout << "Component: " << component.role << " ("
                  << component.kind << ")\n";
    std::cout << "Manifest: " << model.manifest_path.string() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        CliOptions options = consume_options(arguments);
        if (arguments.empty()) usage();
        const std::string command = arguments.front();
        if (command == "--version" || command == "version") {
            if (arguments.size() != 1) usage();
            std::cout << product::format_cli_version(
                product::current_version_info());
            return 0;
        }
        if (command == "serve") {
            vrhino::api::ServerConfig config =
                parse_serve_config(arguments, options.cache_root);
#if VRHINO_PRODUCT_RUN_ENABLED
            config.run_executor = [](
                const product::ResolvedRunnableModel& runnable,
                const product::RunOptions& run_options,
                product::RunEventSink events) {
                return product::run_runnable_model(
                    runnable, run_options, std::move(events));
            };
#endif
            if (const std::optional<std::string> warning =
                    vrhino::api::non_loopback_bind_warning(config.host)) {
                std::cerr << *warning << '\n';
            }
            vrhino::api::Server server(config);
            if (server.bind() < 0) {
                std::cerr << "VRhino API could not bind to " << config.host << ':'
                          << config.port << '\n';
                return 1;
            }
            std::cout << "VRhino Native API listening on http://" << config.host
                      << ':' << config.port << '\n' << std::flush;

            serve_stop_requested = 0;
            const auto previous_handler = std::signal(SIGINT, serve_stop_handler);
            if (previous_handler == SIG_ERR) {
                std::cerr << "VRhino API could not install its shutdown handler\n";
                return 1;
            }
            std::atomic<bool> control_done{false};
            std::thread control([&] {
                while (!control_done.load(std::memory_order_acquire)) {
                    if (serve_stop_requested != 0) {
                        server.stop();
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            });
            const bool listened = server.listen();
            control_done.store(true, std::memory_order_release);
            control.join();
            std::signal(SIGINT, previous_handler);
            if (serve_stop_requested != 0) return 0;
            if (!listened) {
                std::cerr << "VRhino API stopped after a server error\n";
                return 1;
            }
            return 0;
        }
#if VRHINO_PRODUCT_RUN_ENABLED
        if (command == "device") {
            if (arguments.size() != 1) usage();
            const product::HardwareSnapshot hardware = product::query_hardware();
            std::cout << "GPU: " << hardware.gpu_name << '\n'
                      << "Compute capability: " << hardware.compute_major << '.'
                      << hardware.compute_minor << '\n'
                      << "VRAM total: " << format_bytes(hardware.total_vram_bytes) << '\n'
                      << "VRAM available: " << format_bytes(hardware.available_vram_bytes) << '\n'
                      << "CUDA driver API: " << hardware.driver_version << '\n'
                      << "CUDA runtime API: " << hardware.runtime_version << '\n';
            return 0;
        }
#endif
        product::LocalModelCache cache(options.cache_root);
        product::LocalComponentCache component_cache(options.cache_root);

#if VRHINO_PRODUCT_RUN_ENABLED
        if (command == "doctor") {
            if (arguments.size() > 2) usage();
            product::DoctorOptions doctor;
            const product::VersionInfo version = product::current_version_info();
            doctor.product_version = version.version;
            doctor.git_head = version.git_head;
            doctor.cache_root = cache.layout().root;
            const char* configured_home = std::getenv("VRHINO_HOME");
            doctor.cache_root_is_default = options.cache_root.empty() &&
                (configured_home == nullptr || *configured_home == '\0');
            if (arguments.size() == 2) {
                doctor.model_reference = arguments[1];
                if (!options.converter_spec_root.empty()) {
                    doctor.converter_spec_root = options.converter_spec_root;
                } else {
                    try {
                        doctor.converter_spec_root = product::discover_converter_spec_root();
                    } catch (const product::ModelPackageError&) {
                        doctor.converter_spec_root.clear();
                    }
                }
            }
            if (const char* configured = std::getenv("VRHINO_FFMPEG");
                configured != nullptr && *configured != '\0') {
                doctor.encoder_path = configured;
            } else {
                try {
                    doctor.encoder_path = component_cache.resolve(
                        product::kMediaComponentReference, true).entrypoint;
                } catch (const product::ModelPackageError& error) {
                    if (error.code() == product::ModelPackageErrorCode::ComponentNotFound) {
                        doctor.encoder_path = product::default_media_encoder_path();
                    } else {
                        doctor.encoder_resolution_error =
                            "installed media component is invalid";
                    }
                }
            }
            const product::DoctorReport report = product::run_doctor(doctor);
            std::cout << report.text;
            return report.exit_code();
        }
#endif

        if (command == "component") {
            if (arguments.size() != 3 || arguments[1] != "install") usage();
            const product::ComponentInstallResult result =
                component_cache.install_archive(arguments[2]);
            std::cout << "Installed component: " << result.identity.reference() << '\n'
                      << "Cache path: " << result.root.string() << '\n'
                      << "Already installed: " << (result.already_installed ? "yes" : "no")
                      << '\n';
            return 0;
        }

        if (command == "pull") {
            if (arguments.size() != 2) usage();
            cancellation_requested = 0;
            std::signal(SIGINT, cancel_handler);
            product::ConsoleProgress progress(std::cout, ::isatty(STDOUT_FILENO) == 1);
            product::UnifiedPullOptions pull_options;
            pull_options.converter_spec_root = options.converter_spec_root;
            pull_options.registry = options.registry;
            pull_options.registry.cancellation_requested = [] {
                return cancellation_requested != 0;
            };
            pull_options.registry.progress = [&progress](const uint64_t completed,
                                                         const uint64_t total) {
                progress.update("Acquiring source", completed, total);
            };
            if (const char* token = std::getenv("HF_TOKEN");
                token != nullptr && *token != '\0')
                pull_options.huggingface_token = token;
            pull_options.cancellation_requested = [] {
                return cancellation_requested != 0;
            };
            pull_options.conversion_progress = [&progress](const uint64_t completed,
                                                           const uint64_t total) {
                progress.update("Converting", completed, total);
            };
            pull_options.finalization_progress = [&progress](const uint64_t completed,
                                                             const uint64_t total) {
                progress.update("Finalizing", completed, total);
            };
            const product::UnifiedPullResult result = product::pull_runnable_model(
                arguments[1], cache, pull_options, &std::cout);
            if (result.source_reclaimed_bytes != 0)
                std::cout << "Reclaimed " << format_bytes(result.source_reclaimed_bytes)
                          << " of source cache\n";
            const char* distribution = "installed-package";
            if (result.distribution == product::PullDistributionKind::SourceBacked)
                distribution = "source-backed";
            if (result.distribution == product::PullDistributionKind::RegistryPackage)
                distribution = "registry-package";
            if (result.distribution ==
                product::PullDistributionKind::MultiComponentSourceBacked)
                distribution = "multi-component-source-backed";
            if (result.distribution ==
                product::PullDistributionKind::PrivateMultiComponentSourceBacked)
                distribution = "private-multi-component-source-backed";
            std::cout << "Model: " << result.identity.reference() << '\n'
                      << "Distribution: " << distribution << '\n';
            if (result.source.has_value())
                std::cout << "Source: " << result.source->canonical() << '\n';
            std::cout << "Source downloaded: "
                      << format_bytes(result.source_downloaded_bytes) << " ("
                      << result.source_downloaded_bytes << " bytes)\n"
                      << "Source resumed: " << format_bytes(result.source_resumed_bytes)
                      << " (" << result.source_resumed_bytes << " bytes)\n"
                      << "Source CAS reused: " << format_bytes(result.source_reused_bytes)
                      << " (" << result.source_reused_bytes << " bytes)\n"
                      << "Registry downloaded: "
                      << format_bytes(result.registry_downloaded_bytes) << " ("
                      << result.registry_downloaded_bytes << " bytes)\n"
                      << "Converter invoked: "
                      << (result.converter_invoked ? "yes" : "no") << '\n'
                      << "Conversion performed: "
                      << (result.conversion_performed ? "yes" : "no") << '\n'
                      << "Resolution time: " << result.resolution_seconds << " s\n"
                      << "Acquisition time: " << result.acquisition_seconds << " s\n"
                      << "Conversion time: " << result.conversion_seconds << " s\n"
                      << "Installation/finalization time: "
                      << result.finalization_seconds << " s\n"
                      << "Total pull time: " << result.total_seconds << " s\n"
                      << "Runtime SHA256: " << result.runtime_vrm_sha256 << '\n'
                      << "Cache path: " << result.manifest_path.parent_path().string() << '\n'
                      << "Already installed: " << (result.already_installed ? "yes" : "no")
                      << '\n';
            return 0;
        }
        if (command == "import") {
            if (arguments.size() != 3) usage();
            cancellation_requested = 0;
            std::signal(SIGINT, cancel_handler);
            product::ConsoleProgress progress(std::cout, ::isatty(STDOUT_FILENO) == 1);
            product::ImportOptions import_options;
            import_options.converter_spec_root = options.converter_spec_root;
            import_options.cancellation_requested = [] {
                return cancellation_requested != 0;
            };
            import_options.progress = [&progress](const uint64_t completed,
                                                  const uint64_t total) {
                progress.update("Converting", completed, total);
            };
            import_options.finalization_progress = [&progress](
                    const uint64_t completed, const uint64_t total) {
                progress.update("Finalizing", completed, total);
            };
            std::filesystem::path source_directory = arguments[2];
            std::optional<product::AcquisitionResult> acquisition;
            if (arguments[2].starts_with("hf://")) {
                const product::SourceReference source =
                    product::parse_source_reference(arguments[2]);
                const std::filesystem::path specification_root =
                    options.converter_spec_root.empty()
                        ? product::discover_converter_spec_root()
                        : std::filesystem::canonical(options.converter_spec_root);
                const product::SourceArtifactPlanDocument plan =
                    product::load_source_artifact_plan(arguments[1], specification_root);
                product::LocalSourceCache source_cache(
                    cache.layout().root / "sources", cache.layout().temporary);
                product::AcquisitionOptions acquisition_options;
                acquisition_options.network = options.registry;
                acquisition_options.network.cancellation_requested = [] {
                    return cancellation_requested != 0;
                };
                acquisition_options.network.progress =
                    [&progress](const uint64_t completed, const uint64_t total) {
                        progress.update("Acquiring source", completed, total);
                    };
                if (const char* token = std::getenv("HF_TOKEN");
                    token != nullptr && *token != '\0')
                    acquisition_options.huggingface_token = token;
                acquisition = source_cache.acquire(source, plan, acquisition_options,
                                                   &std::cout);
                source_directory = acquisition->materialized_directory;
                std::cout << "Acquired source: " << acquisition->source.canonical() << '\n'
                          << "Source downloaded: "
                          << format_bytes(acquisition->downloaded_bytes) << " ("
                          << acquisition->downloaded_bytes << " bytes)\n"
                          << "Source reused: " << format_bytes(acquisition->reused_bytes)
                          << " (" << acquisition->reused_bytes << " bytes)\n"
                          << "Source resumed: " << format_bytes(acquisition->resumed_bytes)
                          << " (" << acquisition->resumed_bytes << " bytes)\n"
                          << "Source acquisition time: "
                          << acquisition->acquisition_seconds << " s\n"
                          << "Source cache path: "
                          << acquisition->materialized_directory.string() << '\n';
            }
            const product::ImportResult result = product::import_local_model(
                arguments[1], source_directory, cache, import_options);
            std::cout << "Imported: " << result.installation.identity.reference() << '\n'
                      << "Source checkpoint: "
                      << format_bytes(result.source_checkpoint_bytes) << " ("
                      << result.source_checkpoint_bytes << " bytes)\n"
                      << "Runtime VRM: " << format_bytes(result.output_vrm_bytes) << " ("
                      << result.output_vrm_bytes << " bytes)\n"
                      << "Runtime SHA256: " << result.runtime_vrm_sha256 << '\n'
                      << "Tensor mappings: " << result.mapping_count << '\n'
                      << "Conversion time: " << result.conversion_seconds << " s\n"
                      << "Installation/finalization time: "
                      << result.finalization_seconds << " s\n"
                      << "CAS created: " << format_bytes(result.copied_artifact_bytes) << '\n'
                      << "CAS reused: " << format_bytes(result.reused_artifact_bytes) << '\n'
                      << "Cache path: "
                      << result.installation.manifest_path.parent_path().string() << '\n';
            return 0;
        }
        if (command == "list") {
            if (arguments.size() != 1) usage();
            const std::vector<product::InstalledPackage> packages = cache.list();
            std::cout << "NAME\tVERSION\tARCHITECTURE\tSIZE\tINSTALLED\n";
            for (const product::InstalledPackage& package : packages) {
                const product::ModelPackageManifest manifest =
                    product::load_model_package_manifest(package.manifest_path);
                std::cout << package.identity.name_space << '/' << package.identity.name << '\t'
                          << package.identity.version << '\t' << package.identity.architecture
                          << '\t' << format_bytes(manifest.logical_size()) << "\tyes\n";
            }
            return 0;
        }
        if (command == "info") {
            const bool json = arguments.size() == 3 && arguments[2] == "--json";
            if (arguments.size() != 2 && !json) usage();
            const product::ResolvedRunnableModel model =
                cache.resolve(arguments[1], false);
            if (json)
                std::cout << product::build_model_info(model).serialize() << '\n';
            else
                print_info(model);
            return 0;
        }
        if (command == "rm") {
            if (arguments.size() != 2) usage();
            cache.remove(arguments[1]);
            std::cout << "Removed package: " << arguments[1] << '\n'
                      << "Blobs retained: yes\n";
            return 0;
        }
#if VRHINO_PRODUCT_RUN_ENABLED
        if (command == "run") {
            product::RunOptions run_options = parse_run_options(arguments);
            cancellation_requested = 0;
            std::signal(SIGINT, cancel_handler);
            const product::ResolvedRunnableModel model =
                cache.resolve(run_options.model_reference, false);
            const char* external_encoder = std::getenv("VRHINO_FFMPEG");
            if (run_options.encoder_path.empty() &&
                (external_encoder == nullptr || *external_encoder == '\0')) {
                try {
                    run_options.encoder_path = component_cache.resolve(
                        product::kMediaComponentReference, true).entrypoint;
                } catch (const product::ModelPackageError& error) {
                    if (error.code() != product::ModelPackageErrorCode::ComponentNotFound) throw;
                    std::cout << "Downloading VRhino media component...\n";
                    product::ComponentRegistryClient registry(
                        cache, component_cache, options.component_registry);
                    const product::ComponentPullResult pulled = registry.pull(
                        product::kMediaComponentReference, &std::cout);
                    run_options.encoder_path = component_cache.resolve(
                        product::kMediaComponentReference, true).entrypoint;
                    std::cout << "Installed media component " << pulled.identity.version << '\n';
                }
            }
            product::RunSession session(
                model, run_options,
                [](const product::ResolvedRunnableModel& runnable,
                   const product::RunOptions& options,
                   product::RunEventSink events) {
                    return product::run_runnable_model(
                        runnable, options, std::move(events));
                },
                [](const product::RunEvent& event) {
                    if (event.kind != product::RunEventKind::State &&
                        !event.message.empty())
                        std::cout << event.message << '\n';
                });
            const product::RunResult result = session.run();
            std::cout << "Model: " << result.identity.reference() << '\n'
                      << "Preset: " << result.preset << '\n'
                      << "Seed: " << result.seed << '\n'
                      << "Resolution: " << result.width << 'x' << result.height << '\n'
                      << "Frames: " << result.frames << '\n'
                      << "FPS: " << result.fps << '\n'
                      << "Hardware: " << preflight_status_name(result.preflight.status) << '\n'
                      << "Package/component validation time: "
                      << result.package_validation_seconds << " s\n"
                      << "Media input time: " << result.media_input_seconds << " s\n"
                      << "Audio conditioning time: " << result.conditioning_seconds << " s\n"
                      << "Face analysis time: " << result.face_analysis_seconds << " s\n"
                      << "Source preparation time: " << result.source_preparation_seconds << " s\n"
                      << "Sampling time: " << result.sampling_seconds << " s\n"
                      << "Decode time: " << result.decode_seconds << " s\n"
                      << "Parser/mask time: " << result.parser_mask_seconds << " s\n"
                      << "Composite time: " << result.composite_seconds << " s\n"
                      << "Encode time: " << result.encoding_seconds << " s\n"
                      << "Total: " << result.total_seconds << " s\n"
                      << "Peak device memory: " << format_bytes(result.peak_device_bytes) << '\n';
            if (model.manifest.product.family == "text_to_video")
                std::cout << "Video range: [" << result.video_minimum << ','
                          << result.video_maximum << "]\n";
            if (model.manifest.product.family == "lip_sync")
                std::cout << "Maximum AlphaMask support change: "
                          << result.maximum_alpha_support_change << '\n'
                          << "Maximum AlphaMask centroid movement: "
                          << result.maximum_alpha_centroid_movement << " pixels\n"
                          << "Invalid AlphaMasks: " << result.invalid_alpha_masks << '\n';
            std::cout << "Output: "
                      << std::filesystem::absolute(
                             session.effective_output_path()).string()
                      << " (" << result.output_bytes << " bytes)\n";
            return 0;
        }
#endif
        usage();
    } catch (const product::ModelPackageError& error) {
        std::cerr << error.what() << '\n';
        if (error.code() == product::ModelPackageErrorCode::Cancelled) return 130;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "CACHE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
