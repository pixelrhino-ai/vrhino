#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vrhino/product/model_package.h"

namespace product = vrhino::product;

namespace {

[[noreturn]] void usage() {
    std::cerr
        << "Usage:\n"
        << "  vrhino-model [--cache-root PATH] install PACKAGE_DIR\n"
        << "  vrhino-model [--cache-root PATH] list\n"
        << "  vrhino-model [--cache-root PATH] info NAMESPACE/NAME:VERSION\n"
        << "  vrhino-model [--cache-root PATH] resolve NAMESPACE/NAME:VERSION [--verify]\n"
        << "  vrhino-model [--cache-root PATH] rm NAMESPACE/NAME:VERSION\n";
    std::exit(2);
}

void print_model(const product::ResolvedRunnableModel& model, const bool include_paths) {
    const auto& manifest = model.manifest;
    std::cout << "reference=" << manifest.identity.reference() << '\n'
              << "architecture=" << manifest.identity.architecture << '\n'
              << "publisher=" << manifest.identity.publisher << '\n'
              << "runtime_contract=" << manifest.runtime_contract << '\n'
              << "vrm_schema=" << manifest.vrm_format_major << '.'
              << manifest.vrm_format_minor << "/schema" << manifest.vrm_metadata_schema << '\n'
              << "default_preset=" << manifest.default_preset << '\n'
              << "source=" << manifest.source_repository << '@' << manifest.source_revision << '\n'
              << "license=" << manifest.license_identifier << '\n'
              << "artifact_count=" << model.artifacts.size() << '\n'
              << "component_count=" << manifest.components.size() << '\n';
    if (include_paths) {
        std::cout << "manifest=" << model.manifest_path.string() << '\n'
                  << "runtime_model=" << model.runtime_model_path.string() << '\n';
        for (const auto& [id, artifact] : model.artifacts) {
            std::cout << "artifact." << id << ".role=" << artifact.declaration.role << '\n'
                      << "artifact." << id << ".path=" << artifact.path.string() << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        std::filesystem::path cache_root;
        if (arguments.size() >= 2 && arguments[0] == "--cache-root") {
            cache_root = arguments[1];
            arguments.erase(arguments.begin(), arguments.begin() + 2);
        }
        if (arguments.empty()) usage();
        product::LocalModelCache cache(cache_root);
        const std::string command = arguments[0];
        if (command == "install") {
            if (arguments.size() != 2) usage();
            const product::InstallResult result = cache.install(arguments[1]);
            std::cout << "installed=" << result.identity.reference() << '\n'
                      << "manifest=" << result.manifest_path.string() << '\n'
                      << "blobs_created=" << result.blobs_created << '\n'
                      << "blobs_reused=" << result.blobs_reused << '\n'
                      << "bytes_created=" << result.bytes_created << '\n'
                      << "bytes_reused=" << result.bytes_reused << '\n';
            return 0;
        }
        if (command == "list") {
            if (arguments.size() != 1) usage();
            const auto packages = cache.list();
            for (const auto& package : packages) {
                std::cout << package.identity.reference() << '\t'
                          << package.identity.architecture << '\t'
                          << package.identity.publisher << '\n';
            }
            std::cout << "packages=" << packages.size() << '\n';
            return 0;
        }
        if (command == "info") {
            if (arguments.size() != 2) usage();
            print_model(cache.resolve(arguments[1]), false);
            return 0;
        }
        if (command == "resolve") {
            if (arguments.size() != 2 && arguments.size() != 3) usage();
            const bool verify = arguments.size() == 3 && arguments[2] == "--verify";
            if (arguments.size() == 3 && !verify) usage();
            print_model(cache.resolve(arguments[1], verify), true);
            return 0;
        }
        if (command == "rm") {
            if (arguments.size() != 2) usage();
            cache.remove(arguments[1]);
            std::cout << "removed=" << arguments[1] << '\n'
                      << "blobs_retained=true\n";
            return 0;
        }
        usage();
    } catch (const product::ModelPackageError& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "CACHE_ERROR: " << error.what() << '\n';
        return 1;
    }
}
