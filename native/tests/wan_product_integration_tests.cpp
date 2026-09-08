#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unistd.h>

#include "vrhino/json.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/pull_orchestration.h"
#include "vrhino/product/source_acquisition.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

constexpr const char* kReference = "vrhino/wan2.1-t2v-1.3b:1.0.0";
constexpr const char* kRepository = "Wan-AI/Wan2.1-T2V-1.3B";
constexpr const char* kRevision = "37ec512624d61f7aa208f7ea8140a131f93afc9a";

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(input.good(), "cannot open test document: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

size_t mapping_rows(const fs::path& path) {
    std::ifstream input(path);
    require_test(input.good(), "cannot open mapping: " + path.string());
    std::string line;
    size_t rows = 0;
    while (std::getline(input, line)) if (!line.empty()) ++rows;
    require_test(rows > 1, "mapping is empty");
    return rows - 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require_test(argc == 2, "usage: wan-product-integration-tests SPEC_ROOT");
        const fs::path root = fs::canonical(argv[1]);
        const fs::path wan = root / "wan2_1_t2v_1_3b";

        const auto pull = product::find_pull_distribution_plan(kReference, root);
        require_test(pull.has_value(), "Wan pull plan was not discovered");
        require_test(pull->model_reference == kReference, "pull identity drift");
        require_test(pull->source.repository == kRepository, "pull repository drift");
        require_test(pull->source.revision == kRevision, "pull revision drift");
        require_test(pull->converter == "native", "Wan converter is not native");

        const product::SourceArtifactPlanDocument source =
            product::load_source_artifact_plan(kReference, wan);
        require_test(source.requested_source.repository == kRepository,
                     "source repository drift");
        require_test(source.requested_source.revision == kRevision,
                     "source revision drift");
        uint64_t logical_source_bytes = 0;
        std::set<std::string> source_ids;
        std::map<std::string, std::string> source_hashes;
        for (const product::SourceArtifactPlan& artifact : source.artifacts) {
            logical_source_bytes += artifact.size;
            source_ids.insert(artifact.id);
            source_hashes.emplace(artifact.id, artifact.sha256);
            require_test(artifact.repository == kRepository,
                         "source artifact repository drift");
            require_test(artifact.revision == kRevision,
                         "source artifact revision drift");
        }
        require_test(logical_source_bytes == 17562449745ULL,
                     "Wan source logical size drift");
        require_test(source_ids == std::set<std::string>({
            "wan-config", "wan-denoiser", "wan-license", "wan-text-encoder",
            "wan-tokenizer", "wan-vae"}), "Wan source artifact set drift");

        const product::ModelPackageManifest manifest =
            product::load_model_package_manifest(wan / "vrhino-model.json");
        require_test(manifest.identity.reference() == kReference, "manifest identity drift");
        require_test(manifest.identity.architecture == "wan", "manifest architecture drift");
        require_test(manifest.runtime_contract == "cuda-v1", "Runtime contract drift");
        require_test(manifest.artifacts.size() == 8, "manifest artifact count drift");
        require_test(manifest.components.size() == 2, "component graph count drift");
        require_test(manifest.runtime_artifact_id == "runtime", "Runtime artifact drift");
        std::set<std::string> installed_hashes;
        for (const product::ArtifactDeclaration& artifact : manifest.artifacts)
            installed_hashes.insert(artifact.sha256);
        require_test(installed_hashes.contains(source_hashes.at("wan-tokenizer")) &&
                     installed_hashes.contains(source_hashes.at("wan-license")),
                     "runtime-required source-derived artifacts are not shared by content");
        for (const std::string& id : {"wan-config", "wan-denoiser",
                                      "wan-text-encoder", "wan-vae"})
            require_test(!installed_hashes.contains(source_hashes.at(id)),
                         std::string("source-only artifact became installed data: ") + id);

        const vrhino::Json profile = vrhino::Json::parse(
            read_text(wan / "execution/default-preset.json"));
        const vrhino::Json& inputs = profile.at("inputs");
        require_test(inputs.at("width").integer() == 832 &&
                     inputs.at("height").integer() == 480 &&
                     inputs.at("frames").integer() == 81 &&
                     inputs.at("steps").integer() == 50,
                     "qualified Wan run profile drift");
        const vrhino::Json& run = profile.at("run");
        require_test(run.at("execution_dtype").string() == "bfloat16",
                     "Wan profile must use the generic BF16 policy");
        require_test(run.at("precision_policy_artifact").string() == "precision-policy",
                     "precision policy association drift");
        require_test(run.at("components").array().size() == 2,
                     "conditioning component count drift");
        const auto& range = run.at("output").at("range").array();
        require_test(range.size() == 2 && range[0].number() == -1.0 &&
                     range[1].number() == 1.0,
                     "Wan decoder output contract drift");

        require_test(mapping_rows(wan / "tensor-mapping.tsv") == 1019,
                     "Wan VRM mapping count drift");
        require_test(mapping_rows(wan / "umt5-mapping.tsv") == 242,
                     "Wan UMT5 mapping count drift");

        const std::string policy = read_text(
            wan / "policy/bf16-operation-contract-v1.json");
        require_test(policy.find("wan") == std::string::npos &&
                     policy.find("Wan") == std::string::npos,
                     "precision policy became architecture-specific");

        const fs::path failure_root = fs::temp_directory_path() /
            ("vrhino-wan-product-test-" + std::to_string(getpid()));
        std::error_code remove_error;
        fs::remove_all(failure_root, remove_error);
        fs::create_directories(failure_root / "source/google/umt5-xxl");
        bool missing_failed = false;
        try {
            product::LocalModelCache cache(failure_root / "cache-missing");
            product::ImportOptions options;
            options.converter_spec_root = root;
            (void)product::import_local_model(kReference, failure_root / "source",
                                              cache, options);
        } catch (const product::ModelPackageError& error) {
            missing_failed = error.code() == product::ModelPackageErrorCode::ArtifactMissing;
        }
        require_test(missing_failed, "missing Wan source did not fail closed");

        for (const fs::path& relative : {
                fs::path("diffusion_pytorch_model.safetensors"),
                fs::path("models_t5_umt5-xxl-enc-bf16.pth"),
                fs::path("Wan2.1_VAE.pth"), fs::path("config.json"),
                fs::path("google/umt5-xxl/tokenizer.json"), fs::path("LICENSE.txt")}) {
            std::ofstream output(failure_root / "source" / relative,
                                 std::ios::binary | std::ios::trunc);
            output << "corrupt";
        }
        bool corrupt_failed = false;
        try {
            product::LocalModelCache cache(failure_root / "cache-corrupt");
            product::ImportOptions options;
            options.converter_spec_root = root;
            (void)product::import_local_model(kReference, failure_root / "source",
                                              cache, options);
        } catch (const product::ModelPackageError& error) {
            corrupt_failed = error.code() == product::ModelPackageErrorCode::PackageInvalid;
        }
        require_test(corrupt_failed, "corrupt Wan source did not fail closed");
        fs::remove_all(failure_root, remove_error);

        std::cout << "Wan product integration tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Wan product integration tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
