#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>

#include "vrhino/json.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/pull_orchestration.h"
#include "vrhino/product/source_acquisition.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

constexpr const char* kReference = "vrhino/mochi-1-preview:1.0.0";
constexpr const char* kRepository = "genmo/mochi-1-preview";
constexpr const char* kRevision = "14be5fcea23095ed330cb214647916a451e38b6e";

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(input.good(), "cannot open test document: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::vector<std::string> split(const std::string& value, const char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(separator, begin);
        result.push_back(value.substr(begin, end == std::string::npos
                                                ? std::string::npos : end - begin));
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

template <typename Operation>
void expect_code(const product::ModelPackageErrorCode expected,
                 Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": operation unexpectedly succeeded");
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, context + ": wrong error code");
    }
}

struct MappingSummary {
    size_t total = 0;
    size_t denoiser = 0;
    size_t vae = 0;
    std::map<std::pair<std::string, std::string>,
             std::pair<std::string, std::string>> source_layouts;
};

MappingSummary inspect_mapping(const fs::path& path) {
    std::ifstream input(path);
    require_test(input.good(), "cannot open Mochi tensor mapping");
    std::string line;
    require_test(std::getline(input, line) && line ==
        "source_file\tsource_name\tsource_dtype\tsource_shape\ttransformation\t"
        "destination_name\tdestination_dtype\tdestination_shape\tcomponent\trole",
        "Mochi tensor mapping header drift");
    MappingSummary result;
    std::set<std::string> destinations;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split(line, '\t');
        require_test(fields.size() == 10, "invalid Mochi tensor mapping row");
        require_test(fields[2] == "BF16" && fields[6] == "BF16" &&
                         fields[3] == fields[7] && fields[4] == "identity_bytes",
                     "Mochi mapping changed qualified BF16 identity semantics");
        require_test(destinations.insert(fields[5]).second,
                     "duplicate Mochi destination tensor");
        require_test(result.source_layouts.emplace(
            std::make_pair(fields[0], fields[1]),
            std::make_pair(fields[2], fields[3])).second,
            "duplicate Mochi source tensor");
        if (fields[5].rfind("denoiser.", 0) == 0) ++result.denoiser;
        else if (fields[5].rfind("component.vae.decoder.", 0) == 0) ++result.vae;
        else require_test(false, "unexpected Mochi tensor destination");
        ++result.total;
    }
    return result;
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::string result;
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) result.push_back(',');
        result += std::to_string(shape[index]);
    }
    return result;
}

void validate_retained_sources(const fs::path& root,
                               const MappingSummary& mapping) {
    const std::vector<std::pair<std::string, fs::path>> sources = {
        {"transformer-1", root / "transformer/diffusion_pytorch_model.bf16-00001-of-00003.safetensors"},
        {"transformer-2", root / "transformer/diffusion_pytorch_model.bf16-00002-of-00003.safetensors"},
        {"transformer-3", root / "transformer/diffusion_pytorch_model.bf16-00003-of-00003.safetensors"},
        {"vae", root / "vae/diffusion_pytorch_model.bf16.safetensors"},
    };
    size_t required = 0;
    for (const auto& [source_name, path] : sources) {
        product::SafeTensorReader reader(path);
        for (const auto& [name, tensor] : reader.tensors()) {
            if (source_name == "vae" && name.rfind("decoder.", 0) != 0) continue;
            ++required;
            const auto found = mapping.source_layouts.find({source_name, name});
            require_test(found != mapping.source_layouts.end(),
                         "retained Mochi tensor is not mapped: " + source_name + ":" + name);
            require_test(found->second.first == tensor.source_dtype &&
                             found->second.second == shape_text(tensor.shape),
                         "retained Mochi tensor type/shape drift: " + name);
        }
    }
    require_test(required == 1233 && mapping.source_layouts.size() == required,
                 "retained Mochi source coverage drift");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require_test(argc == 2 || argc == 3,
                     "usage: mochi-product-integration-tests SPEC_ROOT [SOURCE_ROOT]");
        const fs::path root = fs::canonical(argv[1]);
        const fs::path mochi = root / "mochi_1_preview";

        const auto pull = product::find_pull_distribution_plan(kReference, root);
        require_test(pull.has_value(), "Mochi pull plan was not discovered");
        require_test(pull->model_reference == kReference, "pull identity drift");
        require_test(pull->source.repository == kRepository, "pull repository drift");
        require_test(pull->source.revision == kRevision, "pull revision drift");
        require_test(pull->converter == "native", "Mochi converter is not native");

        const product::SourceArtifactPlanDocument source =
            product::load_source_artifact_plan(kReference, mochi);
        require_test(source.requested_source.provider == "huggingface" &&
                         source.requested_source.repository == kRepository &&
                         source.requested_source.revision == kRevision,
                     "Mochi source identity drift");
        uint64_t logical_source_bytes = 0;
        std::set<std::string> source_ids;
        std::map<std::string, std::string> source_hashes;
        for (const product::SourceArtifactPlan& artifact : source.artifacts) {
            logical_source_bytes += artifact.size;
            source_ids.insert(artifact.id);
            source_hashes.emplace(artifact.id, artifact.sha256);
            require_test(artifact.repository == kRepository &&
                             artifact.revision == kRevision,
                         "Mochi source artifact provenance drift");
        }
        require_test(source.artifacts.size() == 20,
                     "Mochi source artifact count drift");
        require_test(logical_source_bytes == 40025270143ULL,
                     "Mochi source logical size drift");
        require_test(source_ids.contains("mochi-transformer-shard-1") &&
                         source_ids.contains("mochi-transformer-shard-2") &&
                         source_ids.contains("mochi-transformer-shard-3") &&
                         source_ids.contains("mochi-vae") &&
                         source_ids.contains("mochi-t5-shard-1") &&
                         source_ids.contains("mochi-t5-shard-4") &&
                         source_ids.contains("mochi-tokenizer") &&
                         source_ids.contains("mochi-license-notice"),
                     "Mochi source artifact set drift");

        const product::ModelPackageManifest manifest =
            product::load_model_package_manifest(mochi / "vrhino-model.json");
        require_test(manifest.identity.reference() == kReference,
                     "manifest identity drift");
        require_test(manifest.identity.architecture == "mochi",
                     "manifest architecture drift");
        require_test(manifest.runtime_contract == "cuda-v1",
                     "Runtime contract drift");
        require_test(manifest.source_repository == kRepository &&
                         manifest.source_revision == kRevision,
                     "manifest source identity drift");
        require_test(manifest.license_identifier == "Apache-2.0",
                     "Mochi license drift");
        require_test(manifest.artifacts.size() == 11,
                     "manifest artifact count drift");
        require_test(manifest.components.size() == 2,
                     "component graph count drift");

        std::set<std::string> installed_hashes;
        for (const product::ArtifactDeclaration& artifact : manifest.artifacts)
            installed_hashes.insert(artifact.sha256);
        for (const std::string& id : {"mochi-t5-shard-1", "mochi-t5-shard-2",
                 "mochi-t5-shard-3", "mochi-t5-shard-4", "mochi-t5-index",
                 "mochi-tokenizer", "mochi-license-notice"})
            require_test(installed_hashes.contains(source_hashes.at(id)),
                         "runtime-required source data is not CAS-shared: " + id);
        uint64_t reclaimable = 0;
        for (const product::SourceArtifactPlan& artifact : source.artifacts)
            if (!installed_hashes.contains(artifact.sha256))
                reclaimable += artifact.size;
        require_test(reclaimable == 20975180045ULL,
                     "Mochi source cleanup classification drift");

        const vrhino::Json profile = vrhino::Json::parse(
            read_text(mochi / "execution/default-preset.json"));
        const vrhino::Json& inputs = profile.at("inputs");
        require_test(inputs.at("width").integer() == 848 &&
                         inputs.at("height").integer() == 480 &&
                         inputs.at("frames").integer() == 163 &&
                         inputs.at("fps").integer() == 30 &&
                         inputs.at("steps").integer() == 64 &&
                         inputs.at("cfg").number() == 6.0,
                     "qualified Mochi run profile drift");
        const vrhino::Json& run = profile.at("run");
        require_test(run.at("default_seed").integer() == 11001 &&
                         run.at("negative_prompt").string().empty(),
                     "qualified Mochi prompt/seed drift");
        require_test(run.at("execution_dtype").string() == "bfloat16" &&
                         run.at("precision_policy_artifact").string() ==
                             "precision-policy",
                     "Mochi generic BF16 binding drift");
        require_test(run.at("memory_runtime").at("device_budget_bytes").integer() ==
                         85899345920LL,
                     "Mochi 80 GiB device budget drift");
        require_test(run.at("components").array().size() == 2,
                     "Mochi conditioning component count drift");
        const vrhino::Json& tiling = run.at("component_execution").at("tiling");
        require_test(tiling.at("planner_policy").string() == "RECURSIVE_BISECTION" &&
                         tiling.at("temporal_policy").string() == "FULL" &&
                         tiling.at("tile_counts").array()[0].integer() == 2 &&
                         tiling.at("tile_counts").array()[1].integer() == 4,
                     "Mochi VAE tiling drift");
        const auto& output_range = run.at("output").at("range").array();
        require_test(output_range.size() == 2 && output_range[0].number() == 0.0 &&
                         output_range[1].number() == 1.0,
                     "Mochi output contract drift");

        const vrhino::Json metadata = vrhino::Json::parse(
            read_text(mochi / "metadata.json"));
        const vrhino::Json graph = vrhino::Json::parse(read_text(mochi / "graph.json"));
        require_test(metadata.at("architecture").string() == "mochi" &&
                         metadata.at("latent_contract").at("channels").integer() == 12,
                     "Mochi metadata drift");
        require_test(graph.at("architecture_graph").at("implementation_id").string() ==
                         "dit_flow.mochi.asymm_dit.v1" &&
                         graph.at("sampling_program").at("implementation_id").string() ==
                         "dit_flow.mochi.sampling.v1" &&
                         graph.at("component_graphs").array()[0].at("implementation_id").string() ==
                         "dit_flow.mochi.asymm_vae_decoder.v1",
                     "Mochi graph contract drift");

        const MappingSummary mapping = inspect_mapping(mochi / "tensor-mapping.tsv");
        require_test(mapping.total == 1233 && mapping.denoiser == 1071 &&
                         mapping.vae == 162,
                     "Mochi canonical tensor count drift");
        if (argc == 3) validate_retained_sources(fs::canonical(argv[2]), mapping);

        const std::string policy = read_text(
            mochi / "policy/bf16-operation-contract-v1.json");
        require_test(policy.find("mochi") == std::string::npos &&
                         policy.find("Mochi") == std::string::npos,
                     "precision policy became architecture-specific");

        expect_code(product::ModelPackageErrorCode::InsufficientDiskSpace, [&] {
            product::require_conversion_disk_space(
                manifest.logical_size(), manifest.logical_size(), 1, "/test/cache");
        }, "Mochi disk admission");

        const fs::path failure_root = fs::temp_directory_path() /
            ("vrhino-mochi-product-test-" + std::to_string(getpid()));
        std::error_code remove_error;
        fs::remove_all(failure_root, remove_error);
        fs::create_directories(failure_root / "empty");
        expect_code(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::LocalModelCache cache(failure_root / "cache-missing");
            product::ImportOptions options;
            options.converter_spec_root = root;
            (void)product::import_local_model(kReference, failure_root / "empty",
                                              cache, options);
        }, "missing Mochi source");

        const fs::path corrupt = failure_root / "corrupt";
        for (const product::SourceArtifactPlan& artifact : source.artifacts) {
            fs::create_directories((corrupt / artifact.local_path).parent_path());
            std::ofstream output(corrupt / artifact.local_path,
                                 std::ios::binary | std::ios::trunc);
            output << "corrupt";
        }
        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            product::LocalModelCache cache(failure_root / "cache-cancelled");
            product::ImportOptions options;
            options.converter_spec_root = root;
            options.cancellation_requested = [] { return true; };
            (void)product::import_local_model(kReference, corrupt, cache, options);
        }, "cancelled Mochi conversion");
        product::LocalModelCache cancelled_cache(failure_root / "cache-cancelled");
        require_test(cancelled_cache.list().empty(),
                     "cancelled Mochi conversion published a package");
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::LocalModelCache cache(failure_root / "cache-corrupt");
            product::ImportOptions options;
            options.converter_spec_root = root;
            (void)product::import_local_model(kReference, corrupt, cache, options);
        }, "corrupt Mochi source");
        fs::remove_all(failure_root, remove_error);

        std::cout << "Mochi product integration tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mochi product integration tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
