#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/json.h"
#include "vrhino/product/input_schema.h"
#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(static_cast<bool>(input), "cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

struct Case {
    std::string reference;
    fs::path successor_manifest;
    fs::path legacy_manifest;
    std::string legacy_manifest_sha256;
    fs::path execution;
    fs::path workflow;
    uint64_t seed;
    std::string family;
};

}  // namespace

int main() {
    try {
        const fs::path root = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        const std::vector<Case> cases = {
            {"vrhino/ltx-video-v0.9.1:1.1.1",
             root / "ltx_v0_9_1/successors/1.1.1/vrhino-model.json",
             root / "ltx_v0_9_1/vrhino-model.json",
             "3e7e6726f10baf2f7d56dc6743d6de0dd13a2bc44a4aecbed85b4a937e1428cb",
             root / "ltx_v0_9_1/execution/default-preset.json", {}, 5703,
             "text_to_video"},
            {"vrhino/wan2.1-t2v-1.3b:1.0.1",
             root / "wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json",
             root / "wan2_1_t2v_1_3b/vrhino-model.json",
             "ad4ebc02522e07322fe6bce2a33eae48ee9dd1eec9dfe357628b8ae947ab1a9b",
             root / "wan2_1_t2v_1_3b/execution/default-preset.json", {}, 5701,
             "text_to_video"},
            {"vrhino/mochi-1-preview:1.0.1",
             root / "mochi_1_preview/successors/1.0.1/vrhino-model.json",
             root / "mochi_1_preview/vrhino-model.json",
             "83cd2bb653e799712c4dc24e4cc4f1d616d6a01e63ad0ddd6144ebcdadb2854c",
             root / "mochi_1_preview/execution/default-preset.json", {}, 11001,
             "text_to_video"},
            {"vrhino/musetalk-v1.5:1.0.1",
             root / "public_musetalk_v15/successors/1.0.1/vrhino-model.json",
             root / "public_musetalk_v15/vrhino-model.json",
             "08cccfb542b4dc6c9d43783f3f4c2b2525838e63b7f3aa226f1cd136dbc7dbe3",
             root / "public_musetalk_v15/successors/1.0.1/execution.json",
             root / "musetalk_v15_workflow_v2/workflow.json", 11001, "lip_sync"},
            {"vrhino/latentsync-1.6:1.0.1",
             root / "public_latentsync_16/successors/1.0.1/vrhino-model.json",
             root / "public_latentsync_16/vrhino-model.json",
             "46d0e78b96641983c1ae41fdf71217ee6a208fa158eadbca8bb1359f8d227a90",
             root / "public_latentsync_16/successors/1.0.1/execution.json",
             root / "latentsync_16_workflow/workflow.json", 1247, "lip_sync"},
        };

        for (const Case& item : cases) {
            const product::ModelPackageManifest successor =
                product::load_model_package_manifest(item.successor_manifest);
            require_test(successor.identity.reference() == item.reference,
                         "successor identity mismatch: " + item.reference);
            require_test(successor.product.family == item.family &&
                             successor.product.input_schema.has_value() &&
                             successor.product.frozen_profile.has_value(),
                         "successor Product contract missing: " + item.reference);
            const product::ProductInputSchema& schema =
                *successor.product.input_schema;
            require_test(schema.identity == product::kProductInputSchemaV1,
                         "schema identity mismatch: " + item.reference);
            require_test(schema.find_parameter("seed") != nullptr &&
                             std::get<uint64_t>(
                                 schema.find_parameter("seed")->default_value) == item.seed,
                         "seed default mismatch: " + item.reference);
            require_test(schema.find_output("output") != nullptr &&
                             !schema.find_output("output")->required &&
                             std::get<std::string>(
                                 schema.find_output("output")->default_value) ==
                                 "output.mp4",
                         "output requiredness/default mismatch: " + item.reference);
            require_test(std::set<std::string>(successor.product.required_inputs.begin(),
                                               successor.product.required_inputs.end()) ==
                             (item.family == "text_to_video"
                                  ? std::set<std::string>{"prompt"}
                                  : std::set<std::string>{"audio", "video"}),
                         "derived required inputs mismatch: " + item.reference);
            const vrhino::Json execution = vrhino::Json::parse(read_text(item.execution));
            std::optional<vrhino::Json> workflow;
            if (!item.workflow.empty())
                workflow = vrhino::Json::parse(read_text(item.workflow));
            product::validate_product_execution_consistency(
                successor.product.family, successor.product.workflow_identity,
                schema, *successor.product.frozen_profile, execution,
                workflow ? &*workflow : nullptr);

            const product::ModelPackageManifest legacy =
                product::load_model_package_manifest(item.legacy_manifest);
            require_test(!legacy.product.input_schema.has_value() &&
                             !legacy.product.frozen_profile.has_value(),
                         "legacy package was inferred as schema v1: " + item.reference);
            require_test(product::sha256_file(item.legacy_manifest) ==
                             item.legacy_manifest_sha256,
                         "legacy manifest bytes changed: " + item.legacy_manifest.string());
        }

        const product::ModelPackageManifest latent =
            product::load_model_package_manifest(
                root / "public_latentsync_16/successors/1.0.1/vrhino-model.json");
        const product::ProductInputSchema& latent_schema =
            *latent.product.input_schema;
        require_test(latent_schema.parameters.size() == 1 &&
                         latent_schema.parameters.front().name == "seed" &&
                         latent_schema.find_input("cfg") == nullptr &&
                         latent_schema.find_parameter("cfg") == nullptr &&
                         latent_schema.find_parameter("steps") == nullptr &&
                         latent_schema.find_parameter("eta") == nullptr &&
                         latent_schema.find_parameter("chunk") == nullptr &&
                         latent_schema.find_parameter("alignment_ema") == nullptr &&
                         latent_schema.find_parameter("detector") == nullptr &&
                         latent_schema.find_parameter("tta") == nullptr,
                     "LatentSync exposed an internal control as adjustable");
        require_test(latent.product.frozen_profile->sampling.has_value() &&
                         latent.product.frozen_profile->sampling->method == "ddim" &&
                         latent.product.frozen_profile->sampling->prediction == "epsilon" &&
                         latent.product.frozen_profile->sampling->steps == 20 &&
                         latent.product.frozen_profile->temporal.has_value() &&
                         latent.product.frozen_profile->temporal->chunk_frames == 16,
                     "LatentSync curated read-only profile is incomplete");

        for (const Case& item : cases) {
            const fs::path successor_root = item.successor_manifest.parent_path();
            const vrhino::Json pull = vrhino::Json::parse(
                read_text(successor_root / "pull-plan.json"));
            const vrhino::Json source = vrhino::Json::parse(
                read_text(successor_root / "source-plan.json"));
            require_test(pull.at("model_reference").string() == item.reference &&
                             source.at("model_reference").string() == item.reference,
                         "successor source/pull identity drift: " + item.reference);
        }

        std::cout << "five successor/legacy Product schema spec tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "five successor/legacy Product schema spec tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
