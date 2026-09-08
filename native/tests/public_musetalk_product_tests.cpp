#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>

#include "vrhino/product/model_package.h"
#include "vrhino/product/pull_orchestration.h"
#include "vrhino/product/source_acquisition.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void check(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot read Public MuseTalk fixture");
    return {std::istreambuf_iterator<char>(input), {}};
}

std::map<std::string, product::SourceArtifactPlan> sources_by_id(
        const product::SourceArtifactPlanDocument& document) {
    std::map<std::string, product::SourceArtifactPlan> result;
    for (const auto& artifact : document.artifacts)
        result.emplace(artifact.id, artifact);
    return result;
}

}  // namespace

int main() {
    try {
        const fs::path root = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        const fs::path public_spec = root / "public_musetalk_v15";
        const product::ModelPackageManifest manifest =
            product::load_model_package_manifest(
                public_spec / product::kModelManifestName);
        check(manifest.identity.reference() ==
                  "vrhino/musetalk-v1.5:1.0.0",
              "Public MuseTalk identity drift");
        check(manifest.product.family == "lip_sync" &&
                  manifest.product.status == "public_supported" &&
                  manifest.product.public_distribution ==
                      "mode_c_local_conversion" &&
                  manifest.product.workflow_identity ==
                      "lip_sync_workflow_v1",
              "Public product/distribution declaration drift");
        check(manifest.components.size() == 6 &&
                  manifest.artifacts.size() == 13 &&
                  !manifest.minimum_vram_bytes.has_value() &&
                  !manifest.recommended_vram_bytes.has_value(),
              "Public component/artifact/VRAM contract drift");

        const std::map<std::string, std::string> expected_roles = {
            {"audio_encoder", "4527260f6727d202f0a964ff062d68bd03a00b6647e0809b18ba318afb76777d"},
            {"image_autoencoder", "49d6d814b8546535f36580543987ff0477ebd89040cc053d2a8f836b68db9b1e"},
            {"neural_edit", "4fc8954eea557b636abe22450076b38ddea8a629fa38f9ab545014b1c1d22c47"},
            {"face_detector", "bbe5a9a5d83b13401a9b9bc7829ccb993af6ac28e2dd37c530001e90fdf06d17"},
            {"pose_estimator", "3f457d8ff6eb49098d9626e232785b7158d8838e8054eb1f85d0684decc6a7c5"},
            {"semantic_segmenter", "e797157d562263919e28f74d4701074243a9c56fcadd6f1ba14703e85cc554f1"},
        };
        std::map<std::string, std::string> artifact_hashes;
        for (const auto& artifact : manifest.artifacts)
            artifact_hashes.emplace(artifact.id, artifact.sha256);
        for (const auto& component : manifest.components) {
            check(component.artifact_ids.size() == 1,
                  "Public component must bind exactly one immutable artifact");
            const auto expected = expected_roles.find(component.role);
            check(expected != expected_roles.end() &&
                      artifact_hashes.at(component.artifact_ids.front()) ==
                          expected->second,
                  "Public component role/hash drift: " + component.role);
        }

        const auto pull = product::find_pull_distribution_plan(
            manifest.identity.reference(), root);
        check(pull.has_value() && pull->kind ==
                  product::PullDistributionKind::MultiComponentSourceBacked,
              "Public Mode C pull plan was not selected");
        const auto public_sources = product::load_source_artifact_plan(
            manifest.identity.reference(), public_spec);
        // Freeze the public source plan independently of excluded research specs.
        check(public_sources.artifacts.size() == 12 &&
                  sources_by_id(public_sources).size() == 12,
              "Public source inventory or unique artifact IDs drift");
        check(product::sha256_file(public_spec / "source-plan.json") ==
                  "92ed908857607ba920d079bd7567b87b7366d126ccd726f59a40e814f6537032",
              "Public source acquisition identities drift");

        const std::string source_text = read_text(public_spec / "source-plan.json");
        for (const char* blocked : {"s3fd", "619a316812", "79999_iter",
                                    "468e13ca", "CelebAMask"})
            check(source_text.find(blocked) == std::string::npos,
                  std::string("blocked asset leaked into Public source plan: ") + blocked);

        check(product::sha256_file(
                  root / "musetalk_v15_workflow_v2/workflow.json") ==
                  "4240b286ee1bbde6b356a9648a9fc4c18b50b691ee88b78a341606b7959ca92b",
              "qualified successor workflow identity drift");
        const std::map<std::string, std::string> legal_hashes = {
            {"THIRD_PARTY_NOTICES.txt", "fcf4afc654e3fc97c6b32d04a17029edee1b3af09e9db5bf9e9d4036f52bbaae"},
            {"licenses/CreativeML-OpenRAIL-M.txt", "be351ebe7ac01bcdbb018639aadcfd38f136b7dc3f2a3d4d3a24db51d1b210ef"},
            {"licenses/Apache-2.0.txt", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"},
            {"licenses/MIT.txt", "042b8ae7ce9a75baa8973d121f5f8f166a89a4e20dac220d1bbd66c04be273a9"},
        };
        for (const auto& [path, hash] : legal_hashes)
            check(product::sha256_file(public_spec / path) == hash,
                  "required Public notice/license identity drift: " + path);

        std::cout << "Public MuseTalk product metadata tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Public MuseTalk product metadata tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
