#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
    check(static_cast<bool>(input), "cannot read Public LatentSync fixture");
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
        const fs::path public_spec = root / "public_latentsync_16";
        const product::ModelPackageManifest manifest =
            product::load_model_package_manifest(
                public_spec / product::kModelManifestName);
        check(manifest.identity.reference() ==
                  "vrhino/latentsync-1.6:1.0.0",
              "Public LatentSync identity drift");
        check(manifest.product.family == "lip_sync" &&
                  manifest.product.status == "public_supported" &&
                  manifest.product.public_distribution ==
                      "mode_c_local_conversion" &&
                  manifest.product.workflow_identity ==
                      "lip_sync_diffusion_workflow_v1",
              "Public product/distribution declaration drift");
        check(manifest.components.size() == 5 &&
                  manifest.artifacts.size() == 17 &&
                  !manifest.minimum_vram_bytes.has_value() &&
                  !manifest.recommended_vram_bytes.has_value(),
              "Public component/artifact/VRAM contract drift");

        const std::map<std::string, std::string> expected_roles = {
            {"audio_encoder", "3322531357af02994de5f180b12a1c7cb72a800dd7ed40ab11ec5865304ed1a0"},
            {"image_autoencoder", "49d6d814b8546535f36580543987ff0477ebd89040cc053d2a8f836b68db9b1e"},
            {"neural_edit", "47508970409f6a3cb8ee366446af243fa985e688a5170549316af8a450a6c467"},
            {"face_detector", "bbe5a9a5d83b13401a9b9bc7829ccb993af6ac28e2dd37c530001e90fdf06d17"},
            {"pose_estimator", "3f457d8ff6eb49098d9626e232785b7158d8838e8054eb1f85d0684decc6a7c5"},
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
                  "2510c9c32b40ee1865c3c2cd391a5025bc27817a58a22e0d4dea9ad490c04ade",
              "Public source acquisition identities drift");

        const std::string source_text = read_text(public_spec / "source-plan.json");
        for (const char* blocked : {
                 "InsightFace", "buffalo_l", "det_10g.onnx", "2d106det.onnx",
                 "sfd_face.pth", "stable_syncnet", "SyncNet", "TREPA", "I3D",
                 "VideoMAE", "VGG"})
            check(source_text.find(blocked) == std::string::npos,
                  std::string("blocked/non-inference asset leaked: ") + blocked);
        check(source_text.find("pixelrhino") == std::string::npos &&
                  source_text.find(".vrm") == std::string::npos,
              "Public source plan contains hosted/converted model material");

        check(product::sha256_file(
                  root / "latentsync_16_workflow/workflow.json") ==
                  "8d556d4173e4e69daaed45dbea0bebbacc975b499b7b52d592d44e82715d1312",
              "qualified LatentSync workflow identity drift");
        const vrhino::Json execution = vrhino::Json::parse(
            read_text(public_spec / "execution.json"));
        check(execution.at("execution_dtype").string() == "bfloat16" &&
                  execution.at("precision_policy_artifact").string() ==
                      "precision-policy" &&
                  product::sha256_file(public_spec /
                      "policy/bf16-heavy-consumer-fp32-state-v1.json") ==
                      "e1423a0a42acff673b6f35c1103331d77e3e8b48c0bbfce38fbe03f35ef5d68f",
              "Public LatentSync generic BF16 precision admission drift");
        const std::map<std::string, std::string> legal_hashes = {
            {"THIRD_PARTY_NOTICES.txt", "b4da9f7052f47892c48abc3400fdeaaa15df1102b82cc86d36ad2ed21019e259"},
            {"licenses/CreativeML-Open-RAIL++-M.txt", "5f44a64473fdb1019fc616d999d351f60bf061a4d77272b4b7309f799dba3676"},
            {"licenses/Apache-2.0.txt", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"},
            {"licenses/MIT.txt", "042b8ae7ce9a75baa8973d121f5f8f166a89a4e20dac220d1bbd66c04be273a9"},
        };
        for (const auto& [path, hash] : legal_hashes)
            check(product::sha256_file(public_spec / path) == hash,
                  "required Public notice/license identity drift: " + path);
        const std::string rail = read_text(
            public_spec / "licenses/CreativeML-Open-RAIL++-M.txt");
        check(rail.find("CreativeML Open RAIL++-M License") != std::string::npos &&
                  rail.find("Attachment A") != std::string::npos &&
                  rail.find("Use Restrictions") != std::string::npos,
              "complete CreativeML Open RAIL++-M terms are unavailable");
        const std::string notices = read_text(
            public_spec / "THIRD_PARTY_NOTICES.txt");
        for (const char* required : {
                 "Copyright (c) 2024 Bytedance Ltd. and/or its affiliates.",
                 "Copyright (c) 2022 OpenAI", "Stability AI", "Google LLC",
                 "IDEA", "Open-MMLab", "Tencent Music Entertainment Group",
                 "Converted from the upstream model representation",
                 "does not distribute", "No upstream owner"})
            check(notices.find(required) != std::string::npos,
                  std::string("required notice mapping is missing: ") + required);

        std::cout << "Public LatentSync product metadata tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Public LatentSync product metadata tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
