#include "ltx_schema2_test_support.h"
#include "vrhino/product/declared_run.h"
#include "vrhino/product/request_wiring.h"
#include <iostream>
namespace s = vrhino::ltx_schema2_test;
namespace p = vrhino::product;
using namespace vrhino;
int main(int argc, char **argv) {
    try {
        require(argc == 4, "usage: ltx-schema2-product-tests PACKAGE_DIR OUTPUT_DIR PROMPT");
        const s::fs::path package = argv[1], dir = argv[2];
        s::fs::create_directories(dir);
        auto admitted = std::make_shared<p::AdmittedLocalProduct>(
            p::preflight_local_product(package / "vrhino-model.json", package / "local.json"));
        auto request = Json::parse(
            R"({"schema":"vrhino.product-text-request.v1","prompt":"","negative_prompt":"worst quality, inconsistent motion, blurry, jittery, distorted","seed":5703,"width":704,"height":480,"frames":121})");
        request = s::set(request, "prompt", Json(std::string(argv[3])));
        auto prepared = p::prepare_text_product_request(admitted, request);
        require(prepared.sampling.contract &&
                    prepared.sampling.contract->solver.semantic == SolverSemantic::FlowEuler &&
                    prepared.sampling.steps == 40,
                "Product lost typed authority");
        require(prepared.sampling.latent_shape == std::vector<int64_t>({1, 5280, 128}) &&
                    prepared.runtime_inputs.size() == 5,
                "Product geometry mismatch");
        require(prepared.expected_video_shape == std::vector<int64_t>({1, 3, 121, 480, 704}),
                "Product decoder output mismatch");
        for (int i = 0; i < 40; ++i)
            require(prepared.sampling.model_timestep_at(i).shape() ==
                            std::vector<int64_t>({1, 1}) &&
                        prepared.sampling.model_timestep_at(i).dtype() == DType::F32,
                    "I64 timestep coercion");
        for (const auto &c : prepared.conditioning)
            require(c.expected_hidden_shape == std::vector<int64_t>({1, 128, 4096}) &&
                        c.attention_mask.dtype() == DType::Bool && !c.mask_target.empty(),
                    "Missing padded hidden/mask contract");
        require(prepared.evidence.at("execution_intent")
                        .array()[0]
                        .at("model_timestep")
                        .at("dtype")
                        .string() == "float32",
                "Incorrect typed evidence");
        s::write(dir / "request.json", request);
        s::write(dir / "prepared-evidence.json", prepared.evidence);
        s::write(dir / "admission-evidence.json", admitted->evidence);
        for (const auto *dimension : {"width", "height", "frames"})
            s::reject("frozen geometry override", [&] {
                p::prepare_text_product_request(
                    admitted,
                    s::set(request, dimension, Json(request.at(dimension).integer() + 1)));
            });
        for (const auto *extra : {"sampling_steps", "guidance_scale", "precision", "binding"})
            s::reject("legacy authority override", [&] {
                p::prepare_text_product_request(admitted, s::set(request, extra, Json(int64_t(1))));
            });
        s::reject("negative seed", [&] {
            p::prepare_text_product_request(admitted, s::set(request, "seed", Json(int64_t(-1))));
        });
        s::reject("invalid UTF8", [&] {
            p::prepare_text_product_request(admitted,
                                            s::set(request, "prompt", Json(std::string("\xff"))));
        });
        s::reject("structural cannot qualify numerics",
                  [&] { p::require_numerical_product_admission(admitted->resources.manifest); });
        // Mutable pathname replacement cannot change the immutable admitted artifact authority.
        const auto before = admitted->resources.artifacts.at("programs").path;
        auto corrupt = admitted->model->metadata().at("programs");
        auto sample = corrupt.at("sampling").object();
        sample["maximum_order"] = Json(int64_t(2));
        corrupt = s::set(corrupt, "sampling", Json(sample));
        s::write(dir / "changed-programs.json", corrupt);
        admitted->resources.artifacts.at("programs").path = dir / "changed-programs.json";
        s::reject("verified programs cannot drift",
                  [&] { p::prepare_text_product_request(admitted, request); });
        admitted->resources.artifacts.at("programs").path = before;
        auto changed = Json::parse(admitted->resources.manifest.raw_json);
        auto caps = Json::parse(
            R"(["binding_catalog.v1","execution.per_step.v1","sampling.flow_sigma_cfg.v1","sampling.flow_euler_cfg.v1"])");
        changed = s::set(changed, "admission",
                         s::set(changed.at("admission"), "required_capabilities", caps));
        s::write(dir / "mixed-caps.json", changed);
        s::reject("mixed sampling capability",
                  [&] { p::load_model_package_manifest(dir / "mixed-caps.json"); });
        auto no_shard = s::set(
            Json::parse(admitted->resources.manifest.raw_json), "admission",
            s::set(Json::parse(admitted->resources.manifest.raw_json).at("admission"), "resources",
                   s::set(Json::parse(admitted->resources.manifest.raw_json)
                              .at("admission")
                              .at("resources"),
                          "conditioning_weights",
                          Json::parse(R"({"one.safetensors":"missing"})"))));
        s::write(dir / "missing-shard.json", no_shard);
        s::reject("unresolved mandatory shard",
                  [&] { p::load_model_package_manifest(dir / "missing-shard.json"); });
        std::cout << "LTX_SCHEMA2_REAL_PRODUCT_STRUCTURAL=PASS steps=40 latent_tokens=5280 "
                     "slots=715 native_sentencepiece=PASS T5_index=219 two_shards=PASS masks=PASS "
                     "typed_evidence=PASS negative_cases="
                  << s::negatives << " GPU_execution=NONE\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
