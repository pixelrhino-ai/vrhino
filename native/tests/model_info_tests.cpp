#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/product/info.h"
#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

product::ResolvedRunnableModel load_projection_fixture(const fs::path& manifest_path) {
    product::ResolvedRunnableModel result;
    result.manifest = product::load_model_package_manifest(manifest_path);
    result.manifest_path = fs::path("/private/cache/path") /
                           result.manifest.identity.version / "vrhino-model.json";
    for (const product::ArtifactDeclaration& declaration : result.manifest.artifacts) {
        result.artifacts.emplace(
            declaration.id,
            product::ResolvedArtifact{
                declaration, fs::path("/private/cas/path") / declaration.sha256});
    }
    return result;
}

const vrhino::Json& declaration(const vrhino::Json& schema,
                                const std::string& group,
                                const std::string& name) {
    for (const vrhino::Json& value : schema.at(group).array())
        if (value.at("name").string() == name) return value;
    throw std::runtime_error("missing Product declaration: " + name);
}

struct SuccessorCase {
    fs::path relative_manifest;
    std::string reference;
    std::string family;
    uint64_t seed;
    uint64_t width;
    uint64_t height;
    uint64_t frames;
    uint64_t fps;
    uint64_t steps;
    double guidance;
};

}  // namespace

int main() {
    try {
        const fs::path specs = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        const std::vector<SuccessorCase> successors = {
            {"ltx_v0_9_1/successors/1.1.1/vrhino-model.json",
             "vrhino/ltx-video-v0.9.1:1.1.1", "text_to_video", 5703,
             704, 480, 121, 25, 40, 3.0},
            {"wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json",
             "vrhino/wan2.1-t2v-1.3b:1.0.1", "text_to_video", 5701,
             832, 480, 81, 16, 50, 5.0},
            {"mochi_1_preview/successors/1.0.1/vrhino-model.json",
             "vrhino/mochi-1-preview:1.0.1", "text_to_video", 11001,
             848, 480, 163, 30, 64, 6.0},
            {"public_musetalk_v15/successors/1.0.1/vrhino-model.json",
             "vrhino/musetalk-v1.5:1.0.1", "lip_sync", 11001,
             0, 0, 0, 25, 0, 0.0},
            {"public_latentsync_16/successors/1.0.1/vrhino-model.json",
             "vrhino/latentsync-1.6:1.0.1", "lip_sync", 1247,
             0, 0, 0, 25, 20, 1.5},
        };

        for (const SuccessorCase& item : successors) {
            const product::ResolvedRunnableModel model =
                load_projection_fixture(specs / item.relative_manifest);
            const vrhino::Json projected = product::build_model_info(model);
            const std::string first = projected.serialize();
            const std::string second = product::build_model_info(model).serialize();
            require_test(first == second, "model info JSON is nondeterministic");
            require_test(vrhino::Json::parse(first).serialize() == first,
                         "model info JSON is not canonical and parseable");
            require_test(first.find(VRHINO_TEST_SOURCE_ROOT) == std::string::npos &&
                             first.find("/private/") == std::string::npos &&
                             first.find("/nonpublic-test-home") == std::string::npos,
                         "model info JSON leaked a local path");

            require_test(projected.at("schema_version").integer() == 1,
                         "model info envelope version drift");
            const vrhino::Json& identity = projected.at("model");
            require_test(identity.at("reference").string() == item.reference &&
                             identity.at("namespace").string() == "vrhino" &&
                             !identity.at("architecture").string().empty(),
                         "model identity projection mismatch");
            const vrhino::Json& product_json = projected.at("product");
            require_test(product_json.at("family").string() == item.family &&
                             product_json.at("input_schema").is_object() &&
                             product_json.at("frozen_profile").is_object(),
                         "Product projection missing canonical schema");

            const vrhino::Json raw = vrhino::Json::parse(model.manifest.raw_json);
            require_test(product_json.at("input_schema").serialize() ==
                             raw.at("product").at("input_schema").serialize() &&
                             product_json.at("frozen_profile").serialize() ==
                             raw.at("product").at("frozen_profile").serialize(),
                         "CLI projection reconstructed or changed Product metadata");

            const vrhino::Json& schema = product_json.at("input_schema");
            require_test(schema.at("schema").string() ==
                             product::kProductInputSchemaV1,
                         "Product schema identity drift");
            const vrhino::Json& seed = declaration(schema, "parameters", "seed");
            const vrhino::Json& output = declaration(schema, "outputs", "output");
            require_test(seed.at("default").integer() ==
                             static_cast<int64_t>(item.seed) &&
                             seed.at("validation").at("maximum").string() ==
                                 "18446744073709551615",
                         "seed projection lost its default or uint64 bound");
            require_test(!output.at("required").boolean() &&
                             output.at("default").string() == "output.mp4",
                         "output projection changed optional default semantics");

            const vrhino::Json& frozen = product_json.at("frozen_profile");
            require_test(frozen.at("output").at("fps").at("numerator").integer() ==
                             static_cast<int64_t>(item.fps) &&
                             frozen.at("output").at("fps").at("denominator").integer() == 1,
                         "frozen rational FPS projection mismatch");
            if (item.family == "text_to_video") {
                require_test(declaration(schema, "inputs", "prompt")
                                     .at("required").boolean() &&
                                 frozen.at("output").at("width").integer() ==
                                     static_cast<int64_t>(item.width) &&
                                 frozen.at("output").at("height").integer() ==
                                     static_cast<int64_t>(item.height) &&
                                 frozen.at("output").at("frames").integer() ==
                                     static_cast<int64_t>(item.frames) &&
                                 frozen.at("sampling").at("steps").integer() ==
                                     static_cast<int64_t>(item.steps) &&
                                 frozen.at("sampling").at("guidance_scale").number() ==
                                     item.guidance,
                             "text-to-video frozen projection mismatch");
                require_test(projected.find("qualification") != nullptr,
                             "structured TTV qualification was not separated");
            } else {
                require_test(declaration(schema, "inputs", "video")
                                     .at("required").boolean() &&
                                 declaration(schema, "inputs", "audio")
                                     .at("validation").at("minimum_duration_ms")
                                     .integer() == 40 &&
                                 frozen.at("output").at("duration").string() ==
                                     "audio_derived",
                             "lip-sync media projection mismatch");
            }
            require_test(projected.at("installation").at("installed").boolean() &&
                             projected.at("artifacts").at("declared_count").integer() ==
                                 static_cast<int64_t>(model.manifest.artifacts.size()) &&
                             projected.at("artifacts").at("present_count").integer() ==
                                 static_cast<int64_t>(model.artifacts.size()),
                         "artifact or installation projection mismatch");
            require_test(projected.at("compatibility")
                                     .at("package_schema_version").integer() == 1 &&
                             !projected.at("source").at("repository").string().empty() &&
                             !projected.at("license").at("identifier").string().empty(),
                         "package metadata projection incomplete");
        }

        const std::vector<fs::path> legacy_manifests = {
            "ltx_v0_9_1/vrhino-model.json",
            "wan2_1_t2v_1_3b/vrhino-model.json",
            "mochi_1_preview/vrhino-model.json",
            "public_musetalk_v15/vrhino-model.json",
            "public_latentsync_16/vrhino-model.json",
        };
        for (const fs::path& relative : legacy_manifests) {
            const vrhino::Json projected = product::build_model_info(
                load_projection_fixture(specs / relative));
            require_test(projected.at("product").at("input_schema").is_null() &&
                             projected.at("product").at("frozen_profile").is_null(),
                         "legacy package was silently upgraded to ProductInputSchema v1");
        }

        const product::ResolvedRunnableModel latent = load_projection_fixture(
            specs / "public_latentsync_16/successors/1.0.1/vrhino-model.json");
        const std::string latent_json = product::build_model_info(latent).serialize();
        for (const char* forbidden : {
                 "alignment", "detector", "\"tta\"", "tensor", "\"rng\""})
            require_test(latent_json.find(forbidden) == std::string::npos,
                         "LatentSync internal field leaked: " +
                             std::string(forbidden));
        require_test(latent_json.find("\"method\":\"ddim\"") != std::string::npos &&
                         latent_json.find("\"prediction\":\"epsilon\"") !=
                             std::string::npos,
                     "LatentSync curated frozen facts are absent");

        const vrhino::Json serializer_fixture(vrhino::Json::Value(vrhino::Json::Object{
            {"z", vrhino::Json(vrhino::Json::Value(std::string("line\n\"quoted\"")))},
            {"a", vrhino::Json(vrhino::Json::Value(3.0))},
        }));
        require_test(serializer_fixture.serialize() ==
                         "{\"a\":3.0,\"z\":\"line\\n\\\"quoted\\\"\"}",
                     "canonical JSON field order/escaping drift");
        bool invalid_utf8_rejected = false;
        try {
            (void)vrhino::Json(vrhino::Json::Value(std::string("\xff", 1))).serialize();
        } catch (const vrhino::Error&) {
            invalid_utf8_rejected = true;
        }
        require_test(invalid_utf8_rejected, "invalid UTF-8 was serialized");

        std::cout << "model info JSON projection tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model info JSON projection tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
