#include "vrhino/product/declared_run.h"
#include "vrhino/product/qualification_precision.h"
#include <cmath>
#include <fstream>
#include <limits>
#include <set>

namespace vrhino::product {
namespace {
[[noreturn]] void invalid(const std::string& message) {
    throw ModelPackageError(ModelPackageErrorCode::PackageInvalid, message);
}
void fields(const Json& j, const std::set<std::string>& required,
            const std::set<std::string>& optional = {}) {
    if (!j.is_object()) invalid("Run declaration must be an object");
    for (const auto& key : required)
        if (!j.find(key)) invalid("Missing run declaration field: " + key);
    for (const auto& [key, ignored] : j.object()) {
        (void)ignored;
        if (!required.contains(key) && !optional.contains(key))
            invalid("Unsupported run declaration field: " + key);
    }
}
size_t bytes(const Json& value) {
    if (!value.is_int() || value.integer() < 0 ||
        static_cast<uint64_t>(value.integer()) > std::numeric_limits<size_t>::max())
        invalid("Run memory bytes must be a nonnegative host-sized integer");
    return static_cast<size_t>(value.integer());
}
Json integer(uint64_t n) {
    if (n <= static_cast<uint64_t>(INT64_MAX)) return Json(static_cast<int64_t>(n));
    return Json(std::to_string(n));
}
}
DeclaredTextRun lower_declared_text_run(const ResolvedRunnableModel& model,
                                       const RunOptions& options) {
    const auto& manifest = model.manifest;
    const auto& product = manifest.product;
    if (manifest.schema_version != 2 || product.family != "text_to_video" ||
        !product.input_schema || !product.frozen_profile || !product.frozen_profile->sampling ||
        !product.frozen_profile->sampling->program_artifact || product.execution_artifact_id.empty())
        invalid("Package has no canonical text Product run declaration");
    if ((!options.model_reference.empty() && options.model_reference != manifest.identity.reference()) ||
        (!options.preset.empty() && options.preset != manifest.default_preset) ||
        !options.video.empty() || !options.audio.empty() || options.prompt.empty() ||
        options.prompt.size() > 65536 || options.prompt.find('\0') != std::string::npos)
        throw ModelPackageError(ModelPackageErrorCode::InvalidInput, "Invalid declared text Product request");
    validate_product_contract_for_family(product.family, product.workflow_identity,
                                        *product.input_schema, *product.frozen_profile);
    const auto found = model.artifacts.find(product.execution_artifact_id);
    if (found == model.artifacts.end()) invalid("Missing text Product run artifact");
    const auto& artifact = found->second;
    const auto& a = artifact.declaration;
    if (a.id != product.execution_artifact_id || !a.required || a.role != "product.execution" ||
        !a.size || a.size > 65536)
        invalid("Invalid text Product run artifact declaration");
    std::ifstream input(artifact.path, std::ios::binary);
    if (!input) invalid("Cannot open text Product run declaration");
    std::string data(static_cast<size_t>(a.size) + 1, '\0');
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!input.eof() || input.gcount() != static_cast<std::streamsize>(a.size))
        invalid("Text Product run declaration size mismatch");
    data.resize(static_cast<size_t>(a.size));
    if (sha256_bytes(data) != a.sha256) invalid("Text Product run declaration SHA256 mismatch");
    const auto declaration = Json::parse(data, JsonParseLimits{});
    fields(declaration, {"schema", "negative_prompt", "precision_artifact", "memory", "media_range"});
    if (declaration.at("schema").string() != "vrhino.product-text-run.v1")
        invalid("Unsupported text Product run declaration");
    const auto& negative = declaration.at("negative_prompt").string();
    if (negative.size() > 65536 || negative.find('\0') != std::string::npos)
        invalid("Invalid negative conditioning default");
    DeclaredTextRun result;
    result.precision_artifact = declaration.at("precision_artifact").string();
    const auto precision = model.artifacts.find(result.precision_artifact);
    if (precision == model.artifacts.end() || !precision->second.declaration.required ||
        precision->second.declaration.role != "precision.policy")
        invalid("Run precision reference must identify a required precision policy artifact");
    // Parse the immutable declaration during structural lowering too. This
    // grants no numerical admission and creates no device resources.
    (void)load_declared_product_precision(model, result.precision_artifact);
    const auto& memory = declaration.at("memory");
    fields(memory, {"device_budget_bytes", "host_budget_bytes", "workspace_bytes", "safety_margin_bytes"},
                  {"weight_cache_budget_bytes"});
    result.memory = MemoryBudget{bytes(memory.at("device_budget_bytes")), 0,
        bytes(memory.at("host_budget_bytes")), bytes(memory.at("workspace_bytes")),
        bytes(memory.at("safety_margin_bytes"))};
    if (const auto* cap = memory.find("weight_cache_budget_bytes")) {
        result.memory.weight_cache_budget_bytes = bytes(*cap);
        if (!result.memory.weight_cache_budget_bytes) invalid("Explicit cache cap must be positive");
    }
    result.memory = constrain_run_memory(result.memory, options.resources);
    const auto& range = declaration.at("media_range").array();
    if (range.size() != 2) invalid("Media range must contain two bounds");
    result.video_minimum = static_cast<float>(range[0].number());
    result.video_maximum = static_cast<float>(range[1].number());
    if (!std::isfinite(result.video_minimum) || !std::isfinite(result.video_maximum) ||
        !(result.video_minimum < result.video_maximum)) invalid("Invalid media range");
    const auto& frozen = *product.frozen_profile;
    if (frozen.output.fps.numerator > static_cast<uint64_t>(INT64_MAX)) invalid("Unsupported media FPS");
    result.fps = static_cast<int64_t>(frozen.output.fps.numerator);
    const auto* seed = product.input_schema->find_parameter("seed");
    const uint64_t selected_seed = options.seed.value_or(std::get<uint64_t>(seed->default_value));
    result.request = Json(Json::Object{{"schema", Json(std::string("vrhino.product-text-request.v1"))},
        {"prompt", Json(options.prompt)}, {"negative_prompt", Json(negative)}, {"seed", integer(selected_seed)},
        {"width", integer(*frozen.output.width)}, {"height", integer(*frozen.output.height)},
        {"frames", integer(*frozen.output.frames)}});
    return result;
}
} // namespace vrhino::product
