#include "vrhino/product/run_request.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <utility>

namespace vrhino::product {
namespace fs = std::filesystem;
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw RunRequestError(RunRequestErrorCode::InvalidRequest, message);
}

const Json::Object& optional_group(const Json& root, const std::string& name) {
    static const Json::Object empty;
    const Json* value = root.find(name);
    if (value == nullptr) return empty;
    if (!value->is_object()) invalid(name + " must be an object");
    return value->object();
}

uint64_t decimal_uint64(const Json& value, const std::string& context) {
    if (value.is_int()) {
        if (value.integer() < 0)
            invalid(context + " must be a non-negative integer");
        return static_cast<uint64_t>(value.integer());
    }
    if (!value.is_string() || value.string().empty())
        invalid(context +
                " must be an integer or canonical decimal uint64 string");
    const std::string& text = value.string();
    if ((text.size() > 1 && text.front() == '0') ||
        !std::all_of(text.begin(), text.end(), [](const unsigned char byte) {
            return byte >= '0' && byte <= '9';
        })) {
        invalid(context + " is not a canonical decimal uint64 string");
    }
    uint64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        invalid(context + " exceeds the uint64 domain");
    return result;
}

Json default_json(const ProductFieldDeclaration& declaration) {
    if (std::holds_alternative<std::string>(declaration.default_value))
        return Json(Json::Value(std::get<std::string>(
            declaration.default_value)));
    if (std::holds_alternative<uint64_t>(declaration.default_value)) {
        const uint64_t value = std::get<uint64_t>(declaration.default_value);
        if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return Json(Json::Value(static_cast<int64_t>(value)));
        return Json(Json::Value(std::to_string(value)));
    }
    invalid("Product declaration has no usable default");
}

void assign_field(const ProductFieldDeclaration& declaration,
                  const Json& value, const std::string& context,
                  const bool explicit_value, RunOptions& options) {
    switch (declaration.type) {
        case ProductValueType::Text: {
            if (!value.is_string() || value.string().find('\0') != std::string::npos)
                invalid(context + " must be text");
            if (declaration.validation.min_length &&
                value.string().size() < *declaration.validation.min_length)
                invalid(context + " is shorter than its declared minimum");
            options.prompt = value.string();
            return;
        }
        case ProductValueType::Integer: {
            const uint64_t parsed = decimal_uint64(value, context);
            if ((declaration.validation.minimum &&
                 parsed < *declaration.validation.minimum) ||
                (declaration.validation.maximum &&
                 parsed > *declaration.validation.maximum))
                invalid(context + " is outside its declared range");
            options.seed = parsed;
            return;
        }
        case ProductValueType::MediaVideo:
        case ProductValueType::MediaAudio:
        case ProductValueType::MediaMp4: {
            if (!value.is_string() || value.string().empty() ||
                value.string().find('\0') != std::string::npos)
                invalid(context + " must be a non-empty local path");
            const fs::path path(value.string());
            // The Product-level default output.mp4 remains canonical metadata,
            // but an omitted API output is replaced by the JobManager's unique
            // absolute destination. Only client-supplied paths reach this rule.
            if (!explicit_value &&
                declaration.type == ProductValueType::MediaMp4)
                return;
            if (!path.is_absolute())
                invalid(context + " must be an absolute local path");
            if (declaration.type != ProductValueType::MediaMp4) {
                std::error_code error;
                if (!fs::is_regular_file(path, error) || error)
                    invalid(context +
                            " must reference an existing regular local file");
            }
            if (declaration.type == ProductValueType::MediaVideo)
                options.video = path;
            else if (declaration.type == ProductValueType::MediaAudio)
                options.audio = path;
            else
                options.output = path;
            return;
        }
    }
    invalid(context + " has an unsupported Product type");
}

void map_group(const std::string& group_name,
               const std::vector<ProductFieldDeclaration>& declarations,
               const Json::Object& supplied, RunOptions& options) {
    std::set<std::string> admitted;
    for (const ProductFieldDeclaration& declaration : declarations)
        admitted.insert(declaration.name);
    for (const auto& [name, ignored] : supplied) {
        (void)ignored;
        if (!admitted.contains(name))
            invalid(group_name + " contains unknown field: " + name);
    }

    for (const ProductFieldDeclaration& declaration : declarations) {
        const auto found = supplied.find(declaration.name);
        if (found != supplied.end()) {
            assign_field(declaration, found->second,
                         group_name + "." + declaration.name, true, options);
            continue;
        }
        if (declaration.required)
            invalid(group_name + " is missing required field: " +
                    declaration.name);
        if (declaration.has_default()) {
            const Json value = default_json(declaration);
            assign_field(declaration, value,
                         group_name + "." + declaration.name, false, options);
        }
    }
}

}  // namespace

ProductRunDocument parse_product_run_document(const Json& value) {
    if (!value.is_object()) invalid("run request must be an object");
    static const std::set<std::string> admitted = {
        "inputs", "model", "outputs", "parameters"};
    for (const auto& [name, ignored] : value.object()) {
        (void)ignored;
        if (!admitted.contains(name))
            invalid("run request contains unknown field: " + name);
    }
    const Json* model = value.find("model");
    if (model == nullptr || !model->is_string() || model->string().empty() ||
        model->string().size() > 192 ||
        model->string().find('\0') != std::string::npos)
        invalid("model must be a bounded canonical package reference");

    ProductRunDocument result;
    result.model_reference = model->string();
    result.inputs = optional_group(value, "inputs");
    result.parameters = optional_group(value, "parameters");
    result.outputs = optional_group(value, "outputs");
    return result;
}

RunOptions map_product_run_options(const ResolvedRunnableModel& model,
                                   const ProductRunDocument& request) {
    if (!model.manifest.product.input_schema ||
        model.manifest.product.input_schema->identity !=
            kProductInputSchemaV1) {
        throw RunRequestError(
            RunRequestErrorCode::ModelUnavailable,
            "model has no Native API ProductInputSchema v1 contract");
    }

    RunOptions result;
    result.model_reference = model.manifest.identity.reference();
    const ProductInputSchema& schema = *model.manifest.product.input_schema;
    map_group("inputs", schema.inputs, request.inputs, result);
    map_group("parameters", schema.parameters, request.parameters, result);
    map_group("outputs", schema.outputs, request.outputs, result);
    result.overwrite = false;
    result.debug = false;
    return result;
}

}  // namespace vrhino::product
