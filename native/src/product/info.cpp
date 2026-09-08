#include "vrhino/product/info.h"

#include <limits>

namespace vrhino::product {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,
                            "model info projection: " + message);
}

Json text(const std::string& value) {
    return Json(Json::Value(value));
}

Json integer(const int64_t value) {
    return Json(Json::Value(value));
}

Json boolean(const bool value) {
    return Json(Json::Value(value));
}

Json object(Json::Object value) {
    return Json(Json::Value(std::move(value)));
}

Json array(Json::Array value) {
    return Json(Json::Value(std::move(value)));
}

Json lossless_uint64(const uint64_t value) {
    if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return integer(static_cast<int64_t>(value));
    return text(std::to_string(value));
}

Json optional_uint64(const std::optional<uint64_t>& value) {
    return value ? lossless_uint64(*value) : Json();
}

const Json& required_object_field(const Json& value, const std::string& name,
                                  const std::string& context) {
    const Json* field = value.find(name);
    if (field == nullptr || !field->is_object())
        fail(context + "." + name + " must be an object");
    return *field;
}

Json project_product_contract(const ModelPackageManifest& manifest,
                              const Json& raw_manifest) {
    Json::Object product{
        {"family", text(manifest.product.family)},
        {"status", text(manifest.product.status)},
        {"workflow_identity", manifest.product.workflow_identity.empty()
            ? Json() : text(manifest.product.workflow_identity)},
    };
    if (!manifest.product.input_schema) {
        product.emplace("frozen_profile", Json());
        product.emplace("input_schema", Json());
        return object(std::move(product));
    }

    const Json& raw_product = required_object_field(
        raw_manifest, "product", "manifest");
    const Json* input_schema = raw_product.find("input_schema");
    const Json* frozen_profile = raw_product.find("frozen_profile");
    if (input_schema == nullptr || !input_schema->is_object() ||
        frozen_profile == nullptr || !frozen_profile->is_object())
        fail("verified Product contract is absent from raw manifest");
    product.emplace("input_schema", *input_schema);
    product.emplace("frozen_profile", *frozen_profile);
    return object(std::move(product));
}

Json::Array project_qualification(const Json& raw_manifest,
                                  const std::string& default_preset) {
    const Json* hardware = raw_manifest.find("hardware");
    if (hardware == nullptr || !hardware->is_object()) return {};
    const Json* presets = hardware->find("presets");
    if (presets == nullptr || !presets->is_object()) return {};
    const Json* preset = presets->find(default_preset);
    if (preset == nullptr || !preset->is_object()) return {};
    const Json* qualifications = preset->find("qualifications");
    if (qualifications == nullptr) return {};
    if (!qualifications->is_array()) fail("hardware qualifications must be an array");

    Json::Array result;
    for (const Json& qualification : qualifications->array()) {
        if (!qualification.is_object())
            fail("hardware qualification must be an object");
        Json::Object projected;
        for (const char* name : {"gpu", "scope", "status"}) {
            if (const Json* value = qualification.find(name); value != nullptr) {
                if (!value->is_string())
                    fail("hardware qualification " + std::string(name) +
                         " must be a string");
                projected.emplace(name, *value);
            }
        }
        if (!projected.empty()) result.push_back(object(std::move(projected)));
    }
    return result;
}

std::optional<std::string> raw_optional_string(const Json& object_value,
                                               const std::string& name,
                                               const std::string& context) {
    const Json* value = object_value.find(name);
    if (value == nullptr) return std::nullopt;
    if (!value->is_string()) fail(context + "." + name + " must be a string");
    return value->string();
}

}  // namespace

Json build_model_info(const ResolvedRunnableModel& model) {
    const ModelPackageManifest& manifest = model.manifest;
    const Json raw_manifest = Json::parse(manifest.raw_json);

    Json::Object document;
    document.emplace("schema_version", integer(kModelInfoJsonSchemaVersion));
    document.emplace("model", object({
        {"architecture", text(manifest.identity.architecture)},
        {"name", text(manifest.identity.name)},
        {"namespace", text(manifest.identity.name_space)},
        {"publisher", text(manifest.identity.publisher)},
        {"reference", text(manifest.identity.reference())},
        {"version", text(manifest.identity.version)},
    }));
    document.emplace("product", project_product_contract(manifest, raw_manifest));
    document.emplace("compatibility", object({
        {"default_preset", text(manifest.default_preset)},
        {"package_schema_version", integer(manifest.schema_version)},
        {"runtime_contract", text(manifest.runtime_contract)},
        {"vrm", object({
            {"format_major", integer(manifest.vrm_format_major)},
            {"format_minor", integer(manifest.vrm_format_minor)},
            {"metadata_schema", integer(manifest.vrm_metadata_schema)},
        })},
    }));
    document.emplace("admission", object({
        {"minimum_vram_bytes", optional_uint64(manifest.minimum_vram_bytes)},
        {"recommended_vram_bytes", optional_uint64(manifest.recommended_vram_bytes)},
    }));

    Json::Array qualifications = project_qualification(
        raw_manifest, manifest.default_preset);
    if (!qualifications.empty())
        document.emplace("qualification", array(std::move(qualifications)));

    document.emplace("distribution", object({
        {"public_distribution", text(manifest.product.public_distribution)},
    }));
    document.emplace("source", object({
        {"repository", text(manifest.source_repository)},
        {"revision", text(manifest.source_revision)},
    }));

    Json::Object license{
        {"artifact", text(manifest.license_artifact_id)},
        {"identifier", text(manifest.license_identifier)},
    };
    const Json& raw_license = required_object_field(
        raw_manifest, "license", "manifest");
    if (const std::optional<std::string> notice = raw_optional_string(
            raw_license, "upstream_notice", "manifest.license"))
        license.emplace("upstream_notice", text(*notice));
    document.emplace("license", object(std::move(license)));

    document.emplace("artifacts", object({
        {"component_count", lossless_uint64(manifest.components.size())},
        {"declared_count", lossless_uint64(manifest.artifacts.size())},
        {"logical_size_bytes", lossless_uint64(manifest.logical_size())},
        {"present_count", lossless_uint64(model.artifacts.size())},
    }));
    document.emplace("installation", object({
        {"installed", boolean(true)},
    }));
    return object(std::move(document));
}

}  // namespace vrhino::product
