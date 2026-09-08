#include "vrhino/product/model_list.h"

#include <algorithm>
#include <utility>

namespace vrhino::product {
namespace {

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

}  // namespace

std::vector<LocalModelSummary> list_local_models(const LocalModelCache& cache) {
    std::vector<LocalModelSummary> result;
    for (const InstalledPackage& installed : cache.list()) {
        const ModelPackageManifest manifest =
            load_model_package_manifest(installed.manifest_path);
        result.push_back(LocalModelSummary{
            manifest.identity.reference(),
            manifest.identity.name_space,
            manifest.identity.name,
            manifest.identity.version,
            manifest.identity.architecture,
            manifest.product.family,
            true,
        });
    }
    std::sort(result.begin(), result.end(),
              [](const LocalModelSummary& left, const LocalModelSummary& right) {
                  return left.reference < right.reference;
              });
    return result;
}

Json build_local_model_list(const LocalModelCache& cache) {
    Json::Array models;
    for (const LocalModelSummary& model : list_local_models(cache)) {
        models.push_back(object({
            {"architecture", text(model.architecture)},
            {"installed", boolean(model.installed)},
            {"name", text(model.name)},
            {"namespace", text(model.name_space)},
            {"product_family", text(model.product_family)},
            {"reference", text(model.reference)},
            {"version", text(model.version)},
        }));
    }
    return object({
        {"models", array(std::move(models))},
        {"schema_version", integer(1)},
    });
}

}  // namespace vrhino::product
