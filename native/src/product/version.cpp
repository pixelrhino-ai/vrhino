#include "vrhino/product/version.h"

#include <sstream>
#include <utility>

#include "vrhino/product/model_package.h"

#ifndef VRHINO_PRODUCT_VERSION
#error "VRHINO_PRODUCT_VERSION must come from the canonical release VERSION file"
#endif
#ifndef VRHINO_GIT_HEAD
#define VRHINO_GIT_HEAD "unknown"
#endif

namespace vrhino::product {
namespace {

Json text(std::string value) {
    return Json(Json::Value(std::move(value)));
}

Json integer(const int64_t value) {
    return Json(Json::Value(value));
}

Json object(Json::Object value) {
    return Json(Json::Value(std::move(value)));
}

}  // namespace

VersionInfo current_version_info() {
    return VersionInfo{
        VRHINO_PRODUCT_VERSION,
        VRHINO_GIT_HEAD,
        kCudaRuntimeContract,
        kVrmFormatMajor,
        kVrmFormatMinor,
        kVrmMetadataSchema,
        kModelPackageSchemaVersion,
    };
}

Json build_version_info() {
    const VersionInfo version = current_version_info();
    return object({
        {"git_head", text(version.git_head)},
        {"model_package_schema", integer(version.model_package_schema)},
        {"runtime_contract", text(version.runtime_contract)},
        {"schema_version", integer(1)},
        {"version", text(version.version)},
        {"vrm", object({
            {"format_major", integer(version.vrm_format_major)},
            {"format_minor", integer(version.vrm_format_minor)},
            {"metadata_schema", integer(version.vrm_metadata_schema)},
        })},
    });
}

std::string format_cli_version(const VersionInfo& version) {
    std::ostringstream output;
    output << "VRhino " << version.version << '\n'
           << "Git HEAD: " << version.git_head << '\n'
           << "CUDA contract: " << version.runtime_contract << '\n'
           << "VRM: " << version.vrm_format_major << '.'
           << version.vrm_format_minor << "/schema"
           << version.vrm_metadata_schema << '\n'
           << "Model Package schema: " << version.model_package_schema << '\n';
    return output.str();
}

}  // namespace vrhino::product
