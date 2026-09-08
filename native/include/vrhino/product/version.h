#pragma once

#include <cstdint>
#include <string>

#include "vrhino/json.h"

namespace vrhino::product {

struct VersionInfo {
    std::string version;
    std::string git_head;
    std::string runtime_contract;
    int64_t vrm_format_major = 0;
    int64_t vrm_format_minor = 0;
    int64_t vrm_metadata_schema = 0;
    int64_t model_package_schema = 0;
};

VersionInfo current_version_info();
Json build_version_info();
std::string format_cli_version(const VersionInfo& version);

}  // namespace vrhino::product
