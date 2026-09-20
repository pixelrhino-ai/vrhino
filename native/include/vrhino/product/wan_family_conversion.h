#pragma once
#include "vrhino/product/converter.h"
#include "vrhino/package_declaration.h"
#include "vrhino/product/program_declaration.h"

namespace vrhino::product {
// Product/Converter APIs. None of these select models inside Shared Runtime.
Json lower_wan_source_config(const Json& config, const Json& patch, int64_t text_limit);
std::vector<TensorMapping> map_wan_parameters(const TensorSource& source,
    const Json& canonical, uint32_t binding);

Json verify_wan_family_structure(const std::filesystem::path& vrm);
Json convert_wan_family_package(const std::filesystem::path& model_root,
    const std::filesystem::path& semantic_source_root,
    const std::filesystem::path& specification,
    const std::filesystem::path& output_directory,
    const WorkProgressCallback& progress = {});
} // namespace vrhino::product
