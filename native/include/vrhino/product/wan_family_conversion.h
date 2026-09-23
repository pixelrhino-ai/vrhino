#pragma once
#include "vrhino/product/converter.h"
#include "vrhino/package_declaration.h"
#include "vrhino/product/program_declaration.h"

namespace vrhino::product {
// Product/Converter APIs. None of these select models inside Shared Runtime.
Json lower_wan_source_config(const Json& config, const Json& patch, int64_t text_limit);
std::vector<TensorMapping> map_wan_parameters(const TensorSource& source,
    const Json& canonical, uint32_t binding);

Json verify_wan_family_structure(const std::filesystem::path& vrm,
    const std::string& expected_manifest_sha256 =
        "4ea6995c08502841565b9dd2e612af0dd97ddfb49ab17721339fc5c077338911",
    const std::string& expected_source_revision =
        "3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7");
Json convert_wan_family_package(const std::filesystem::path& model_root,
    const std::filesystem::path& semantic_source_root,
    const std::filesystem::path& specification,
    const std::filesystem::path& output_directory,
    const WorkProgressCallback& progress = {},
    const std::string& source_contract_name = "source-contract.json",
    const std::string& source_manifest_name = "source-manifest.tsv");
ImportResult import_wan22_a14b_product(
    const std::filesystem::path& qualified_source_directory,
    LocalModelCache& cache,
    const ImportOptions& options = {});
} // namespace vrhino::product
