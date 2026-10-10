#pragma once
#include "vrhino/product/converter.h"
namespace vrhino::product {
std::filesystem::path checked_ltx_source(const std::filesystem::path &root,
                                         const std::filesystem::path &relative);
std::vector<TensorMapping>
load_ltx_source_mappings(const std::filesystem::path &path);
Json::Object ltx_source_bindings(const std::vector<TensorMapping> &mappings,
                                 const std::string &prefix);
void validate_ltx_source_mappings(const SafeTensorReader &reader,
                                  const std::vector<TensorMapping> &mappings);
} // namespace vrhino::product
