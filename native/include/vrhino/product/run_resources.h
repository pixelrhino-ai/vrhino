#pragma once

#include <filesystem>
#include <optional>

#include "vrhino/json.h"
#include "vrhino/memory.h"

namespace vrhino::product {

// Machine-local resource constraints, independent of model/package semantics.
// Omission preserves the package budget; a declaration may only tighten it.
struct RunResources {
    std::optional<size_t> weight_cache_budget_bytes;
};

RunResources parse_run_resources(const Json& declaration);
RunResources load_run_resources(const std::filesystem::path& path);
MemoryBudget constrain_run_memory(const MemoryBudget& declared,
                                  const RunResources& resources);

}  // namespace vrhino::product
