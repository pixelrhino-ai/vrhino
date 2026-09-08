#pragma once

#include "vrhino/json.h"
#include "vrhino/product/model_package.h"

namespace vrhino::product {

inline constexpr int64_t kModelInfoJsonSchemaVersion = 1;

// Builds the stable, transport-neutral model information projection. CLI and
// future read-only transports serialize this data; they do not reconstruct the
// Product contract independently.
Json build_model_info(const ResolvedRunnableModel& model);

}  // namespace vrhino::product
