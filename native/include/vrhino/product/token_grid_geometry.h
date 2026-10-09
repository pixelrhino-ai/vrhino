#pragma once
#include "vrhino/bundle.h"
#include "vrhino/json.h"
#include <cstdint>

namespace vrhino::product {
struct TokenGridGeometry {
    int64_t width, height, frames;
    std::vector<int64_t> latent_shape, expected_video_shape;
    TensorBundle runtime_inputs;
    uint64_t tensor_bytes_bound;
};
// A single closed host lowering; neither expressions nor operator dispatch.
// The caller supplies its already admitted resource bound before allocation.
TokenGridGeometry admit_token_grid_geometry(const Json &declaration, int64_t width, int64_t height,
                                            int64_t frames, uint64_t seed,
                                            uint64_t tensor_bytes_bound);
void validate_token_grid_geometry(const TokenGridGeometry &geometry);
} // namespace vrhino::product
