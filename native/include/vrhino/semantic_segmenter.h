#pragma once

#include <map>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct SemanticSegmenter2DResult { Tensor logits; };
struct SemanticSegmenter2DObservation { std::map<std::string, Tensor> tensors; };

// Generic frozen 2D semantic segmentation component. Image preprocessing,
// label decoding, and product mask policy remain outside Shared Runtime.
class SemanticSegmenter2DComponentExecutor {
public:
    SemanticSegmenter2DComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    SemanticSegmenter2DResult execute(
        const Json& graph, const Tensor& normalized_image,
        SemanticSegmenter2DObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
