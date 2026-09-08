#pragma once

#include <map>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct PoseEstimator2DResult {
    Tensor simcc_x;
    Tensor simcc_y;
};

struct PoseEstimator2DObservation {
    std::map<std::string, Tensor> tensors;
};

// Generic frozen top-down 2D pose component. Image affine preprocessing,
// flip-TTA combination, SimCC decode, and product geometry policy stay outside
// Shared Runtime.
class PoseEstimator2DComponentExecutor {
public:
    PoseEstimator2DComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    PoseEstimator2DResult execute(
        const Json& graph, const Tensor& normalized_image,
        PoseEstimator2DObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
