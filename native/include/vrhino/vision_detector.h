#pragma once

#include <map>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct VisionDetectorScaleOutput {
    Tensor confidence_logits;
    Tensor localization;
};

struct VisionDetectorResult {
    std::vector<VisionDetectorScaleOutput> scales;
};

struct DenseVisionDetectorResult {
    Tensor regressors;
    Tensor classification_logits;
};

struct VisionDetectorObservation {
    std::map<std::string, Tensor> tensors;
};

// Generic frozen multi-scale convolutional vision detector. It returns raw
// neural head tensors; anchor decoding, thresholds, NMS, and ROI selection are
// deliberately outside Shared Runtime.
class VisionDetectorComponentExecutor {
public:
    VisionDetectorComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    VisionDetectorResult execute(const Json& graph, const Tensor& image,
                                 VisionDetectorObservation* observation = nullptr);
    DenseVisionDetectorResult execute_dense(
        const Json& graph, const Tensor& image,
        VisionDetectorObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
