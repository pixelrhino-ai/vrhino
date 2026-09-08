#pragma once

#include <functional>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct TemporalConditionalUNetObservation {
    std::function<void(const std::string&, const Tensor&)> capture;
};

struct TemporalConditionalUNetResult {
    Tensor epsilon;
};

// Model-neutral executor for a temporal conditional 2D UNet. Spatial
// convolution is performed over flattened B*F frames; temporal behavior is a
// composition of generic reshape/permute/attention/linear operations.
class TemporalConditionalUNet2DComponentExecutor {
public:
    TemporalConditionalUNet2DComponentExecutor(Backend& backend, WeightMap weights)
        : TemporalConditionalUNet2DComponentExecutor(
              backend, std::move(weights), PrecisionPolicy::fp32()) {}
    TemporalConditionalUNet2DComponentExecutor(
        Backend& backend, WeightMap weights, PrecisionPolicy policy)
        : backend_(backend), weights_(std::move(weights)),
          policy_(std::move(policy)) {}

    TemporalConditionalUNetResult execute(
        const Json& graph, const Tensor& latent, const Tensor& timestep,
        const Tensor& conditioning,
        TemporalConditionalUNetObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
    PrecisionPolicy policy_;
};

}  // namespace vrhino
