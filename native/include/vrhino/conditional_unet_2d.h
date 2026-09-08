#pragma once

#include <map>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct ConditionalUNet2DObservation {
    std::map<std::string, Tensor> tensors;
};

struct ConditionalUNet2DResult {
    Tensor predicted_latent;
};

// Model-name-free executor for a frozen conditional 2D UNet graph. The
// topology is data; all numerical work is composed from generic Backend
// primitives shared with the rest of the Native Runtime.
class ConditionalUNet2DComponentExecutor {
public:
    ConditionalUNet2DComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    ConditionalUNet2DResult execute(
        const Json& graph, const Tensor& latent, const Tensor& timestep,
        const Tensor& conditioning,
        ConditionalUNet2DObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
