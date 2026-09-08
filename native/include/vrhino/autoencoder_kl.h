#pragma once

#include <map>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct AutoencoderKLObservation {
    std::map<std::string, Tensor> tensors;
};

struct AutoencoderKLEncoderResult {
    Tensor posterior_mean;
    Tensor posterior_logvar;
};

struct AutoencoderKLPosteriorSample {
    Tensor epsilon;
    Tensor sampled_latent;
    Tensor scaled_latent;
};

struct AutoencoderKLDecoderResult {
    Tensor decoded;
    Tensor rgb_0_1;
};

// Shared model-name-free executor for bounded AutoencoderKL encode/decode
// entry points. The graph provides topology/configuration; numerical work is
// exclusively composed from generic Backend primitives.
class AutoencoderKLComponentExecutor {
public:
    AutoencoderKLComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    AutoencoderKLEncoderResult encode(
        const Json& graph, const Tensor& normalized_rgb,
        AutoencoderKLObservation* observation = nullptr);
    AutoencoderKLPosteriorSample sample(
        const Json& graph, const Tensor& mean, const Tensor& logvar,
        RngState& component_rng, const Tensor* fixed_epsilon = nullptr);
    AutoencoderKLDecoderResult decode(
        const Json& graph, const Tensor& scaled_latent,
        AutoencoderKLObservation* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
