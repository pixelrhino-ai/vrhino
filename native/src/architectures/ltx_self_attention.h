#pragma once

#include "vrhino/architecture.h"

namespace vrhino::ltx_internal {
using BlockBoundaries = std::map<int, TensorBundle>;
// Qualification boundaries share the sole production Graph path; no mode switch.
Tensor transformer_block_forward(Backend&, const PrecisionPolicy&, const WeightMap&,
    const Tensor& input, const Tensor& context, const Tensor& timestep,
    const Tensor& cosine, const Tensor& sine, const Tensor& mask,
    TensorBundle* trace = nullptr);
Tensor denoiser_forward(Backend&, const PrecisionPolicy&, const WeightMap&,
    const Tensor& latent, const Tensor& cosine, const Tensor& sine,
    const Tensor& text, const Tensor& mask, const Tensor& sigma,
    BlockBoundaries* boundaries = nullptr);
}
