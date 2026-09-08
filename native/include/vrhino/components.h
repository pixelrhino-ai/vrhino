#pragma once

#include <map>
#include <string>
#include <vector>

#include "vrhino/architecture.h"

namespace vrhino {

struct ComponentState {
    RngState rng;
    std::map<std::string, Tensor> temporal_cache;
};

// Shared, model-name-free component executor. Architecture graphs select and
// compose these operations; all numerical work remains owned by Backend.
class ComponentExecutor {
public:
    ComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    const Tensor& weight(const std::string& key) const { return weights_.at(key); }
    bool contains(const std::string& key) const { return weights_.contains(key); }
    Tensor conv3d(const Tensor& x, const std::string& prefix,
                  const std::vector<int>& padding = {0, 0, 0},
                  const std::vector<int>& stride = {1, 1, 1});
    // Mutable/rvalue inputs transfer their semantic last-use ownership to the
    // executor. Const inputs are borrowed and preserve the caller's branch.
    Tensor causal_conv3d(Tensor& x, const std::string& prefix,
                         PadMode mode = PadMode::Constant, bool causal = true,
                         const std::vector<int>& stride = {1, 1, 1},
                         ComponentState* state = nullptr,
                         const std::string& cache_key = {});
    Tensor causal_conv3d(Tensor&& x, const std::string& prefix,
                         PadMode mode = PadMode::Constant, bool causal = true,
                         const std::vector<int>& stride = {1, 1, 1},
                         ComponentState* state = nullptr,
                         const std::string& cache_key = {});
    Tensor causal_conv3d(const Tensor& x, const std::string& prefix,
                         PadMode mode = PadMode::Constant, bool causal = true,
                         const std::vector<int>& stride = {1, 1, 1},
                         ComponentState* state = nullptr,
                         const std::string& cache_key = {});
    Tensor conv2d_frames(const Tensor& x, const std::string& prefix);
    Tensor group_norm(const Tensor& x, const std::string& prefix,
                      int groups = 32, float eps = 1e-6f);
    Tensor rms_channel(const Tensor& x, const std::string& prefix,
                       float eps = 1e-12f);
    Tensor pixel_norm(const Tensor& x, float eps = 1e-8f);
    Tensor silu(const Tensor& x);
    Tensor linear(const Tensor& x, const std::string& prefix);
    Tensor timestep_embedding(const Tensor& timestep, const std::string& prefix);
    Tensor nearest(const Tensor& x, const std::vector<double>& factors);
    Tensor pixel_shuffle(const Tensor& x, const std::vector<int64_t>& factors);
    Tensor spatial_attention_conv(const Tensor& x, const std::string& prefix);
    Tensor sequence_attention_linear(const Tensor& x, const std::string& prefix,
                                     bool causal_frames);
    Tensor spatial_noise(const Tensor& x, const Tensor& scale, ComponentState& state);
    Tensor video_range(const Tensor& x, float source_min = -1.0f,
                       float source_max = 1.0f, float target_min = 0.0f,
                       float target_max = 1.0f);
    Tensor clamp(const Tensor& x, float low, float high);

private:
    Tensor causal_conv3d_owned(Tensor source, const std::string& prefix,
                               PadMode mode, bool causal,
                               const std::vector<int>& stride,
                               ComponentState* state,
                               const std::string& cache_key);
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
