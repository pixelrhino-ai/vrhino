#pragma once

#include <map>
#include <memory>
#include <string>

#include "vrhino/backend.h"
#include "vrhino/bundle.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/sampling.h"

namespace vrhino {

class WeightMap {
public:
    WeightMap() = default;
    explicit WeightMap(std::map<std::string, const Tensor*> values, std::string prefix = {})
        : values_(std::move(values)), prefix_(std::move(prefix)) {}
    const Tensor& at(const std::string& name) const;
    const Tensor* find(const std::string& name) const;
    bool contains(const std::string& name) const { return find(name) != nullptr; }
    WeightMap prefix(const std::string& value) const { return WeightMap(values_, prefix_ + value); }
private:
    std::map<std::string, const Tensor*> values_;
    std::string prefix_;
};

class Architecture {
public:
    virtual ~Architecture() = default;
    virtual std::unique_ptr<Denoiser> create_denoiser(Backend& backend,
                                                       const PrecisionPolicy& policy,
                                                       const TensorBundle& input) = 0;
    virtual SamplingProgram create_program(const TensorBundle& input) = 0;
    virtual Tensor decode(Backend& backend, const PrecisionPolicy& policy,
                          const Tensor& latent,
                          const TensorBundle& input) = 0;
    // Optional component instrumentation consumed by the shared runtime.
    virtual TensorBundle take_trace() { return {}; }
};

std::unique_ptr<Architecture> create_architecture(const VrmModel& model);
Tensor operation_linear(Backend& backend, const PrecisionPolicy& policy,
                        PrecisionOperation operation, PrecisionSemantic output_semantic,
                        const Tensor& input, const Tensor& weight,
                        const Tensor* bias = nullptr);
Tensor operation_conv2d(Backend& backend, const PrecisionPolicy& policy,
                        PrecisionSemantic output_semantic,
                        const Tensor& input, const Tensor& weight,
                        const Tensor* bias, const std::vector<int>& stride,
                        const std::vector<int>& padding,
                        const std::vector<int>& dilation = {1, 1},
                        int groups = 1);
Tensor modulate(Backend& backend, const PrecisionPolicy& policy,
                const Tensor& x, const Tensor& shift, const Tensor& scale);
Tensor attention_norm(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& input, const Tensor* weight,
                      float eps, int64_t axis = -1);
Tensor operation_rope(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& input, const Tensor& cosine,
                      const Tensor& sine);
Tensor operation_attention(Backend& backend, const PrecisionPolicy& policy,
                           const Tensor& query, const Tensor& key,
                           const Tensor& value, const Tensor* mask = nullptr,
                           bool causal = false, float scale = 0.0f,
                           const Tensor* bias = nullptr);
// Model-neutral Boolean validity-mask lowering for generic Attention. The
// source is a host [B,S] mask where true means a valid token. The self mask is
// [B,1,S,S]; the key-padding mask is [B,1,prefix+S].
Tensor boolean_self_attention_mask(const Tensor& validity,
                                   bool first_key_always_allowed = false);
Tensor boolean_key_padding_mask(const Tensor& validity,
                                int64_t unmasked_prefix_tokens = 0);
Tensor residual_add(Backend& backend, const PrecisionPolicy& policy,
                    const Tensor& residual, const Tensor& branch);
Tensor gated_residual(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& residual, const Tensor& branch, const Tensor& gate);
Tensor producer_linear(Backend& backend, const PrecisionPolicy& policy,
                       PrecisionSemantic output_semantic, const Tensor& input,
                       const Tensor& weight, const Tensor* bias = nullptr);
Tensor split_heads(Backend& backend, const Tensor& x, int heads);
Tensor merge_heads(Backend& backend, const Tensor& x);
std::pair<Tensor, Tensor> standard_rope(Backend& backend,
                                       const PrecisionPolicy& policy,
                                       const std::vector<std::vector<float>>& coordinates,
                                       const std::vector<int>& axis_dims, double theta,
                                       bool frequency_f64);
std::pair<Tensor, Tensor> fractional_rope(Backend& backend,
                                         const PrecisionPolicy& policy,
                                         const Tensor& coordinates,
                                         const std::vector<float>& max_positions,
                                         int width, float theta, bool centered);
PreparedTensorHandle prepare_fractional_rope(
    PreparedTensorCache& cache, Backend& backend,
    const PrecisionPolicy& policy, const Tensor& coordinates,
    const std::vector<float>& max_positions, int width, float theta,
    bool centered);
std::unique_ptr<Architecture> make_wan_architecture(const VrmModel& model);
std::unique_ptr<Architecture> make_hunyuan_video_architecture(const VrmModel& model);
std::unique_ptr<Architecture> make_ltx_architecture(const VrmModel& model);
std::unique_ptr<Architecture> make_mochi_architecture(const VrmModel& model);
std::unique_ptr<Architecture> make_cogvideox_architecture(const VrmModel& model);

}  // namespace vrhino
