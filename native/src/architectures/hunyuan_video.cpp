#include "vrhino/architecture.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vrhino/components.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

Tensor time_embedding(Backend& backend, const PrecisionPolicy& policy,
                      const WeightMap& weights, const Tensor& values) {
    Tensor x = backend.sinusoidal_embedding(backend.reshape(values, {-1}), 256,
                                             true, 0.0, false);
    x = operation_linear(backend, policy, PrecisionOperation::Modulation,
                         PrecisionSemantic::TemporaryCompute, x,
                         weights.at("mlp.0.weight"), &weights.at("mlp.0.bias"));
    x = backend.activation(x, Activation::Silu);
    return operation_linear(backend, policy, PrecisionOperation::Modulation,
                            PrecisionSemantic::TemporaryCompute, x,
                            weights.at("mlp.2.weight"), &weights.at("mlp.2.bias"));
}

Tensor text_projection(Backend& backend, const WeightMap& weights, const Tensor& values) {
    Tensor x = backend.linear(values, weights.at("linear_1.weight"),
                              &weights.at("linear_1.bias"));
    x = backend.activation(x, Activation::Silu);
    return backend.linear(x, weights.at("linear_2.weight"), &weights.at("linear_2.bias"));
}

Tensor token_refiner_block(Backend& backend, const PrecisionPolicy& policy,
                           const WeightMap& weights,
                           const Tensor& input, const Tensor& conditioning,
                           const Tensor& attention_mask) {
    Tensor gate = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(conditioning, Activation::Silu),
        weights.at("adaLN_modulation.1.weight"),
        &weights.at("adaLN_modulation.1.bias"));
    auto gates = backend.split(gate, {gate.dim(-1) / 2, gate.dim(-1) / 2}, -1);
    Tensor norm = backend.layer_norm(input, &weights.at("norm1.weight"),
                                     &weights.at("norm1.bias"), 1e-6f);
    Tensor qkv = backend.linear(norm, weights.at("self_attn_qkv.weight"),
                                &weights.at("self_attn_qkv.bias"));
    const int64_t batch = qkv.dim(0), tokens = qkv.dim(1), width = input.dim(-1), heads = 24;
    qkv = backend.permute(backend.reshape(qkv, {batch, tokens, 3, heads, width / heads}),
                          {2, 0, 1, 3, 4});
    auto qkv_parts = backend.split(qkv, {1, 1, 1}, 0);
    for (Tensor& part : qkv_parts) part = backend.reshape(part, {batch, tokens, heads, width / heads});
    Tensor attended = merge_heads(backend, operation_attention(
        backend, policy, qkv_parts[0], qkv_parts[1], qkv_parts[2],
        &attention_mask));
    attended = backend.linear(attended, weights.at("self_attn_proj.weight"),
                              &weights.at("self_attn_proj.bias"));
    Tensor hidden = gated_residual(backend, policy, input, attended,
                                   backend.reshape(gates[0], {batch, 1, width}));
    norm = backend.layer_norm(hidden, &weights.at("norm2.weight"),
                              &weights.at("norm2.bias"), 1e-6f);
    BackendProfileRegion mlp_profile(backend, "mlp.total");
    Tensor mlp = backend.linear(norm, weights.at("mlp.fc1.weight"), &weights.at("mlp.fc1.bias"));
    mlp = backend.activation(mlp, Activation::Silu);
    mlp = backend.linear(mlp, weights.at("mlp.fc2.weight"), &weights.at("mlp.fc2.bias"));
    return gated_residual(backend, policy, hidden, mlp,
                          backend.reshape(gates[1], {batch, 1, width}));
}

Tensor text_refiner(Backend& backend, const PrecisionPolicy& policy,
                    const WeightMap& weights, const Tensor& text,
                    const Tensor& timestep, const Tensor& text_mask,
                    const Tensor& attention_mask) {
    Tensor time = time_embedding(backend, policy, weights.prefix("t_embedder."), timestep);
    Tensor mask = backend.reshape(backend.cast(text_mask,
        policy.operation_compute_dtype(PrecisionOperation::Conditioning)),
                                  {text_mask.dim(0), text_mask.dim(1), 1});
    Tensor context = backend.div(backend.reduce_sum(backend.mul(text, mask), 1),
                                 backend.reduce_sum(mask, 1));
    context = text_projection(backend, weights.prefix("c_embedder."), context);
    Tensor conditioning = backend.add(time, context);
    Tensor hidden = backend.linear(text, weights.at("input_embedder.weight"),
                                   &weights.at("input_embedder.bias"));
    for (int index = 0; index < 2; ++index)
        hidden = token_refiner_block(backend, policy,
            weights.prefix("individual_token_refiner.blocks." + std::to_string(index) + "."),
            hidden, conditioning, attention_mask);
    return hidden;
}

struct QKV { Tensor q, k, v; };

QKV stream_qkv(Backend& backend, const PrecisionPolicy& policy,
               const WeightMap& weights, const std::string& stream,
               const Tensor& input, TensorBundle* trace = nullptr) {
    Tensor qkv = backend.linear(input, weights.at(stream + "_attn_qkv.weight"),
                                weights.find(stream + "_attn_qkv.bias"));
    if (trace) (*trace)["block.0." + stream + ".qkv_projection"] = qkv;
    const int64_t b = qkv.dim(0), tokens = qkv.dim(1), heads = 24,
                  width = input.dim(-1), head_width = width / heads;
    qkv = backend.permute(backend.reshape(qkv, {b, tokens, 3, heads, head_width}),
                          {2, 0, 1, 3, 4});
    auto values = backend.split(qkv, {1, 1, 1}, 0);
    for (Tensor& value : values) value = backend.reshape(value, {b, tokens, heads, head_width});
    values[0] = attention_norm(backend, policy, values[0],
                               &weights.at(stream + "_attn_q_norm.weight"), 1e-6f);
    values[1] = attention_norm(backend, policy, values[1],
                               &weights.at(stream + "_attn_k_norm.weight"), 1e-6f);
    if (trace) {
        (*trace)["block.0." + stream + ".q_norm"] = values[0];
        (*trace)["block.0." + stream + ".k_norm"] = values[1];
    }
    return {values[0], values[1], values[2]};
}

Tensor stream_update(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const std::string& stream,
                     const Tensor& states, const Tensor& attention_output,
                     const std::vector<Tensor>& mod,
                     TensorBundle* trace = nullptr) {
    Tensor projected = backend.linear(attention_output, weights.at(stream + "_attn_proj.weight"),
                                      weights.find(stream + "_attn_proj.bias"));
    if (trace) (*trace)["block.0." + stream + ".attention_output_projection"] = projected;
    Tensor hidden = gated_residual(backend, policy, states, projected,
                                   backend.reshape(mod[2], {states.dim(0), 1, states.dim(-1)}));
    if (trace) (*trace)["block.0." + stream + ".first_residual"] = hidden;
    Tensor normalized = backend.layer_norm(hidden, nullptr, nullptr, 1e-6f);
    if (trace) (*trace)["block.0." + stream + ".pre_ffn_norm"] = normalized;
    Tensor mlp_input = modulate(backend, policy, normalized,
        backend.reshape(mod[3], {states.dim(0), 1, states.dim(-1)}),
        backend.reshape(mod[4], {states.dim(0), 1, states.dim(-1)}));
    BackendProfileRegion mlp_profile(backend, "mlp.total");
    Tensor mlp = backend.linear(mlp_input, weights.at(stream + "_mlp.fc1.weight"),
                                &weights.at(stream + "_mlp.fc1.bias"));
    if (trace) (*trace)["block.0." + stream + ".ffn_projection_in"] = mlp;
    mlp = backend.activation(mlp, Activation::GeluTanh);
    if (trace) (*trace)["block.0." + stream + ".ffn_activation"] = mlp;
    mlp = backend.linear(mlp, weights.at(stream + "_mlp.fc2.weight"),
                         &weights.at(stream + "_mlp.fc2.bias"));
    if (trace) (*trace)["block.0." + stream + ".ffn_projection_out"] = mlp;
    Tensor output = gated_residual(backend, policy, hidden, mlp,
                                   backend.reshape(mod[5], {states.dim(0), 1, states.dim(-1)}));
    if (trace) (*trace)["block.0." + stream + ".output"] = output;
    return output;
}

std::pair<Tensor, Tensor> double_block(Backend& backend, const PrecisionPolicy& policy,
                                      const WeightMap& weights,
                                      const Tensor& image, const Tensor& text,
                                      const Tensor& conditioning, const Tensor& rope_cos,
                                      const Tensor& rope_sin, const Tensor& attention_mask,
                                      TensorBundle* trace = nullptr) {
    if (trace) {
        (*trace)["block.0.input.image"] = image;
        (*trace)["block.0.input.text"] = text;
    }
    Tensor im = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(conditioning, Activation::Silu),
        weights.at("img_mod.linear.weight"), &weights.at("img_mod.linear.bias"));
    Tensor tm = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(conditioning, Activation::Silu),
        weights.at("txt_mod.linear.weight"), &weights.at("txt_mod.linear.bias"));
    std::vector<int64_t> sections(6, image.dim(-1));
    auto image_mod = backend.split(im, sections, -1), text_mod = backend.split(tm, sections, -1);
    Tensor image_input = modulate(backend, policy, backend.layer_norm(image, nullptr, nullptr, 1e-6f),
        backend.reshape(image_mod[0], {image.dim(0), 1, image.dim(-1)}),
        backend.reshape(image_mod[1], {image.dim(0), 1, image.dim(-1)}));
    if (trace) (*trace)["block.0.img.pre_attention_norm"] = image_input;
    QKV iqkv = stream_qkv(backend, policy, weights, "img", image_input, trace);
    iqkv.q = operation_rope(backend, policy, iqkv.q, rope_cos, rope_sin);
    iqkv.k = operation_rope(backend, policy, iqkv.k, rope_cos, rope_sin);
    Tensor text_input = modulate(backend, policy, backend.layer_norm(text, nullptr, nullptr, 1e-6f),
        backend.reshape(text_mod[0], {text.dim(0), 1, text.dim(-1)}),
        backend.reshape(text_mod[1], {text.dim(0), 1, text.dim(-1)}));
    if (trace) (*trace)["block.0.txt.pre_attention_norm"] = text_input;
    QKV tqkv = stream_qkv(backend, policy, weights, "txt", text_input, trace);
    Tensor attention_heads = operation_attention(backend, policy,
        backend.concat({iqkv.q, tqkv.q}, 1), backend.concat({iqkv.k, tqkv.k}, 1),
        backend.concat({iqkv.v, tqkv.v}, 1), &attention_mask);
    if (trace) (*trace)["block.0.attention_heads"] = attention_heads;
    Tensor attended = merge_heads(backend, attention_heads);
    if (trace) (*trace)["block.0.attention_merged"] = attended;
    auto streams = backend.split(attended, {image.dim(1), text.dim(1)}, 1);
    if (trace) {
        (*trace)["block.0.img.attention_merged"] = streams[0];
        (*trace)["block.0.txt.attention_merged"] = streams[1];
    }
    return {stream_update(backend, policy, weights, "img", image, streams[0], image_mod, trace),
            stream_update(backend, policy, weights, "txt", text, streams[1], text_mod, trace)};
}

Tensor single_block(Backend& backend, const PrecisionPolicy& policy,
                    const WeightMap& weights, const Tensor& states,
                    const Tensor& conditioning, int64_t text_length,
                    const Tensor& rope_cos, const Tensor& rope_sin,
                    const Tensor& attention_mask) {
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(conditioning, Activation::Silu),
        weights.at("modulation.linear.weight"), &weights.at("modulation.linear.bias"));
    std::vector<int64_t> sections(3, states.dim(-1));
    auto mod = backend.split(modulation, sections, -1);
    Tensor input = modulate(backend, policy, backend.layer_norm(states, nullptr, nullptr, 1e-6f),
        backend.reshape(mod[0], {states.dim(0), 1, states.dim(-1)}),
        backend.reshape(mod[1], {states.dim(0), 1, states.dim(-1)}));
    Tensor projected = backend.linear(input, weights.at("linear1.weight"),
                                      &weights.at("linear1.bias"));
    const int64_t b = states.dim(0), tokens = states.dim(1), width = states.dim(-1), heads = 24;
    auto fused = backend.split(projected, {3 * width, projected.dim(-1) - 3 * width}, -1);
    Tensor qkv = backend.permute(backend.reshape(fused[0], {b, tokens, 3, heads, width / heads}),
                                 {2, 0, 1, 3, 4});
    auto parts = backend.split(qkv, {1, 1, 1}, 0);
    for (Tensor& value : parts) value = backend.reshape(value, {b, tokens, heads, width / heads});
    parts[0] = attention_norm(backend, policy, parts[0],
                              &weights.at("q_norm.weight"), 1e-6f);
    parts[1] = attention_norm(backend, policy, parts[1],
                              &weights.at("k_norm.weight"), 1e-6f);
    const int64_t image_length = tokens - text_length;
    Tensor iq = operation_rope(backend, policy,
        backend.slice(parts[0], 1, 0, image_length), rope_cos, rope_sin);
    Tensor ik = operation_rope(backend, policy,
        backend.slice(parts[1], 1, 0, image_length), rope_cos, rope_sin);
    parts[0] = backend.concat({iq, backend.slice(parts[0], 1, image_length, tokens)}, 1);
    parts[1] = backend.concat({ik, backend.slice(parts[1], 1, image_length, tokens)}, 1);
    Tensor attended = merge_heads(backend, operation_attention(
        backend, policy, parts[0], parts[1], parts[2], &attention_mask));
    BackendProfileRegion mlp_profile(backend, "mlp.total");
    Tensor mlp = backend.activation(fused[1], Activation::GeluTanh);
    Tensor output = backend.linear(backend.concat({attended, mlp}, -1),
                                   weights.at("linear2.weight"), &weights.at("linear2.bias"));
    return gated_residual(backend, policy, states, output,
                          backend.reshape(mod[2], {b, 1, width}));
}

Tensor full_denoiser(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const Tensor& latent,
                     const Tensor& timestep, const Tensor& text, const Tensor& mask,
                     const Tensor& host_mask, const Tensor& token_attention_mask,
                     const Tensor& pooled, float embedded_guidance,
                     TensorBundle* trace = nullptr) {
    Tensor vec = time_embedding(backend, policy, weights.prefix("time_in."), timestep);
    Tensor pooled_projected = backend.linear(pooled, weights.at("vector_in.in_layer.weight"),
                                              &weights.at("vector_in.in_layer.bias"));
    pooled_projected = backend.activation(pooled_projected, Activation::Silu);
    pooled_projected = backend.linear(pooled_projected, weights.at("vector_in.out_layer.weight"),
                                      &weights.at("vector_in.out_layer.bias"));
    vec = backend.add(vec, pooled_projected);
    vec = backend.add(vec, time_embedding(backend, policy, weights.prefix("guidance_in."),
                                          host_f32({1}, {embedded_guidance})));
    Tensor image = backend.conv3d(latent, weights.at("img_in.proj.weight"),
                                  &weights.at("img_in.proj.bias"), {1, 2, 2}, {0, 0, 0});
    const int64_t b = image.dim(0), width = image.dim(1), frames = image.dim(2),
                  height = image.dim(3), spatial_width = image.dim(4), tokens = frames * height * spatial_width;
    image = backend.permute(backend.reshape(image, {b, width, tokens}), {0, 2, 1});
    if (trace) (*trace)["input_projection"] = image;
    Tensor context = text_refiner(
        backend, policy, weights.prefix("txt_in."), text, timestep, mask,
        token_attention_mask);
    Tensor joint_attention_mask = backend.copy_to_device(
        boolean_key_padding_mask(host_mask, image.dim(1)), DType::Bool);
    std::vector<std::vector<float>> coordinates;
    for (int64_t t = 0; t < frames; ++t) for (int64_t h = 0; h < height; ++h)
        for (int64_t w = 0; w < spatial_width; ++w)
            coordinates.push_back({static_cast<float>(t), static_cast<float>(h), static_cast<float>(w)});
    auto [rope_cos, rope_sin] = standard_rope(
        backend, policy, coordinates, {16, 56, 56}, 256.0, false);
    for (int index = 0; index < 20; ++index) {
        auto result = double_block(backend, policy,
            weights.prefix("double_blocks." + std::to_string(index) + "."),
                                   image, context, vec, rope_cos, rope_sin,
                                   joint_attention_mask,
                                   trace && index == 0 ? trace : nullptr);
        image = std::move(result.first); context = std::move(result.second);
        if (trace) (*trace)["double_block." + std::to_string(index)] =
            backend.concat({image, context}, 1);
    }
    const int64_t image_length = image.dim(1), text_length = context.dim(1);
    Tensor hidden = backend.concat({image, context}, 1);
    for (int index = 0; index < 40; ++index) {
        hidden = single_block(backend, policy,
            weights.prefix("single_blocks." + std::to_string(index) + "."),
                              hidden, vec, text_length, rope_cos, rope_sin,
                              joint_attention_mask);
        if (trace) (*trace)["single_block." + std::to_string(index)] = hidden;
    }
    image = backend.slice(hidden, 1, 0, image_length);
    WeightMap final = weights.prefix("final_layer.");
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(vec, Activation::Silu),
        final.at("adaLN_modulation.1.weight"), &final.at("adaLN_modulation.1.bias"));
    auto parts = backend.split(modulation, {width, width}, -1);
    image = producer_linear(backend, policy, PrecisionSemantic::DenoiserOutput,
        modulate(backend, policy, backend.layer_norm(image, nullptr, nullptr, 1e-6f),
        backend.reshape(parts[0], {b, 1, width}), backend.reshape(parts[1], {b, 1, width})),
        final.at("linear.weight"), &final.at("linear.bias"));
    image = backend.reshape(image, {b, frames, height, spatial_width, 16, 1, 2, 2});
    image = backend.permute(image, {0, 4, 1, 5, 2, 6, 3, 7});
    image = backend.reshape(image, {b, 16, frames, height * 2, spatial_width * 2});
    if (trace) (*trace)["output_projection"] = image;
    return image;
}

Tensor decoder_residual(Backend& backend, ComponentExecutor& e, const WeightMap& weights,
                        const Tensor& input, const std::string& prefix) {
    Tensor y = e.group_norm(input, prefix + ".norm1");
    y = e.causal_conv3d(e.silu(y), prefix + ".conv1.conv", PadMode::Replicate);
    y = e.group_norm(y, prefix + ".norm2");
    y = e.causal_conv3d(e.silu(y), prefix + ".conv2.conv", PadMode::Replicate);
    Tensor identity = input;
    if (weights.contains(prefix + ".conv_shortcut.conv.weight"))
        identity = e.causal_conv3d(identity, prefix + ".conv_shortcut.conv", PadMode::Replicate);
    return backend.add(identity, y);
}

using UpsampleScale = std::array<int64_t, 3>;

Tensor first_frame_preserving_nearest(Backend& backend, ComponentExecutor& e,
                                      const Tensor& input, const UpsampleScale& scale) {
    require(scale[0] == 1 || scale[0] == 2,
            "Hunyuan temporal upsample scale must be 1 or 2");
    require(scale[1] > 0 && scale[2] > 0,
            "Hunyuan spatial upsample scales must be positive");
    if (scale[0] == 1 || input.dim(2) == 1)
        return e.nearest(input, {1.0, static_cast<double>(scale[1]),
                                 static_cast<double>(scale[2])});
    Tensor first = e.nearest(backend.slice(input, 2, 0, 1),
                             {1.0, static_cast<double>(scale[1]),
                              static_cast<double>(scale[2])});
    Tensor rest = e.nearest(backend.slice(input, 2, 1, input.dim(2)),
                            {static_cast<double>(scale[0]),
                             static_cast<double>(scale[1]),
                             static_cast<double>(scale[2])});
    Tensor output = backend.concat({first, rest}, 2);
    require(output.dim(2) == (input.dim(2) - 1) * scale[0] + 1,
            "Hunyuan temporal upsample shape contract mismatch");
    return output;
}

Tensor decode_hunyuan(Backend& backend, const WeightMap& weights, const Tensor& raw,
                      const std::array<UpsampleScale, 3>& upsample_scales) {
    ComponentExecutor e(backend, weights);
    Tensor x = backend.mul(raw, scalar_f32(1.0f / 0.476986f));
    x = e.conv3d(x, "post_quant_conv");
    x = e.causal_conv3d(x, "decoder.conv_in.conv", PadMode::Replicate);
    x = decoder_residual(backend, e, weights, x, "decoder.mid_block.resnets.0");
    x = e.sequence_attention_linear(x, "decoder.mid_block.attentions.0", false);
    x = decoder_residual(backend, e, weights, x, "decoder.mid_block.resnets.1");
    for (int block = 0; block < 4; ++block) {
        for (int residual = 0; residual < 3; ++residual)
            x = decoder_residual(backend, e, weights, x,
                "decoder.up_blocks." + std::to_string(block) + ".resnets." + std::to_string(residual));
        if (block < 3) {
            x = first_frame_preserving_nearest(
                backend, e, x, upsample_scales.at(static_cast<size_t>(block)));
            x = e.causal_conv3d(x, "decoder.up_blocks." + std::to_string(block) +
                                ".upsamplers.0.conv.conv", PadMode::Replicate);
        }
    }
    x = e.silu(e.group_norm(x, "decoder.conv_norm_out"));
    return e.video_range(e.causal_conv3d(x, "decoder.conv_out.conv", PadMode::Replicate));
}

class HunyuanDenoiser final : public Denoiser {
public:
    HunyuanDenoiser(Backend& backend, const PrecisionPolicy& policy,
                    WeightMap weights, const Tensor& text,
                    const Tensor& mask, const Tensor& pooled,
                    float embedded_guidance, bool trace_enabled)
        : backend_(backend), weights_(std::move(weights)),
          policy_(policy),
          text_(backend.copy_to_device(text, policy.boundary_dtype(
              PrecisionSemantic::Conditioning))),
          mask_(backend.copy_to_device(mask, DType::Bool)),
          host_mask_(mask),
          token_attention_mask_(backend.copy_to_device(
              boolean_self_attention_mask(mask, true), DType::Bool)),
          pooled_(backend.copy_to_device(pooled, policy.boundary_dtype(
              PrecisionSemantic::Conditioning))),
          embedded_guidance_(embedded_guidance), trace_enabled_(trace_enabled) {}
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        trace_.clear();
        return {full_denoiser(backend_, policy_, weights_, latent, timestep,
                              text_, mask_, host_mask_, token_attention_mask_, pooled_,
                              embedded_guidance_,
                              trace_enabled_ ? &trace_ : nullptr)};
    }
    TensorBundle take_trace() override {
        TensorBundle result = std::move(trace_); trace_.clear(); return result;
    }
private:
    Backend& backend_; WeightMap weights_; const PrecisionPolicy& policy_;
    Tensor text_, mask_, host_mask_, token_attention_mask_, pooled_;
    float embedded_guidance_ = 6000.0f;
    bool trace_enabled_ = false;
    TensorBundle trace_;
};

class HunyuanArchitecture final : public Architecture {
public:
    explicit HunyuanArchitecture(const VrmModel& model)
        : denoiser_(model.bindings(model.graph().at("architecture_graph"))),
          component_(model.bindings(model.graph().at("component_graphs").array().at(0))) {
        const Json& component = model.graph().at("component_graphs").array().at(0);
        const auto& stages = component.at("config").at("upsample_stages").array();
        require(stages.size() == upsample_scales_.size(),
                "Hunyuan VAE must declare exactly three upsample stages");
        for (size_t index = 0; index < stages.size(); ++index) {
            const int64_t temporal = stages[index].at("temporal_scale").integer();
            const int64_t spatial = stages[index].at("spatial_scale").integer();
            require((temporal == 1 || temporal == 2) && spatial > 0,
                    "Invalid Hunyuan VAE upsample stage declaration");
            upsample_scales_[index] = {temporal, spatial, spatial};
        }
        const Json& config = model.graph().at("sampling_program").at("config");
        default_steps_ = static_cast<int>(config.at("steps").integer());
        default_flow_shift_ = static_cast<float>(config.at("flow_shift").number());
        if (const Json* guidance = config.find("embedded_guidance"))
            default_embedded_guidance_ = static_cast<float>(guidance->number());
        else
            default_embedded_guidance_ =
                static_cast<float>(config.at("embedded_guidance_scale").number() * 1000.0);
        require(default_steps_ >= 2 && default_steps_ <= 10000,
                "Hunyuan declared sampling steps are outside [2,10000]");
        require(std::isfinite(default_flow_shift_) && default_flow_shift_ > 0.0f,
                "Hunyuan declared flow shift must be finite and positive");
        require(std::isfinite(default_embedded_guidance_),
                "Hunyuan declared embedded guidance must be finite");
    }
    std::unique_ptr<Denoiser> create_denoiser(
            Backend& backend, const PrecisionPolicy& policy,
            const TensorBundle& input) override {
        const float embedded_guidance = input.contains("embedded_guidance")
            ? read_scalar_f32(input.at("embedded_guidance"))
            : default_embedded_guidance_;
        require(std::isfinite(embedded_guidance),
                "Hunyuan embedded guidance must be finite");
        return std::make_unique<HunyuanDenoiser>(backend, policy, denoiser_,
            input.at("text"), input.at("mask"), input.at("pooled"),
            embedded_guidance,
            input.contains("audit_trace") && read_scalar_i64(input.at("audit_trace")) != 0);
    }
    SamplingProgram create_program(const TensorBundle& input) override {
        SamplingProgram p; p.latent_shape = {1, 16, 1, 4, 4};
        if (input.contains("latent_shape")) {
            const Tensor& shape = input.at("latent_shape");
            require(shape.device() == Device::CPU && shape.dtype() == DType::I64 &&
                    shape.shape() == std::vector<int64_t>({5}), "Hunyuan latent_shape contract mismatch");
            p.latent_shape.assign(shape.data_as<int64_t>(), shape.data_as<int64_t>() + 5);
            require(p.latent_shape[0] == 1 && p.latent_shape[1] == 16 && p.latent_shape[2] > 0 &&
                    p.latent_shape[3] > 0 && p.latent_shape[4] > 0 &&
                    p.latent_shape[3] % 2 == 0 && p.latent_shape[4] % 2 == 0,
                    "Hunyuan latent_shape is not legal");
        }
        p.seed = read_scalar_i64(input.at("seed"));
        p.steps = input.contains("sampling_steps")
            ? static_cast<int>(read_scalar_i64(input.at("sampling_steps")))
            : default_steps_;
        require(p.steps >= 2 && p.steps <= 10000,
                "Hunyuan sampling steps are outside [2,10000]");
        const float flow_shift = input.contains("flow_shift")
            ? read_scalar_f32(input.at("flow_shift"))
            : default_flow_shift_;
        require(std::isfinite(flow_shift) && flow_shift > 0.0f,
                "Hunyuan flow shift must be finite and positive");
        p.guidance_mode = GuidanceMode::Linear; p.guidance_coefficients = {1.0f};
        p.scheduler = SchedulerKind::Euler;
        for (int index = 0; index <= p.steps; ++index) {
            const float sigma = 1.0f - static_cast<float>(index) / p.steps;
            p.sigmas.push_back(
                flow_shift * sigma / (1.0f + (flow_shift - 1.0f) * sigma));
        }
        for (int index = 0; index < p.steps; ++index) {
            p.model_timesteps.push_back(host_f32({}, {p.sigmas[index] * 1000.0f}));
            p.update_deltas.push_back(p.sigmas[index + 1] - p.sigmas[index]);
        }
        return p;
    }
    Tensor decode(Backend& backend, const PrecisionPolicy&, const Tensor& latent,
                  const TensorBundle&) override {
        return decode_hunyuan(backend, component_, latent, upsample_scales_);
    }
private:
    WeightMap denoiser_, component_;
    std::array<UpsampleScale, 3> upsample_scales_{};
    int default_steps_ = 3;
    float default_flow_shift_ = 5.0f;
    float default_embedded_guidance_ = 6000.0f;
};

}  // namespace

std::unique_ptr<Architecture> make_hunyuan_video_architecture(const VrmModel& model) {
    return std::make_unique<HunyuanArchitecture>(model);
}

}  // namespace vrhino
