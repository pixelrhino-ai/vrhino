#include "vrhino/architecture.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vrhino/components.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

constexpr int kHeads = 30;
constexpr int kWidth = 1920;
constexpr int kTextTokens = 226;

struct ModulatedPair {
    Tensor image;
    Tensor text;
    Tensor image_gate;
    Tensor text_gate;
};

ModulatedPair layer_norm_zero(Backend& backend, const PrecisionPolicy& policy,
                              const WeightMap& weights, const Tensor& image,
                              const Tensor& text, const Tensor& time) {
    Tensor values = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(time, Activation::Silu),
        weights.at("linear.weight"), &weights.at("linear.bias"));
    auto parts = backend.split(values, {kWidth, kWidth, kWidth, kWidth, kWidth, kWidth}, -1);
    Tensor image_norm = backend.layer_norm(
        image, &weights.at("norm.weight"), &weights.at("norm.bias"), 1e-5f);
    Tensor text_norm = backend.layer_norm(
        text, &weights.at("norm.weight"), &weights.at("norm.bias"), 1e-5f);
    return {
        modulate(backend, policy, image_norm,
                 backend.reshape(parts[0], {image.dim(0), 1, kWidth}),
                 backend.reshape(parts[1], {image.dim(0), 1, kWidth})),
        modulate(backend, policy, text_norm,
                 backend.reshape(parts[3], {text.dim(0), 1, kWidth}),
                 backend.reshape(parts[4], {text.dim(0), 1, kWidth})),
        backend.reshape(parts[2], {image.dim(0), 1, kWidth}),
        backend.reshape(parts[5], {text.dim(0), 1, kWidth}),
    };
}

std::pair<Tensor, Tensor> joint_attention(
        Backend& backend, const PrecisionPolicy& policy, const WeightMap& weights,
        const Tensor& image, const Tensor& text, TensorBundle* trace) {
    Tensor joint = backend.concat({text, image}, 1);
    Tensor q = backend.linear(joint, weights.at("to_q.weight"), &weights.at("to_q.bias"));
    Tensor k = backend.linear(joint, weights.at("to_k.weight"), &weights.at("to_k.bias"));
    Tensor v = backend.linear(joint, weights.at("to_v.weight"), &weights.at("to_v.bias"));
    q = split_heads(backend, q, kHeads);
    k = split_heads(backend, k, kHeads);
    v = split_heads(backend, v, kHeads);
    q = backend.layer_norm(q, &weights.at("norm_q.weight"),
                           &weights.at("norm_q.bias"), 1e-6f);
    k = backend.layer_norm(k, &weights.at("norm_k.weight"),
                           &weights.at("norm_k.bias"), 1e-6f);
    if (trace) {
        (*trace)["block.0.attention.q"] = q;
        (*trace)["block.0.attention.k"] = k;
    }
    Tensor attended = merge_heads(backend, operation_attention(
        backend, policy, q, k, v, nullptr, false));
    attended = backend.linear(attended, weights.at("to_out.0.weight"),
                              &weights.at("to_out.0.bias"));
    auto result = backend.split(attended, {text.dim(1), image.dim(1)}, 1);
    return {result[1], result[0]};
}

std::pair<Tensor, Tensor> transformer_block(
        Backend& backend, const PrecisionPolicy& policy, const WeightMap& weights,
        const Tensor& image_input, const Tensor& text_input, const Tensor& time,
        TensorBundle* trace) {
    ModulatedPair attention_input = layer_norm_zero(
        backend, policy, weights.prefix("norm1."), image_input, text_input, time);
    auto attended = joint_attention(
        backend, policy, weights.prefix("attn1."),
        attention_input.image, attention_input.text, trace);
    Tensor image = gated_residual(
        backend, policy, image_input, attended.first, attention_input.image_gate);
    Tensor text = gated_residual(
        backend, policy, text_input, attended.second, attention_input.text_gate);
    ModulatedPair feed_input = layer_norm_zero(
        backend, policy, weights.prefix("norm2."), image, text, time);
    Tensor joint = backend.concat({feed_input.text, feed_input.image}, 1);
    Tensor feed = backend.linear(joint, weights.at("ff.net.0.proj.weight"),
                                 &weights.at("ff.net.0.proj.bias"));
    feed = backend.activation(feed, Activation::GeluTanh);
    feed = backend.linear(feed, weights.at("ff.net.2.weight"),
                          &weights.at("ff.net.2.bias"));
    auto feed_parts = backend.split(feed, {text.dim(1), image.dim(1)}, 1);
    image = gated_residual(backend, policy, image, feed_parts[1], feed_input.image_gate);
    text = gated_residual(backend, policy, text, feed_parts[0], feed_input.text_gate);
    return {image, text};
}

Tensor patch_embed(Backend& backend, const WeightMap& weights,
                   const Tensor& latent, const Tensor& text) {
    const int64_t batch = latent.dim(0), frames = latent.dim(2);
    Tensor image = backend.reshape(backend.permute(latent, {0, 2, 1, 3, 4}),
                                   {batch * frames, latent.dim(1), latent.dim(3), latent.dim(4)});
    image = backend.conv2d(image, weights.at("patch_embed.proj.weight"),
                           &weights.at("patch_embed.proj.bias"), {2, 2}, {0, 0});
    image = backend.permute(
        backend.reshape(image, {batch, frames, image.dim(1), image.dim(2), image.dim(3)}),
        {0, 1, 3, 4, 2});
    image = backend.reshape(image, {batch, -1, image.dim(-1)});
    Tensor projected_text = backend.linear(
        text, weights.at("patch_embed.text_proj.weight"),
        &weights.at("patch_embed.text_proj.bias"));
    Tensor joint = backend.concat({projected_text, image}, 1);
    const Tensor& position = weights.at("position_embedding");
    require(position.ndim() == 3 && position.dim(0) == 1 &&
                position.dim(2) == kWidth && position.dim(1) >= joint.dim(1),
            "CogVideoX position embedding contract mismatch");
    return backend.add(joint, backend.slice(position, 1, 0, joint.dim(1)));
}

Tensor full_denoiser(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const Tensor& latent,
                     const Tensor& timestep, const Tensor& text,
                     TensorBundle* trace) {
    require(latent.ndim() == 5 && latent.dim(1) == 16 &&
                latent.dim(3) % 2 == 0 && latent.dim(4) % 2 == 0,
            "CogVideoX latent contract mismatch");
    require(text.ndim() == 3 && text.dim(0) == latent.dim(0) &&
                text.dim(1) == kTextTokens && text.dim(2) == 4096,
            "CogVideoX conditioning contract mismatch");
    Tensor time = backend.sinusoidal_embedding(
        backend.reshape(timestep, {-1}), kWidth, true, 0.0, false);
    time = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, time,
        weights.at("time_embedding.linear_1.weight"),
        &weights.at("time_embedding.linear_1.bias"));
    time = backend.activation(time, Activation::Silu);
    time = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, time,
        weights.at("time_embedding.linear_2.weight"),
        &weights.at("time_embedding.linear_2.bias"));
    Tensor joint = patch_embed(backend, weights, latent, text);
    Tensor text_state = backend.slice(joint, 1, 0, kTextTokens);
    Tensor image = backend.slice(joint, 1, kTextTokens, joint.dim(1));
    if (trace) (*trace)["input_projection"] = image;
    for (int index = 0; index < 30; ++index) {
        auto output = transformer_block(
            backend, policy,
            weights.prefix("transformer_blocks." + std::to_string(index) + "."),
            image, text_state, time, trace && index == 0 ? trace : nullptr);
        image = std::move(output.first);
        text_state = std::move(output.second);
        if (trace) (*trace)["block." + std::to_string(index)] = image;
    }
    image = backend.layer_norm(image, &weights.at("norm_final.weight"),
                               &weights.at("norm_final.bias"), 1e-5f);
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(time, Activation::Silu),
        weights.at("norm_out.linear.weight"), &weights.at("norm_out.linear.bias"));
    auto parts = backend.split(modulation, {kWidth, kWidth}, -1);
    image = modulate(
        backend, policy,
        backend.layer_norm(image, &weights.at("norm_out.norm.weight"),
                           &weights.at("norm_out.norm.bias"), 1e-5f),
        backend.reshape(parts[0], {image.dim(0), 1, kWidth}),
        backend.reshape(parts[1], {image.dim(0), 1, kWidth}));
    image = producer_linear(backend, policy, PrecisionSemantic::DenoiserOutput,
                            image, weights.at("proj_out.weight"),
                            &weights.at("proj_out.bias"));
    const int64_t batch = latent.dim(0), frames = latent.dim(2);
    const int64_t height = latent.dim(3) / 2, width = latent.dim(4) / 2;
    image = backend.reshape(image, {batch, frames, height, width, 16, 2, 2});
    image = backend.permute(image, {0, 4, 1, 2, 5, 3, 6});
    image = backend.reshape(image, {batch, 16, frames, height * 2, width * 2});
    if (trace) (*trace)["output_projection"] = image;
    return image;
}

Tensor causal_first_conv(Backend& backend, ComponentExecutor& executor,
                         const Tensor& input, const std::string& prefix) {
    const Tensor& kernel = executor.weight(prefix + ".weight");
    require(kernel.ndim() == 5, "CogVideoX causal Conv3D weight rank mismatch");
    Tensor value = input;
    if (kernel.dim(2) > 1) {
        Tensor first = backend.slice(input, 2, 0, 1);
        std::vector<Tensor> pieces(static_cast<size_t>(kernel.dim(2) - 1), first);
        pieces.push_back(input);
        value = backend.concat(pieces, 2);
    }
    if (kernel.dim(3) > 1 || kernel.dim(4) > 1)
        value = backend.pad(value, {kernel.dim(4) / 2, kernel.dim(4) / 2,
                                    kernel.dim(3) / 2, kernel.dim(3) / 2});
    return executor.conv3d(value, prefix);
}

Tensor resize_like(Backend& backend, const Tensor& input, const Tensor& target) {
    require(input.ndim() == 5 && target.ndim() == 5, "CogVideoX resize rank mismatch");
    return backend.interpolate_nearest(input, {
        static_cast<double>(target.dim(2)) / input.dim(2),
        static_cast<double>(target.dim(3)) / input.dim(3),
        static_cast<double>(target.dim(4)) / input.dim(4),
    });
}

Tensor spatial_norm(Backend& backend, ComponentExecutor& executor,
                    const Tensor& input, const Tensor& latent,
                    const std::string& prefix) {
    Tensor resized = resize_like(backend, latent, input);
    Tensor scale = causal_first_conv(backend, executor, resized, prefix + ".conv_y.conv");
    Tensor shift = causal_first_conv(backend, executor, resized, prefix + ".conv_b.conv");
    Tensor normalized = executor.group_norm(input, prefix + ".norm_layer", 32, 1e-6f);
    return backend.add(backend.mul(normalized, scale), shift);
}

Tensor decoder_residual(Backend& backend, ComponentExecutor& executor,
                        const Tensor& input, const Tensor& latent,
                        const std::string& prefix) {
    Tensor hidden = spatial_norm(backend, executor, input, latent, prefix + ".norm1");
    hidden = causal_first_conv(backend, executor, executor.silu(hidden), prefix + ".conv1.conv");
    hidden = spatial_norm(backend, executor, hidden, latent, prefix + ".norm2");
    hidden = causal_first_conv(backend, executor, executor.silu(hidden), prefix + ".conv2.conv");
    Tensor residual = input;
    if (executor.contains(prefix + ".conv_shortcut.weight"))
        residual = executor.conv3d(input, prefix + ".conv_shortcut");
    return backend.add(residual, hidden);
}

Tensor decoder_upsample(Backend& backend, ComponentExecutor& executor,
                        const Tensor& input, int block) {
    Tensor hidden = input;
    if (block < 2 && input.dim(2) > 1) {
        if (input.dim(2) % 2 == 1) {
            Tensor first = backend.interpolate_nearest(
                backend.slice(input, 2, 0, 1), {1.0, 2.0, 2.0});
            Tensor rest = backend.interpolate_nearest(
                backend.slice(input, 2, 1, input.dim(2)), {2.0, 2.0, 2.0});
            hidden = backend.concat({first, rest}, 2);
        } else {
            hidden = backend.interpolate_nearest(input, {2.0, 2.0, 2.0});
        }
    } else {
        hidden = backend.interpolate_nearest(input, {1.0, 2.0, 2.0});
    }
    return executor.conv2d_frames(
        hidden, "decoder.up_blocks." + std::to_string(block) + ".upsamplers.0.conv");
}

Tensor decode_video(Backend& backend, const PrecisionPolicy&, WeightMap weights,
                    const Tensor& encoded) {
    ComponentExecutor executor(backend, std::move(weights));
    Tensor latent = backend.mul(encoded, scalar_f32(1.0f / 1.15258426f));
    Tensor hidden = causal_first_conv(backend, executor, latent, "decoder.conv_in.conv");
    for (int index = 0; index < 2; ++index)
        hidden = decoder_residual(backend, executor, hidden, latent,
                                  "decoder.mid_block.resnets." + std::to_string(index));
    for (int block = 0; block < 4; ++block) {
        for (int index = 0; index < 4; ++index)
            hidden = decoder_residual(
                backend, executor, hidden, latent,
                "decoder.up_blocks." + std::to_string(block) +
                    ".resnets." + std::to_string(index));
        if (block < 3) hidden = decoder_upsample(backend, executor, hidden, block);
    }
    hidden = spatial_norm(backend, executor, hidden, latent, "decoder.norm_out");
    hidden = causal_first_conv(backend, executor, executor.silu(hidden), "decoder.conv_out.conv");
    return executor.video_range(hidden);
}

class CogVideoXDenoiser final : public Denoiser {
public:
    CogVideoXDenoiser(Backend& backend, const PrecisionPolicy& policy,
                      WeightMap weights, const Tensor& positive,
                      const Tensor& negative, bool trace)
        : backend_(backend), policy_(policy), weights_(std::move(weights)), trace_enabled_(trace) {
        const DType dtype = policy.boundary_dtype(PrecisionSemantic::Conditioning);
        text_ = backend.concat({backend.copy_to_device(negative, dtype),
                                backend.copy_to_device(positive, dtype)}, 0);
    }
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        trace_.clear();
        Tensor doubled = backend_.concat({latent, latent}, 0);
        float value = timestep.dtype() == DType::I64
            ? static_cast<float>(read_scalar_i64(timestep)) : read_scalar_f32(timestep);
        Tensor time = host_f32({2}, {value, value});
        Tensor prediction = full_denoiser(
            backend_, policy_, weights_, doubled, time, text_,
            trace_enabled_ ? &trace_ : nullptr);
        return backend_.split(prediction, {1, 1}, 0);
    }
    TensorBundle take_trace() override {
        TensorBundle result = std::move(trace_); trace_.clear(); return result;
    }
private:
    Backend& backend_;
    const PrecisionPolicy& policy_;
    WeightMap weights_;
    Tensor text_;
    bool trace_enabled_;
    TensorBundle trace_;
};

class CogVideoXArchitecture final : public Architecture {
public:
    explicit CogVideoXArchitecture(const VrmModel& model)
        : denoiser_(model.bindings(model.graph().at("architecture_graph"))),
          component_(model.bindings(model.graph().at("component_graphs").array().at(0))) {}
    std::unique_ptr<Denoiser> create_denoiser(
            Backend& backend, const PrecisionPolicy& policy,
            const TensorBundle& input) override {
        return std::make_unique<CogVideoXDenoiser>(
            backend, policy, denoiser_, input.at("positive"), input.at("negative"),
            input.contains("audit_trace") && read_scalar_i64(input.at("audit_trace")) != 0);
    }
    SamplingProgram create_program(const TensorBundle& input) override {
        const Tensor& shape = input.at("latent_shape");
        const Tensor& timesteps = input.at("timesteps");
        const Tensor& alpha = input.at("alpha_cumprod");
        const Tensor& previous = input.at("previous_alpha_cumprod");
        require(shape.device() == Device::CPU && shape.dtype() == DType::I64 &&
                    shape.shape() == std::vector<int64_t>({5}),
                "CogVideoX latent shape declaration mismatch");
        require(timesteps.device() == Device::CPU && timesteps.dtype() == DType::I64 &&
                    timesteps.ndim() == 1 && alpha.device() == Device::CPU &&
                    previous.device() == Device::CPU && alpha.dtype() == DType::F32 &&
                    previous.dtype() == DType::F32 && alpha.shape() == timesteps.shape() &&
                    previous.shape() == timesteps.shape(),
                "CogVideoX typed AlphaCumprod table declaration mismatch");
        SamplingProgram program;
        program.latent_shape.assign(shape.data_as<int64_t>(), shape.data_as<int64_t>() + 5);
        program.seed = static_cast<uint64_t>(read_scalar_i64(input.at("seed")));
        program.steps = static_cast<int>(timesteps.dim(0));
        const float cfg = read_scalar_f32(input.at("cfg_scale"));
        program.guidance_mode = GuidanceMode::CFG;
        program.guidance_coefficients = {1.0f - cfg, cfg};
        std::vector<AlphaCumprodScheduleTransition> transitions;
        for (int index = 0; index < program.steps; ++index)
            transitions.push_back({scalar_i64(timesteps.data_as<int64_t>()[index]),
                                   alpha.data_as<float>()[index],
                                   previous.data_as<float>()[index]});
        program.contract.emplace(
            PredictionContract{PredictionSemantic::V},
            SolverContract{SolverSemantic::AffineFirstOrder, 1},
            ScheduleContract::alpha_cumprod(std::move(transitions)));
        return program;
    }
    Tensor decode(Backend& backend, const PrecisionPolicy& policy,
                  const Tensor& latent, const TensorBundle&) override {
        return decode_video(backend, policy, component_, latent);
    }
private:
    WeightMap denoiser_;
    WeightMap component_;
};

}  // namespace

std::unique_ptr<Architecture> make_cogvideox_architecture(const VrmModel& model) {
    return std::make_unique<CogVideoXArchitecture>(model);
}

}  // namespace vrhino
