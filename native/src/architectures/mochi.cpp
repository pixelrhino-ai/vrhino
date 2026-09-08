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

constexpr int kHeads = 24;
constexpr int kHeadWidth = 128;

float cpu_value(const Tensor& tensor, int64_t index) {
    require(tensor.device() == Device::CPU, "Mochi host parameter expected");
    if (tensor.dtype() == DType::F32) return tensor.data_as<float>()[index];
    require(tensor.dtype() == DType::BF16, "Mochi host parameter dtype mismatch");
    uint32_t bits = static_cast<uint32_t>(tensor.data_as<uint16_t>()[index]) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::pair<Tensor, Tensor> learned_mixed_rope(
        Backend& backend, const PrecisionPolicy& policy, const Tensor& frequencies,
                                             int64_t frames, int64_t height, int64_t width) {
    require(frequencies.shape() == std::vector<int64_t>({3, kHeads, kHeadWidth / 2}),
            "Mochi learned RoPE frequency shape mismatch");
    const int64_t tokens = frames * height * width;
    std::vector<float> cosine(static_cast<size_t>(tokens * kHeads * kHeadWidth));
    std::vector<float> sine(cosine.size());
    const float scale = std::sqrt(192.0f * 192.0f / static_cast<float>(height * width));
    int64_t token = 0;
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t h = 0; h < height; ++h) {
            const float hp = (-static_cast<float>(height) / 2.0f + h + 0.5f) * scale;
            for (int64_t w = 0; w < width; ++w, ++token) {
                const float wp = (-static_cast<float>(width) / 2.0f + w + 0.5f) * scale;
                for (int head = 0; head < kHeads; ++head) {
                    for (int pair = 0; pair < kHeadWidth / 2; ++pair) {
                        const int64_t stride = kHeads * (kHeadWidth / 2);
                        const float phase = static_cast<float>(t) * cpu_value(frequencies, head * 64 + pair)
                            + hp * cpu_value(frequencies, stride + head * 64 + pair)
                            + wp * cpu_value(frequencies, 2 * stride + head * 64 + pair);
                        const size_t base = (static_cast<size_t>(token) * kHeads + head) * kHeadWidth + 2 * pair;
                        cosine[base] = cosine[base + 1] = std::cos(phase);
                        sine[base] = sine[base + 1] = std::sin(phase);
                    }
                }
            }
        }
    }
    const DType dtype = policy.operation_output_dtype(
        PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute);
    return {
        backend.copy_to_device(host_f32({1, tokens, kHeads, kHeadWidth}, cosine), dtype),
        backend.copy_to_device(host_f32({1, tokens, kHeads, kHeadWidth}, sine), dtype),
    };
}

PreparedTensorHandle prepare_learned_mixed_rope(
        PreparedTensorCache& cache, Backend& backend,
        const PrecisionPolicy& policy, const Tensor& frequencies,
        int64_t frames, int64_t height, int64_t width) {
    require(frequencies.device().is_host(),
            "Prepared learned mixed RoPE frequencies must be host-resident");
    const DType dtype = policy.operation_output_dtype(
        PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute);
    PreparedTensorKey key;
    key.operation = "positional.learned_mixed_rope.v1";
    key.input_shape = frequencies.shape();
    key.input_strides = frequencies.strides();
    key.input_dtype = frequencies.dtype();
    key.input_device_type = frequencies.device().type;
    key.input_device_index = frequencies.device().index;
    key.input_identity = reinterpret_cast<uintptr_t>(frequencies.data());
    key.input_content_hash = hash_host_tensor_content(frequencies);
    key.parameter_words = {
        static_cast<uint64_t>(frames), static_cast<uint64_t>(height),
        static_cast<uint64_t>(width), static_cast<uint64_t>(kHeads),
        static_cast<uint64_t>(kHeadWidth)};
    key.target_dtype = dtype;
    key.target_backend = backend.name();
    key.target_device_index = 0;
    return cache.prepare(key, [&] {
        auto [cosine, sine] = learned_mixed_rope(
            backend, policy, frequencies, frames, height, width);
        const uint64_t transfer_bytes =
            static_cast<uint64_t>(frames) * static_cast<uint64_t>(height) *
            static_cast<uint64_t>(width) * static_cast<uint64_t>(kHeads) *
            static_cast<uint64_t>(kHeadWidth) * sizeof(float) * 2;
        return PreparedTensorMaterialization{
            {std::move(cosine), std::move(sine)}, 1, 2, transfer_bytes};
    });
}

Tensor mask_with_prefix(const Tensor& mask, int64_t prefix) {
    require(mask.device() == Device::CPU && mask.dtype() == DType::Bool && mask.ndim() == 2,
            "Mochi conditioning mask contract mismatch");
    const int64_t batch = mask.dim(0), length = mask.dim(1);
    std::vector<uint8_t> values(static_cast<size_t>(batch * (prefix + length)), 1);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < length; ++i)
            values[static_cast<size_t>(b * (prefix + length) + prefix + i)] =
                mask.data_as<uint8_t>()[b * length + i];
    }
    return host_bool({batch, 1, prefix + length}, values);
}

Tensor time_embedding(Backend& backend, const PrecisionPolicy& policy,
                      const WeightMap& weights, const Tensor& timestep) {
    Tensor x = backend.sinusoidal_embedding(backend.reshape(timestep, {-1}), 256, true, 0.0, false);
    x = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, x,
        weights.at("time_embed.timestep_embedder.linear_1.weight"),
        &weights.at("time_embed.timestep_embedder.linear_1.bias"));
    x = backend.activation(x, Activation::Silu);
    return operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, x,
        weights.at("time_embed.timestep_embedder.linear_2.weight"),
        &weights.at("time_embed.timestep_embedder.linear_2.bias"));
}

std::pair<Tensor, Tensor> conditioning(Backend& backend, const PrecisionPolicy& policy,
                                       const WeightMap& weights,
                                       const Tensor& timestep, const Tensor& text,
                                       const Tensor& mask) {
    const DType compute_dtype = policy.operation_compute_dtype(
        PrecisionOperation::Conditioning);
    Tensor mask_value = backend.reshape(backend.cast(mask, compute_dtype),
                                        {mask.dim(0), mask.dim(1), 1});
    Tensor count = backend.clamp(backend.reduce_sum(mask_value, 1, true), 1.0f, 1.0e9f);
    Tensor pooled = backend.div(backend.reduce_sum(backend.mul(text, mask_value), 1, true), count);
    Tensor tokens = backend.concat({pooled, text}, 1);
    Tensor kv = backend.linear(tokens, weights.at("time_embed.pooler.to_kv.weight"),
                               &weights.at("time_embed.pooler.to_kv.bias"));
    auto kv_parts = backend.split(kv, {4096, 4096}, -1);
    Tensor key = backend.reshape(kv_parts[0], {text.dim(0), tokens.dim(1), 8, 512});
    Tensor value = backend.reshape(kv_parts[1], {text.dim(0), tokens.dim(1), 8, 512});
    Tensor query = backend.linear(pooled, weights.at("time_embed.pooler.to_q.weight"),
                                  &weights.at("time_embed.pooler.to_q.bias"));
    query = backend.reshape(query, {text.dim(0), 1, 8, 512});
    Tensor pool_mask = mask_with_prefix(mask, 1);
    pooled = backend.reshape(operation_attention(
                                 backend, policy, query, key, value, &pool_mask),
                             {text.dim(0), 1, 4096});
    pooled = backend.reshape(backend.linear(pooled, weights.at("time_embed.pooler.to_out.weight"),
                                            &weights.at("time_embed.pooler.to_out.bias")),
                             {text.dim(0), 3072});
    Tensor context = backend.linear(text, weights.at("time_embed.caption_proj.weight"),
                                    &weights.at("time_embed.caption_proj.bias"));
    return {backend.add(time_embedding(backend, policy, weights, timestep), pooled), context};
}

struct QKV { Tensor q, k, v; };

QKV qkv(Backend& backend, const PrecisionPolicy& policy,
        const WeightMap& weights, const Tensor& input, bool context,
        TensorBundle* trace = nullptr, const std::string& trace_prefix = {}) {
    const std::string norm_prefix = context ? "added_" : "";
    const std::string q_name = context ? "attn1.add_q_proj.weight" : "attn1.to_q.weight";
    const std::string k_name = context ? "attn1.add_k_proj.weight" : "attn1.to_k.weight";
    const std::string v_name = context ? "attn1.add_v_proj.weight" : "attn1.to_v.weight";
    Tensor q_projected = backend.linear(input, weights.at(q_name));
    Tensor k_projected = backend.linear(input, weights.at(k_name));
    Tensor v_projected = backend.linear(input, weights.at(v_name));
    if (trace) {
        (*trace)[trace_prefix + ".q_projection"] = q_projected;
        (*trace)[trace_prefix + ".k_projection"] = k_projected;
        (*trace)[trace_prefix + ".v_projection"] = v_projected;
    }
    Tensor q = split_heads(backend, q_projected, kHeads);
    Tensor k = split_heads(backend, k_projected, kHeads);
    Tensor v = split_heads(backend, v_projected, kHeads);
    q = attention_norm(backend, policy, q,
                       &weights.at("attn1.norm_" + norm_prefix + "q.weight"), 1e-5f);
    k = attention_norm(backend, policy, k,
                       &weights.at("attn1.norm_" + norm_prefix + "k.weight"), 1e-5f);
    if (trace) {
        (*trace)[trace_prefix + ".q_norm"] = q;
        (*trace)[trace_prefix + ".k_norm"] = k;
    }
    return {q, k, v};
}

Tensor swiglu(Backend& backend, const WeightMap& weights, const Tensor& input,
              const std::string& prefix) {
    BackendProfileRegion profile(backend, "mlp.total");
    Tensor projected = backend.linear(input, weights.at(prefix + ".net.0.proj.weight"));
    auto parts = backend.split(projected, {projected.dim(-1) / 2, projected.dim(-1) / 2}, -1);
    Tensor hidden = backend.mul(parts[0], backend.activation(parts[1], Activation::Silu));
    return backend.linear(hidden, weights.at(prefix + ".net.2.weight"));
}

Tensor normalized_gated_residual(Backend& backend, const PrecisionPolicy& policy,
                                 const Tensor& residual,
                                 const Tensor& branch, const Tensor& gate) {
    Tensor normalized = backend.rms_norm(branch, nullptr, 1e-6f);
    Tensor bounded = backend.activation(gate, Activation::Tanh);
    return gated_residual(backend, policy, residual, normalized,
                          backend.reshape(bounded, {residual.dim(0), 1, residual.dim(-1)}));
}

std::pair<Tensor, Tensor> asymmetric_block(Backend& backend, const PrecisionPolicy& policy,
                                           const WeightMap& weights,
                                           const Tensor& image, const Tensor& context,
                                           const Tensor& cond, const Tensor& text_mask,
                                           const Tensor& rope_cos, const Tensor& rope_sin,
                                           bool context_pre_only,
                                           TensorBundle* trace = nullptr) {
    if (trace) {
        (*trace)["block.0.input.image"] = image;
        (*trace)["block.0.input.context"] = context;
    }
    Tensor image_mod = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(cond, Activation::Silu),
        weights.at("norm1.linear.weight"), &weights.at("norm1.linear.bias"));
    auto im = backend.split(image_mod, {image.dim(-1), image.dim(-1), image.dim(-1), image.dim(-1)}, -1);
    Tensor image_input = backend.mul(backend.rms_norm(image, nullptr, 1e-6f),
        backend.add(scalar_f32(1.0f), backend.reshape(im[0], {image.dim(0), 1, image.dim(-1)})));
    if (trace) (*trace)["block.0.pre_attention_norm.image"] = image_input;

    Tensor context_input;
    std::vector<Tensor> cm;
    if (context_pre_only) {
        Tensor scale = operation_linear(
            backend, policy, PrecisionOperation::Modulation,
            PrecisionSemantic::TemporaryCompute,
            backend.activation(cond, Activation::Silu),
            weights.at("norm1_context.linear_1.weight"),
            &weights.at("norm1_context.linear_1.bias"));
        context_input = backend.mul(backend.rms_norm(context, nullptr, 1e-6f),
            backend.add(scalar_f32(1.0f), backend.reshape(scale, {context.dim(0), 1, context.dim(-1)})));
    } else {
        Tensor context_mod = operation_linear(
            backend, policy, PrecisionOperation::Modulation,
            PrecisionSemantic::TemporaryCompute,
            backend.activation(cond, Activation::Silu),
            weights.at("norm1_context.linear.weight"),
            &weights.at("norm1_context.linear.bias"));
        cm = backend.split(context_mod,
                           {context.dim(-1), context.dim(-1), context.dim(-1), context.dim(-1)}, -1);
        context_input = backend.mul(backend.rms_norm(context, nullptr, 1e-6f),
            backend.add(scalar_f32(1.0f), backend.reshape(cm[0], {context.dim(0), 1, context.dim(-1)})));
    }
    if (trace) (*trace)["block.0.pre_attention_norm.context"] = context_input;

    QKV iqkv = qkv(backend, policy, weights, image_input, false, trace, "block.0.image_attention");
    iqkv.q = operation_rope(backend, policy, iqkv.q, rope_cos, rope_sin);
    iqkv.k = operation_rope(backend, policy, iqkv.k, rope_cos, rope_sin);
    QKV cqkv = qkv(backend, policy, weights, context_input, true, trace, "block.0.context_attention");
    Tensor joint_mask = mask_with_prefix(text_mask, image.dim(1));
    Tensor attention_heads = operation_attention(backend, policy,
        backend.concat({iqkv.q, cqkv.q}, 1), backend.concat({iqkv.k, cqkv.k}, 1),
        backend.concat({iqkv.v, cqkv.v}, 1), &joint_mask);
    if (trace) (*trace)["block.0.attention_heads"] = attention_heads;
    Tensor attended = merge_heads(backend, attention_heads);
    if (trace) (*trace)["block.0.attention_merged"] = attended;
    auto outputs = backend.split(attended, {image.dim(1), context.dim(1)}, 1);
    if (trace) {
        (*trace)["block.0.image_attention.attention_merged"] = outputs[0];
        (*trace)["block.0.context_attention.attention_merged"] = outputs[1];
    }

    Tensor image_branch = backend.linear(outputs[0], weights.at("attn1.to_out.0.weight"),
                                         &weights.at("attn1.to_out.0.bias"));
    if (trace) (*trace)["block.0.image_attention.output_projection"] = image_branch;
    Tensor image_hidden = normalized_gated_residual(
        backend, policy, image, image_branch, im[1]);
    if (trace) (*trace)["block.0.image_first_residual"] = image_hidden;
    Tensor image_ff_input = backend.mul(backend.rms_norm(image_hidden, nullptr, 1e-6f),
        backend.add(scalar_f32(1.0f), backend.reshape(im[2], {image.dim(0), 1, image.dim(-1)})));
    Tensor image_ff = swiglu(backend, weights, image_ff_input, "ff");
    if (trace) (*trace)["block.0.image_ffn_output"] = image_ff;
    image_hidden = normalized_gated_residual(
        backend, policy, image_hidden, image_ff, im[3]);
    if (trace) (*trace)["block.0.image_output"] = image_hidden;

    Tensor context_hidden = context;
    if (!context_pre_only) {
        Tensor context_branch = backend.linear(outputs[1], weights.at("attn1.to_add_out.weight"),
                                               &weights.at("attn1.to_add_out.bias"));
        if (trace) (*trace)["block.0.context_attention.output_projection"] = context_branch;
        context_hidden = normalized_gated_residual(
            backend, policy, context, context_branch, cm[1]);
        Tensor context_ff_input = backend.mul(backend.rms_norm(context_hidden, nullptr, 1e-6f),
            backend.add(scalar_f32(1.0f), backend.reshape(cm[2], {context.dim(0), 1, context.dim(-1)})));
        Tensor context_ff = swiglu(backend, weights, context_ff_input, "ff_context");
        if (trace) (*trace)["block.0.context_ffn_output"] = context_ff;
        context_hidden = normalized_gated_residual(
            backend, policy, context_hidden, context_ff, cm[3]);
        if (trace) (*trace)["block.0.context_output"] = context_hidden;
    }
    return {image_hidden, context_hidden};
}

Tensor patch_embed(Backend& backend, const WeightMap& weights, const Tensor& latent) {
    const int64_t b = latent.dim(0), t = latent.dim(2), h = latent.dim(3), w = latent.dim(4);
    Tensor frames = backend.reshape(backend.permute(latent, {0, 2, 1, 3, 4}),
                                    {b * t, latent.dim(1), h, w});
    Tensor patched = backend.conv2d(frames, weights.at("patch_embed.proj.weight"),
                                    &weights.at("patch_embed.proj.bias"), {2, 2}, {0, 0});
    patched = backend.reshape(patched, {b, t, patched.dim(1), patched.dim(2), patched.dim(3)});
    patched = backend.permute(patched, {0, 1, 3, 4, 2});
    return backend.reshape(patched, {b, t * patched.dim(2) * patched.dim(3), patched.dim(4)});
}

Tensor full_denoiser(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const Tensor& latent,
                     const Tensor& timestep, const Tensor& text,
                     const Tensor& text_mask, const Tensor& rope_cos,
                     const Tensor& rope_sin,
                     TensorBundle* trace) {
    require(latent.ndim() == 5 && latent.dim(1) == 12 && latent.dim(3) % 2 == 0 &&
            latent.dim(4) % 2 == 0, "Mochi latent contract mismatch");
    auto [cond, context] = conditioning(
        backend, policy, weights, timestep, text, text_mask);
    if (trace) {
        (*trace)["conditioning.temb"] = cond;
        (*trace)["conditioning.context"] = context;
    }
    Tensor image = patch_embed(backend, weights, latent);
    if (trace) (*trace)["input_projection"] = image;
    for (int index = 0; index < 48; ++index) {
        auto output = asymmetric_block(backend, policy,
            weights.prefix("transformer_blocks." + std::to_string(index) + "."),
            image, context, cond, text_mask, rope_cos, rope_sin, index == 47,
            trace && index == 0 ? trace : nullptr);
        image = std::move(output.first);
        context = std::move(output.second);
        if (trace)
            (*trace)["block." + std::to_string(index)] = image;
    }
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(cond, Activation::Silu),
        weights.at("norm_out.linear.weight"), &weights.at("norm_out.linear.bias"));
    auto parts = backend.split(modulation, {3072, 3072}, -1);
    image = modulate(backend, policy, backend.layer_norm(image, nullptr, nullptr, 1e-6f),
                     backend.reshape(parts[1], {image.dim(0), 1, 3072}),
                     backend.reshape(parts[0], {image.dim(0), 1, 3072}));
    image = producer_linear(backend, policy, PrecisionSemantic::DenoiserOutput,
                            image, weights.at("proj_out.weight"),
                            &weights.at("proj_out.bias"));
    const int64_t b = latent.dim(0), t = latent.dim(2), h = latent.dim(3) / 2, w = latent.dim(4) / 2;
    image = backend.reshape(image, {b, t, h, w, 2, 2, 12});
    image = backend.permute(image, {0, 6, 1, 2, 4, 3, 5});
    image = backend.reshape(image, {b, 12, t, h * 2, w * 2});
    if (trace) (*trace)["output_projection"] = image;
    return image;
}

Tensor frame_group_norm(Backend& backend, ComponentExecutor& executor, const Tensor& input,
                        const std::string& prefix) {
    const int64_t b = input.dim(0), c = input.dim(1), t = input.dim(2), h = input.dim(3), w = input.dim(4);
    Tensor frames = backend.reshape(backend.permute(input, {0, 2, 1, 3, 4}), {b * t, c, h, w});
    frames = executor.group_norm(frames, prefix + ".norm_layer", 32, 1e-5f);
    return backend.permute(backend.reshape(frames, {b, t, c, h, w}), {0, 2, 1, 3, 4});
}

Tensor decoder_residual(Backend& backend, ComponentExecutor& executor, const Tensor& input,
                        const std::string& prefix) {
    Tensor hidden = frame_group_norm(backend, executor, input, prefix + ".norm1");
    hidden = executor.causal_conv3d(executor.silu(hidden), prefix + ".conv1.conv", PadMode::Replicate);
    hidden = frame_group_norm(backend, executor, hidden, prefix + ".norm2");
    hidden = executor.causal_conv3d(executor.silu(hidden), prefix + ".conv2.conv", PadMode::Replicate);
    return backend.add(input, hidden);
}

Tensor channel_linear(Backend& backend, ComponentExecutor& executor, const Tensor& input,
                      const std::string& prefix) {
    Tensor channels_last = backend.permute(input, {0, 2, 3, 4, 1});
    return backend.permute(executor.linear(channels_last, prefix), {0, 4, 1, 2, 3});
}

Tensor decode_mochi(Backend& backend, const PrecisionPolicy& policy,
                    const WeightMap& weights, const Tensor& latent,
                    TensorBundle* trace) {
    static const std::vector<float> mean = {
        -0.0673089595f, -0.0380113815f, -0.0747782091f, -0.0556526447f,
         0.0127672315f, -0.0470354275f,  0.0438969679f, -0.0934630571f,
        -0.0991831476f, -0.0087297934f, -0.0119315563f, -0.0321993392f,
    };
    static const std::vector<float> stddev = {
        0.9263795028f, 0.9248894543f, 0.9393059391f, 0.9592537328f,
        0.8244560133f, 0.9172599754f, 0.9294154431f, 1.3720942358f,
        0.8813936689f, 0.9168315692f, 0.9185249279f, 0.9274757571f,
    };
    ComponentExecutor executor(backend, weights);
    const DType vae_input_dtype = policy.boundary_dtype(
        PrecisionSemantic::VaeInput);
    Tensor x = backend.add(backend.mul(backend.cast(latent, vae_input_dtype),
                                      host_f32({1, 12, 1, 1, 1}, stddev)),
                           host_f32({1, 12, 1, 1, 1}, mean));
    x = executor.conv3d(x, "decoder.conv_in");
    for (int index = 0; index < 3; ++index)
        x = decoder_residual(backend, executor, x, "decoder.block_in.resnets." + std::to_string(index));
    if (trace) (*trace)["vae.representative_block"] = x;
    const int layers[] = {6, 4, 3};
    const std::vector<std::vector<int64_t>> factors = {{3, 2, 2}, {2, 2, 2}, {1, 2, 2}};
    for (int block = 0; block < 3; ++block) {
        for (int index = 0; index < layers[block]; ++index)
            x = decoder_residual(backend, executor, x,
                "decoder.up_blocks." + std::to_string(block) + ".resnets." + std::to_string(index));
        x = executor.pixel_shuffle(channel_linear(backend, executor, x,
            "decoder.up_blocks." + std::to_string(block) + ".proj"), factors[block]);
    }
    for (int index = 0; index < 3; ++index)
        x = decoder_residual(backend, executor, x, "decoder.block_out.resnets." + std::to_string(index));
    x = channel_linear(backend, executor, executor.silu(x), "decoder.proj_out");
    if (x.dim(2) >= 6) x = backend.slice(x, 2, 5, x.dim(2));
    x = executor.video_range(x);
    x = backend.cast(x, policy.boundary_dtype(PrecisionSemantic::VideoOutput));
    if (trace) (*trace)["vae.output"] = x;
    return x;
}

std::vector<float> linear_quadratic_schedule(
        int steps, float threshold, int linear_steps) {
    const double threshold_value = static_cast<double>(threshold);
    std::vector<float> sigma;
    for (int i = 0; i < linear_steps; ++i)
        sigma.push_back(static_cast<float>(
            1.0 - i * threshold_value / linear_steps));
    const double diff = linear_steps - threshold_value * steps;
    const int quadratic_steps = steps - linear_steps;
    const double a = diff /
        (linear_steps * quadratic_steps * quadratic_steps);
    const double b = threshold_value / linear_steps -
        2.0 * diff / (quadratic_steps * quadratic_steps);
    const double c = a * linear_steps * linear_steps;
    for (int i = linear_steps; i < steps; ++i)
        sigma.push_back(static_cast<float>(
            1.0 - (a * i * i + b * i + c)));
    sigma.push_back(0.0f);
    return sigma;
}

struct MochiSamplingDeclaration {
    int steps = 2;
    float guidance_scale = 4.5f;
    float threshold_noise = 0.025f;
    int linear_steps = 0;

    static MochiSamplingDeclaration from_input(const TensorBundle& input) {
        MochiSamplingDeclaration declaration;
        if (input.contains("sampling_steps")) {
            const int64_t steps = read_scalar_i64(input.at("sampling_steps"));
            require(steps >= 2 && steps <= 1000,
                    "Mochi sampling_steps must be in [2, 1000]");
            declaration.steps = static_cast<int>(steps);
        }
        if (input.contains("guidance_scale"))
            declaration.guidance_scale =
                read_scalar_f32(input.at("guidance_scale"));
        if (input.contains("threshold_noise"))
            declaration.threshold_noise =
                read_scalar_f32(input.at("threshold_noise"));
        if (input.contains("linear_steps")) {
            const int64_t linear_steps =
                read_scalar_i64(input.at("linear_steps"));
            require(linear_steps >= 1 && linear_steps < declaration.steps,
                    "Mochi linear_steps must be in [1, sampling_steps)");
            declaration.linear_steps = static_cast<int>(linear_steps);
        } else {
            declaration.linear_steps = declaration.steps / 2;
        }
        declaration.validate();
        return declaration;
    }

    void validate() const {
        require(steps >= 2 && steps <= 1000,
                "Mochi sampling_steps must be in [2, 1000]");
        require(std::isfinite(guidance_scale),
                "Mochi guidance_scale must be finite");
        require(std::isfinite(threshold_noise) && threshold_noise > 0.0f &&
                    threshold_noise < 1.0f,
                "Mochi threshold_noise must be finite and in (0, 1)");
        require(linear_steps >= 1 && linear_steps < steps,
                "Mochi linear_steps must be in [1, sampling_steps)");
    }
};

class MochiDenoiser final : public Denoiser {
public:
    MochiDenoiser(Backend& backend, const PrecisionPolicy& policy,
                  WeightMap weights, const Tensor& positive,
                  const Tensor& negative, const Tensor& positive_mask,
                  const Tensor& negative_mask, int64_t frames,
                  int64_t height, int64_t width, bool trace_enabled)
        : backend_(backend), policy_(policy), weights_(std::move(weights)),
          trace_enabled_(trace_enabled) {
        require(positive.shape() == negative.shape() && positive.ndim() == 3 &&
                positive.dim(0) == 1 && positive.dim(2) == 4096,
                "Mochi conditioning tensor contract mismatch");
        require(positive_mask.shape() == negative_mask.shape() &&
                positive_mask.shape() == std::vector<int64_t>({1, positive.dim(1)}),
                "Mochi conditioning mask shape mismatch");
        const DType conditioning_dtype = policy_.boundary_dtype(
            PrecisionSemantic::Conditioning);
        text_ = backend_.concat({backend_.copy_to_device(negative, conditioning_dtype),
                                 backend_.copy_to_device(positive, conditioning_dtype)}, 0);
        std::vector<uint8_t> mask_values(static_cast<size_t>(2 * positive.dim(1)));
        std::memcpy(mask_values.data(), negative_mask.data(), static_cast<size_t>(positive.dim(1)));
        std::memcpy(mask_values.data() + positive.dim(1), positive_mask.data(), static_cast<size_t>(positive.dim(1)));
        mask_ = host_bool({2, positive.dim(1)}, mask_values);
        rope_ = prepare_learned_mixed_rope(
            prepared_tensors(), backend_, policy_, weights_.at("pos_frequencies"),
            frames, height, width);
    }
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        trace_.clear();
        const std::vector<Tensor>& rope = prepared_tensors().reuse(rope_);
        require(rope.size() == 2,
                "Learned mixed RoPE prepared tensor arity mismatch");
        Tensor doubled = backend_.concat({latent, latent}, 0);
        const float value = read_scalar_f32(timestep);
        Tensor expanded = host_f32({2}, {value, value});
        Tensor output = full_denoiser(backend_, policy_, weights_, doubled,
                                      expanded, text_, mask_, rope[0], rope[1],
                                      trace_enabled_ ? &trace_ : nullptr);
        return backend_.split(output, {1, 1}, 0);
    }
    TensorBundle take_trace() override {
        TensorBundle result = std::move(trace_); trace_.clear(); return result;
    }
private:
    Backend& backend_;
    const PrecisionPolicy& policy_;
    WeightMap weights_;
    Tensor text_, mask_;
    PreparedTensorHandle rope_;
    bool trace_enabled_ = false;
    TensorBundle trace_;
};

class MochiArchitecture final : public Architecture {
public:
    explicit MochiArchitecture(const VrmModel& model)
        : denoiser_(model.bindings(model.graph().at("architecture_graph"))),
          component_(model.bindings(model.graph().at("component_graphs").array().at(0))) {}
    std::unique_ptr<Denoiser> create_denoiser(
            Backend& backend, const PrecisionPolicy& policy,
            const TensorBundle& input) override {
        trace_enabled_ = input.contains("audit_trace") && read_scalar_i64(input.at("audit_trace")) != 0;
        std::vector<int64_t> latent_shape = {1, 12, 1, 2, 2};
        if (input.contains("latent_shape")) {
            const Tensor& shape = input.at("latent_shape");
            require(shape.device() == Device::CPU && shape.dtype() == DType::I64 &&
                    shape.shape() == std::vector<int64_t>({5}),
                    "Mochi latent_shape contract mismatch");
            latent_shape.assign(shape.data_as<int64_t>(),
                                shape.data_as<int64_t>() + 5);
        }
        require(latent_shape[0] == 1 && latent_shape[1] == 12 &&
                    latent_shape[2] > 0 && latent_shape[3] > 0 &&
                    latent_shape[4] > 0 && latent_shape[3] % 2 == 0 &&
                    latent_shape[4] % 2 == 0,
                "Mochi latent_shape is not legal");
        return std::make_unique<MochiDenoiser>(backend, policy, denoiser_, input.at("positive"),
            input.at("negative"), input.at("positive_mask"), input.at("negative_mask"),
            latent_shape[2], latent_shape[3] / 2, latent_shape[4] / 2,
            trace_enabled_);
    }
    SamplingProgram create_program(const TensorBundle& input) override {
        const MochiSamplingDeclaration declaration =
            MochiSamplingDeclaration::from_input(input);
        SamplingProgram program;
        program.latent_shape = {1, 12, 1, 2, 2};
        if (input.contains("latent_shape")) {
            const Tensor& shape = input.at("latent_shape");
            require(shape.device() == Device::CPU && shape.dtype() == DType::I64 &&
                    shape.shape() == std::vector<int64_t>({5}), "Mochi latent_shape contract mismatch");
            program.latent_shape.assign(shape.data_as<int64_t>(), shape.data_as<int64_t>() + 5);
            require(program.latent_shape[0] == 1 && program.latent_shape[1] == 12 &&
                    program.latent_shape[2] > 0 && program.latent_shape[3] > 0 &&
                    program.latent_shape[4] > 0 && program.latent_shape[3] % 2 == 0 &&
                    program.latent_shape[4] % 2 == 0, "Mochi latent_shape is not legal");
        }
        program.seed = read_scalar_i64(input.at("seed"));
        program.steps = declaration.steps;
        program.guidance_mode = GuidanceMode::CFG;
        program.guidance_coefficients = {
            1.0f - declaration.guidance_scale,
            declaration.guidance_scale};
        program.scheduler = SchedulerKind::Euler;
        program.sigmas = linear_quadratic_schedule(
            program.steps, declaration.threshold_noise,
            declaration.linear_steps);
        for (int index = 0; index < program.steps; ++index) {
            program.model_timesteps.push_back(scalar_f32((1.0f - program.sigmas[index]) * 1000.0f));
            program.update_deltas.push_back(program.sigmas[index] - program.sigmas[index + 1]);
        }
        return program;
    }
    Tensor decode(Backend& backend, const PrecisionPolicy& policy,
                  const Tensor& latent, const TensorBundle&) override {
        trace_.clear();
        return decode_mochi(
            backend, policy, component_, latent, trace_enabled_ ? &trace_ : nullptr);
    }
    TensorBundle take_trace() override {
        TensorBundle result = std::move(trace_); trace_.clear(); return result;
    }
private:
    WeightMap denoiser_, component_;
    bool trace_enabled_ = false;
    TensorBundle trace_;
};

}  // namespace

std::unique_ptr<Architecture> make_mochi_architecture(const VrmModel& model) {
    return std::make_unique<MochiArchitecture>(model);
}

}  // namespace vrhino
