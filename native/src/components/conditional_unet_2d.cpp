#include "vrhino/conditional_unet_2d.h"

#include <cmath>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

const Tensor* optional(const WeightMap& weights, const std::string& name) {
    return weights.find(name);
}

Tensor conv2d(Backend& backend, const WeightMap& weights, const Tensor& input,
              const std::string& prefix, int stride = 1, int padding = 1) {
    return backend.conv2d(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          {stride, stride}, {padding, padding});
}

Tensor linear(Backend& backend, const WeightMap& weights, const Tensor& input,
              const std::string& prefix) {
    return backend.linear(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          DType::F32, DType::F32);
}

Tensor group_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix,
                  float epsilon) {
    return backend.group_norm(input, 32,
                              optional(weights, prefix + ".weight"),
                              optional(weights, prefix + ".bias"), epsilon);
}

Tensor layer_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix) {
    return backend.layer_norm(input, optional(weights, prefix + ".weight"),
                              optional(weights, prefix + ".bias"), 1.0e-5f);
}

void observe(ConditionalUNet2DObservation* observation,
             const std::string& name, const Tensor& tensor) {
    if (observation) observation->tensors.emplace(name, tensor);
}

Tensor resnet(Backend& backend, const WeightMap& weights, const Tensor& input,
              const Tensor& time_embedding, const std::string& prefix,
              float epsilon) {
    Tensor hidden = group_norm(backend, weights, input, prefix + ".norm1", epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = conv2d(backend, weights, hidden, prefix + ".conv1");

    Tensor time = backend.activation(time_embedding, Activation::Silu);
    time = linear(backend, weights, time, prefix + ".time_emb_proj");
    time = backend.reshape(time, {time.dim(0), time.dim(1), 1, 1});
    hidden = backend.add(hidden, time);

    hidden = group_norm(backend, weights, hidden, prefix + ".norm2", epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = conv2d(backend, weights, hidden, prefix + ".conv2");
    Tensor residual = input;
    if (weights.contains(prefix + ".conv_shortcut.weight"))
        residual = conv2d(backend, weights, input, prefix + ".conv_shortcut", 1, 0);
    return backend.add(residual, hidden);
}

Tensor attention(Backend& backend, const WeightMap& weights,
                 const Tensor& query_input, const Tensor& key_value_input,
                 const std::string& prefix, int64_t heads) {
    const int64_t batch = query_input.dim(0);
    const int64_t query_tokens = query_input.dim(1);
    const int64_t key_tokens = key_value_input.dim(1);
    const int64_t width = query_input.dim(2);
    require(width % heads == 0, "conditional UNet attention head mismatch");
    const int64_t head_width = width / heads;
    Tensor query = backend.reshape(linear(backend, weights, query_input,
        prefix + ".to_q"), {batch, query_tokens, heads, head_width});
    Tensor key = backend.reshape(linear(backend, weights, key_value_input,
        prefix + ".to_k"), {batch, key_tokens, heads, head_width});
    Tensor value = backend.reshape(linear(backend, weights, key_value_input,
        prefix + ".to_v"), {batch, key_tokens, heads, head_width});
    Tensor hidden = backend.attention(query, key, value, nullptr, false,
        1.0f / std::sqrt(static_cast<float>(head_width)));
    hidden = backend.reshape(hidden, {batch, query_tokens, width});
    return linear(backend, weights, hidden, prefix + ".to_out.0");
}

Tensor transformer(Backend& backend, const WeightMap& weights,
                   const Tensor& input, const Tensor& conditioning,
                   const std::string& prefix, int64_t heads,
                   ConditionalUNet2DObservation* observation,
                   bool observe_first_attention) {
    require(input.ndim() == 4 && conditioning.ndim() == 3,
            "conditional UNet transformer input mismatch");
    const int64_t batch = input.dim(0), channels = input.dim(1);
    const int64_t height = input.dim(2), width = input.dim(3);
    Tensor hidden = group_norm(backend, weights, input, prefix + ".norm", 1.0e-6f);
    hidden = conv2d(backend, weights, hidden, prefix + ".proj_in", 1, 0);
    hidden = backend.permute(hidden, {0, 2, 3, 1});
    hidden = backend.reshape(hidden, {batch, height * width, channels});

    const std::string block = prefix + ".transformer_blocks.0";
    Tensor normalized = layer_norm(backend, weights, hidden, block + ".norm1");
    Tensor branch = attention(backend, weights, normalized, normalized,
                              block + ".attn1", heads);
    if (observe_first_attention)
        observe(observation, "first_self_attention", branch);
    hidden = backend.add(hidden, branch);

    normalized = layer_norm(backend, weights, hidden, block + ".norm2");
    branch = attention(backend, weights, normalized, conditioning,
                       block + ".attn2", heads);
    if (observe_first_attention)
        observe(observation, "first_audio_cross_attention", branch);
    hidden = backend.add(hidden, branch);

    normalized = layer_norm(backend, weights, hidden, block + ".norm3");
    Tensor projected = linear(backend, weights, normalized,
                              block + ".ff.net.0.proj");
    const int64_t half = projected.dim(-1) / 2;
    std::vector<Tensor> geglu = backend.split(projected, {half, half}, -1);
    Tensor feed_forward = backend.mul(
        geglu[0], backend.activation(geglu[1], Activation::Gelu));
    feed_forward = linear(backend, weights, feed_forward, block + ".ff.net.2");
    hidden = backend.add(hidden, feed_forward);

    hidden = backend.reshape(hidden, {batch, height, width, channels});
    hidden = backend.permute(hidden, {0, 3, 1, 2});
    hidden = conv2d(backend, weights, hidden, prefix + ".proj_out", 1, 0);
    return backend.add(input, hidden);
}

void validate_graph(const Json& graph) {
    const Json& config = graph.at("config");
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "conditional_unet_2d" &&
            graph.at("entry_point").string() == "execute" &&
            config.at("dtype").string() == "float32" &&
            config.at("input_channels").integer() == 8 &&
            config.at("output_channels").integer() == 4 &&
            config.at("conditioning_width").integer() == 384 &&
            config.at("layers_per_block").integer() == 2 &&
            config.at("attention_heads").integer() == 8 &&
            config.at("norm_num_groups").integer() == 32,
            "unsupported conditional UNet2D component graph");
}

}  // namespace

ConditionalUNet2DResult ConditionalUNet2DComponentExecutor::execute(
        const Json& graph, const Tensor& latent, const Tensor& timestep,
        const Tensor& conditioning,
        ConditionalUNet2DObservation* observation) {
    validate_graph(graph);
    require(latent.ndim() == 4 && latent.dtype() == DType::F32 &&
            latent.dim(1) == 8 && conditioning.ndim() == 3 &&
            conditioning.dtype() == DType::F32 && conditioning.dim(0) == latent.dim(0) &&
            conditioning.dim(2) == 384 && timestep.ndim() == 1 &&
            timestep.dim(0) == latent.dim(0),
            "conditional UNet2D execution input mismatch");
    const float epsilon = static_cast<float>(
        graph.at("config").at("norm_epsilon").number());
    const int64_t heads = graph.at("config").at("attention_heads").integer();

    Tensor time = backend_.cast(timestep, DType::F32);
    time = backend_.sinusoidal_embedding(time, 320, true, 0.0, false);
    time = linear(backend_, weights_, time, "time_embedding.linear_1");
    time = backend_.activation(time, Activation::Silu);
    time = linear(backend_, weights_, time, "time_embedding.linear_2");
    observe(observation, "timestep_embedding", time);

    Tensor hidden = conv2d(backend_, weights_, latent, "conv_in");
    observe(observation, "input_conv", hidden);
    std::vector<Tensor> residuals{hidden};

    for (int block = 0; block < 4; ++block) {
        for (int layer = 0; layer < 2; ++layer) {
            const std::string root = "down_blocks." + std::to_string(block);
            hidden = resnet(backend_, weights_, hidden, time,
                root + ".resnets." + std::to_string(layer), epsilon);
            if (block == 0 && layer == 0)
                observe(observation, "first_down_resnet", hidden);
            if (block < 3)
                hidden = transformer(backend_, weights_, hidden, conditioning,
                    root + ".attentions." + std::to_string(layer), heads,
                    observation, block == 0 && layer == 0);
            residuals.push_back(hidden);
        }
        if (block < 3) {
            hidden = conv2d(backend_, weights_, hidden,
                "down_blocks." + std::to_string(block) +
                    ".downsamplers.0.conv", 2, 1);
            if (block == 0) observe(observation, "first_downsample", hidden);
            if (block == 2) observe(observation, "deep_down_path", hidden);
            residuals.push_back(hidden);
        }
    }

    hidden = resnet(backend_, weights_, hidden, time,
                    "mid_block.resnets.0", epsilon);
    hidden = transformer(backend_, weights_, hidden, conditioning,
                         "mid_block.attentions.0", heads, observation, false);
    observe(observation, "mid_block", hidden);
    hidden = resnet(backend_, weights_, hidden, time,
                    "mid_block.resnets.1", epsilon);

    for (int block = 0; block < 4; ++block) {
        for (int layer = 0; layer < 3; ++layer) {
            require(!residuals.empty(), "conditional UNet skip stack underflow");
            Tensor skip = residuals.back();
            residuals.pop_back();
            hidden = backend_.concat({hidden, skip}, 1);
            const std::string root = "up_blocks." + std::to_string(block);
            hidden = resnet(backend_, weights_, hidden, time,
                root + ".resnets." + std::to_string(layer), epsilon);
            if (block == 3 && layer == 2)
                observe(observation, "deep_up_path", hidden);
            if (block > 0)
                hidden = transformer(backend_, weights_, hidden, conditioning,
                    root + ".attentions." + std::to_string(layer), heads,
                    observation, false);
            if (block == 2 && layer == 0)
                observe(observation, "representative_up_block", hidden);
        }
        if (block < 3) {
            hidden = backend_.interpolate_nearest(hidden, {2.0, 2.0});
            hidden = conv2d(backend_, weights_, hidden,
                "up_blocks." + std::to_string(block) +
                    ".upsamplers.0.conv");
        }
    }
    require(residuals.empty(), "conditional UNet skip stack was not exhausted");

    hidden = group_norm(backend_, weights_, hidden, "conv_norm_out", epsilon);
    observe(observation, "final_normalization", hidden);
    hidden = backend_.activation(hidden, Activation::Silu);
    observe(observation, "final_activation", hidden);
    hidden = conv2d(backend_, weights_, hidden, "conv_out");
    observe(observation, "final_conv", hidden);
    return {hidden};
}

}  // namespace vrhino
