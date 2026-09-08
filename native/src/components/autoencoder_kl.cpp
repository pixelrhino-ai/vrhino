#include "vrhino/autoencoder_kl.h"

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

Tensor norm(Backend& backend, const WeightMap& weights, const Tensor& input,
            const std::string& prefix, int groups, float epsilon) {
    return backend.group_norm(input, groups,
                              optional(weights, prefix + ".weight"),
                              optional(weights, prefix + ".bias"), epsilon);
}

Tensor resnet(Backend& backend, const WeightMap& weights, const Tensor& input,
              const std::string& prefix, int groups, float epsilon) {
    Tensor hidden = norm(backend, weights, input, prefix + ".norm1", groups, epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = conv2d(backend, weights, hidden, prefix + ".conv1");
    hidden = norm(backend, weights, hidden, prefix + ".norm2", groups, epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = conv2d(backend, weights, hidden, prefix + ".conv2");
    Tensor residual = input;
    if (weights.contains(prefix + ".conv_shortcut.weight"))
        residual = conv2d(backend, weights, residual,
                          prefix + ".conv_shortcut", 1, 0);
    return backend.add(residual, hidden);
}

Tensor spatial_attention(Backend& backend, const WeightMap& weights,
                         const Tensor& input, const std::string& prefix,
                         int groups, float epsilon) {
    require(input.ndim() == 4, "AutoencoderKL attention expects NCHW");
    const int64_t batch = input.dim(0), channels = input.dim(1);
    const int64_t height = input.dim(2), width = input.dim(3);
    Tensor hidden = norm(backend, weights, input, prefix + ".group_norm",
                         groups, epsilon);
    hidden = backend.permute(backend.reshape(hidden,
        {batch, channels, height * width}), {0, 2, 1});
    Tensor query = backend.reshape(linear(backend, weights, hidden,
        prefix + ".query"), {batch, height * width, 1, channels});
    Tensor key = backend.reshape(linear(backend, weights, hidden,
        prefix + ".key"), {batch, height * width, 1, channels});
    Tensor value = backend.reshape(linear(backend, weights, hidden,
        prefix + ".value"), {batch, height * width, 1, channels});
    hidden = backend.attention(query, key, value, nullptr, false,
                               1.0f / std::sqrt(static_cast<float>(channels)));
    hidden = backend.reshape(hidden, {batch, height * width, channels});
    hidden = linear(backend, weights, hidden, prefix + ".proj_attn");
    hidden = backend.reshape(backend.permute(hidden, {0, 2, 1}),
                             {batch, channels, height, width});
    return backend.add(input, hidden);
}

Tensor mid_block(Backend& backend, const WeightMap& weights, const Tensor& input,
                 const std::string& prefix, int groups, float epsilon) {
    Tensor hidden = resnet(backend, weights, input,
                           prefix + ".resnets.0", groups, epsilon);
    hidden = spatial_attention(backend, weights, hidden,
                               prefix + ".attentions.0", groups, epsilon);
    return resnet(backend, weights, hidden,
                  prefix + ".resnets.1", groups, epsilon);
}

void observe(AutoencoderKLObservation* observation, const std::string& name,
             const Tensor& tensor) {
    if (observation) observation->tensors.emplace(name, tensor);
}

std::vector<int64_t> channels(const Json& graph) {
    std::vector<int64_t> result;
    for (const Json& value : graph.at("config").at("block_out_channels").array())
        result.push_back(value.integer());
    require(result.size() == 4, "AutoencoderKL requires four channel stages");
    return result;
}

void validate_graph(const Json& graph) {
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "autoencoder_kl" &&
            graph.at("config").at("dtype").string() == "float32" &&
            graph.at("config").at("latent_channels").integer() == 4,
            "Unsupported AutoencoderKL component graph");
    const auto& entries = graph.at("entry_points").array();
    require(entries.size() == 2 && entries[0].string() == "encode" &&
            entries[1].string() == "decode",
            "AutoencoderKL entry-point contract mismatch");
}

}  // namespace

AutoencoderKLEncoderResult AutoencoderKLComponentExecutor::encode(
        const Json& graph, const Tensor& normalized_rgb,
        AutoencoderKLObservation* observation) {
    validate_graph(graph);
    require(normalized_rgb.ndim() == 4 && normalized_rgb.dtype() == DType::F32 &&
            normalized_rgb.dim(1) == 3,
            "AutoencoderKL encode expects FP32 NCHW RGB");
    const Json& config = graph.at("config");
    const int groups = static_cast<int>(config.at("norm_num_groups").integer());
    const float epsilon = static_cast<float>(config.at("norm_epsilon").number());
    const int layers = static_cast<int>(config.at("layers_per_block").integer());
    const std::vector<int64_t> stage_channels = channels(graph);
    Tensor hidden = conv2d(backend_, weights_, normalized_rgb, "encoder.conv_in");
    observe(observation, "initial_conv", hidden);
    for (size_t block = 0; block < stage_channels.size(); ++block) {
        for (int layer = 0; layer < layers; ++layer)
            hidden = resnet(backend_, weights_, hidden,
                "encoder.down_blocks." + std::to_string(block) + ".resnets." +
                    std::to_string(layer), groups, epsilon);
        if (block + 1 < stage_channels.size()) {
            hidden = backend_.pad(hidden, {0, 1, 0, 1}, 0.0f,
                                  PadMode::Constant);
            hidden = conv2d(backend_, weights_, hidden,
                "encoder.down_blocks." + std::to_string(block) +
                    ".downsamplers.0.conv", 2, 0);
        }
        observe(observation, "down_block_" + std::to_string(block), hidden);
    }
    hidden = mid_block(backend_, weights_, hidden, "encoder.mid_block",
                       groups, epsilon);
    observe(observation, "mid_block", hidden);
    hidden = norm(backend_, weights_, hidden, "encoder.conv_norm_out",
                  groups, epsilon);
    hidden = backend_.activation(hidden, Activation::Silu);
    hidden = conv2d(backend_, weights_, hidden, "encoder.conv_out");
    observe(observation, "pre_posterior", hidden);
    hidden = conv2d(backend_, weights_, hidden, "quant_conv", 1, 0);
    observe(observation, "quant_conv", hidden);
    std::vector<Tensor> moments = backend_.split(hidden, {4, 4}, 1);
    require(moments.size() == 2, "AutoencoderKL posterior split failed");
    Tensor logvar = backend_.clamp(moments[1], -30.0f, 20.0f);
    observe(observation, "posterior_mean", moments[0]);
    observe(observation, "posterior_logvar", logvar);
    return {moments[0], logvar};
}

AutoencoderKLPosteriorSample AutoencoderKLComponentExecutor::sample(
        const Json& graph, const Tensor& mean, const Tensor& logvar,
        RngState& component_rng, const Tensor* fixed_epsilon) {
    validate_graph(graph);
    require(mean.shape() == logvar.shape() && mean.dtype() == DType::F32 &&
            logvar.dtype() == DType::F32,
            "AutoencoderKL posterior parameter mismatch");
    Tensor epsilon = fixed_epsilon
        ? backend_.copy_to_device(*fixed_epsilon, DType::F32)
        : backend_.rng_normal(component_rng, mean.shape(), DType::F32);
    require(epsilon.shape() == mean.shape() && epsilon.dtype() == DType::F32,
            "AutoencoderKL epsilon mismatch");
    Tensor deviation = backend_.exp(backend_.mul(logvar, scalar_f32(0.5f)));
    Tensor sampled = backend_.add(mean, backend_.mul(deviation, epsilon));
    const float scale = static_cast<float>(
        graph.at("config").at("scaling_factor").number());
    return {epsilon, sampled, backend_.mul(sampled, scalar_f32(scale))};
}

AutoencoderKLDecoderResult AutoencoderKLComponentExecutor::decode(
        const Json& graph, const Tensor& scaled_latent,
        AutoencoderKLObservation* observation) {
    validate_graph(graph);
    require(scaled_latent.ndim() == 4 && scaled_latent.dtype() == DType::F32 &&
            scaled_latent.dim(1) == 4,
            "AutoencoderKL decode expects FP32 NCHW latent");
    const Json& config = graph.at("config");
    const int groups = static_cast<int>(config.at("norm_num_groups").integer());
    const float epsilon = static_cast<float>(config.at("norm_epsilon").number());
    const int layers = static_cast<int>(
        config.at("decoder_layers_per_block").integer());
    const float scale = static_cast<float>(config.at("scaling_factor").number());
    Tensor hidden = backend_.div(scaled_latent, scalar_f32(scale));
    observe(observation, "predecode_unscaled", hidden);
    hidden = conv2d(backend_, weights_, hidden, "post_quant_conv", 1, 0);
    hidden = conv2d(backend_, weights_, hidden, "decoder.conv_in");
    observe(observation, "initial_conv", hidden);
    hidden = mid_block(backend_, weights_, hidden, "decoder.mid_block",
                       groups, epsilon);
    observe(observation, "mid_block", hidden);
    const std::vector<int64_t> stage_channels = channels(graph);
    for (size_t block = 0; block < stage_channels.size(); ++block) {
        for (int layer = 0; layer < layers; ++layer)
            hidden = resnet(backend_, weights_, hidden,
                "decoder.up_blocks." + std::to_string(block) + ".resnets." +
                    std::to_string(layer), groups, epsilon);
        if (block + 1 < stage_channels.size()) {
            hidden = backend_.interpolate_nearest(hidden, {2.0, 2.0});
            hidden = conv2d(backend_, weights_, hidden,
                "decoder.up_blocks." + std::to_string(block) +
                    ".upsamplers.0.conv");
        }
        observe(observation, "up_block_" + std::to_string(block), hidden);
    }
    hidden = norm(backend_, weights_, hidden, "decoder.conv_norm_out",
                  groups, epsilon);
    hidden = backend_.activation(hidden, Activation::Silu);
    observe(observation, "final_norm_activation", hidden);
    Tensor decoded = conv2d(backend_, weights_, hidden, "decoder.conv_out");
    observe(observation, "decoded", decoded);
    Tensor rgb = backend_.clamp(backend_.add(
        backend_.mul(decoded, scalar_f32(0.5f)), scalar_f32(0.5f)), 0.0f, 1.0f);
    observe(observation, "rgb_0_1", rgb);
    return {decoded, rgb};
}

}  // namespace vrhino
