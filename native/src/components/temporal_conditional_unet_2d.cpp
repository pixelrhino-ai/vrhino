#include "vrhino/temporal_conditional_unet_2d.h"

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

void observe(TemporalConditionalUNetObservation* observation,
             const std::string& name, const Tensor& tensor) {
    if (observation && observation->capture) observation->capture(name, tensor);
}

Tensor linear(Backend& backend, const PrecisionPolicy& policy,
              const WeightMap& weights, const Tensor& input,
              const std::string& prefix,
              PrecisionOperation operation = PrecisionOperation::Linear) {
    return operation_linear(
        backend, policy, operation, PrecisionSemantic::TemporaryCompute,
        input, weights.at(prefix + ".weight"),
        optional(weights, prefix + ".bias"));
}

Tensor flatten_frames(Backend& backend, const Tensor& input) {
    require(input.ndim() == 5, "temporal UNet frame flatten requires rank five");
    Tensor output = backend.permute(input, {0, 2, 1, 3, 4});
    return backend.reshape(output, {input.dim(0) * input.dim(2), input.dim(1),
                                    input.dim(3), input.dim(4)});
}

Tensor unflatten_frames(Backend& backend, const Tensor& input,
                        int64_t batch, int64_t frames) {
    require(input.ndim() == 4 && input.dim(0) == batch * frames,
            "temporal UNet frame unflatten mismatch");
    Tensor output = backend.reshape(input, {batch, frames, input.dim(1),
                                            input.dim(2), input.dim(3)});
    return backend.permute(output, {0, 2, 1, 3, 4});
}

Tensor spatial_conv2d(Backend& backend, const PrecisionPolicy& policy,
                      const WeightMap& weights,
                      const Tensor& input, const std::string& prefix,
                      int stride = 1, int padding = 1) {
    const int64_t batch = input.dim(0), frames = input.dim(2);
    Tensor flat = flatten_frames(backend, input);
    flat = operation_conv2d(
        backend, policy, PrecisionSemantic::TemporaryCompute, flat,
        weights.at(prefix + ".weight"), optional(weights, prefix + ".bias"),
        {stride, stride}, {padding, padding});
    return unflatten_frames(backend, flat, batch, frames);
}

Tensor group_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix,
                  float epsilon) {
    return backend.group_norm(input, 32, optional(weights, prefix + ".weight"),
                              optional(weights, prefix + ".bias"), epsilon);
}

Tensor layer_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix) {
    return backend.layer_norm(input, optional(weights, prefix + ".weight"),
                              optional(weights, prefix + ".bias"), 1.0e-5f);
}

Tensor attention(Backend& backend, const PrecisionPolicy& policy,
                 const WeightMap& weights,
                 const Tensor& query_input, const Tensor& key_value_input,
                 const std::string& prefix, int64_t heads) {
    require(query_input.ndim() == 3 && key_value_input.ndim() == 3 &&
                query_input.dim(0) == key_value_input.dim(0),
            "temporal conditional UNet attention input mismatch");
    const int64_t batch = query_input.dim(0);
    const int64_t query_tokens = query_input.dim(1);
    const int64_t key_tokens = key_value_input.dim(1);
    const int64_t width = query_input.dim(2);
    require(width % heads == 0, "temporal conditional UNet head mismatch");
    const int64_t head_width = width / heads;
    Tensor query = backend.reshape(linear(
        backend, policy, weights, query_input, prefix + ".to_q",
        PrecisionOperation::AttentionProjection),
        {batch, query_tokens, heads, head_width});
    Tensor key = backend.reshape(linear(
        backend, policy, weights, key_value_input, prefix + ".to_k",
        PrecisionOperation::AttentionProjection),
        {batch, key_tokens, heads, head_width});
    Tensor value = backend.reshape(linear(
        backend, policy, weights, key_value_input, prefix + ".to_v",
        PrecisionOperation::AttentionProjection),
        {batch, key_tokens, heads, head_width});
    Tensor hidden = operation_attention(
        backend, policy, query, key, value, nullptr, false,
        1.0f / std::sqrt(static_cast<float>(head_width)));
    hidden = backend.reshape(hidden, {batch, query_tokens, width});
    return linear(backend, policy, weights, hidden, prefix + ".to_out.0",
                  PrecisionOperation::AttentionProjection);
}

Tensor geglu(Backend& backend, const PrecisionPolicy& policy,
             const WeightMap& weights, const Tensor& input,
             const std::string& prefix) {
    Tensor projected = linear(
        backend, policy, weights, input, prefix + ".net.0.proj");
    const int64_t half = projected.dim(-1) / 2;
    require(projected.dim(-1) == half * 2, "GEGLU projection width is odd");
    std::vector<Tensor> halves = backend.split(projected, {half, half}, -1);
    Tensor hidden = backend.mul(halves[0],
        backend.activation(halves[1], Activation::Gelu));
    return linear(backend, policy, weights, hidden, prefix + ".net.2");
}

Tensor resnet(Backend& backend, const PrecisionPolicy& policy,
              const WeightMap& weights, const Tensor& input,
              const Tensor& time_embedding, const std::string& prefix,
              float epsilon) {
    Tensor hidden = group_norm(backend, weights, input, prefix + ".norm1", epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = spatial_conv2d(
        backend, policy, weights, hidden, prefix + ".conv1");
    Tensor time = backend.activation(time_embedding, Activation::Silu);
    time = linear(backend, policy, weights, time, prefix + ".time_emb_proj",
                  PrecisionOperation::Modulation);
    time = backend.reshape(time, {time.dim(0), time.dim(1), 1, 1, 1});
    hidden = backend.add(hidden, time);
    hidden = group_norm(backend, weights, hidden, prefix + ".norm2", epsilon);
    hidden = backend.activation(hidden, Activation::Silu);
    hidden = spatial_conv2d(
        backend, policy, weights, hidden, prefix + ".conv2");
    Tensor residual = input;
    if (weights.contains(prefix + ".conv_shortcut.weight"))
        residual = spatial_conv2d(backend, policy, weights, input,
                                  prefix + ".conv_shortcut", 1, 0);
    return residual_add(backend, policy, residual, hidden);
}

Tensor spatial_transformer(Backend& backend, const PrecisionPolicy& policy,
                           const WeightMap& weights,
                           const Tensor& input, const Tensor& conditioning,
                           const std::string& prefix, int64_t heads,
                           TemporalConditionalUNetObservation* observation,
                           bool first) {
    const int64_t batch = input.dim(0), channels = input.dim(1);
    const int64_t frames = input.dim(2), height = input.dim(3), width = input.dim(4);
    require(conditioning.shape() == std::vector<int64_t>(
                {batch, frames, 50, 384}),
            "temporal conditional UNet audio conditioning mismatch");
    Tensor flat = flatten_frames(backend, input);
    Tensor hidden = group_norm(backend, weights, flat, prefix + ".norm", 1.0e-6f);
    hidden = operation_conv2d(
        backend, policy, PrecisionSemantic::TemporaryCompute, hidden,
        weights.at(prefix + ".proj_in.weight"),
        optional(weights, prefix + ".proj_in.bias"), {1, 1}, {0, 0});
    hidden = backend.permute(hidden, {0, 2, 3, 1});
    hidden = backend.reshape(hidden, {batch * frames, height * width, channels});
    Tensor audio = backend.reshape(conditioning,
        {batch * frames, conditioning.dim(2), conditioning.dim(3)});

    const std::string block = prefix + ".transformer_blocks.0";
    Tensor normalized = layer_norm(backend, weights, hidden, block + ".norm1");
    Tensor branch = attention(backend, policy, weights, normalized, normalized,
                              block + ".attn1", heads);
    if (first) observe(observation, "representative_spatial_self_attention", branch);
    hidden = residual_add(backend, policy, hidden, branch);
    normalized = layer_norm(backend, weights, hidden, block + ".norm2");
    branch = attention(backend, policy, weights, normalized, audio,
                       block + ".attn2", heads);
    if (first) observe(observation, "representative_audio_cross_attention", branch);
    hidden = residual_add(backend, policy, hidden, branch);
    normalized = layer_norm(backend, weights, hidden, block + ".norm3");
    hidden = residual_add(
        backend, policy, hidden,
        geglu(backend, policy, weights, normalized, block + ".ff"));

    hidden = backend.reshape(hidden, {batch * frames, height, width, channels});
    hidden = backend.permute(hidden, {0, 3, 1, 2});
    hidden = operation_conv2d(
        backend, policy, PrecisionSemantic::TemporaryCompute, hidden,
        weights.at(prefix + ".proj_out.weight"),
        optional(weights, prefix + ".proj_out.bias"), {1, 1}, {0, 0});
    flat = residual_add(backend, policy, flat, hidden);
    return unflatten_frames(backend, flat, batch, frames);
}

Tensor temporal_attention(Backend& backend, const PrecisionPolicy& policy,
                          const WeightMap& weights,
                          const Tensor& input, const std::string& prefix,
                          int64_t batch, int64_t frames, int64_t spatial,
                          int64_t heads,
                          TemporalConditionalUNetObservation* observation,
                          int sublayer, bool first) {
    Tensor normalized = layer_norm(backend, weights, input,
        prefix.substr(0, prefix.rfind(".attention_blocks.")) + ".norms." +
            std::to_string(sublayer));
    if (first && sublayer == 0)
        observe(observation, "temporal_before_reshape", normalized);
    Tensor temporal = backend.reshape(normalized,
        {batch, frames, spatial, normalized.dim(2)});
    temporal = backend.permute(temporal, {0, 2, 1, 3});
    temporal = backend.reshape(temporal,
        {batch * spatial, frames, normalized.dim(2)});
    const Tensor& full_position = weights.at(prefix + ".pos_encoder.pe");
    require(full_position.shape() == std::vector<int64_t>(
                {1, 24, normalized.dim(2)}) && frames <= 24,
            "temporal positional encoding contract mismatch");
    Tensor position = backend.slice(full_position, 1, 0, frames);
    temporal = backend.add(temporal, position);
    if (first && sublayer == 0)
        observe(observation, "temporal_positional_encoding", temporal);
    Tensor branch = attention(
        backend, policy, weights, temporal, temporal, prefix, heads);
    branch = backend.reshape(branch,
        {batch, spatial, frames, normalized.dim(2)});
    branch = backend.permute(branch, {0, 2, 1, 3});
    branch = backend.reshape(branch,
        {batch * frames, spatial, normalized.dim(2)});
    if (first) observe(observation,
        sublayer == 0 ? "temporal_attention_0_output" :
                        "temporal_attention_1_output", branch);
    return residual_add(backend, policy, input, branch);
}

Tensor motion_module(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights,
                     const Tensor& input, const std::string& prefix,
                     int64_t heads,
                     TemporalConditionalUNetObservation* observation,
                     bool first) {
    require(input.ndim() == 5 && input.dim(2) >= 1 && input.dim(2) <= 24,
            "motion module temporal extent is unsupported");
    if (first) observe(observation, "temporal_module_input_5d", input);
    const int64_t batch = input.dim(0), frames = input.dim(2);
    const int64_t height = input.dim(3), width = input.dim(4);
    const int64_t spatial = height * width, channels = input.dim(1);
    const std::string root = prefix + ".temporal_transformer";
    Tensor flat = flatten_frames(backend, input);
    const Tensor residual = flat;
    Tensor hidden = group_norm(backend, weights, flat, root + ".norm", 1.0e-6f);
    hidden = backend.permute(hidden, {0, 2, 3, 1});
    hidden = backend.reshape(hidden, {batch * frames, spatial, channels});
    hidden = linear(backend, policy, weights, hidden, root + ".proj_in");
    const std::string block = root + ".transformer_blocks.0";
    for (int sublayer = 0; sublayer < 2; ++sublayer)
        hidden = temporal_attention(backend, policy, weights, hidden,
            block + ".attention_blocks." + std::to_string(sublayer),
            batch, frames, spatial, heads, observation, sublayer, first);
    Tensor normalized = layer_norm(backend, weights, hidden, block + ".ff_norm");
    Tensor feed_forward = geglu(
        backend, policy, weights, normalized, block + ".ff");
    if (first) observe(observation, "temporal_geglu_ffn_output", feed_forward);
    hidden = residual_add(backend, policy, hidden, feed_forward);
    if (first) observe(observation, "temporal_block_final", hidden);
    hidden = linear(backend, policy, weights, hidden, root + ".proj_out");
    hidden = backend.reshape(hidden, {batch * frames, height, width, channels});
    hidden = backend.permute(hidden, {0, 3, 1, 2});
    hidden = residual_add(backend, policy, hidden, residual);
    return unflatten_frames(backend, hidden, batch, frames);
}

Tensor expanded_timestep(Backend& backend, const Tensor& timestep, int64_t batch) {
    require(timestep.device().is_host() && timestep.numel() == 1,
            "temporal UNet timestep must be a host scalar or batch vector");
    Tensor expanded = Tensor::host({batch}, DType::F32);
    float value = 0.0f;
    if (timestep.dtype() == DType::F32) value = timestep.data_as<float>()[0];
    else if (timestep.dtype() == DType::I64)
        value = static_cast<float>(timestep.data_as<int64_t>()[0]);
    else if (timestep.dtype() == DType::I32)
        value = static_cast<float>(timestep.data_as<int32_t>()[0]);
    else require(false, "temporal UNet timestep dtype is unsupported");
    for (int64_t index = 0; index < batch; ++index)
        expanded.data_as<float>()[index] = value;
    return backend.copy_to_device(expanded, DType::F32);
}

void validate_graph(const Json& graph) {
    const Json& config = graph.at("config");
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "temporal_conditional_unet_2d" &&
            graph.at("entry_point").string() == "execute" &&
            config.at("dtype").string() == "float32" &&
            config.at("input_channels").integer() == 13 &&
            config.at("output_channels").integer() == 4 &&
            config.at("sample_size").integer() == 64 &&
            config.at("conditioning_width").integer() == 384 &&
            config.at("layers_per_block").integer() == 2 &&
            config.at("up_layers_per_block").integer() == 3 &&
            config.at("attention_heads").integer() == 8 &&
            config.at("norm_num_groups").integer() == 32 &&
            config.at("motion_module_count").integer() == 20 &&
            config.at("temporal_position_maximum").integer() == 24,
            "unsupported temporal conditional UNet component graph");
}

}  // namespace

TemporalConditionalUNetResult TemporalConditionalUNet2DComponentExecutor::execute(
        const Json& graph, const Tensor& latent, const Tensor& timestep,
        const Tensor& conditioning,
        TemporalConditionalUNetObservation* observation) {
    validate_graph(graph);
    require(latent.ndim() == 5 && latent.dtype() == DType::F32 &&
                latent.dim(1) == 13 && latent.dim(2) >= 1 && latent.dim(2) <= 24 &&
                latent.dim(3) == 64 && latent.dim(4) == 64 &&
                conditioning.ndim() == 4 && conditioning.dtype() == DType::F32 &&
                conditioning.dim(0) == latent.dim(0) &&
                conditioning.dim(1) == latent.dim(2) &&
                conditioning.dim(2) == 50 && conditioning.dim(3) == 384,
            "temporal conditional UNet execution input mismatch");
    const float epsilon = static_cast<float>(
        graph.at("config").at("norm_epsilon").number());
    const int64_t heads = graph.at("config").at("attention_heads").integer();
    Tensor time = expanded_timestep(backend_, timestep, latent.dim(0));
    time = backend_.sinusoidal_embedding(time, 320, true, 0.0, false);
    time = linear(backend_, policy_, weights_, time, "time_embedding.linear_1",
                  PrecisionOperation::Modulation);
    time = backend_.activation(time, Activation::Silu);
    time = linear(backend_, policy_, weights_, time, "time_embedding.linear_2",
                  PrecisionOperation::Modulation);

    Tensor hidden = spatial_conv2d(
        backend_, policy_, weights_, latent, "conv_in");
    observe(observation, "initial_conv", hidden);
    const DType residual_dtype = policy_.persistent_state_dtype(
        PrecisionSemantic::ResidualState);
    std::vector<Tensor> residuals{backend_.cast(hidden, residual_dtype)};
    int motion_count = 0;
    for (int block = 0; block < 4; ++block) {
        const std::string root = "down_blocks." + std::to_string(block);
        for (int layer = 0; layer < 2; ++layer) {
            hidden = resnet(backend_, policy_, weights_, hidden, time,
                            root + ".resnets." + std::to_string(layer), epsilon);
            if (block == 0 && layer == 0)
                observe(observation, "first_down_resnet", hidden);
            if (block < 3)
                hidden = spatial_transformer(backend_, policy_, weights_, hidden,
                    conditioning, root + ".attentions." + std::to_string(layer),
                    heads, observation, block == 0 && layer == 0);
            hidden = motion_module(backend_, policy_, weights_, hidden,
                root + ".motion_modules." + std::to_string(layer), heads,
                observation, block == 0 && layer == 0);
            if (layer == 0 && block == 0)
                observe(observation, "motion_width320", hidden);
            if (layer == 0 && block == 1)
                observe(observation, "motion_width640", hidden);
            if (layer == 0 && block == 2)
                observe(observation, "motion_width1280", hidden);
            ++motion_count;
            residuals.push_back(backend_.cast(hidden, residual_dtype));
        }
        if (block < 3) {
            hidden = spatial_conv2d(backend_, policy_, weights_, hidden,
                root + ".downsamplers.0.conv", 2, 1);
            residuals.push_back(backend_.cast(hidden, residual_dtype));
        }
        if (block == 3) observe(observation, "deepest_down_path", hidden);
    }

    hidden = resnet(backend_, policy_, weights_, hidden, time,
                    "mid_block.resnets.0", epsilon);
    hidden = spatial_transformer(backend_, policy_, weights_, hidden, conditioning,
        "mid_block.attentions.0", heads, observation, false);
    hidden = resnet(backend_, policy_, weights_, hidden, time,
                    "mid_block.resnets.1", epsilon);
    observe(observation, "mid_block", hidden);

    for (int block = 0; block < 4; ++block) {
        const std::string root = "up_blocks." + std::to_string(block);
        for (int layer = 0; layer < 3; ++layer) {
            require(!residuals.empty(), "temporal UNet skip stack underflow");
            Tensor skip = residuals.back();
            residuals.pop_back();
            hidden = backend_.concat({hidden, skip}, 1);
            hidden = resnet(backend_, policy_, weights_, hidden, time,
                            root + ".resnets." + std::to_string(layer), epsilon);
            if (block > 0)
                hidden = spatial_transformer(backend_, policy_, weights_, hidden,
                    conditioning, root + ".attentions." + std::to_string(layer),
                    heads, observation, false);
            hidden = motion_module(backend_, policy_, weights_, hidden,
                root + ".motion_modules." + std::to_string(layer), heads,
                observation, false);
            if (block == 1 && layer == 0)
                observe(observation, "representative_up_motion", hidden);
            ++motion_count;
        }
        if (block < 3) {
            hidden = backend_.interpolate_nearest(hidden, {1.0, 2.0, 2.0});
            hidden = spatial_conv2d(backend_, policy_, weights_, hidden,
                                    root + ".upsamplers.0.conv");
        }
    }
    require(residuals.empty() && motion_count == 20,
            "temporal UNet topology execution mismatch");
    hidden = group_norm(backend_, weights_, hidden, "conv_norm_out", epsilon);
    observe(observation, "final_norm", hidden);
    hidden = backend_.activation(hidden, Activation::Silu);
    hidden = spatial_conv2d(
        backend_, policy_, weights_, hidden, "conv_out");
    observe(observation, "final_epsilon", hidden);
    return {hidden};
}

}  // namespace vrhino
