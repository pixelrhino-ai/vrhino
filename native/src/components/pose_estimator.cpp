#include "vrhino/pose_estimator.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

const Tensor* optional(const WeightMap& weights, const std::string& name) {
    return weights.find(name);
}

Tensor conv(Backend& backend, const WeightMap& weights, const Tensor& input,
            const std::string& prefix, int kernel, int stride = 1,
            int padding = 0, int groups = 1) {
    (void)kernel;
    return backend.conv2d(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          {stride, stride}, {padding, padding}, {1, 1}, groups);
}

Tensor linear(Backend& backend, const WeightMap& weights, const Tensor& input,
              const std::string& prefix) {
    return backend.linear(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          DType::F32, DType::F32);
}

Tensor channel_parameter(Backend& backend, const Tensor& value) {
    return backend.reshape(value, {1, value.numel(), 1, 1});
}

Tensor batch_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix) {
    Tensor mean = channel_parameter(backend, weights.at(prefix + ".running_mean"));
    Tensor variance = channel_parameter(backend, weights.at(prefix + ".running_var"));
    Tensor weight = channel_parameter(backend, weights.at(prefix + ".weight"));
    Tensor bias = channel_parameter(backend, weights.at(prefix + ".bias"));
    Tensor centered = backend.add(input, backend.mul(mean, scalar_f32(-1.0f)));
    Tensor denominator = backend.sqrt(backend.add(variance, scalar_f32(1.0e-5f)));
    return backend.add(backend.mul(backend.div(centered, denominator), weight), bias);
}

Tensor conv_module(Backend& backend, const WeightMap& weights,
                   const Tensor& input, const std::string& prefix,
                   int kernel, int stride = 1, int padding = 0,
                   int groups = 1) {
    Tensor hidden = conv(backend, weights, input, prefix + ".conv", kernel,
                         stride, padding, groups);
    hidden = batch_norm(backend, weights, hidden, prefix + ".bn");
    return backend.activation(hidden, Activation::Silu);
}

Tensor depthwise_module(Backend& backend, const WeightMap& weights,
                        const Tensor& input, const std::string& prefix,
                        int kernel) {
    Tensor hidden = conv_module(backend, weights, input,
        prefix + ".depthwise_conv", kernel, 1, kernel / 2,
        static_cast<int>(input.dim(1)));
    return conv_module(backend, weights, hidden,
                       prefix + ".pointwise_conv", 1);
}

Tensor global_average(Backend& backend, const Tensor& input) {
    Tensor sum = backend.reduce_sum(input, 2, true);
    sum = backend.reduce_sum(sum, 3, true);
    return backend.mul(sum, scalar_f32(1.0f /
        static_cast<float>(input.dim(2) * input.dim(3))));
}

Tensor hard_sigmoid(Backend& backend, const Tensor& input) {
    return backend.clamp(backend.mul(backend.add(input, scalar_f32(3.0f)),
                                    scalar_f32(1.0f / 6.0f)), 0.0f, 1.0f);
}

Tensor channel_attention(Backend& backend, const WeightMap& weights,
                         const Tensor& input, const std::string& prefix) {
    Tensor gate = global_average(backend, input);
    gate = conv(backend, weights, gate, prefix + ".fc", 1);
    return backend.mul(input, hard_sigmoid(backend, gate));
}

Tensor csp_block(Backend& backend, const WeightMap& weights,
                 const Tensor& input, const std::string& prefix,
                 int blocks, bool residual) {
    Tensor main = conv_module(backend, weights, input, prefix + ".main_conv", 1);
    Tensor shortcut = conv_module(backend, weights, input,
                                  prefix + ".short_conv", 1);
    for (int index = 0; index < blocks; ++index) {
        const std::string root = prefix + ".blocks." + std::to_string(index);
        Tensor branch = conv_module(backend, weights, main,
                                    root + ".conv1", 3, 1, 1);
        branch = depthwise_module(backend, weights, branch,
                                  root + ".conv2", 5);
        main = residual ? backend.add(main, branch) : branch;
    }
    Tensor hidden = backend.concat({main, shortcut}, 1);
    hidden = channel_attention(backend, weights, hidden, prefix + ".attention");
    return conv_module(backend, weights, hidden, prefix + ".final_conv", 1);
}

Tensor spatial_pyramid(Backend& backend, const WeightMap& weights,
                       const Tensor& input, const std::string& prefix) {
    Tensor hidden = conv_module(backend, weights, input, prefix + ".conv1", 1);
    Tensor p5 = backend.max_pool2d(hidden, {5, 5}, {1, 1}, {2, 2});
    Tensor p9 = backend.max_pool2d(hidden, {9, 9}, {1, 1}, {4, 4});
    Tensor p13 = backend.max_pool2d(hidden, {13, 13}, {1, 1}, {6, 6});
    return conv_module(backend, weights, backend.concat({hidden, p5, p9, p13}, 1),
                       prefix + ".conv2", 1);
}

Tensor scale_norm(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix) {
    Tensor squared = backend.mul(input, input);
    Tensor norm = backend.sqrt(backend.reduce_sum(squared, -1, true));
    norm = backend.mul(norm, scalar_f32(1.0f /
        std::sqrt(static_cast<float>(input.dim(-1)))));
    norm = backend.clamp(norm, 1.0e-5f, std::numeric_limits<float>::max());
    return backend.mul(backend.div(input, norm), weights.at(prefix + ".g"));
}

void observe(PoseEstimator2DObservation* observation, const std::string& name,
             const Tensor& tensor) {
    if (observation) observation->tensors.emplace(name, tensor);
}

void validate_graph(const Json& graph) {
    const Json& config = graph.at("config");
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "pose_estimator_2d" &&
            graph.at("entry_point").string() == "execute" &&
            config.at("dtype").string() == "float32" &&
            config.at("input_height").integer() == 384 &&
            config.at("input_width").integer() == 288 &&
            config.at("keypoint_count").integer() == 133 &&
            config.at("simcc_x_bins").integer() == 576 &&
            config.at("simcc_y_bins").integer() == 768,
            "unsupported pose estimator graph");
}

}  // namespace

PoseEstimator2DResult PoseEstimator2DComponentExecutor::execute(
        const Json& graph, const Tensor& image,
        PoseEstimator2DObservation* observation) {
    validate_graph(graph);
    require(image.ndim() == 4 && image.dtype() == DType::F32 &&
            image.dim(1) == 3 && image.dim(2) == 384 && image.dim(3) == 288,
            "pose estimator input must be [B,3,384,288] FP32");

    Tensor hidden = conv_module(backend_, weights_, image, "backbone.stem.0", 3, 2, 1);
    hidden = conv_module(backend_, weights_, hidden, "backbone.stem.1", 3, 1, 1);
    hidden = conv_module(backend_, weights_, hidden, "backbone.stem.2", 3, 1, 1);
    observe(observation, "stem", hidden);

    constexpr int block_counts[4] = {3, 6, 6, 3};
    for (int stage = 1; stage <= 4; ++stage) {
        const std::string stage_root = "backbone.stage" + std::to_string(stage);
        hidden = conv_module(backend_, weights_, hidden, stage_root + ".0", 3, 2, 1);
        const int csp_index = stage == 4 ? 2 : 1;
        if (stage == 4) {
            hidden = spatial_pyramid(backend_, weights_, hidden,
                                     stage_root + ".1");
            observe(observation, "spp", hidden);
        }
        hidden = csp_block(backend_, weights_, hidden,
                           stage_root + "." + std::to_string(csp_index),
                           block_counts[stage - 1], stage != 4);
        if (stage == 2) observe(observation, "backbone_stage", hidden);
    }
    observe(observation, "backbone_final", hidden);

    hidden = conv(backend_, weights_, hidden, "head.final_layer", 7, 1, 3);
    observe(observation, "head_projection", hidden);
    hidden = backend_.reshape(hidden, {hidden.dim(0), 133, 108});
    hidden = scale_norm(backend_, weights_, hidden, "head.mlp.0");
    hidden = linear(backend_, weights_, hidden, "head.mlp.1");

    Tensor normalized = scale_norm(backend_, weights_, hidden, "head.gau.ln");
    Tensor uv = backend_.activation(linear(backend_, weights_, normalized,
                                           "head.gau.uv"), Activation::Silu);
    auto sections = backend_.split(uv, {512, 512, 128}, -1);
    Tensor u = sections[0], value = sections[1], base = sections[2];
    base = backend_.reshape(base, {base.dim(0), base.dim(1), 1, 128});
    Tensor gamma = backend_.reshape(weights_.at("head.gau.gamma"), {1, 1, 2, 128});
    Tensor beta = backend_.reshape(weights_.at("head.gau.beta"), {1, 1, 2, 128});
    auto query_key = backend_.split(backend_.add(backend_.mul(base, gamma), beta),
                                    {1, 1}, 2);
    Tensor query = backend_.reshape(query_key[0], {base.dim(0), base.dim(1), 128});
    Tensor key = backend_.reshape(query_key[1], {base.dim(0), base.dim(1), 128});
    key = backend_.permute(key, {0, 2, 1});
    Tensor kernel = backend_.batched_matmul(query, key);
    kernel = backend_.activation(backend_.mul(kernel,
        scalar_f32(1.0f / std::sqrt(128.0f))), Activation::Relu);
    kernel = backend_.mul(kernel, kernel);
    Tensor gau = backend_.mul(u, backend_.batched_matmul(kernel, value));
    gau = linear(backend_, weights_, gau, "head.gau.o");
    Tensor residual_scale = backend_.reshape(weights_.at("head.gau.res_scale.scale"),
                                              {1, 1, 256});
    hidden = backend_.add(backend_.mul(hidden, residual_scale), gau);
    observe(observation, "pose_head", hidden);

    PoseEstimator2DResult result{
        linear(backend_, weights_, hidden, "head.cls_x"),
        linear(backend_, weights_, hidden, "head.cls_y")};
    observe(observation, "raw_x", result.simcc_x);
    observe(observation, "raw_y", result.simcc_y);
    return result;
}

}  // namespace vrhino
