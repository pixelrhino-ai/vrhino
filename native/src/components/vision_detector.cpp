#include "vrhino/vision_detector.h"

#include <array>
#include <string>
#include <vector>

#include "vrhino/error.h"

namespace vrhino {
namespace {

const Tensor* optional(const WeightMap& weights, const std::string& name) {
    return weights.find(name);
}

Tensor conv(Backend& backend, const WeightMap& weights, const Tensor& input,
            const std::string& prefix, int stride = 1, int padding = 1) {
    return backend.conv2d(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          {stride, stride}, {padding, padding});
}

Tensor relu_conv(Backend& backend, const WeightMap& weights, const Tensor& input,
                 const std::string& prefix, int stride = 1, int padding = 1) {
    return backend.activation(conv(backend, weights, input, prefix, stride, padding),
                              Activation::Relu);
}

void observe(VisionDetectorObservation* observation, const std::string& name,
             const Tensor& tensor) {
    if (observation) observation->tensors.emplace(name, tensor);
}

Tensor normalized(Backend& backend, const WeightMap& weights, const Tensor& input,
                  const std::string& prefix) {
    Tensor result = backend.l2_normalize(input, 1, 1.0e-10f);
    Tensor scale = weights.at(prefix + ".weight");
    scale = backend.reshape(scale, {1, scale.numel(), 1, 1});
    return backend.mul(result, scale);
}

void validate_graph(const Json& graph) {
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "vision_detector_multiscale" &&
            graph.at("entry_point").string() == "execute" &&
            graph.at("config").at("dtype").string() == "float32" &&
            graph.at("config").at("input_channels").integer() == 3 &&
            graph.at("config").at("scale_count").integer() == 6,
            "unsupported multi-scale vision detector graph");
}

void validate_dense_graph(const Json& graph) {
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "vision_detector_dense_anchors" &&
            graph.at("entry_point").string() == "execute" &&
            graph.at("config").at("dtype").string() == "float32" &&
            graph.at("config").at("input_channels").integer() == 3 &&
            graph.at("config").at("input_height").integer() == 128 &&
            graph.at("config").at("input_width").integer() == 128 &&
            graph.at("config").at("block_count").integer() == 16 &&
            graph.at("config").at("anchor_count").integer() == 896 &&
            graph.at("config").at("regression_coordinates").integer() == 16,
            "unsupported dense-anchor vision detector graph");
}

Tensor dense_conv(Backend& backend, const WeightMap& weights,
                  const Tensor& input, const std::string& prefix,
                  int stride = 1, int groups = 1) {
    return backend.conv2d(input, weights.at(prefix + ".weight"),
                          optional(weights, prefix + ".bias"),
                          {stride, stride}, {0, 0}, {1, 1}, groups);
}

}  // namespace

VisionDetectorResult VisionDetectorComponentExecutor::execute(
        const Json& graph, const Tensor& image,
        VisionDetectorObservation* observation) {
    validate_graph(graph);
    require(image.ndim() == 4 && image.dtype() == DType::F32 && image.dim(1) == 3,
            "vision detector input must be NCHW FP32 RGB");

    Tensor hidden = relu_conv(backend_, weights_, image, "conv1_1");
    observe(observation, "early_backbone", hidden);
    hidden = relu_conv(backend_, weights_, hidden, "conv1_2");
    hidden = backend_.max_pool2d(hidden, {2, 2}, {2, 2});
    hidden = relu_conv(backend_, weights_, hidden, "conv2_1");
    hidden = relu_conv(backend_, weights_, hidden, "conv2_2");
    hidden = backend_.max_pool2d(hidden, {2, 2}, {2, 2});
    hidden = relu_conv(backend_, weights_, hidden, "conv3_1");
    hidden = relu_conv(backend_, weights_, hidden, "conv3_2");
    Tensor f3 = relu_conv(backend_, weights_, hidden, "conv3_3");
    observe(observation, "pre_normalized_feature", f3);
    hidden = backend_.max_pool2d(f3, {2, 2}, {2, 2});
    hidden = relu_conv(backend_, weights_, hidden, "conv4_1");
    hidden = relu_conv(backend_, weights_, hidden, "conv4_2");
    Tensor f4 = relu_conv(backend_, weights_, hidden, "conv4_3");
    hidden = backend_.max_pool2d(f4, {2, 2}, {2, 2});
    hidden = relu_conv(backend_, weights_, hidden, "conv5_1");
    hidden = relu_conv(backend_, weights_, hidden, "conv5_2");
    Tensor f5 = relu_conv(backend_, weights_, hidden, "conv5_3");
    hidden = backend_.max_pool2d(f5, {2, 2}, {2, 2});
    hidden = relu_conv(backend_, weights_, hidden, "fc6", 1, 3);
    Tensor ffc7 = relu_conv(backend_, weights_, hidden, "fc7", 1, 0);
    hidden = relu_conv(backend_, weights_, ffc7, "conv6_1", 1, 0);
    Tensor f6 = relu_conv(backend_, weights_, hidden, "conv6_2", 2, 1);
    hidden = relu_conv(backend_, weights_, f6, "conv7_1", 1, 0);
    Tensor f7 = relu_conv(backend_, weights_, hidden, "conv7_2", 2, 1);

    f3 = normalized(backend_, weights_, f3, "conv3_3_norm");
    f4 = normalized(backend_, weights_, f4, "conv4_3_norm");
    f5 = normalized(backend_, weights_, f5, "conv5_3_norm");
    observe(observation, "normalized_feature", f3);

    std::vector<Tensor> features{f3, f4, f5, ffc7, f6, f7};
    std::vector<std::string> roots{
        "conv3_3_norm_mbox", "conv4_3_norm_mbox", "conv5_3_norm_mbox",
        "fc7_mbox", "conv6_2_mbox", "conv7_2_mbox"};
    VisionDetectorResult result;
    for (size_t index = 0; index < features.size(); ++index) {
        Tensor confidence = conv(backend_, weights_, features[index],
                                 roots[index] + "_conf");
        Tensor localization = conv(backend_, weights_, features[index],
                                   roots[index] + "_loc");
        if (index == 0) {
            auto channels = backend_.split(confidence, {1, 1, 1, 1}, 1);
            Tensor background = backend_.maximum(
                backend_.maximum(channels[0], channels[1]), channels[2]);
            confidence = backend_.concat({background, channels[3]}, 1);
            observe(observation, "first_confidence_maxout", confidence);
            observe(observation, "first_localization_head", localization);
        }
        result.scales.push_back({confidence, localization});
    }
    return result;
}

DenseVisionDetectorResult VisionDetectorComponentExecutor::execute_dense(
        const Json& graph, const Tensor& image,
        VisionDetectorObservation* observation) {
    validate_dense_graph(graph);
    require(image.shape() == std::vector<int64_t>({1, 3, 128, 128}) &&
            image.dtype() == DType::F32,
            "dense vision detector input must be [1,3,128,128] FP32 RGB");

    Tensor hidden = backend_.pad(image, {1, 2, 1, 2});
    hidden = dense_conv(backend_, weights_, hidden, "stem", 2);
    hidden = backend_.activation(hidden, Activation::Relu);
    observe(observation, "early_backbone", hidden);

    constexpr std::array<int, 16> input_channels{
        24,24,28,32,36,42,48,56,64,72,80,88,96,96,96,96};
    constexpr std::array<int, 16> output_channels{
        24,28,32,36,42,48,56,64,72,80,88,96,96,96,96,96};
    constexpr std::array<int, 16> strides{
        1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1};
    Tensor feature_16;
    for (int block = 0; block < 16; ++block) {
        Tensor residual = hidden;
        Tensor depthwise_input = strides[block] == 2
            ? backend_.pad(hidden, {0, 1, 0, 1})
            : backend_.pad(hidden, {1, 1, 1, 1});
        const std::string prefix = "blocks." + std::to_string(block);
        Tensor transformed = dense_conv(backend_, weights_, depthwise_input,
                                        prefix + ".depthwise", strides[block],
                                        input_channels[block]);
        if (block == 0) observe(observation, "representative_depthwise", transformed);
        transformed = dense_conv(backend_, weights_, transformed,
                                 prefix + ".pointwise");
        if (strides[block] == 2)
            residual = backend_.max_pool2d(residual, {2, 2}, {2, 2});
        const int channel_padding = output_channels[block] - input_channels[block];
        if (channel_padding > 0)
            residual = backend_.pad(residual,
                                    {0, 0, 0, 0, 0, channel_padding});
        hidden = backend_.activation(backend_.add(residual, transformed),
                                     Activation::Relu);
        if (block == 10) {
            feature_16 = hidden;
            observe(observation, "deeper_feature", hidden);
        }
        if (block == 15) observe(observation, "final_feature", hidden);
    }

    Tensor classification_8 = dense_conv(
        backend_, weights_, feature_16, "heads.classification_8");
    Tensor classification_16 = dense_conv(
        backend_, weights_, hidden, "heads.classification_16");
    Tensor regression_8 = dense_conv(
        backend_, weights_, feature_16, "heads.regression_8");
    Tensor regression_16 = dense_conv(
        backend_, weights_, hidden, "heads.regression_16");
    classification_8 = backend_.reshape(
        backend_.permute(classification_8, {0, 2, 3, 1}), {1, 512, 1});
    classification_16 = backend_.reshape(
        backend_.permute(classification_16, {0, 2, 3, 1}), {1, 384, 1});
    regression_8 = backend_.reshape(
        backend_.permute(regression_8, {0, 2, 3, 1}), {1, 512, 16});
    regression_16 = backend_.reshape(
        backend_.permute(regression_16, {0, 2, 3, 1}), {1, 384, 16});
    DenseVisionDetectorResult result{
        backend_.concat({regression_8, regression_16}, 1),
        backend_.concat({classification_8, classification_16}, 1)};
    observe(observation, "raw_regressors", result.regressors);
    observe(observation, "raw_classification_logits", result.classification_logits);
    return result;
}

}  // namespace vrhino
