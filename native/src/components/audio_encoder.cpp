#include "vrhino/audio_encoder.h"

#include <string>
#include <utility>

#include "vrhino/error.h"

namespace vrhino {
namespace {

const Tensor* optional_weight(const WeightMap& weights, const Json& descriptor,
                              const std::string& key) {
    const Json* value = descriptor.find(key);
    return value && !value->is_null() ? &weights.at(value->string()) : nullptr;
}

Tensor linear(Backend& backend, const WeightMap& weights, const Tensor& input,
              const Json& descriptor) {
    return backend.linear(input, weights.at(descriptor.at("weight").string()),
                          optional_weight(weights, descriptor, "bias"),
                          DType::F32, DType::F32);
}

Tensor norm(Backend& backend, const WeightMap& weights, const Tensor& input,
            const Json& descriptor) {
    require(descriptor.at("kind").string() == "layer_norm",
            "Audio encoder supports only generic LayerNorm");
    return backend.layer_norm(input,
        optional_weight(weights, descriptor, "weight"),
        optional_weight(weights, descriptor, "bias"),
        static_cast<float>(descriptor.at("eps").number()));
}

Tensor conv1d_ncl(Backend& backend, const WeightMap& weights, const Tensor& input,
                  const Json& descriptor) {
    require(input.ndim() == 3, "Conv1D expects NCL input");
    const Tensor& source_weight = weights.at(descriptor.at("weight").string());
    require(source_weight.ndim() == 3 && source_weight.dim(1) == input.dim(1),
            "Conv1D weight/input channel mismatch");
    const int stride = static_cast<int>(descriptor.at("stride").integer());
    const int padding = static_cast<int>(descriptor.at("padding").integer());
    require(stride > 0 && padding >= 0, "Invalid Conv1D stride/padding");
    Tensor input_2d = backend.reshape(input,
        {input.dim(0), input.dim(1), 1, input.dim(2)});
    Tensor weight_2d = backend.reshape(source_weight,
        {source_weight.dim(0), source_weight.dim(1), 1, source_weight.dim(2)});
    Tensor result = backend.conv2d(input_2d, weight_2d,
        optional_weight(weights, descriptor, "bias"), {1, stride}, {0, padding});
    require(result.dim(2) == 1, "Conv1D lowering produced non-singleton height");
    return backend.reshape(result, {result.dim(0), result.dim(1), result.dim(3)});
}

void bind_output(AudioEncoderComponentResult& result, const Json& descriptor,
                 const std::map<std::string, Tensor>& values) {
    const std::string name = descriptor.at("name").string();
    const std::string source = descriptor.at("source").string();
    const auto found = values.find(source);
    require(found != values.end(), "Unknown AudioEncoder output source: " + source);
    require(result.outputs.emplace(name, found->second).second,
            "Duplicate AudioEncoder output binding: " + name);
}

}  // namespace

AudioEncoderComponentResult AudioEncoderComponentExecutor::execute(
        const Json& graph, const Tensor& log_mel_features) {
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "audio_encoder_transformer",
            "Unsupported AudioEncoder component graph");
    require(log_mel_features.ndim() == 3 &&
            log_mel_features.dtype() == DType::F32,
            "AudioEncoder input must be FP32 NCL");

    const Json& frontend = graph.at("frontend");
    Tensor hidden = conv1d_ncl(backend_, weights_, log_mel_features,
                               frontend.at("conv1"));
    hidden = backend_.activation(hidden, Activation::Gelu);
    hidden = conv1d_ncl(backend_, weights_, hidden, frontend.at("conv2"));
    hidden = backend_.activation(hidden, Activation::Gelu);
    hidden = backend_.permute(hidden, {0, 2, 1});
    const Tensor& positions = weights_.at(
        graph.at("absolute_position_embedding").at("weight").string());
    require(positions.ndim() == 2 && hidden.dim(1) == positions.dim(0) &&
            hidden.dim(2) == positions.dim(1),
            "AudioEncoder positional embedding shape mismatch");
    hidden = backend_.add(hidden, positions);

    std::map<std::string, Tensor> semantic_values;
    semantic_values.emplace("frontend", hidden);
    const auto& blocks = graph.at("blocks").array();
    for (size_t index = 0; index < blocks.size(); ++index) {
        const Json& block = blocks[index];
        Tensor normalized = norm(backend_, weights_, hidden, block.at("input_norm"));
        const Json& attention = block.at("attention");
        Tensor query = split_heads(backend_, linear(backend_, weights_, normalized,
                                                   attention.at("query")),
                                   static_cast<int>(attention.at("heads").integer()));
        Tensor key = split_heads(backend_, linear(backend_, weights_, normalized,
                                                 attention.at("key")),
                                 static_cast<int>(attention.at("heads").integer()));
        Tensor value = split_heads(backend_, linear(backend_, weights_, normalized,
                                                   attention.at("value")),
                                   static_cast<int>(attention.at("heads").integer()));
        Tensor attended = backend_.attention(query, key, value, nullptr, false,
            static_cast<float>(attention.at("scale").number()));
        attended = linear(backend_, weights_, merge_heads(backend_, attended),
                          attention.at("output"));
        hidden = backend_.add(hidden, attended);

        normalized = norm(backend_, weights_, hidden,
                          block.at("post_attention_norm"));
        Tensor branch = linear(backend_, weights_, normalized,
                               block.at("mlp").at("input"));
        branch = backend_.activation(branch, Activation::Gelu);
        branch = linear(backend_, weights_, branch, block.at("mlp").at("output"));
        hidden = backend_.add(hidden, branch);
        semantic_values.emplace("block." + std::to_string(index), hidden);
    }

    hidden = norm(backend_, weights_, hidden, graph.at("final_norm"));
    semantic_values.emplace("final", hidden);
    AudioEncoderComponentResult result;
    for (const Json& output : graph.at("outputs").array())
        bind_output(result, output, semantic_values);
    require(!result.outputs.empty(), "AudioEncoder graph has no semantic outputs");
    return result;
}

}  // namespace vrhino
