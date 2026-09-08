#include "vrhino/conditioning.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

const Json* optional(const Json& object, const std::string& key) { return object.find(key); }

const Tensor* optional_weight(const WeightMap& weights, const Json& descriptor,
                              const std::string& key) {
    const Json* value = optional(descriptor, key);
    return value && !value->is_null() ? &weights.at(value->string()) : nullptr;
}

Tensor linear(Backend& backend, const WeightMap& weights, const Tensor& input,
              const Json& descriptor) {
    return backend.linear(input, weights.at(descriptor.at("weight").string()),
                          optional_weight(weights, descriptor, "bias"));
}

Tensor normalize(Backend& backend, const WeightMap& weights, const Tensor& input,
                 const Json& descriptor) {
    const std::string kind = descriptor.at("kind").string();
    const float eps = static_cast<float>(descriptor.at("eps").number());
    const Tensor* weight = optional_weight(weights, descriptor, "weight");
    if (kind == "rms_norm") return backend.rms_norm(input, weight, eps);
    if (kind == "layer_norm")
        return backend.layer_norm(input, weight, optional_weight(weights, descriptor, "bias"), eps);
    throw Error("Unsupported conditioning norm semantic: " + kind);
}

Tensor activation(Backend& backend, const Tensor& input, const std::string& kind) {
    if (kind == "silu") return backend.activation(input, Activation::Silu);
    if (kind == "gelu") return backend.activation(input, Activation::Gelu);
    if (kind == "gelu_tanh") return backend.activation(input, Activation::GeluTanh);
    if (kind == "quick_gelu") {
        Tensor factor = scalar_f32(1.702f);
        return backend.div(backend.activation(backend.mul(input, factor), Activation::Silu), factor);
    }
    throw Error("Unsupported conditioning activation semantic: " + kind);
}

Tensor mask_b1k(const Tensor& mask) {
    require(mask.device().is_host() && mask.ndim() == 2,
            "Conditioning attention mask must be a host BxS tensor");
    std::vector<uint8_t> values(mask.numel());
    for (int64_t index = 0; index < mask.numel(); ++index) {
        if (mask.dtype() == DType::I64) values[index] = mask.data_as<int64_t>()[index] != 0;
        else if (mask.dtype() == DType::I32) values[index] = mask.data_as<int32_t>()[index] != 0;
        else if (mask.dtype() == DType::Bool || mask.dtype() == DType::U8)
            values[index] = mask.data_as<uint8_t>()[index] != 0;
        else throw Error("Conditioning attention mask must be integer/bool");
    }
    return host_bool({mask.dim(0), 1, mask.dim(1)}, values);
}

float finite_mask_sentinel(DType target_dtype) {
    if (target_dtype == DType::F32) return std::numeric_limits<float>::lowest();
    if (target_dtype == DType::F16) return -65504.0f;
    if (target_dtype == DType::BF16)
        return -std::bit_cast<float>(uint32_t{0x7f7f0000U});
    throw Error("AdditiveFinite mask consumer dtype must be FP32, FP16, or BF16");
}

Tensor additive_mask_b11k(const Tensor& mask, DType target_dtype) {
    require(mask.device().is_host() && mask.ndim() == 2,
            "Conditioning additive mask must be a host BxS tensor");
    const float sentinel = finite_mask_sentinel(target_dtype);
    std::vector<float> values(mask.numel());
    for (int64_t index = 0; index < mask.numel(); ++index) {
        bool valid;
        if (mask.dtype() == DType::I64) valid = mask.data_as<int64_t>()[index] != 0;
        else if (mask.dtype() == DType::I32) valid = mask.data_as<int32_t>()[index] != 0;
        else if (mask.dtype() == DType::Bool || mask.dtype() == DType::U8)
            valid = mask.data_as<uint8_t>()[index] != 0;
        else throw Error("Conditioning attention mask must be integer/bool");
        values[index] = valid ? 0.0f : sentinel;
    }
    return host_f32({mask.dim(0), 1, 1, mask.dim(1)}, values);
}

Tensor absolute_positions(const Tensor& ids) {
    std::vector<int64_t> positions(ids.numel());
    for (int64_t batch = 0; batch < ids.dim(0); ++batch)
        for (int64_t token = 0; token < ids.dim(1); ++token)
            positions[batch * ids.dim(1) + token] = token;
    return host_i64(ids.shape(), positions);
}

Tensor relative_buckets(int64_t query_tokens, int64_t key_tokens, int64_t buckets,
                        int64_t max_distance, bool bidirectional) {
    require(buckets > 0 && max_distance > 0, "Invalid relative-position bucket config");
    std::vector<int64_t> output(query_tokens * key_tokens);
    int64_t active_buckets = buckets;
    if (bidirectional) active_buckets /= 2;
    const int64_t exact = active_buckets / 2;
    require(exact > 0 && max_distance > exact, "Invalid relative-position bucket range");
    for (int64_t query = 0; query < query_tokens; ++query) {
        for (int64_t key = 0; key < key_tokens; ++key) {
            int64_t relative = key - query;
            int64_t bucket = 0;
            if (bidirectional) {
                if (relative > 0) bucket += active_buckets;
                relative = std::abs(relative);
            } else relative = std::max<int64_t>(-relative, 0);
            if (relative < exact) bucket += relative;
            else {
                const double fraction = std::log(static_cast<double>(relative) / exact) /
                                        std::log(static_cast<double>(max_distance) / exact);
                int64_t large = exact + static_cast<int64_t>(fraction * (active_buckets - exact));
                bucket += std::min<int64_t>(large, active_buckets - 1);
            }
            output[query * key_tokens + key] = bucket;
        }
    }
    return host_i64({1, query_tokens, key_tokens}, output);
}

Tensor repeat_kv_heads(Backend& backend, const Tensor& input, int64_t query_heads) {
    const int64_t kv_heads = input.dim(2);
    require(query_heads % kv_heads == 0, "Attention query/KV head ratio is not integral");
    if (query_heads == kv_heads) return input;
    const int64_t repeats = query_heads / kv_heads;
    std::vector<Tensor> heads;
    heads.reserve(query_heads);
    for (int64_t head = 0; head < kv_heads; ++head) {
        Tensor value = backend.slice(input, 2, head, head + 1);
        for (int64_t repeat = 0; repeat < repeats; ++repeat) heads.push_back(value);
    }
    return backend.concat(heads, 2);
}

Tensor llama_rope_layout(Backend& backend, const Tensor& input, const Tensor& cosine,
                         const Tensor& sine) {
    const int64_t width = input.dim(-1), half = width / 2;
    require(width % 2 == 0, "Llama RoPE width must be even");
    Tensor first = backend.reshape(backend.slice(input, -1, 0, half),
                                   {input.dim(0), input.dim(1), input.dim(2), half, 1});
    Tensor second = backend.reshape(backend.slice(input, -1, half, width),
                                    {input.dim(0), input.dim(1), input.dim(2), half, 1});
    Tensor interleaved = backend.reshape(backend.concat({first, second}, -1), input.shape());
    interleaved = backend.rope_nd(interleaved, cosine, sine);
    Tensor paired = backend.reshape(interleaved,
                                    {input.dim(0), input.dim(1), input.dim(2), half, 2});
    first = backend.reshape(backend.slice(paired, -1, 0, 1),
                            {input.dim(0), input.dim(1), input.dim(2), half});
    second = backend.reshape(backend.slice(paired, -1, 1, 2),
                             {input.dim(0), input.dim(1), input.dim(2), half});
    return backend.concat({first, second}, -1);
}

std::pair<Tensor, Tensor> llama_frequencies(Backend& backend, int64_t tokens,
                                            int64_t width, double theta) {
    std::vector<float> cosine(tokens * width), sine(tokens * width);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t pair = 0; pair < width / 2; ++pair) {
            const double inverse = 1.0 / std::pow(theta, static_cast<double>(2 * pair) / width);
            const float c = static_cast<float>(std::cos(token * inverse));
            const float s = static_cast<float>(std::sin(token * inverse));
            cosine[token * width + 2 * pair] = cosine[token * width + 2 * pair + 1] = c;
            sine[token * width + 2 * pair] = sine[token * width + 2 * pair + 1] = s;
        }
    }
    return {backend.copy_to_device(host_f32({1, tokens, 1, width}, cosine), backend.execution_dtype()),
            backend.copy_to_device(host_f32({1, tokens, 1, width}, sine), backend.execution_dtype())};
}

Tensor pool_token_argmax(Backend& backend, const Tensor& hidden, const Tensor& ids) {
    require(ids.device().is_host() && ids.ndim() == 2 && ids.dtype() == DType::I64,
            "token_argmax pooling requires host int64 IDs");
    std::vector<int64_t> indices(ids.dim(0));
    for (int64_t batch = 0; batch < ids.dim(0); ++batch) {
        int64_t best = 0;
        for (int64_t token = 1; token < ids.dim(1); ++token)
            if (ids.data_as<int64_t>()[batch * ids.dim(1) + token] >
                ids.data_as<int64_t>()[batch * ids.dim(1) + best]) best = token;
        indices[batch] = best;
    }
    return backend.indexed_gather(hidden, host_i64({ids.dim(0)}, indices));
}

}  // namespace

MaskSemantic conditioning_mask_semantic(const Json& attention_descriptor) {
    const Json* value = attention_descriptor.find("mask_semantic");
    if (!value) return MaskSemantic::Boolean;
    require(value->is_string(), "Conditioning mask semantic must be a string");
    if (value->string() == "none") return MaskSemantic::None;
    if (value->string() == "boolean") return MaskSemantic::Boolean;
    if (value->string() == "additive_finite") return MaskSemantic::AdditiveFinite;
    throw Error("Unsupported conditioning mask semantic: " + value->string());
}

ConditioningComponentResult ConditioningComponentExecutor::execute(
    const Json& graph, const Tensor& input_ids, const Tensor& attention_mask,
    const ConditioningObservationRequest* observation) {
    require(graph.at("schema_version").integer() == 1 &&
            graph.at("kind").string() == "pre_norm_transformer",
            "Unsupported conditioning component graph");
    require(input_ids.device().is_host() && input_ids.dtype() == DType::I64 &&
            input_ids.ndim() == 2, "Conditioning input IDs must be host int64 BxS");
    require(attention_mask.shape() == input_ids.shape(), "Conditioning mask shape mismatch");

    ConditioningComponentResult result;
    Tensor hidden = backend_.indexed_gather(
        weights_.at(graph.at("embedding").at("weight").string()), input_ids);
    if (const Json* position = optional(graph, "absolute_position_embedding"))
        hidden = backend_.add(hidden, backend_.indexed_gather(
            weights_.at(position->at("weight").string()), absolute_positions(input_ids)));
    result.captures.emplace("embedding", hidden);
    // Constructed lazily only when an Attention block consumes the tokenizer
    // mask. A graph using MaskSemantic::None must not synthesize an all-valid
    // mask or otherwise conflate mask production with mask consumption.
    Tensor bool_mask;

    const auto& blocks = graph.at("blocks").array();
    if (observation) {
        require(observation->block_index >= 0 || !observation->output_block_indices.empty(),
                "Conditioning observation request is empty");
        if (observation->block_index >= 0)
            require(observation->block_index < static_cast<int64_t>(blocks.size()),
                    "Conditioning observation block index is out of range");
        for (size_t index = 0; index < observation->output_block_indices.size(); ++index) {
            const int64_t block = observation->output_block_indices[index];
            require(block >= 0 && block < static_cast<int64_t>(blocks.size()),
                    "Conditioning output observation block index is out of range");
            require(std::find(observation->output_block_indices.begin(),
                              observation->output_block_indices.begin() + index, block) ==
                        observation->output_block_indices.begin() + index,
                    "Duplicate conditioning output observation block index");
        }
    }
    Tensor official;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        const Json& block = blocks[block_index];
        const Json& attention = block.at("attention");
        const bool observe = observation &&
            observation->block_index == static_cast<int64_t>(block_index);
        const std::string observation_prefix = "block" + std::to_string(block_index) + ".";
        const auto record = [&](const std::string& name, const Tensor& value) {
            if (observe)
                require(result.captures.emplace(observation_prefix + name, value).second,
                        "Duplicate conditioning observation key");
        };

        record("input", hidden);
        Tensor normalized = normalize(backend_, weights_, hidden, block.at("input_norm"));
        record("pre_attention_norm", normalized);
        Tensor q_projection = linear(backend_, weights_, normalized, attention.at("q"));
        Tensor k_projection = linear(backend_, weights_, normalized, attention.at("k"));
        Tensor v_projection = linear(backend_, weights_, normalized, attention.at("v"));
        record("q_projection", q_projection);
        record("k_projection", k_projection);
        record("v_projection", v_projection);
        Tensor q = split_heads(backend_, q_projection, attention.at("query_heads").integer());
        Tensor k = split_heads(backend_, k_projection, attention.at("kv_heads").integer());
        Tensor v = split_heads(backend_, v_projection, attention.at("kv_heads").integer());
        record("q_heads", q);
        record("k_heads", k);
        record("v_heads", v);
        Tensor position_bias, position_bias_lookup;
        const Tensor* position_bias_pointer = nullptr;
        if (const Json* position = optional(attention, "position")) {
            const std::string kind = position->at("kind").string();
            if (kind == "relative_bucket") {
                Tensor buckets = relative_buckets(input_ids.dim(1), input_ids.dim(1),
                    position->at("buckets").integer(), position->at("max_distance").integer(),
                    position->at("bidirectional").boolean());
                position_bias_lookup = backend_.indexed_gather(
                    weights_.at(position->at("weight").string()), buckets);
                position_bias = backend_.permute(position_bias_lookup, {0, 3, 1, 2});
                record("relative_bucket_ids", buckets);
                record("position_bias_lookup", position_bias_lookup);
                record("position_bias", position_bias);
                position_bias_pointer = &position_bias;
            } else if (kind == "llama_rope") {
                auto [cosine, sine] = llama_frequencies(backend_, input_ids.dim(1), q.dim(-1),
                                                        position->at("theta").number());
                q = llama_rope_layout(backend_, q, cosine, sine);
                k = llama_rope_layout(backend_, k, cosine, sine);
            } else throw Error("Unsupported conditioning position semantic: " + kind);
        }
        k = repeat_kv_heads(backend_, k, q.dim(2));
        v = repeat_kv_heads(backend_, v, q.dim(2));
        const MaskSemantic mask_semantic = conditioning_mask_semantic(attention);
        const Tensor* bool_mask_pointer = nullptr;
        Tensor combined_bias, padding_bias;
        if (mask_semantic == MaskSemantic::Boolean) {
            if (!bool_mask.defined()) bool_mask = mask_b1k(attention_mask);
            bool_mask_pointer = &bool_mask;
        } else if (mask_semantic == MaskSemantic::AdditiveFinite) {
            const DType target_dtype = position_bias_pointer
                ? position_bias_pointer->dtype() : backend_.execution_dtype();
            padding_bias = additive_mask_b11k(attention_mask, target_dtype);
            combined_bias = position_bias_pointer ? backend_.add(*position_bias_pointer, padding_bias)
                                                  : backend_.copy_to_device(padding_bias, backend_.execution_dtype());
            record("padding_bias", padding_bias);
            record("combined_bias", combined_bias);
            position_bias_pointer = &combined_bias;
        }
        AttentionObservation attention_observation;
        Tensor attended = backend_.attention(q, k, v, bool_mask_pointer,
            attention.at("causal").boolean(), static_cast<float>(attention.at("scale").number()),
            position_bias_pointer, observe ? &attention_observation : nullptr);
        if (observe) {
            record("attention_score", attention_observation.score);
            record("biased_score", attention_observation.biased_score);
            record("softmax", attention_observation.softmax);
        }
        record("attention_heads", attended);
        Tensor attention_merged = merge_heads(backend_, attended);
        record("attention_merged", attention_merged);
        Tensor output_projection = linear(backend_, weights_, attention_merged,
                                          attention.at("output"));
        record("output_projection", output_projection);
        hidden = backend_.add(hidden, output_projection);
        record("first_residual", hidden);
        normalized = normalize(backend_, weights_, hidden, block.at("post_attention_norm"));
        record("ffn_norm", normalized);
        const Json& mlp = block.at("mlp");
        Tensor activation_projection = linear(backend_, weights_, normalized,
                                              mlp.at("activation_input"));
        record("ffn_activation_projection", activation_projection);
        Tensor branch = activation(backend_, activation_projection, mlp.at("activation").string());
        record("ffn_activation", branch);
        if (const Json* gate = optional(mlp, "gate_input")) {
            Tensor gate_projection = linear(backend_, weights_, normalized, *gate);
            record("ffn_gate_projection", gate_projection);
            branch = backend_.mul(branch, gate_projection);
            record("ffn_gated", branch);
        }
        branch = linear(backend_, weights_, branch, mlp.at("output"));
        record("ffn_output", branch);
        hidden = backend_.add(hidden, branch);
        record("output", hidden);
        if (observation && !observe &&
            std::find(observation->output_block_indices.begin(),
                      observation->output_block_indices.end(),
                      static_cast<int64_t>(block_index)) !=
                observation->output_block_indices.end())
            require(result.captures.emplace(observation_prefix + "output", hidden).second,
                    "Duplicate conditioning output observation key");
        if (const Json* capture = optional(block, "capture"))
            result.captures.emplace(capture->string(), hidden);
        if (const Json* selected = optional(graph, "official_output_after_block"))
            if (selected->integer() == static_cast<int64_t>(block_index)) official = hidden;
    }
    if (const Json* final_norm = optional(graph, "final_norm")) {
        Tensor final = normalize(backend_, weights_, hidden, *final_norm);
        result.captures.emplace("final_norm", final);
        hidden = final;
    }
    result.hidden_states = official.defined() ? official : hidden;
    if (const Json* start = optional(graph, "output_slice_start"))
        result.hidden_states = backend_.slice(result.hidden_states, 1, start->integer(),
                                               result.hidden_states.dim(1));
    if (const Json* trim = optional(graph, "output_trim_to_mask")) {
        require(trim->boolean() && input_ids.dim(0) == 1,
                "output_trim_to_mask currently requires batch size 1");
        int64_t valid = 0;
        for (int64_t token = 0; token < attention_mask.dim(1); ++token) {
            if (attention_mask.dtype() == DType::I64) valid += attention_mask.data_as<int64_t>()[token] != 0;
            else if (attention_mask.dtype() == DType::I32) valid += attention_mask.data_as<int32_t>()[token] != 0;
            else valid += attention_mask.data_as<uint8_t>()[token] != 0;
        }
        result.hidden_states = backend_.slice(result.hidden_states, 1, 0, valid);
    }
    result.captures.emplace("official_output", result.hidden_states);
    if (const Json* pooling = optional(graph, "pooling")) {
        require(pooling->at("kind").string() == "token_argmax", "Unsupported conditioning pooling semantic");
        result.pooled_output = pool_token_argmax(backend_, result.hidden_states, input_ids);
        result.captures.emplace("pooled_output", result.pooled_output);
    }
    return result;
}

}  // namespace vrhino
