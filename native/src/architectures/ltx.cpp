#include "vrhino/architecture.h"
#include "ltx_self_attention_graph.h"

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

#include "vrhino/components.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {
void validate_self_attention_context(Backend& backend, const PrecisionPolicy& policy) {
    require(policy.requested_dtype() == backend.execution_dtype() &&
            (backend.execution_dtype() == DType::F32 || backend.execution_dtype() == DType::BF16),
            "LTX NeuralGraph requires matching F32 or BF16 execution context");
}

std::pair<Tensor, Tensor> timestep_conditioning(Backend& backend,
                                                const PrecisionPolicy& policy,
                                                const WeightMap& weights,
                                                const Tensor& sigma) {
    const int64_t batch = sigma.dim(0), tokens = sigma.numel() / batch;
    Tensor embedding = backend.sinusoidal_embedding(
        backend.reshape(backend.mul(sigma, scalar_f32(1000.0f)), {-1}), 256, true, 0.0, false);
    embedding = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, embedding,
        weights.at("emb.timestep_embedder.linear_1.weight"),
        &weights.at("emb.timestep_embedder.linear_1.bias"));
    embedding = backend.activation(embedding, Activation::Silu);
    embedding = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute, embedding,
        weights.at("emb.timestep_embedder.linear_2.weight"),
        &weights.at("emb.timestep_embedder.linear_2.bias"));
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(embedding, Activation::Silu),
        weights.at("linear.weight"), &weights.at("linear.bias"));
    return {backend.reshape(modulation, {batch, tokens, modulation.dim(-1)}),
            backend.reshape(embedding, {batch, tokens, embedding.dim(-1)})};
}

Tensor projected_attention(Backend& backend, const PrecisionPolicy& policy,
                           const WeightMap& weights,
                           const std::string& prefix, const Tensor& query_states,
                           const Tensor& key_value_states, int heads,
                           const Tensor* mask = nullptr,
                           TensorBundle* trace = nullptr,
                           const std::string& trace_prefix = {}) {
    Tensor q = backend.linear(query_states, weights.at(prefix + ".to_q.weight"),
                              &weights.at(prefix + ".to_q.bias"));
    Tensor k = backend.linear(key_value_states, weights.at(prefix + ".to_k.weight"),
                              &weights.at(prefix + ".to_k.bias"));
    Tensor v = backend.linear(key_value_states, weights.at(prefix + ".to_v.weight"),
                              &weights.at(prefix + ".to_v.bias"));
    if (trace) {
        (*trace)[trace_prefix + ".q_projection"] = q;
        (*trace)[trace_prefix + ".k_projection"] = k;
        (*trace)[trace_prefix + ".v_projection"] = v;
    }
    q = attention_norm(backend, policy, q,
                       &weights.at(prefix + ".q_norm.weight"), 1e-5f);
    k = attention_norm(backend, policy, k,
                       &weights.at(prefix + ".k_norm.weight"), 1e-5f);
    if (trace) {
        (*trace)[trace_prefix + ".q_norm"] = q;
        (*trace)[trace_prefix + ".k_norm"] = k;
    }
    if (trace) {
        (*trace)[trace_prefix + ".q_positioned"] = q;
        (*trace)[trace_prefix + ".k_positioned"] = k;
    }
    q = split_heads(backend, q, heads); k = split_heads(backend, k, heads);
    v = split_heads(backend, v, heads);
    if (trace) {
        (*trace)[trace_prefix + ".q_heads"] = q;
        (*trace)[trace_prefix + ".k_heads"] = k;
        (*trace)[trace_prefix + ".v_heads"] = v;
    }
    Tensor attended = operation_attention(backend, policy, q, k, v, mask);
    if (trace) (*trace)[trace_prefix + ".attention_heads"] = attended;
    Tensor output = merge_heads(backend, attended);
    if (trace) (*trace)[trace_prefix + ".attention_merged"] = output;
    output = backend.linear(output, weights.at(prefix + ".to_out.0.weight"),
                            &weights.at(prefix + ".to_out.0.bias"));
    if (trace) (*trace)[trace_prefix + ".output_projection"] = output;
    return output;
}

Tensor transformer_block(Backend& backend, const PrecisionPolicy& policy,
                         const WeightMap& weights, const Tensor& input,
                         const Tensor& context, const Tensor& timestep,
                         const Tensor& rope_cos, const Tensor& rope_sin,
                         const Tensor& context_mask,
                         TensorBundle* trace = nullptr) {
    validate_self_attention_context(backend, policy);
    if (trace) (*trace)["block.0.input"] = input;
    const auto contract = ltx_internal::self_attention_contract(backend.execution_dtype(), input.dtype());
    const auto graph = ltx_internal::self_attention_graph(
        input.dim(0), input.dim(1), timestep.dim(1), trace != nullptr, true, contract);
    std::vector<Tensor> parameters;
    for (const char* name : ltx_internal::self_attention_parameters)
        // Legacy F32 materialization is unchanged. Mixed bindings are exact,
        // borrowed production tensors; the evaluator validates before upload.
        parameters.push_back(contract == ltx_internal::SelfAttentionContract::F32
            ? backend.copy_to_device(weights.at(name), DType::F32) : weights.at(name));
    auto outputs = neural_graph::evaluate(backend, graph,
        {input, timestep, rope_cos, rope_sin}, parameters);
    Tensor hidden = trace ? outputs.back() : outputs.front();
    if (trace)
        for (size_t i = 0; i < outputs.size(); ++i)
            (*trace)[ltx_internal::self_attention_checkpoints[i]] = outputs[i];
    // v18 is shared with FFN: select its remaining rows once, never rebuild it.
    std::vector<Tensor> parts(6);
    for (int i = 3; i < 6; ++i)
        parts[i] = backend.reshape(backend.slice(outputs[1], 2, i, i + 1),
            {input.dim(0), timestep.dim(1), input.dim(2)});
    if (trace) {
        (*trace)["block.0.ffn_shift"] = parts[3];
        (*trace)["block.0.ffn_scale"] = parts[4];
        (*trace)["block.0.ffn_gate"] = parts[5];
    }
    Tensor cross_output = projected_attention(
        backend, policy, weights, "attn2", hidden, context, 32,
        &context_mask, trace, "block.0.cross_attention");
    hidden = residual_add(backend, policy, hidden, cross_output);
    if (trace) (*trace)["block.0.cross_residual"] = hidden;
    Tensor normalized = backend.rms_norm(hidden, nullptr, 1e-6f);
    if (trace) (*trace)["block.0.pre_ffn_norm"] = normalized;
    BackendProfileRegion mlp_profile(backend, "mlp.total");
    Tensor ff = backend.linear(modulate(backend, policy, normalized, parts[3], parts[4]),
                               weights.at("ff.net.0.proj.weight"), &weights.at("ff.net.0.proj.bias"));
    if (trace) (*trace)["block.0.ffn_projection_in"] = ff;
    ff = backend.activation(ff, Activation::GeluTanh);
    if (trace) (*trace)["block.0.ffn_activation"] = ff;
    ff = backend.linear(ff, weights.at("ff.net.2.weight"), &weights.at("ff.net.2.bias"));
    if (trace) (*trace)["block.0.ffn_projection_out"] = ff;
    Tensor output = gated_residual(backend, policy, hidden, ff, parts[5]);
    if (trace) (*trace)["block.0.output"] = output;
    return output;
}

Tensor full_denoiser(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const Tensor& latent,
                     const Tensor& rope_cos, const Tensor& rope_sin,
                     const Tensor& text,
                     const Tensor& text_mask, const Tensor& sigma,
                     TensorBundle* trace = nullptr,
                     ltx_internal::BlockBoundaries* boundaries = nullptr) {
    validate_self_attention_context(backend, policy);
    Tensor hidden = backend.linear(latent, weights.at("patchify_proj.weight"),
                                   &weights.at("patchify_proj.bias"));
    if (trace) (*trace)["input_projection"] = hidden;
    auto [timestep, embedded] = timestep_conditioning(
        backend, policy, weights.prefix("adaln_single."), sigma);
    if (trace) {
        (*trace)["conditioning.modulation"] = timestep;
        (*trace)["conditioning.embedded"] = embedded;
    }
    WeightMap caption = weights.prefix("caption_projection.");
    Tensor context = backend.linear(text, caption.at("linear_1.weight"), &caption.at("linear_1.bias"));
    context = backend.activation(context, Activation::GeluTanh);
    context = backend.linear(context, caption.at("linear_2.weight"), &caption.at("linear_2.bias"));
    if (trace) (*trace)["conditioning.context"] = context;
    for (int index = 0; index < 28; ++index) {
        if (boundaries && boundaries->contains(index))
            boundaries->at(index) = {{"input", hidden}, {"context", context},
                {"timestep", timestep}, {"cosine", rope_cos}, {"sine", rope_sin},
                {"mask", text_mask}};
        TensorBundle block_trace;
        const bool detailed = trace && (index == 14 || index == 27);
        hidden = transformer_block(backend, policy,
            weights.prefix("transformer_blocks." + std::to_string(index) + "."),
            hidden, context, timestep, rope_cos, rope_sin, text_mask,
            trace && index == 0 ? trace : (detailed ? &block_trace : nullptr));
        if (detailed)
            for (const auto& [name, tensor] : block_trace)
                (*trace)["block." + std::to_string(index) + "." + name.substr(8)] = tensor;
        if (trace) (*trace)["block." + std::to_string(index)] = hidden;
    }
    Tensor table = backend.reshape(weights.at("scale_shift_table"), {1, 1, 2, hidden.dim(-1)});
    Tensor external = backend.reshape(embedded, {embedded.dim(0), embedded.dim(1), 1, hidden.dim(-1)});
    auto parts = backend.split(backend.add(table, external), {1, 1}, 2);
    Tensor shift = backend.reshape(parts[0], {parts[0].dim(0), parts[0].dim(1), hidden.dim(-1)});
    Tensor scale = backend.reshape(parts[1], {parts[1].dim(0), parts[1].dim(1), hidden.dim(-1)});
    hidden = modulate(backend, policy, backend.layer_norm(hidden, nullptr, nullptr, 1e-6f), shift, scale);
    if (trace) (*trace)["final_norm"] = hidden;
    Tensor output = producer_linear(backend, policy, PrecisionSemantic::DenoiserOutput,
                                    hidden, weights.at("proj_out.weight"),
                                    &weights.at("proj_out.bias"));
    if (trace) (*trace)["output_projection"] = output;
    return output;
}

Tensor edge_time_conv(Backend& backend, ComponentExecutor& e, const Tensor& input,
                      const std::string& prefix) {
    const Tensor& kernel = e.weight(prefix + ".weight");
    const int64_t temporal = (kernel.dim(2) - 1) / 2;
    Tensor x = input;
    if (temporal) x = backend.pad(x, {0, 0, 0, 0, temporal, temporal}, 0.0f, PadMode::Replicate);
    if (kernel.dim(3) > 1 || kernel.dim(4) > 1)
        x = backend.pad(x, {kernel.dim(4) / 2, kernel.dim(4) / 2,
                            kernel.dim(3) / 2, kernel.dim(3) / 2}, 0.0f);
    return e.conv3d(x, prefix);
}

Tensor decoder_residual(Backend& backend, ComponentExecutor& e, const WeightMap& weights,
                        const PrecisionPolicy& policy,
                        const Tensor& input, const std::string& prefix,
                        const Tensor& timestep, ComponentState& state, bool inject_noise) {
    Tensor y = e.pixel_norm(input);
    Tensor table = backend.reshape(weights.at(prefix + ".scale_shift_table"),
                                    {1, 4, input.dim(1), 1, 1, 1});
    Tensor external = backend.reshape(timestep, {input.dim(0), 4, input.dim(1), 1, 1, 1});
    auto parts = backend.split(backend.add(table, external), {1, 1, 1, 1}, 1);
    for (Tensor& part : parts) part = backend.reshape(part, {input.dim(0), input.dim(1), 1, 1, 1});
    y = e.silu(modulate(backend, policy, y, parts[0], parts[1]));
    y = edge_time_conv(backend, e, y, prefix + ".conv1.conv");
    if (inject_noise) y = e.spatial_noise(y, weights.at(prefix + ".per_channel_scale1"), state);
    y = e.pixel_norm(y);
    y = e.silu(modulate(backend, policy, y, parts[2], parts[3]));
    y = edge_time_conv(backend, e, y, prefix + ".conv2.conv");
    if (inject_noise) y = e.spatial_noise(y, weights.at(prefix + ".per_channel_scale2"), state);
    return backend.add(input, y);
}

Tensor residual_stack(Backend& backend, ComponentExecutor& e, const WeightMap& weights,
                      const PrecisionPolicy& policy,
                      Tensor x, int block, int count, const Tensor& scaled,
                      ComponentState& state, bool inject_noise) {
    const std::string base = "decoder.up_blocks." + std::to_string(block);
    Tensor embedded = e.timestep_embedding(scaled, base + ".time_embedder");
    embedded = backend.reshape(embedded, {x.dim(0), embedded.dim(-1), 1, 1, 1});
    for (int index = 0; index < count; ++index)
        x = decoder_residual(backend, e, weights, policy, x,
            base + ".res_blocks." + std::to_string(index), embedded, state, inject_noise);
    return x;
}

Tensor depth_to_space(Backend& backend, ComponentExecutor& e, const Tensor& input, int block) {
    Tensor residual = e.pixel_shuffle(input, {2, 2, 2});
    residual = backend.concat({residual, residual, residual, residual}, 1);
    residual = backend.slice(residual, 2, 1, residual.dim(2));
    Tensor x = edge_time_conv(backend, e, input,
        "decoder.up_blocks." + std::to_string(block) + ".conv.conv");
    x = e.pixel_shuffle(x, {2, 2, 2});
    x = backend.slice(x, 2, 1, x.dim(2));
    return backend.add(x, residual);
}

Tensor decode_ltx(Backend& backend, const PrecisionPolicy& policy,
                  const WeightMap& weights, const Tensor& tokens,
                  uint64_t seed, const std::vector<int64_t>& grid) {
    ComponentExecutor e(backend, weights);
    ComponentState state; state.rng = {seed, 0, "pytorch_compat.v1"};
    require(grid.size() == 3 && grid[0] * grid[1] * grid[2] == tokens.dim(1),
            "LTX latent grid/token mismatch");
    Tensor latent = backend.permute(
        backend.reshape(tokens, {tokens.dim(0), grid[0], grid[1], grid[2], 128}),
        {0, 4, 1, 2, 3});
    Tensor stddev = backend.reshape(weights.at("per_channel_statistics.std-of-means"), {1, 128, 1, 1, 1});
    Tensor mean = backend.reshape(weights.at("per_channel_statistics.mean-of-means"), {1, 128, 1, 1, 1});
    latent = backend.add(backend.mul(latent, stddev), mean);
    Tensor x = edge_time_conv(backend, e, latent, "decoder.conv_in.conv");
    Tensor scaled = backend.mul(host_f32({1}, {0.0f}), weights.at("decoder.timestep_scale_multiplier"));
    x = residual_stack(backend, e, weights, policy, x, 0, 8, scaled, state, false);
    x = depth_to_space(backend, e, x, 1);
    x = residual_stack(backend, e, weights, policy, x, 2, 7, scaled, state, true);
    x = depth_to_space(backend, e, x, 3);
    x = residual_stack(backend, e, weights, policy, x, 4, 6, scaled, state, true);
    x = depth_to_space(backend, e, x, 5);
    x = residual_stack(backend, e, weights, policy, x, 6, 5, scaled, state, true);
    x = e.pixel_norm(x);
    Tensor embedded = e.timestep_embedding(scaled, "decoder.last_time_embedder");
    Tensor table = backend.reshape(weights.at("decoder.last_scale_shift_table"),
                                    {1, 2, x.dim(1), 1, 1, 1});
    auto parts = backend.split(backend.add(table,
        backend.reshape(embedded, {x.dim(0), 2, x.dim(1), 1, 1, 1})), {1, 1}, 1);
    Tensor shift = backend.reshape(parts[0], {x.dim(0), x.dim(1), 1, 1, 1});
    Tensor scale = backend.reshape(parts[1], {x.dim(0), x.dim(1), 1, 1, 1});
    x = e.silu(modulate(backend, policy, x, shift, scale));
    x = edge_time_conv(backend, e, x, "decoder.conv_out.conv");
    const int64_t b = x.dim(0), t = x.dim(2), h = x.dim(3), w = x.dim(4);
    x = backend.reshape(x, {b, 3, 1, 4, 4, t, h, w});
    x = backend.permute(x, {0, 1, 5, 2, 6, 4, 7, 3});
    x = backend.reshape(x, {b, 3, t, h * 4, w * 4});
    return e.video_range(x);
}

class LtxDenoiser final : public Denoiser {
public:
    LtxDenoiser(Backend& backend, const PrecisionPolicy& policy,
                WeightMap weights, const Tensor& positive,
                const Tensor& negative, const Tensor& positive_mask,
                const Tensor& negative_mask, const Tensor& coordinates,
                const std::array<float, 3>& coordinate_scale)
        : backend_(backend), weights_(std::move(weights)), policy_(policy),
          text_(backend.concat({backend.copy_to_device(negative, policy.boundary_dtype(
                                    PrecisionSemantic::Conditioning)),
                                backend.copy_to_device(positive, policy.boundary_dtype(
                                    PrecisionSemantic::Conditioning))}, 0)),
          text_mask_(backend.copy_to_device(combine_masks(negative_mask, positive_mask),
                                             DType::Bool)),
          trace_enabled_(false) {
        require(coordinates.device() == Device::CPU && coordinates.dtype() == DType::F32 &&
                coordinates.ndim() == 3 && coordinates.dim(0) == 1 && coordinates.dim(1) == 3 &&
                coordinates.dim(2) > 0, "LTX coordinates contract mismatch");
        const int64_t tokens = coordinates.dim(2);
        const int64_t values = coordinates.numel();
        std::vector<float> transformed(static_cast<size_t>(values));
        const float* source = coordinates.data_as<float>();
        for (int64_t axis = 0; axis < 3; ++axis)
            for (int64_t token = 0; token < tokens; ++token)
                transformed[static_cast<size_t>(axis * tokens + token)] =
                    source[axis * tokens + token] * coordinate_scale[static_cast<size_t>(axis)];
        std::vector<float> doubled(static_cast<size_t>(values * 2));
        std::memcpy(doubled.data(), transformed.data(), values * sizeof(float));
        std::memcpy(doubled.data() + values, transformed.data(), values * sizeof(float));
        coordinates_ = host_f32({2, 3, coordinates.dim(2)}, doubled);
        rope_ = prepare_fractional_rope(
            prepared_tensors(), backend_, policy_, coordinates_,
            {20.0f, 2048.0f, 2048.0f}, 2048, 10000.0f, true);
    }
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        trace_.clear();
        const std::vector<Tensor>& rope = prepared_tensors().reuse(rope_);
        require(rope.size() == 2,
                "Fractional RoPE prepared tensor arity mismatch");
        Tensor doubled = backend_.concat({latent, latent}, 0);
        const float value = read_scalar_f32(timestep);
        Tensor sigma = host_f32({2, 1}, {value, value});
        Tensor prediction = full_denoiser(
            backend_, policy_, weights_, doubled, rope[0], rope[1], text_,
            text_mask_, sigma,
            trace_enabled_ ? &trace_ : nullptr);
        if (trace_enabled_) {
            trace_["denoiser.latent"] = doubled;
            trace_["denoiser.scalar_sigma"] = timestep;
            trace_["denoiser.sigma"] = sigma;
            trace_["denoiser.cosine"] = rope[0]; trace_["denoiser.sine"] = rope[1];
            trace_["denoiser.text"] = text_; trace_["denoiser.mask"] = text_mask_;
        }
        return backend_.split(prediction, {1, 1}, 0);
    }
    TensorBundle take_trace() override {
        TensorBundle result = std::move(trace_); trace_.clear(); return result;
    }
    void enable_trace() { trace_enabled_ = true; }
private:
    static Tensor combine_masks(const Tensor& first, const Tensor& second) {
        require(first.device() == Device::CPU && second.device() == Device::CPU &&
                first.dtype() == DType::Bool && second.dtype() == DType::Bool &&
                first.shape() == second.shape() && first.ndim() == 2,
                "LTX conditioning mask contract mismatch");
        std::vector<uint8_t> values(static_cast<size_t>(first.numel() + second.numel()));
        std::memcpy(values.data(), first.data(), first.bytes());
        std::memcpy(values.data() + first.numel(), second.data(), second.bytes());
        return host_bool({first.dim(0) + second.dim(0), 1, first.dim(1)}, values);
    }
    Backend& backend_; WeightMap weights_; const PrecisionPolicy& policy_;
    Tensor text_, text_mask_, coordinates_;
    PreparedTensorHandle rope_;
    bool trace_enabled_ = false;
    TensorBundle trace_;
};

Tensor full_mask(const Tensor& text) {
    return host_bool({text.dim(0), text.dim(1)},
                     std::vector<uint8_t>(static_cast<size_t>(text.dim(0) * text.dim(1)), 1));
}

struct LtxSamplingDeclaration {
    int steps = 3;
    float guidance_scale = 4.5f;
    int64_t resolution_shift_min_tokens = 1024;
    int64_t resolution_shift_max_tokens = 4096;
    float resolution_shift_min = 0.95f;
    float resolution_shift_max = 2.05f;

    static LtxSamplingDeclaration from_input(const TensorBundle& input) {
        LtxSamplingDeclaration declaration;
        if (input.contains("sampling_steps")) {
            const int64_t steps = read_scalar_i64(input.at("sampling_steps"));
            require(steps > 0 && steps <= 1000,
                    "LTX sampling_steps must be in [1, 1000]");
            declaration.steps = static_cast<int>(steps);
        }
        if (input.contains("guidance_scale"))
            declaration.guidance_scale = read_scalar_f32(input.at("guidance_scale"));
        if (input.contains("resolution_shift_min_tokens"))
            declaration.resolution_shift_min_tokens =
                read_scalar_i64(input.at("resolution_shift_min_tokens"));
        if (input.contains("resolution_shift_max_tokens"))
            declaration.resolution_shift_max_tokens =
                read_scalar_i64(input.at("resolution_shift_max_tokens"));
        if (input.contains("resolution_shift_min"))
            declaration.resolution_shift_min =
                read_scalar_f32(input.at("resolution_shift_min"));
        if (input.contains("resolution_shift_max"))
            declaration.resolution_shift_max =
                read_scalar_f32(input.at("resolution_shift_max"));
        declaration.validate();
        return declaration;
    }

    void validate() const {
        require(steps > 0 && steps <= 1000,
                "LTX sampling_steps must be in [1, 1000]");
        require(std::isfinite(guidance_scale),
                "LTX guidance_scale must be finite");
        require(resolution_shift_min_tokens > 0 &&
                resolution_shift_max_tokens > resolution_shift_min_tokens,
                "LTX resolution-shift token bounds are not legal");
        require(std::isfinite(resolution_shift_min) &&
                std::isfinite(resolution_shift_max) &&
                resolution_shift_max >= resolution_shift_min,
                "LTX resolution-shift range is not legal");
    }
};

class LtxArchitecture final : public Architecture {
public:
    explicit LtxArchitecture(const VrmModel& model)
        : denoiser_(model.bindings(model.graph().at("architecture_graph"))),
          component_(model.bindings(model.graph().at("component_graphs").array().at(0))) {
        const Json& config = model.graph().at("architecture_graph").at("config");
        if (const Json* declared = config.find("rope_coordinate_scale")) {
            require(declared->is_array() && declared->array().size() == coordinate_scale_.size(),
                    "LTX rope_coordinate_scale must declare exactly three axes");
            for (size_t axis = 0; axis < coordinate_scale_.size(); ++axis) {
                coordinate_scale_[axis] = static_cast<float>(declared->array()[axis].number());
                require(std::isfinite(coordinate_scale_[axis]) && coordinate_scale_[axis] > 0.0f,
                        "LTX rope_coordinate_scale values must be finite and positive");
            }
        }
    }
    std::unique_ptr<Denoiser> create_denoiser(
            Backend& backend, const PrecisionPolicy& policy,
            const TensorBundle& input) override {
        const Tensor positive_mask = input.contains("positive_mask")
            ? input.at("positive_mask") : full_mask(input.at("positive"));
        const Tensor negative_mask = input.contains("negative_mask")
            ? input.at("negative_mask") : full_mask(input.at("negative"));
        auto result = std::make_unique<LtxDenoiser>(backend, policy, denoiser_, input.at("positive"),
                                                    input.at("negative"), positive_mask,
                                                    negative_mask, input.at("coordinates"),
                                                    coordinate_scale_);
        if (input.contains("audit_trace") && read_scalar_i64(input.at("audit_trace")) != 0)
            result->enable_trace();
        return result;
    }
    SamplingProgram create_program(const TensorBundle& input) override {
        const int64_t tokens = input.at("coordinates").dim(2);
        require(tokens > 0, "LTX token count must be positive");
        const LtxSamplingDeclaration declaration =
            LtxSamplingDeclaration::from_input(input);
        SamplingProgram p; p.latent_shape = {1, tokens, 128};
        p.seed = read_scalar_i64(input.at("seed"));
        p.steps = declaration.steps;
        p.guidance_mode = GuidanceMode::CFG;
        p.guidance_coefficients = {
            1.0f - declaration.guidance_scale, declaration.guidance_scale};
        p.scheduler = SchedulerKind::Euler; p.subtract_prediction = true;
        const float min_tokens = static_cast<float>(declaration.resolution_shift_min_tokens);
        const float max_tokens = static_cast<float>(declaration.resolution_shift_max_tokens);
        const float mu = (declaration.resolution_shift_max - declaration.resolution_shift_min) /
            (max_tokens - min_tokens) * (tokens - min_tokens) +
            declaration.resolution_shift_min;
        const float shift = std::exp(mu);
        require(std::isfinite(shift) && shift > 0.0f,
                "LTX resolution shift is not finite and positive");
        for (int index = 0; index < p.steps; ++index) {
            const float sigma = 1.0f - static_cast<float>(index) / p.steps;
            const float shifted = shift / (shift + (1.0f / sigma - 1.0f));
            p.sigmas.push_back(shifted); p.model_timesteps.push_back(host_f32({1, 1}, {shifted}));
        }
        p.sigmas.push_back(0.0f);
        for (int index = 0; index < p.steps; ++index)
            p.update_deltas.push_back(p.sigmas[index] - p.sigmas[index + 1]);
        return p;
    }
    Tensor decode(Backend& backend, const PrecisionPolicy& policy, const Tensor& latent,
                  const TensorBundle& input) override {
        std::vector<int64_t> grid = {1, 2, 2};
        if (input.contains("latent_grid")) {
            const Tensor& encoded = input.at("latent_grid");
            require(encoded.device() == Device::CPU && encoded.dtype() == DType::I64 &&
                    encoded.shape() == std::vector<int64_t>({3}), "LTX latent_grid contract mismatch");
            grid.assign(encoded.data_as<int64_t>(), encoded.data_as<int64_t>() + 3);
            require(grid[0] > 0 && grid[1] > 0 && grid[2] > 0,
                    "LTX latent_grid is not legal");
        }
        return decode_ltx(backend, policy, component_, latent,
                          read_scalar_i64(input.at("decode_seed")), grid);
    }
private:
    WeightMap denoiser_, component_;
    std::array<float, 3> coordinate_scale_{1.0f, 1.0f, 1.0f};
};

}  // namespace

Tensor ltx_internal::transformer_block_forward(Backend& backend, const PrecisionPolicy& policy,
    const WeightMap& weights, const Tensor& input, const Tensor& context,
    const Tensor& timestep, const Tensor& cosine, const Tensor& sine, const Tensor& mask,
    TensorBundle* trace) {
    return transformer_block(backend, policy, weights, input, context, timestep,
                             cosine, sine, mask, trace);
}

Tensor ltx_internal::denoiser_forward(Backend& backend, const PrecisionPolicy& policy,
    const WeightMap& weights, const Tensor& latent, const Tensor& cosine, const Tensor& sine,
    const Tensor& text, const Tensor& mask, const Tensor& sigma,
    BlockBoundaries* boundaries) {
    return full_denoiser(backend, policy, weights, latent, cosine, sine, text, mask,
                         sigma, nullptr, boundaries);
}

std::unique_ptr<Architecture> make_ltx_architecture(const VrmModel& model) {
    return std::make_unique<LtxArchitecture>(model);
}

}  // namespace vrhino
