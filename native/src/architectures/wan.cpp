#include "vrhino/architecture.h"
#include "wan_self_attention_graph.h"
#include "wan_execution.h"
#include "vrhino/package_declaration.h"
#include "vrhino/product/program_declaration.h"

#include <cmath>
#include <memory>
#include <limits>
#include <utility>

#include "vrhino/components.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

Tensor project_attention(Backend& backend, const PrecisionPolicy& policy,
                         const WeightMap& weights, const std::string& prefix,
                         const Tensor& query_states, const Tensor& key_value_states,
                         int heads, TensorBundle* trace = nullptr,
                         const std::string& trace_prefix = {},
                         const Tensor* q_cos = nullptr, const Tensor* q_sin = nullptr,
                         const Tensor* k_cos = nullptr, const Tensor* k_sin = nullptr,
                         float epsilon = 1e-6f) {
    Tensor q = backend.linear(query_states, weights.at(prefix + ".q.weight"),
                              &weights.at(prefix + ".q.bias"));
    Tensor k = backend.linear(key_value_states, weights.at(prefix + ".k.weight"),
                              &weights.at(prefix + ".k.bias"));
    Tensor v = backend.linear(key_value_states, weights.at(prefix + ".v.weight"),
                              &weights.at(prefix + ".v.bias"));
    if (trace) {
        (*trace)[trace_prefix + ".q_linear"] = q;
        (*trace)[trace_prefix + ".k_linear"] = k;
        (*trace)[trace_prefix + ".v_linear"] = v;
    }
    q = attention_norm(backend, policy, q,
                       &weights.at(prefix + ".norm_q.weight"), epsilon);
    k = attention_norm(backend, policy, k,
                       &weights.at(prefix + ".norm_k.weight"), epsilon);
    if (trace) {
        (*trace)[trace_prefix + ".q_norm"] = q;
        (*trace)[trace_prefix + ".k_norm"] = k;
    }
    q = split_heads(backend, q, heads); k = split_heads(backend, k, heads); v = split_heads(backend, v, heads);
    if (trace) {
        (*trace)[trace_prefix + ".q_heads"] = q;
        (*trace)[trace_prefix + ".k_heads"] = k;
        (*trace)[trace_prefix + ".v_heads"] = v;
    }
    if (q_cos) q = operation_rope(backend, policy, q, *q_cos, *q_sin);
    if (k_cos) k = operation_rope(backend, policy, k, *k_cos, *k_sin);
    if (trace) {
        (*trace)[trace_prefix + ".q_positioned"] = q;
        (*trace)[trace_prefix + ".k_positioned"] = k;
    }
    Tensor attention_heads = operation_attention(backend, policy, q, k, v);
    if (trace) (*trace)[trace_prefix + ".attention_heads"] = attention_heads;
    Tensor attended = merge_heads(backend, attention_heads);
    if (trace) (*trace)[trace_prefix + ".attention_merged"] = attended;
    Tensor output = backend.linear(attended, weights.at(prefix + ".o.weight"),
        &weights.at(prefix + ".o.bias"));
    if (trace) (*trace)[trace_prefix + ".output"] = output;
    return output;
}

Tensor block(Backend& backend, const PrecisionPolicy& policy,
             const WeightMap& weights, const Tensor& input,
             const Tensor& modulation_input, const Tensor& context,
             const Tensor& rope_cos, const Tensor& rope_sin,
             TensorBundle* trace = nullptr,
             const std::string& trace_prefix = {},
             const wan_family::Config& config = {}) {
    if (trace) (*trace)[trace_prefix + ".input"] = input;
    Tensor combined, x;
    std::vector<Tensor> modulation(6);
    {
        auto values = wan_internal::self_attention(backend, policy, weights, input,
            modulation_input, rope_cos, rope_sin, trace != nullptr, config);
        x = values[trace ? 21 : 0];
        combined = values[1];
        if (trace) for (size_t i = 0; i < values.size(); ++i)
            (*trace)[trace_prefix + "." + wan_internal::checkpoints[i]] = values[i];
        // The Graph owns combined modulation. Only continuation rows are extracted here.
        for (int i = 3; i < 6; ++i) modulation[i] = backend.slice(combined, 1, i, i + 1);
    }
    if (trace) {
        (*trace)[trace_prefix + ".combined"] = combined;
        (*trace)[trace_prefix + ".modulation_input"] = modulation_input;
        (*trace)[trace_prefix + ".context"] = context;
        (*trace)[trace_prefix + ".cosine"] = rope_cos;
        (*trace)[trace_prefix + ".sine"] = rope_sin;
    }
    if (trace) (*trace)[trace_prefix + ".first_residual"] = x;
    const Tensor* norm_weight = weights.find("norm3.weight");
    const Tensor* norm_bias = weights.find("norm3.bias");
    Tensor cross_input = backend.layer_norm(x, norm_weight, norm_bias, config.epsilon);
    if (trace) (*trace)[trace_prefix + ".pre_cross_attention_norm"] = cross_input;
    Tensor cross_output = project_attention(backend, policy, weights, "cross_attn", cross_input,
                                             context, static_cast<int>(config.heads), trace,
                                             trace_prefix + ".cross_attention", nullptr, nullptr, nullptr, nullptr, config.epsilon);
    x = residual_add(backend, policy, x, cross_output);
    if (trace) (*trace)[trace_prefix + ".cross_residual"] = x;
    Tensor normalized = backend.layer_norm(x, nullptr, nullptr, config.epsilon);
    if (trace) (*trace)[trace_prefix + ".pre_ffn_norm"] = normalized;
    Tensor ff_input = modulate(backend, policy, normalized, modulation[3], modulation[4]);
    if (trace) (*trace)[trace_prefix + ".ffn_input"] = ff_input;
    BackendProfileRegion mlp_profile(backend, "mlp.total");
    Tensor ff = backend.linear(ff_input, weights.at("ffn.0.weight"), &weights.at("ffn.0.bias"));
    if (trace) (*trace)[trace_prefix + ".ffn_projection_in"] = ff;
    ff = backend.activation(ff, Activation::GeluTanh);
    if (trace) (*trace)[trace_prefix + ".ffn_activation"] = ff;
    ff = backend.linear(ff, weights.at("ffn.2.weight"), &weights.at("ffn.2.bias"));
    if (trace) (*trace)[trace_prefix + ".ffn_projection_out"] = ff;
    Tensor output = gated_residual(backend, policy, x, ff, modulation[5]);
    if (trace) (*trace)[trace_prefix + ".output"] = output;
    return output;
}

Tensor full_denoiser(Backend& backend, const PrecisionPolicy& policy,
                     const WeightMap& weights, const Tensor& video,
                     const Tensor& timestep, const Tensor& raw_context,
                     TensorBundle* trace = nullptr, int trace_block = -1,
                     const wan_family::Config& config = {}) {
    Tensor patch = backend.conv3d(video, weights.at("patch_embedding.weight"),
                                  &weights.at("patch_embedding.bias"),
                                  {config.patch[0], config.patch[1], config.patch[2]}, {0, 0, 0});
    if (trace) (*trace)["input_projection"] = patch;
    const int64_t batch = patch.dim(0), width = patch.dim(1), frames = patch.dim(2),
                  height = patch.dim(3), spatial_width = patch.dim(4), tokens = frames * height * spatial_width;
    Tensor hidden = backend.permute(backend.reshape(patch, {batch, width, tokens}), {0, 2, 1});
    const DType temporary_full = policy.operation_contract(
        PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute).temporary_dtype;
    Tensor time = backend.sinusoidal_embedding(
        backend.reshape(backend.cast(timestep, temporary_full), {-1}),
        config.frequency, true, 0.0, true);
    time = operation_linear(backend, policy, PrecisionOperation::Modulation,
                            PrecisionSemantic::TemporaryCompute, time,
                            weights.at("time_embedding.0.weight"),
                            &weights.at("time_embedding.0.bias"));
    time = backend.activation(time, Activation::Silu);
    time = operation_linear(backend, policy, PrecisionOperation::Modulation,
                            PrecisionSemantic::TemporaryCompute, time,
                            weights.at("time_embedding.2.weight"),
                            &weights.at("time_embedding.2.bias"));
    if (trace) (*trace)["conditioning.time"] = time;
    Tensor modulation = operation_linear(
        backend, policy, PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute,
        backend.activation(time, Activation::Silu),
        weights.at("time_projection.1.weight"),
        &weights.at("time_projection.1.bias"));
    modulation = backend.reshape(modulation, {batch, 6, width});
    if (trace) (*trace)["conditioning.modulation"] = modulation;
    require(raw_context.ndim() == 3 && raw_context.dim(1) <= config.text_length,
            "Wan conditioning sequence contract mismatch");
    Tensor context = backend.pad(raw_context,
                                 {0, 0, 0, config.text_length - raw_context.dim(1)}, 0.0f);
    context = backend.linear(context, weights.at("text_embedding.0.weight"), &weights.at("text_embedding.0.bias"));
    context = backend.activation(context, Activation::GeluTanh);
    context = backend.linear(context, weights.at("text_embedding.2.weight"), &weights.at("text_embedding.2.bias"));
    if (trace) (*trace)["conditioning.context"] = context;
    std::vector<std::vector<float>> coordinates;
    for (int64_t t = 0; t < frames; ++t) for (int64_t h = 0; h < height; ++h)
        for (int64_t w = 0; w < spatial_width; ++w) coordinates.push_back({static_cast<float>(t), static_cast<float>(h), static_cast<float>(w)});
    const int d = static_cast<int>(config.dim / config.heads);
    auto [rope_cos, rope_sin] = standard_rope(
        backend, policy, coordinates, {d - 4 * (d / 6), 2 * (d / 6), 2 * (d / 6)}, config.rope_theta, true);
    rope_cos = backend.cast(rope_cos, temporary_full);
    rope_sin = backend.cast(rope_sin, temporary_full);
    for (int index = 0; index < config.layers; ++index) {
        const std::string block_prefix = "block." + std::to_string(index);
        hidden = block(backend, policy,
                       weights.prefix("blocks." + std::to_string(index) + "."),
                       hidden, modulation, context, rope_cos, rope_sin,
                       trace && index == trace_block ? trace : nullptr, block_prefix, config);
        if (trace) (*trace)[block_prefix + ".output"] = hidden;
        if (trace && index == 0) (*trace)["block_0"] = hidden;
        if (trace && index == config.layers / 2) (*trace)["middle_block"] = hidden;
        if (trace && index == config.layers - 1) (*trace)["last_block"] = hidden;
    }
    const DType modulation_dtype = policy.operation_contract(
        PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute).temporary_dtype;
    Tensor head_external = backend.reshape(backend.cast(time, modulation_dtype), {batch, 1, width});
    auto head_parts = backend.split(backend.add(
        backend.cast(weights.at("head.modulation"), modulation_dtype), head_external),
        {1, 1}, 1);
    Tensor normalized = backend.layer_norm(hidden, nullptr, nullptr, config.epsilon);
    Tensor projected = producer_linear(backend, policy,
        PrecisionSemantic::DenoiserOutput,
        modulate(backend, policy, normalized, head_parts[0], head_parts[1]),
        weights.at("head.head.weight"), &weights.at("head.head.bias"));
    if (trace) (*trace)["final_projection"] = projected;
    projected = backend.reshape(projected, {batch, frames, height, spatial_width, config.patch[0], config.patch[1], config.patch[2], config.output_channels});
    projected = backend.permute(projected, {0, 7, 1, 4, 2, 5, 3, 6});
    return backend.reshape(projected, {batch, config.output_channels, frames * config.patch[0], height * config.patch[1], spatial_width * config.patch[2]});
}

Tensor decoder_residual(Backend& backend, ComponentExecutor& executor,
                        const WeightMap& weights, const Tensor& input,
                        const std::string& prefix, ComponentState& state) {
    Tensor identity = input;
    if (weights.contains(prefix + ".shortcut.weight"))
        identity = executor.causal_conv3d(input, prefix + ".shortcut",
            PadMode::Constant, true, {1, 1, 1}, &state, prefix + ".shortcut");
    Tensor y = executor.silu(executor.rms_channel(input, prefix + ".residual.0"));
    y = executor.causal_conv3d(y, prefix + ".residual.2",
        PadMode::Constant, true, {1, 1, 1}, &state, prefix + ".residual.2");
    y = executor.silu(executor.rms_channel(y, prefix + ".residual.3"));
    y = executor.causal_conv3d(y, prefix + ".residual.6",
        PadMode::Constant, true, {1, 1, 1}, &state, prefix + ".residual.6");
    return backend.add(y, identity);
}

Tensor decoder_resample(Backend& backend, ComponentExecutor& executor,
                        const Tensor& input, const std::string& prefix,
                        bool temporal, ComponentState& state) {
    Tensor expanded = input;
    if (temporal) {
        const std::string cache_key = prefix + ".time_conv";
        const auto found = state.temporal_cache.find(cache_key);
        if (found == state.temporal_cache.end()) {
            // The first streaming chunk is emitted once. A zero history marker
            // makes the next generic cached causal convolution identical to
            // the official two-zero-frame left padding.
            state.temporal_cache.emplace(cache_key, backend.mul(
                backend.slice(input, 2, 0, 1), scalar_f32(0.0f)));
        } else {
            Tensor packed = executor.causal_conv3d(input, prefix + ".time_conv",
                PadMode::Constant, true, {1, 1, 1}, &state, cache_key);
            const int64_t b = packed.dim(0), channels = packed.dim(1) / 2,
                          t = packed.dim(2), h = packed.dim(3), w = packed.dim(4);
            expanded = backend.reshape(backend.permute(
                backend.reshape(packed, {b, 2, channels, t, h, w}),
                {0, 2, 3, 1, 4, 5}), {b, channels, t * 2, h, w});
        }
    }
    const int64_t b = expanded.dim(0), c = expanded.dim(1), t = expanded.dim(2),
                  h = input.dim(3), w = input.dim(4);
    Tensor frames = backend.reshape(backend.permute(expanded, {0, 2, 1, 3, 4}),
                                    {b * t, c, h, w});
    frames = executor.nearest(frames, {2.0, 2.0});
    Tensor x = backend.permute(backend.reshape(frames, {b, t, c, h * 2, w * 2}),
                               {0, 2, 1, 3, 4});
    return executor.conv2d_frames(x, prefix + ".resample.1");
}

Tensor decode_wan_chunk(Backend& backend, ComponentExecutor& executor,
                        const WeightMap& weights, const Tensor& input,
                        ComponentState& state, TensorBundle* trace) {
    Tensor x = executor.causal_conv3d(input, "decoder.conv1", PadMode::Constant,
                                      true, {1, 1, 1}, &state, "decoder.conv1");
    x = decoder_residual(backend, executor, weights, x, "decoder.middle.0", state);
    x = executor.spatial_attention_conv(x, "decoder.middle.1");
    x = decoder_residual(backend, executor, weights, x, "decoder.middle.2", state);
    if (trace) (*trace)["vae.middle"] = x;
    for (int index : {0, 1, 2}) x = decoder_residual(backend, executor, weights, x,
        "decoder.upsamples." + std::to_string(index), state);
    x = decoder_resample(backend, executor, x, "decoder.upsamples.3", true, state);
    for (int index : {4, 5, 6}) x = decoder_residual(backend, executor, weights, x,
        "decoder.upsamples." + std::to_string(index), state);
    x = decoder_resample(backend, executor, x, "decoder.upsamples.7", true, state);
    for (int index : {8, 9, 10}) x = decoder_residual(backend, executor, weights, x,
        "decoder.upsamples." + std::to_string(index), state);
    x = decoder_resample(backend, executor, x, "decoder.upsamples.11", false, state);
    for (int index : {12, 13, 14}) x = decoder_residual(backend, executor, weights, x,
        "decoder.upsamples." + std::to_string(index), state);
    x = executor.silu(executor.rms_channel(x, "decoder.head.0"));
    return executor.causal_conv3d(x, "decoder.head.2", PadMode::Constant,
                                  true, {1, 1, 1}, &state, "decoder.head.2");
}

Tensor decode_wan(Backend& backend, const PrecisionPolicy& policy,
                  const WeightMap& weights, const Tensor& raw_latent,
                  TensorBundle* trace = nullptr) {
    static const std::vector<float> mean = {
        -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
        0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
    static const std::vector<float> stddev = {
        2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
        3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
    ComponentExecutor executor(backend, weights);
    const DType vae_input_dtype = policy.boundary_dtype(
        PrecisionSemantic::VaeInput);
    Tensor latent = backend.add(backend.mul(backend.cast(raw_latent, vae_input_dtype),
        host_f32({1, 16, 1, 1, 1}, stddev)), host_f32({1, 16, 1, 1, 1}, mean));
    if (trace) (*trace)["vae.input_scaled"] = latent;
    Tensor x = executor.conv3d(latent, "conv2");
    if (trace) (*trace)["vae.early"] = x;
    ComponentState state;
    std::vector<Tensor> decoded_chunks;
    decoded_chunks.reserve(static_cast<size_t>(x.dim(2)));
    for (int64_t frame = 0; frame < x.dim(2); ++frame)
        decoded_chunks.push_back(decode_wan_chunk(
            backend, executor, weights, backend.slice(x, 2, frame, frame + 1),
            state, trace));
    Tensor decoded = decoded_chunks.size() == 1 ? decoded_chunks.front() :
        backend.concat(decoded_chunks, 2);
    if (trace) (*trace)["vae.decoded"] = decoded;
    return executor.clamp(decoded, -1.0f, 1.0f);
}

class WanDenoiser final : public Denoiser {
public:
    WanDenoiser(Backend& backend, const PrecisionPolicy& policy,
                std::shared_ptr<const wan_family::Definition> definition,
                std::shared_ptr<const AdmittedArchitectureBinding> binding, const Tensor& positive,
                const Tensor& negative, bool trace_enabled,
                int trace_step, int trace_block)
        : backend_(backend), policy_(policy), definition_(std::move(definition)), binding_(std::move(binding)),
          weights_(binding_->weights()),
          positive_(backend.copy_to_device(positive, policy.boundary_dtype(
              PrecisionSemantic::Conditioning))),
          negative_(backend.copy_to_device(negative, policy.boundary_dtype(
              PrecisionSemantic::Conditioning))),
          trace_enabled_(trace_enabled), trace_step_(trace_step),
          trace_block_(trace_block) {}
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        return evaluate_at(latent, timestep, evaluation_index_);
    }
    std::vector<Tensor> evaluate_step(const Tensor& latent, const StepExecutionContext& context) override {
        if (!context.selected_component.graph()) return evaluate(latent,context.timestep);
        return evaluate_at(latent, context.timestep, static_cast<int>(context.step_index));
    }
    std::vector<Tensor> evaluate_at(const Tensor& latent, const Tensor& timestep, int global_step) {
        trace_.clear();
        TensorBundle negative_trace, positive_trace;
        const bool trace_this_step = trace_enabled_ &&
            (trace_step_ < 0 || global_step == trace_step_);
        Tensor unconditional = full_denoiser(backend_, policy_, weights_, latent,
            timestep, negative_,
            trace_this_step ? &negative_trace : nullptr, trace_block_, definition_->config);
        Tensor conditional = full_denoiser(backend_, policy_, weights_, latent,
            timestep, positive_,
            trace_this_step ? &positive_trace : nullptr, trace_block_, definition_->config);
        ++evaluation_index_;
        for (auto& [name, tensor] : negative_trace)
            trace_["branch.0." + name] = std::move(tensor);
        for (auto& [name, tensor] : positive_trace)
            trace_["branch.1." + name] = std::move(tensor);
        return {unconditional, conditional};
    }
    TensorBundle take_trace() override { TensorBundle result = std::move(trace_); trace_.clear(); return result; }
private:
    Backend& backend_; const PrecisionPolicy& policy_;
    std::shared_ptr<const wan_family::Definition> definition_;
    std::shared_ptr<const AdmittedArchitectureBinding> binding_;
    WeightMap weights_;
    Tensor positive_, negative_;
    bool trace_enabled_ = false;
    int trace_step_ = -1;
    int trace_block_ = -1;
    int evaluation_index_ = 0;
    TensorBundle trace_;
};

class WanArchitecture final : public Architecture {
public:
    explicit WanArchitecture(const VrmModel& model)
        : definition_(wan_family::lower(model.canonical_architecture())),
          denoiser_(AdmittedArchitectureBinding::admit(definition_->parameters,
              model.bindings(model.graph().at("architecture_graph")), BorrowedBindingLifetime::CallerRetained)),
          component_(model.bindings(model.graph().at("component_graphs").array().at(0))) {}
    explicit WanArchitecture(std::shared_ptr<const VrmModel> model) : owner_(std::move(model)) {
        package_.emplace(PackageDeclaration::parse(owner_->graph()));
        package_->validate_tensor_references(*owner_);
        // This family adapter currently supports one shared topology per package.
        require(package_->graphs().size() == 1, "Family adapter requires one shared graph");
        definition_ = wan_family::lower(CanonicalArchitectureDeclaration::parse(
            package_->graphs().begin()->second));
        for (const auto& [id, binding] : package_->bindings()) {
            std::map<std::string, const Tensor*> weights;
            for (const auto& [slot, tensor] : binding.parameters)
                weights.emplace(slot, &owner_->tensor(tensor));
            bindings_.emplace(id, AdmittedArchitectureBinding::admit(definition_->parameters,
                weights, BorrowedBindingLifetime::ExplicitOwners, {owner_}));
        }
        programs_.emplace(product::admit_program_declaration(owner_->metadata().at("programs"), *package_));
        const auto& shared = owner_->metadata().at("shared_components");
        const auto& decoder = shared.at("decoder");
        require(decoder.at("implementation_id").string() == "dit_flow.wan.vae_decoder.v1" &&
            decoder.at("latent_contract").at("channels").integer() == definition_->config.output_channels &&
            decoder.at("latent_contract").at("layout").string() == "BCTHW",
            "Unsupported family decoder interface");
        auto decoder_weights = owner_->bindings(decoder);
        std::vector<ArchitectureParameterSlot> slots;
        for (const auto& slot : shared.at("decoder_slots").array()) {
            std::vector<int64_t> shape;
            for (const auto& n : slot.at("shape").array()) shape.push_back(n.integer());
            const auto& name = slot.at("name").string();
            std::string role;
            for (const auto& [key, value] : decoder.at("runtime_tensor_bindings").object())
                if (value.string() == name) { require(role.empty(), "Duplicate decoder tensor alias"); role = key; }
            require(!role.empty(), "Decoder slot has no binding");
            require(slot.at("dtype").string() == "float32", "Unsupported decoder storage dtype");
            slots.push_back({role, shape, DType::F32});
        }
        decoder_binding_ = AdmittedArchitectureBinding::admit(
            std::make_shared<ArchitectureBindingDeclaration>(std::move(slots)), decoder_weights,
            BorrowedBindingLifetime::ExplicitOwners, {owner_});
        component_ = WeightMap(decoder_binding_->weights());
    }
    std::unique_ptr<Denoiser> create_denoiser(
            Backend& backend, const PrecisionPolicy& policy,
            const TensorBundle& input) override {
        require(!package_, "Catalog package requires admitted execution setup; singleton fallback forbidden");
        const auto [trace_step, trace_block] = tracing(input);
        return wan_family::bound_denoiser(definition_, denoiser_, backend, policy, latent_shape(input),
            input.at("positive"), input.at("negative"), trace_enabled_, trace_step, trace_block);
    }
    ExecutionSetup create_execution_setup(Backend& backend, const PrecisionPolicy& policy,
                                         const TensorBundle& input) override {
        const auto [trace_step, trace_block] = tracing(input);
        auto execution = wan_family::realize(definition_, backend, policy, latent_shape(input),
            input.at("positive"), input.at("negative"), trace_enabled_, trace_step, trace_block);
        std::vector<ComponentInstance> instances;
        if (package_) {
            for (const auto& [id, instance] : package_->instances())
                instances.push_back(execution.bind({id}, {instance.binding}, bindings_.at(instance.binding)));
            return {ExecutionContext(std::move(instances)), programs_->execution};
        }
        instances.push_back(execution.bind({0}, {0}, denoiser_));
        return {ExecutionContext(std::move(instances)), ExecutionProgram::uniform({0})};
    }
    SamplingProgram create_program(const TensorBundle& input) override {
        if (programs_) {
            require(!input.contains("sampling_steps") && !input.contains("guidance_scale") &&
                    !input.contains("flow_shift"), "Declared programs cannot be overridden by legacy sampling inputs");
            auto program = programs_->sampling;
            program.latent_shape = latent_shape(input);
            program.seed = read_scalar_i64(input.at("seed"));
            return program;
        }
        SamplingProgram program; program.latent_shape = latent_shape(input);
        program.seed = read_scalar_i64(input.at("seed"));
        program.steps = input.contains("sampling_steps") ?
            static_cast<int>(read_scalar_i64(input.at("sampling_steps"))) : 3;
        const float guidance = input.contains("guidance_scale") ?
            read_scalar_f32(input.at("guidance_scale")) : 5.0f;
        program.guidance_mode = GuidanceMode::CFG;
        program.guidance_coefficients = {1.0f - guidance, guidance};
        constexpr double start = 0.999, stop = 0.0;
        const double shift = input.contains("flow_shift") ?
            read_scalar_f32(input.at("flow_shift")) : 5.0;
        std::vector<float> sigmas;
        std::vector<Tensor> model_timesteps;
        for (int index = 0; index < program.steps; ++index) {
            const double value = start + (stop - start) * index / program.steps;
            const double shifted = shift * value / (1.0 + (shift - 1.0) * value);
            sigmas.push_back(static_cast<float>(shifted));
            model_timesteps.push_back(
                scalar_i64(static_cast<int64_t>(shifted * 1000.0)));
        }
        sigmas.push_back(0.0f);
        std::vector<FlowScheduleTransition> transitions;
        transitions.reserve(static_cast<size_t>(program.steps));
        for (int index = 0; index < program.steps; ++index)
            transitions.push_back({std::move(model_timesteps.at(index)),
                                   sigmas.at(index), sigmas.at(index + 1)});
        program.contract.emplace(
            PredictionContract{PredictionSemantic::Flow},
            SolverContract{SolverSemantic::MultistepPredictorCorrector, 2},
            ScheduleContract::flow_sigma(std::move(transitions)));
        return program;
    }
    Tensor decode(Backend& backend, const PrecisionPolicy& policy,
                  const Tensor& latent, const TensorBundle&) override {
        trace_.clear();
        return decode_wan(
            backend, policy, component_, latent, trace_enabled_ ? &trace_ : nullptr);
    }
    TensorBundle take_trace() override { TensorBundle result = std::move(trace_); trace_.clear(); return result; }
private:
    std::vector<int64_t> latent_shape(const TensorBundle& input) const {
        const auto& c = definition_->config;
        std::vector<int64_t> shape{1,c.input_channels,c.patch[0],2*c.patch[1],2*c.patch[2]};
        if (input.contains("latent_shape")) {
            const auto& value = input.at("latent_shape");
            validate_architecture_tensor(value,{5},DType::I64);
            shape.assign(value.data_as<int64_t>(),value.data_as<int64_t>()+5);
        }
        wan_family::validate_latent_shape(*definition_,shape);
        require(c.output_channels == 16, "Existing decoder requires 16 latent channels");
        return shape;
    }
    std::pair<int,int> tracing(const TensorBundle& input) {
        trace_enabled_ = input.contains("audit_trace") && read_scalar_i64(input.at("audit_trace")) != 0;
        const int64_t step = input.contains("audit_trace_step") ? read_scalar_i64(input.at("audit_trace_step")) : -1;
        const int64_t block_index = input.contains("audit_trace_block") ? read_scalar_i64(input.at("audit_trace_block")) : -1;
        require(step >= -1 && step <= std::numeric_limits<int>::max(), "Wan audit trace step outside supported bound");
        require(block_index >= -1 && block_index < definition_->config.layers, "Wan audit trace block outside graph");
        return {static_cast<int>(step),static_cast<int>(block_index)};
    }
    std::shared_ptr<const VrmModel> owner_;
    std::optional<PackageDeclaration> package_;
    std::optional<product::DeclaredPrograms> programs_;
    std::map<uint32_t, std::shared_ptr<const AdmittedArchitectureBinding>> bindings_;
    std::shared_ptr<const AdmittedArchitectureBinding> decoder_binding_;
    std::shared_ptr<const wan_family::Definition> definition_;
    std::shared_ptr<const AdmittedArchitectureBinding> denoiser_;
    WeightMap component_;
    bool trace_enabled_ = false;
    TensorBundle trace_;
};

}  // namespace

namespace wan_family {
std::unique_ptr<Denoiser> bound_denoiser(std::shared_ptr<const Definition> definition,
    std::shared_ptr<const AdmittedArchitectureBinding> binding, Backend& backend,
    const PrecisionPolicy& policy, const std::vector<int64_t>& latent_shape,
    const Tensor& positive, const Tensor& negative, bool trace_enabled, int trace_step, int trace_block) {
    require(definition && binding, "Missing family definition/binding");
    (void)binding->parameters_for(*definition->parameters);
    validate_request(*definition,latent_shape,positive,negative);
    require(trace_step >= -1 && trace_block >= -1 && trace_block < definition->config.layers, "Invalid family trace declaration");
    require(policy.requested_dtype() == backend.execution_dtype(), "Family policy/backend context mismatch");
    return std::make_unique<WanDenoiser>(backend,policy,std::move(definition),std::move(binding),
        positive,negative,trace_enabled,trace_step,trace_block);
}

ExecutionDefinition realize(std::shared_ptr<const Definition> definition, Backend& backend,
    const PrecisionPolicy& policy, const std::vector<int64_t>& latent_shape,
    const Tensor& positive, const Tensor& negative, bool trace_enabled, int trace_step, int trace_block) {
    require(definition != nullptr, "Missing family graph definition");
    validate_request(*definition,latent_shape,positive,negative);
    std::vector<ExecutionTensorContract> slots;
    for (const auto& slot : definition->parameters->slots()) slots.push_back({slot.shape,slot.dtype});
    // Raw endpoint output follows the existing producer contract. Sampling still
    // performs its unchanged consumer cast. Timestep representation is exact I64.
    const auto output = policy.operation_contract(PrecisionOperation::Linear,PrecisionSemantic::DenoiserOutput).output_dtype;
    // Injective within the admitted 21-bit field bounds: versioned padded-text
    // interface, feature width and sequence bound. No model or storage identity.
    const uint64_t conditioning_contract = (uint64_t{1} << 42) |
        (static_cast<uint64_t>(definition->config.text_dim) << 21) |
        static_cast<uint64_t>(definition->config.text_length);
    ComponentInterface interface{{latent_shape,policy.persistent_state_dtype(PrecisionSemantic::SamplingState)},
        {{},DType::I64},{{latent_shape,output},{latent_shape,output}},GuidanceMode::CFG,PredictionSemantic::Flow,conditioning_contract};
    auto graph = std::make_shared<const ComponentGraphDefinition>(interface,std::move(slots),
        [definition,&backend,&policy,latent_shape,positive,negative,trace_enabled,trace_step,trace_block]
        (const std::vector<Tensor>& parameters) {
            require(parameters.size()==definition->parameters->slots().size(),"Family factory slot count mismatch");
            std::map<std::string,const Tensor*> named;
            for (size_t i=0;i<parameters.size();++i) named.emplace(definition->parameters->slots()[i].role,&parameters[i]);
            auto binding=AdmittedArchitectureBinding::admit(definition->parameters,named,BorrowedBindingLifetime::CallerRetained);
            return bound_denoiser(definition,std::move(binding),backend,policy,latent_shape,positive,negative,
                                  trace_enabled,trace_step,trace_block);
        });
    return {std::move(definition),std::move(graph)};
}

ComponentInstance ExecutionDefinition::bind(ComponentInstanceID instance, BindingID binding,
    std::shared_ptr<const AdmittedArchitectureBinding> parameters) const {
    require(family && graph && parameters,"Missing family execution declaration");
    const auto& handles=parameters->parameters_for(*family->parameters);
    return ComponentInstance::bind(instance,binding,graph,handles,{parameters});
}
}  // namespace wan_family

std::unique_ptr<Architecture> make_wan_package_architecture(std::shared_ptr<const VrmModel> model) {
    return std::make_unique<WanArchitecture>(std::move(model));
}

std::unique_ptr<Architecture> make_wan_architecture(const VrmModel& model) {
    return std::make_unique<WanArchitecture>(model);
}

}  // namespace vrhino
