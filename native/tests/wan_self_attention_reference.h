#pragma once
// Test-only producer and imperative self-attention reference.
// Production code is included for shared helpers and Graph block comparisons.
#define make_wan_architecture make_wan_test_architecture
#include "../src/architectures/wan.cpp"
#undef make_wan_architecture
namespace vrhino::wan_test {
inline TensorBundle producer(Backend& backend, const PrecisionPolicy& policy,
                             const WeightMap& weights, const Tensor& video,
                             const Tensor& timestep, const Tensor& raw_context) {
    TensorBundle* trace = nullptr;
    Tensor patch = backend.conv3d(video, weights.at("patch_embedding.weight"),
                                  &weights.at("patch_embedding.bias"), {1, 2, 2}, {0, 0, 0});
    if (trace) (*trace)["input_projection"] = patch;
    const int64_t batch = patch.dim(0), width = patch.dim(1), frames = patch.dim(2),
                  height = patch.dim(3), spatial_width = patch.dim(4), tokens = frames * height * spatial_width;
    Tensor hidden = backend.permute(backend.reshape(patch, {batch, width, tokens}), {0, 2, 1});
    const DType temporary_full = policy.operation_contract(
        PrecisionOperation::Modulation,
        PrecisionSemantic::TemporaryCompute).temporary_dtype;
    Tensor time = backend.sinusoidal_embedding(
        backend.reshape(backend.cast(timestep, temporary_full), {-1}),
        256, true, 0.0, true);
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
    require(raw_context.ndim() == 3 && raw_context.dim(1) <= 512,
            "Wan conditioning sequence contract mismatch");
    Tensor context = backend.pad(raw_context,
                                 {0, 0, 0, 512 - raw_context.dim(1)}, 0.0f);
    context = backend.linear(context, weights.at("text_embedding.0.weight"), &weights.at("text_embedding.0.bias"));
    context = backend.activation(context, Activation::GeluTanh);
    context = backend.linear(context, weights.at("text_embedding.2.weight"), &weights.at("text_embedding.2.bias"));
    if (trace) (*trace)["conditioning.context"] = context;
    std::vector<std::vector<float>> coordinates;
    for (int64_t t = 0; t < frames; ++t) for (int64_t h = 0; h < height; ++h)
        for (int64_t w = 0; w < spatial_width; ++w) coordinates.push_back({static_cast<float>(t), static_cast<float>(h), static_cast<float>(w)});
    auto [rope_cos, rope_sin] = standard_rope(
        backend, policy, coordinates, {44, 42, 42}, 10000.0, true);
    rope_cos = backend.cast(rope_cos, temporary_full);
    rope_sin = backend.cast(rope_sin, temporary_full);
    return {{"hidden",hidden},{"modulation",modulation},{"context",context},
            {"cosine",rope_cos},{"sine",rope_sin}};
}
inline TensorBundle reference(Backend& backend, const PrecisionPolicy& policy,
                              const WeightMap& weights, const TensorBundle& x) {
    TensorBundle trace;
    const Tensor& input=x.at("hidden");
    trace["input"]=input;
    const DType modulation_dtype=policy.operation_contract(
        PrecisionOperation::Modulation,PrecisionSemantic::TemporaryCompute).temporary_dtype;
    Tensor combined=backend.add(backend.cast(weights.at("modulation"),modulation_dtype),
                                 backend.cast(x.at("modulation"),modulation_dtype));
    auto modulation=backend.split(combined,{1,1,1,1,1,1},1);
    for(Tensor& value:modulation) value=backend.reshape(value,{combined.dim(0),1,combined.dim(2)});
    trace["combined"]=combined;trace["shift"]=modulation[0];trace["scale"]=modulation[1];trace["gate"]=modulation[2];
    Tensor normalized=backend.layer_norm(input,nullptr,nullptr,1e-6f);
    trace["pre_attention_norm"]=normalized;
    Tensor self_input=modulate(backend,policy,normalized,modulation[0],modulation[1]);
    trace["self_attention_input"]=self_input;
    Tensor output=project_attention(backend,policy,weights,"self_attn",self_input,self_input,12,
        &trace,"self_attention",&x.at("cosine"),&x.at("sine"),&x.at("cosine"),&x.at("sine"));
    trace["first_residual"]=gated_residual(backend,policy,input,output,modulation[2]);
    // Diagnostic-only gate multiplication; excluded from timed reference execution.
    return trace;
}
}
