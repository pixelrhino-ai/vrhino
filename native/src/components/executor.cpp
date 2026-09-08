#include "vrhino/components.h"

#include <utility>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {

Tensor ComponentExecutor::conv3d(const Tensor& x, const std::string& prefix,
                                 const std::vector<int>& padding,
                                 const std::vector<int>& stride) {
    const Tensor* bias = weights_.find(prefix + ".bias");
    return backend_.conv3d(x, weight(prefix + ".weight"), bias, stride, padding);
}

Tensor ComponentExecutor::causal_conv3d(Tensor& x, const std::string& prefix,
                                        PadMode mode, bool causal,
                                        const std::vector<int>& stride,
                                        ComponentState* state,
                                        const std::string& cache_key) {
    return causal_conv3d_owned(std::move(x), prefix, mode, causal, stride,
                               state, cache_key);
}

Tensor ComponentExecutor::causal_conv3d(Tensor&& x, const std::string& prefix,
                                        PadMode mode, bool causal,
                                        const std::vector<int>& stride,
                                        ComponentState* state,
                                        const std::string& cache_key) {
    return causal_conv3d_owned(std::move(x), prefix, mode, causal, stride,
                               state, cache_key);
}

Tensor ComponentExecutor::causal_conv3d(const Tensor& x, const std::string& prefix,
                                        PadMode mode, bool causal,
                                        const std::vector<int>& stride,
                                        ComponentState* state,
                                        const std::string& cache_key) {
    return causal_conv3d_owned(x, prefix, mode, causal, stride, state,
                               cache_key);
}

Tensor ComponentExecutor::causal_conv3d_owned(
    Tensor source, const std::string& prefix, PadMode mode, bool causal,
    const std::vector<int>& stride, ComponentState* state,
    const std::string& cache_key) {
    const Tensor& kernel = weight(prefix + ".weight");
    require(kernel.ndim() == 5, "causal_conv3d weight rank mismatch");
    int64_t left_t = causal ? kernel.dim(2) - 1 : (kernel.dim(2) - 1) / 2;
    const int64_t right_t = causal ? 0 : (kernel.dim(2) - 1) / 2;
    if (state && !cache_key.empty()) {
        const auto found = state->temporal_cache.find(cache_key);
        if (found != state->temporal_cache.end()) {
            left_t = std::max<int64_t>(0, left_t - found->second.dim(2));
            source = backend_.concat({found->second, source}, 2);
        }
        const int64_t keep = std::min<int64_t>(kernel.dim(2) - 1, source.dim(2));
        if (keep > 0)
            state->temporal_cache[cache_key] = backend_.slice(
                source, 2, source.dim(2) - keep, source.dim(2));
    }
    Tensor padded = backend_.pad(source, {kernel.dim(4) / 2, kernel.dim(4) / 2,
                                     kernel.dim(3) / 2, kernel.dim(3) / 2,
                                     left_t, right_t}, 0.0f, mode);
    // pad has consumed source on the same backend stream. Dropping this handle
    // is safe without a host sync: an unaliased allocation becomes available
    // only to later enqueues; aliases/fan-out retain their shared owner.
    source = Tensor{};
    return conv3d(padded, prefix, {0, 0, 0}, stride);
}

Tensor ComponentExecutor::conv2d_frames(const Tensor& x, const std::string& prefix) {
    require(x.ndim() == 5, "conv2d_frames expects BCTHW");
    const int64_t b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4);
    Tensor frames = backend_.reshape(backend_.permute(x, {0, 2, 1, 3, 4}), {b * t, c, h, w});
    const Tensor* bias = weights_.find(prefix + ".bias");
    Tensor y = backend_.conv2d(frames, weight(prefix + ".weight"), bias,
                               {1, 1}, {1, 1});
    return backend_.permute(backend_.reshape(y, {b, t, y.dim(1), y.dim(2), y.dim(3)}),
                            {0, 2, 1, 3, 4});
}

Tensor ComponentExecutor::group_norm(const Tensor& x, const std::string& prefix,
                                     int groups, float eps) {
    return backend_.group_norm(x, groups, weights_.find(prefix + ".weight"),
                               weights_.find(prefix + ".bias"), eps);
}

Tensor ComponentExecutor::rms_channel(const Tensor& x, const std::string& prefix,
                                      float eps) {
    return backend_.rms_norm(x, &weight(prefix + ".gamma"), eps, 1);
}

Tensor ComponentExecutor::pixel_norm(const Tensor& x, float eps) {
    return backend_.pixel_norm(x, 1, eps);
}
Tensor ComponentExecutor::silu(const Tensor& x) { return backend_.activation(x, Activation::Silu); }
Tensor ComponentExecutor::linear(const Tensor& x, const std::string& prefix) {
    return backend_.linear(x, weight(prefix + ".weight"), weights_.find(prefix + ".bias"));
}
Tensor ComponentExecutor::timestep_embedding(const Tensor& timestep, const std::string& prefix) {
    Tensor embedded = backend_.sinusoidal_embedding(backend_.reshape(timestep, {-1}), 256,
                                                     true, 0.0, false);
    embedded = linear(embedded, prefix + ".timestep_embedder.linear_1");
    return linear(silu(embedded), prefix + ".timestep_embedder.linear_2");
}
Tensor ComponentExecutor::nearest(const Tensor& x, const std::vector<double>& factors) {
    return backend_.interpolate_nearest(x, factors);
}
Tensor ComponentExecutor::pixel_shuffle(const Tensor& x, const std::vector<int64_t>& factors) {
    return backend_.pixel_shuffle_nd(x, factors);
}

Tensor ComponentExecutor::spatial_attention_conv(const Tensor& input,
                                                 const std::string& prefix) {
    require(input.ndim() == 5, "spatial attention expects BCTHW");
    const int64_t b = input.dim(0), c = input.dim(1), t = input.dim(2),
                  h = input.dim(3), w = input.dim(4);
    Tensor x = backend_.reshape(backend_.permute(input, {0, 2, 1, 3, 4}), {b * t, c, h, w});
    x = rms_channel(x, prefix + ".norm");
    Tensor qkv = backend_.conv2d(x, weight(prefix + ".to_qkv.weight"),
                                 &weight(prefix + ".to_qkv.bias"), {1, 1}, {0, 0});
    qkv = backend_.permute(backend_.reshape(qkv, {b * t, 1, 3 * c, h * w}), {0, 1, 3, 2});
    auto parts = backend_.split(qkv, {c, c, c}, -1);
    for (Tensor& part : parts) part = backend_.permute(part, {0, 2, 1, 3});
    Tensor y = backend_.attention(parts[0], parts[1], parts[2]);
    y = backend_.reshape(backend_.permute(backend_.reshape(y, {b * t, h * w, c}), {0, 2, 1}),
                         {b * t, c, h, w});
    y = backend_.conv2d(y, weight(prefix + ".proj.weight"), &weight(prefix + ".proj.bias"),
                        {1, 1}, {0, 0});
    y = backend_.permute(backend_.reshape(y, {b, t, c, h, w}), {0, 2, 1, 3, 4});
    return backend_.add(y, input);
}

Tensor ComponentExecutor::sequence_attention_linear(const Tensor& input,
                                                     const std::string& prefix,
                                                     bool causal_frames) {
    require(input.ndim() == 5, "sequence attention expects BCTHW");
    const int64_t b = input.dim(0), c = input.dim(1), t = input.dim(2),
                  h = input.dim(3), w = input.dim(4), tokens = t * h * w;
    Tensor x = group_norm(input, prefix + ".group_norm");
    Tensor seq = backend_.permute(backend_.reshape(x, {b, c, tokens}), {0, 2, 1});
    Tensor q = backend_.reshape(linear(seq, prefix + ".to_q"), {b, tokens, 1, c});
    Tensor k = backend_.reshape(linear(seq, prefix + ".to_k"), {b, tokens, 1, c});
    Tensor v = backend_.reshape(linear(seq, prefix + ".to_v"), {b, tokens, 1, c});
    Tensor y = backend_.attention(q, k, v, nullptr, causal_frames);
    y = linear(backend_.reshape(y, {b, tokens, c}), prefix + ".to_out.0");
    y = backend_.reshape(backend_.permute(y, {0, 2, 1}), {b, c, t, h, w});
    return backend_.add(y, input);
}

Tensor ComponentExecutor::spatial_noise(const Tensor& x, const Tensor& scale,
                                        ComponentState& state) {
    Tensor noise = backend_.rng_normal(state.rng, {x.dim(-2), x.dim(-1)}, DType::F32);
    noise = backend_.reshape(noise, {1, 1, 1, x.dim(-2), x.dim(-1)});
    return backend_.add(x, backend_.mul(noise,
        backend_.reshape(scale, {1, scale.numel(), 1, 1, 1})));
}

Tensor ComponentExecutor::video_range(const Tensor& x, float source_min,
                                      float source_max, float target_min,
                                      float target_max) {
    Tensor shifted = backend_.add(x, scalar_f32(-source_min));
    shifted = backend_.mul(shifted, scalar_f32((target_max - target_min) /
                                               (source_max - source_min)));
    return backend_.clamp(backend_.add(shifted, scalar_f32(target_min)),
                          target_min, target_max);
}
Tensor ComponentExecutor::clamp(const Tensor& x, float low, float high) {
    return backend_.clamp(x, low, high);
}

}  // namespace vrhino
