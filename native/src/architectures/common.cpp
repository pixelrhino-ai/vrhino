#include "vrhino/architecture.h"

#include <bit>
#include <cmath>
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {

const Tensor& WeightMap::at(const std::string& name) const {
    const auto found = values_.find(prefix_ + name);
    require(found != values_.end(), "Missing architecture weight: " + prefix_ + name);
    return *found->second;
}
const Tensor* WeightMap::find(const std::string& name) const {
    const auto found = values_.find(prefix_ + name);
    return found == values_.end() ? nullptr : found->second;
}
Tensor operation_linear(Backend& backend, const PrecisionPolicy& policy,
                        PrecisionOperation operation, PrecisionSemantic output_semantic,
                        const Tensor& input, const Tensor& weight, const Tensor* bias) {
    const PrecisionOperationContract contract =
        policy.operation_contract(operation, output_semantic);
    const Tensor operand = contract.cast_boundary == PrecisionCastBoundary::ConsumerOperand
        ? backend.cast(input, contract.operand_dtype) : input;
    return backend.linear(operand, weight, bias,
                          contract.compute_dtype, contract.output_dtype);
}
Tensor operation_conv2d(Backend& backend, const PrecisionPolicy& policy,
                        PrecisionSemantic output_semantic,
                        const Tensor& input, const Tensor& weight,
                        const Tensor* bias, const std::vector<int>& stride,
                        const std::vector<int>& padding,
                        const std::vector<int>& dilation, int groups) {
    const PrecisionOperationContract contract = policy.operation_contract(
        PrecisionOperation::Convolution, output_semantic);
    const Tensor operand =
        contract.cast_boundary == PrecisionCastBoundary::ConsumerOperand
        ? backend.cast(input, contract.operand_dtype) : input;
    Tensor output = backend.conv2d(
        operand, weight, bias, stride, padding, dilation, groups);
    return output.dtype() == contract.output_dtype
        ? output : backend.cast(output, contract.output_dtype);
}
Tensor modulate(Backend& backend, const PrecisionPolicy& policy,
                const Tensor& x, const Tensor& shift, const Tensor& scale) {
    const PrecisionOperationContract contract = policy.operation_contract(
        PrecisionOperation::Modulation, PrecisionSemantic::TemporaryCompute);
    Tensor temporary = backend.add(
        backend.mul(backend.cast(x, contract.temporary_dtype),
                    backend.add(scalar_f32(1.0f),
                                backend.cast(scale, contract.temporary_dtype))),
        backend.cast(shift, contract.temporary_dtype));
    return backend.cast(temporary, contract.output_dtype);
}
Tensor attention_norm(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& input, const Tensor* weight,
                      float eps, int64_t axis) {
    const PrecisionOperationContract contract = policy.operation_contract(
        PrecisionOperation::Normalization, PrecisionSemantic::AttentionStatistic);
    return backend.rms_norm(input, weight, eps, axis, contract.output_dtype);
}
Tensor operation_rope(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& input, const Tensor& cosine,
                      const Tensor& sine) {
    const PrecisionOperationContract contract = policy.operation_contract(
        PrecisionOperation::Rope, PrecisionSemantic::TemporaryCompute);
    return backend.rope_nd(backend.cast(input, contract.operand_dtype),
                           cosine, sine);
}
Tensor operation_attention(Backend& backend, const PrecisionPolicy& policy,
                           const Tensor& query, const Tensor& key,
                           const Tensor& value, const Tensor* mask,
                           bool causal, float scale, const Tensor* bias) {
    const PrecisionOperationContract contract = policy.operation_contract(
        PrecisionOperation::Attention, PrecisionSemantic::TemporaryCompute);
    return backend.attention(backend.cast(query, contract.operand_dtype),
                             backend.cast(key, contract.operand_dtype),
                             backend.cast(value, contract.operand_dtype),
                             mask, causal, scale, bias);
}
Tensor boolean_self_attention_mask(const Tensor& validity,
                                   bool first_key_always_allowed) {
    require(validity.device() == DeviceId::host() &&
                validity.dtype() == DType::Bool && validity.ndim() == 2,
            "Attention validity mask must be host [B,S] Bool");
    const int64_t batch = validity.dim(0), tokens = validity.dim(1);
    require(batch > 0 && tokens > 0,
            "Attention validity mask dimensions must be positive");
    const uint8_t* source = validity.data_as<uint8_t>();
    const int64_t elements = shape_numel({batch, 1, tokens, tokens});
    std::vector<uint8_t> values(static_cast<size_t>(elements), 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t q = 0; q < tokens; ++q) {
            for (int64_t k = 0; k < tokens; ++k) {
                values[static_cast<size_t>((b * tokens + q) * tokens + k)] =
                    static_cast<uint8_t>(source[b * tokens + q] &&
                                         source[b * tokens + k]);
            }
            if (first_key_always_allowed)
                values[static_cast<size_t>((b * tokens + q) * tokens)] = 1;
        }
    }
    return host_bool({batch, 1, tokens, tokens}, values);
}
Tensor boolean_key_padding_mask(const Tensor& validity,
                                int64_t unmasked_prefix_tokens) {
    require(validity.device() == DeviceId::host() &&
                validity.dtype() == DType::Bool && validity.ndim() == 2,
            "Attention validity mask must be host [B,S] Bool");
    require(unmasked_prefix_tokens >= 0,
            "Attention key-padding prefix must be non-negative");
    const int64_t batch = validity.dim(0), tokens = validity.dim(1);
    require(batch > 0 && tokens > 0,
            "Attention validity mask dimensions must be positive");
    require(unmasked_prefix_tokens <= INT64_MAX - tokens,
            "Attention key-padding length overflow");
    const int64_t total = unmasked_prefix_tokens + tokens;
    const int64_t elements = shape_numel({batch, 1, total});
    std::vector<uint8_t> values(static_cast<size_t>(elements), 1);
    const uint8_t* source = validity.data_as<uint8_t>();
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t token = 0; token < tokens; ++token)
            values[static_cast<size_t>(b * total + unmasked_prefix_tokens + token)] =
                source[b * tokens + token];
    return host_bool({batch, 1, total}, values);
}
Tensor residual_add(Backend& backend, const PrecisionPolicy& policy,
                    const Tensor& residual, const Tensor& branch) {
    const DType dtype = policy.persistent_state_dtype(PrecisionSemantic::ResidualState);
    return backend.add(backend.cast(residual, dtype), backend.cast(branch, dtype));
}
Tensor gated_residual(Backend& backend, const PrecisionPolicy& policy,
                      const Tensor& residual, const Tensor& branch, const Tensor& gate) {
    const DType dtype = policy.persistent_state_dtype(PrecisionSemantic::ResidualState);
    return backend.add(backend.cast(residual, dtype),
                       backend.mul(backend.cast(branch, dtype), backend.cast(gate, dtype)));
}
Tensor producer_linear(Backend& backend, const PrecisionPolicy& policy,
                       PrecisionSemantic output_semantic, const Tensor& input,
                       const Tensor& weight, const Tensor* bias) {
    return operation_linear(backend, policy, PrecisionOperation::Linear,
                            output_semantic, input, weight, bias);
}
Tensor split_heads(Backend& backend, const Tensor& x, int heads) {
    require(x.ndim() == 3 && x.dim(2) % heads == 0, "split_heads shape mismatch");
    return backend.reshape(x, {x.dim(0), x.dim(1), heads, x.dim(2) / heads});
}
Tensor merge_heads(Backend& backend, const Tensor& x) {
    require(x.ndim() == 4, "merge_heads shape mismatch");
    return backend.reshape(x, {x.dim(0), x.dim(1), x.dim(2) * x.dim(3)});
}

std::pair<Tensor, Tensor> standard_rope(Backend& backend,
                                       const PrecisionPolicy& policy,
                                       const std::vector<std::vector<float>>& coordinates,
                                       const std::vector<int>& axis_dims, double theta,
                                       bool frequency_f64) {
    require(!coordinates.empty() && coordinates[0].size() == axis_dims.size(), "standard RoPE coordinate mismatch");
    int width = 0;
    for (int dimension : axis_dims) { require(dimension % 2 == 0, "RoPE axis width odd"); width += dimension; }
    std::vector<float> cosine, sine; cosine.reserve(coordinates.size() * width); sine.reserve(coordinates.size() * width);
    for (const auto& coordinate : coordinates) {
        for (size_t axis = 0; axis < axis_dims.size(); ++axis) {
            for (int pair = 0; pair < axis_dims[axis] / 2; ++pair) {
                double inverse;
                if (frequency_f64) inverse = 1.0 / std::pow(theta, static_cast<double>(2 * pair) / axis_dims[axis]);
                else inverse = 1.0f / std::pow(static_cast<float>(theta), static_cast<float>(2 * pair) / axis_dims[axis]);
                const double phase = coordinate[axis] * inverse;
                const float c = static_cast<float>(std::cos(phase)), s = static_cast<float>(std::sin(phase));
                cosine.push_back(c); cosine.push_back(c); sine.push_back(s); sine.push_back(s);
            }
        }
    }
    Tensor c = host_f32({1, static_cast<int64_t>(coordinates.size()), 1, width}, cosine);
    Tensor s = host_f32({1, static_cast<int64_t>(coordinates.size()), 1, width}, sine);
    const DType dtype = policy.operation_output_dtype(
        PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute);
    return {backend.copy_to_device(c, dtype), backend.copy_to_device(s, dtype)};
}

std::pair<Tensor, Tensor> fractional_rope(Backend& backend,
                                         const PrecisionPolicy& policy,
                                         const Tensor& coordinates,
                                         const std::vector<float>& max_positions,
                                         int width, float theta, bool centered) {
    require(coordinates.device() == Device::CPU && coordinates.dtype() == DType::F32 && coordinates.ndim() == 3,
            "fractional RoPE expects CPU B-Axes-Tokens coordinates");
    const int batch = coordinates.dim(0), axes = coordinates.dim(1), tokens = coordinates.dim(2);
    require(axes == static_cast<int>(max_positions.size()), "fractional RoPE axes mismatch");
    const int pairs_per_axis = width / (2 * axes), used = pairs_per_axis * axes * 2, remainder = width - used;
    std::vector<float> cosine(static_cast<size_t>(batch) * tokens * width, 1.0f), sine(cosine.size(), 0.0f);
    const float* values = coordinates.data_as<float>();
    for (int b = 0; b < batch; ++b) for (int token = 0; token < tokens; ++token) {
        int output = remainder;
        for (int pair = 0; pair < pairs_per_axis; ++pair) {
            for (int axis = 0; axis < axes; ++axis) {
                float fraction = values[(b * axes + axis) * tokens + token] / max_positions[axis];
                if (centered) fraction = fraction * 2.0f - 1.0f;
                const float exponent = static_cast<float>(pair) / (pairs_per_axis - 1);
                const float index = std::pow(theta, exponent) * static_cast<float>(M_PI_2);
                const float phase = index * fraction, c = std::cos(phase), s = std::sin(phase);
                const size_t base = (static_cast<size_t>(b) * tokens + token) * width + output;
                cosine[base] = c; cosine[base + 1] = c; sine[base] = s; sine[base + 1] = s; output += 2;
            }
        }
    }
    const DType dtype = policy.operation_output_dtype(
        PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute);
    return {backend.copy_to_device(host_f32({batch, tokens, width}, cosine), dtype),
            backend.copy_to_device(host_f32({batch, tokens, width}, sine), dtype)};
}

PreparedTensorHandle prepare_fractional_rope(
        PreparedTensorCache& cache, Backend& backend,
        const PrecisionPolicy& policy, const Tensor& coordinates,
        const std::vector<float>& max_positions, int width, float theta,
        bool centered) {
    require(coordinates.device().is_host(),
            "Prepared fractional RoPE coordinates must be host-resident");
    const DType dtype = policy.operation_output_dtype(
        PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute);
    PreparedTensorKey key;
    key.operation = "positional.fractional_rope.v1";
    key.input_shape = coordinates.shape();
    key.input_strides = coordinates.strides();
    key.input_dtype = coordinates.dtype();
    key.input_device_type = coordinates.device().type;
    key.input_device_index = coordinates.device().index;
    key.input_identity = reinterpret_cast<uintptr_t>(coordinates.data());
    key.input_content_hash = hash_host_tensor_content(coordinates);
    key.parameter_words.push_back(static_cast<uint64_t>(width));
    key.parameter_words.push_back(static_cast<uint64_t>(centered));
    key.parameter_words.push_back(std::bit_cast<uint32_t>(theta));
    key.parameter_words.push_back(max_positions.size());
    for (float value : max_positions)
        key.parameter_words.push_back(std::bit_cast<uint32_t>(value));
    key.target_dtype = dtype;
    key.target_backend = backend.name();
    key.target_device_index = 0;
    return cache.prepare(key, [&] {
        auto [cosine, sine] = fractional_rope(
            backend, policy, coordinates, max_positions, width, theta,
            centered);
        const uint64_t transfer_bytes =
            static_cast<uint64_t>(coordinates.dim(0)) *
            static_cast<uint64_t>(coordinates.dim(2)) *
            static_cast<uint64_t>(width) * sizeof(float) * 2;
        return PreparedTensorMaterialization{
            {std::move(cosine), std::move(sine)}, 1, 2, transfer_bytes};
    });
}

}  // namespace vrhino
