#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if VRHINO_USE_METAL_BACKEND
#include "vrhino/backend/metal_backend.h"
#else
#include "vrhino/backend/cuda_backend.h"
#endif
#include "vrhino/error.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"

namespace {

#if VRHINO_USE_METAL_BACKEND
using TestBackend = vrhino::MetalBackend;
#else
using TestBackend = vrhino::CudaBackend;
#endif

using vrhino::Tensor;

Tensor host(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    return vrhino::host_f32(shape, values);
}

Tensor host_i64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) {
    Tensor output = Tensor::host(shape, vrhino::DType::I64);
    vrhino::require(output.numel() == static_cast<int64_t>(values.size()), "i64 test tensor size mismatch");
    std::memcpy(output.data(), values.data(), values.size() * sizeof(int64_t));
    return output;
}

#if VRHINO_USE_METAL_BACKEND
Tensor host_bf16_exact(const std::vector<int64_t>& shape,
                       const std::vector<float>& values) {
    Tensor output = Tensor::host(shape, vrhino::DType::BF16);
    vrhino::require(output.numel() == static_cast<int64_t>(values.size()),
                    "bf16 test tensor size mismatch");
    auto* destination = output.data_as<uint16_t>();
    for (size_t index = 0; index < values.size(); ++index) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        vrhino::require((bits & 0xffffU) == 0,
                        "bf16 exact test value is not representable");
        destination[index] = static_cast<uint16_t>(bits >> 16);
    }
    return output;
}
#endif

std::vector<float> sequence(int64_t count, float start = -0.7f, float step = 0.13f) {
    std::vector<float> output(count);
    for (int64_t index = 0; index < count; ++index) output[index] = start + step * index;
    return output;
}

class Tests {
public:
    struct NumericalStat {
        double max_abs = 0.0;
        double sum_abs = 0.0;
        uint64_t elements = 0;
        float atol = 0.0f;
        float rtol = 0.0f;
    };
    TestBackend backend;
    std::map<std::string, int> cases;
    std::map<std::string, int> dtype_cases;
    std::map<std::string, NumericalStat> numerical;

    explicit Tests(vrhino::DType dtype) { backend.set_execution_dtype(dtype); }

    void record_dtype(const std::string& name, const Tensor& actual) {
        if (backend.execution_dtype() == vrhino::DType::BF16 && !name.starts_with("cast"))
            vrhino::require(actual.dtype() == vrhino::DType::BF16,
                            name + " did not preserve BF16 execution dtype");
        ++dtype_cases[name + "." + vrhino::dtype_name(actual.dtype())];
    }

    void close(const std::string& name, const Tensor& actual,
               const std::vector<float>& expected, float atol = 2e-5f,
               float rtol = 2e-5f) {
        record_dtype(name, actual);
        if (backend.execution_dtype() == vrhino::DType::BF16 && !name.starts_with("cast")) {
            atol = std::max(atol, 5e-2f); rtol = std::max(rtol, 5e-2f);
        }
        Tensor fp32 = actual.dtype() == vrhino::DType::F32 ? actual : backend.cast(actual, vrhino::DType::F32);
        Tensor value = fp32.device() == vrhino::Device::CPU ? fp32 : backend.copy_to_host(fp32);
        vrhino::require(value.numel() == static_cast<int64_t>(expected.size()), name + " output size mismatch");
        auto& stat = numerical[name];
        stat.atol = std::max(stat.atol, atol); stat.rtol = std::max(stat.rtol, rtol);
        for (int64_t index = 0; index < value.numel(); ++index) {
            const float difference = std::abs(value.data_as<float>()[index] - expected[index]);
            stat.max_abs = std::max(stat.max_abs, static_cast<double>(difference));
            stat.sum_abs += difference; ++stat.elements;
            vrhino::require(difference <= atol + rtol * std::abs(expected[index]),
                name + " numerical mismatch at " + std::to_string(index) +
                ": native=" + std::to_string(value.data_as<float>()[index]) +
                " reference=" + std::to_string(expected[index]));
        }
        ++cases[name];
    }

    void report_numerical() const {
        for (const auto& [name, stat] : numerical)
            std::cout << "numerical=" << name << ",max_abs=" << stat.max_abs
                      << ",mean_abs=" << stat.sum_abs / stat.elements
                      << ",allclose=true,atol=" << stat.atol << ",rtol=" << stat.rtol << "\n";
    }

    void shape(const std::string& name, const Tensor& actual,
               const std::vector<int64_t>& expected) {
        record_dtype(name, actual);
        vrhino::require(actual.shape() == expected, name + " shape mismatch"); ++cases[name];
    }

    void exact_repeat(const std::string& name, const Tensor& first,
                      const Tensor& second) {
        vrhino::require(first.shape() == second.shape() &&
                            first.dtype() == second.dtype(),
                        name + " repeat metadata mismatch");
        Tensor a = first.device() == vrhino::Device::CPU
            ? first : backend.copy_to_host(first);
        Tensor b = second.device() == vrhino::Device::CPU
            ? second : backend.copy_to_host(second);
        const size_t bytes = static_cast<size_t>(a.numel()) *
                             vrhino::dtype_size(a.dtype());
        vrhino::require(std::memcmp(a.data(), b.data(), bytes) == 0,
                        name + " repeat was not bit-exact");
        ++cases[name];
    }
};

std::vector<float> layer_norm_ref(const std::vector<float>& x, int width, float eps,
                                  const std::vector<float>* weight = nullptr,
                                  const std::vector<float>* bias = nullptr) {
    std::vector<float> out(x.size());
    for (size_t row = 0; row < x.size() / width; ++row) {
        double sum = 0, square = 0;
        for (int c = 0; c < width; ++c) {
            float v = x[row * width + c];
            sum += v;
            square += static_cast<double>(v) * v;
        }
        float mean = sum / width;
        float inv = 1.0f / std::sqrt(static_cast<float>(
            square / width - static_cast<double>(mean) * mean) + eps);
        for (int c = 0; c < width; ++c) {
            float v = (x[row * width + c] - mean) * inv;
            if (weight) v *= (*weight)[c]; if (bias) v += (*bias)[c]; out[row * width + c] = v;
        }
    }
    return out;
}

std::vector<float> rms_ref(const std::vector<float>& x, int rows, int width, float eps,
                           const std::vector<float>* weight = nullptr) {
    std::vector<float> out(x.size());
    for (int row = 0; row < rows; ++row) {
        double square = 0;
        for (int c = 0; c < width; ++c)
            square += static_cast<double>(x[row * width + c]) * x[row * width + c];
        float inv = 1.0f / std::sqrt(square / width + eps);
        for (int c = 0; c < width; ++c) out[row * width + c] = x[row * width + c] * inv * (weight ? (*weight)[c] : 1.0f);
    }
    return out;
}

std::vector<float> rms_axis_ref(const std::vector<float>& x,
                                const std::vector<int64_t>& shape, int axis,
                                float eps, const std::vector<float>* weight) {
    const int64_t width = shape[axis];
    int64_t inner = 1;
    for (size_t dim = static_cast<size_t>(axis + 1); dim < shape.size(); ++dim)
        inner *= shape[dim];
    const int64_t outer = static_cast<int64_t>(x.size()) / (width * inner);
    std::vector<float> out(x.size());
    for (int64_t o = 0; o < outer; ++o) for (int64_t i = 0; i < inner; ++i) {
        double square = 0.0;
        for (int64_t c = 0; c < width; ++c) {
            const float value = x[(o * width + c) * inner + i];
            square += static_cast<double>(value) * value;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(square / width) + eps);
        for (int64_t c = 0; c < width; ++c) {
            const int64_t index = (o * width + c) * inner + i;
            out[index] = x[index] * inv * (weight ? (*weight)[c] : 1.0f);
        }
    }
    return out;
}

std::vector<float> attention_ref(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, int batch, int qs, int ks,
                                 int heads, int width,
                                 const std::vector<uint8_t>* mask = nullptr,
                                 bool causal = false, float requested_scale = 0.0f) {
    std::vector<float> out(static_cast<size_t>(batch) * qs * heads * width);
    const float scale = requested_scale == 0.0f
        ? 1.0f / std::sqrt(static_cast<float>(width)) : requested_scale;
    for (int b = 0; b < batch; ++b) for (int i = 0; i < qs; ++i) for (int h = 0; h < heads; ++h) {
        std::vector<float> score(ks); float maximum = -INFINITY;
        for (int j = 0; j < ks; ++j) {
            const bool valid = (!causal || j <= i) &&
                (!mask || (*mask)[(b * qs + i) * ks + j]);
            float s = -INFINITY;
            if (valid) {
                s = 0; for (int d = 0; d < width; ++d)
                    s += q[((b * qs + i) * heads + h) * width + d] * k[((b * ks + j) * heads + h) * width + d];
                s *= scale;
            }
            score[j] = s; maximum = std::max(maximum, score[j]);
        }
        float denominator = 0; for (float& s : score) { s = std::exp(s - maximum); denominator += s; }
        for (int d = 0; d < width; ++d) for (int j = 0; j < ks; ++j)
            out[((b * qs + i) * heads + h) * width + d] += score[j] / denominator * v[((b * ks + j) * heads + h) * width + d];
    }
    return out;
}

void tensor_primitives(Tests& t) {
    for (auto shape : {std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 3, 2}}) {
        auto x_values = sequence(vrhino::shape_numel(shape)); Tensor x = host(shape, x_values);
        Tensor w = host({3, 2}, {1, 0, 0, 1, 1, 1});
        std::vector<float> expected; for (size_t i = 0; i < x_values.size(); i += 2)
            expected.insert(expected.end(), {x_values[i], x_values[i + 1], x_values[i] + x_values[i + 1]});
        t.close("linear", t.backend.linear(x, w), expected);
    }
    {
        const auto x_values = sequence(32, -0.4f, 0.025f);
        const auto w_values = sequence(256, -0.2f, 0.002f);
        std::vector<float> expected(32, 0.0f);
        for (int row = 0; row < 2; ++row)
            for (int column = 0; column < 16; ++column)
                for (int inner = 0; inner < 16; ++inner)
                    expected[row * 16 + column] +=
                        x_values[row * 16 + inner] * w_values[column * 16 + inner];
        t.close("linear", t.backend.linear(host({2, 16}, x_values), host({16, 16}, w_values)),
                expected, 2e-5f, 2e-5f);
    }
#if !VRHINO_USE_METAL_BACKEND
    {
        // Both paths use the same exactly representable BF16 operands and
        // FP32 GEMM accumulation. The direct path writes the accumulator to
        // FP32; the widened path first rounds it to BF16 and cannot recover
        // the discarded low bits.
        const Tensor input = host({1, 2}, {1.0f, 1.0f});
        const Tensor weight = host({1, 2}, {1.0f, 0.00390625f});
        const Tensor direct = t.backend.linear(
            input, weight, nullptr, vrhino::DType::BF16, vrhino::DType::F32);
        const Tensor rounded = t.backend.linear(
            input, weight, nullptr, vrhino::DType::BF16, vrhino::DType::BF16);
        const Tensor widened = t.backend.cast(rounded, vrhino::DType::F32);
        const Tensor direct_host = t.backend.copy_to_host(direct);
        const Tensor widened_host = t.backend.copy_to_host(widened);
        vrhino::require(direct.dtype() == vrhino::DType::F32 &&
                        rounded.dtype() == vrhino::DType::BF16 &&
                        widened.dtype() == vrhino::DType::F32,
                        "Linear producer-output dtype contract mismatch");
        vrhino::require(std::abs(direct_host.data_as<float>()[0] - 1.00390625f) < 1e-7f,
                        "Direct FP32 producer output lost its accumulator result");
        vrhino::require(direct_host.data_as<float>()[0] !=
                        widened_host.data_as<float>()[0],
                        "Direct FP32 producer output is indistinguishable from BF16 widening");
        ++t.cases["linear.producer_output_rounding"];
    }
#endif
#if VRHINO_USE_METAL_BACKEND
    {
        // Covers the production-size MPSMatrixMultiplication dispatch with a
        // model-neutral, cancellation-heavy, contiguous [M,K] x [N,K]^T case.
        constexpr int64_t m = 512, n = 4096, k = 4096;
        std::vector<float> base_input(k), input(m * k), weight(n * k);
        for (int64_t inner = 0; inner < k; ++inner)
            base_input[inner] = static_cast<float>((inner * 17) % 257 - 128) / 256.0f;
        for (int64_t row = 0; row < m; ++row) {
            const float scale = (row % 4 == 0) ? 1.0f :
                                (row % 4 == 1) ? -1.0f :
                                (row % 4 == 2) ? 0.5f : -0.5f;
            for (int64_t inner = 0; inner < k; ++inner)
                input[row * k + inner] = scale * base_input[inner];
        }
        std::vector<double> column_dot(n, 0.0);
        for (int64_t column = 0; column < n; ++column) {
            for (int64_t inner = 0; inner < k; ++inner) {
                const float value = static_cast<float>(
                    (inner * 13 + column * 7) % 251 - 125) / 512.0f;
                weight[column * k + inner] = value;
                column_dot[column] += static_cast<double>(base_input[inner]) * value;
            }
        }
        std::vector<float> expected(m * n);
        for (int64_t row = 0; row < m; ++row) {
            const double scale = (row % 4 == 0) ? 1.0 :
                                 (row % 4 == 1) ? -1.0 :
                                 (row % 4 == 2) ? 0.5 : -0.5;
            for (int64_t column = 0; column < n; ++column)
                expected[row * n + column] = static_cast<float>(scale * column_dot[column]);
        }
        t.close("linear.mps_large_cancellation",
                t.backend.linear(host({m, k}, input), host({n, k}, weight)),
                expected);
    }
#endif
#if !VRHINO_USE_METAL_BACKEND
    const Tensor quant_input = host({2, 2}, {2, 3, -1, 4});
    const std::vector<float> quant_expected = {2, 3, 5, -1, 4, 3};
    auto check_quantized = [&](const std::string& name, std::vector<uint8_t> packed,
                               vrhino::QuantType kind, const std::string& layout) {
        Tensor weight = Tensor::borrowed(packed.data(), packed.size(), {3, 2}, vrhino::DType::U8);
        auto info = std::make_shared<vrhino::QuantizationInfo>();
        info->type = kind; info->logical_dtype = vrhino::DType::F32;
        info->compute_dtype = vrhino::DType::BF16; info->accumulation_dtype = vrhino::DType::F32;
        info->group_size = 2; info->axis = 1; info->symmetric = true;
        info->granularity = "per_channel"; info->zero_point_mode = "none";
        info->packing_layout = layout;
        info->scales = host({3, 1}, {1, 1, 1});
        weight.set_quantization(info);
        t.close(name, t.backend.linear(quant_input, weight), quant_expected, 1e-3f, 1e-3f);
    };
    check_quantized("quantized_linear.fp8", {0x38, 0, 0, 0x38, 0x38, 0x38},
                    vrhino::QuantType::FP8E4M3FN, "byte_e4m3fn");
    check_quantized("quantized_linear.int8", {1, 0, 0, 1, 1, 1},
                    vrhino::QuantType::INT8Symmetric, "byte_twos_complement");
    check_quantized("quantized_linear.int4", {0x01, 0x10, 0x11},
                    vrhino::QuantType::INT4Symmetric, "nibble_low_first_twos_complement");
#endif
    for (auto shape : {std::vector<int64_t>{2, 2}, std::vector<int64_t>{2, 1, 2}}) {
        auto a = sequence(vrhino::shape_numel(shape), 1, 1); Tensor x = host(shape, a), y = host({2}, {2, 4});
        std::vector<float> add(a.size()), mul(a.size()), div(a.size());
        for (size_t i = 0; i < a.size(); ++i) { add[i] = a[i] + (i % 2 ? 4 : 2); mul[i] = a[i] * (i % 2 ? 4 : 2); div[i] = a[i] / (i % 2 ? 4 : 2); }
        t.close("add", t.backend.add(x, y), add); t.close("mul", t.backend.mul(x, y), mul); t.close("div", t.backend.div(x, y), div);
    }
    Tensor base = host({2, 3, 2}, sequence(12, 0, 1));
    for (auto target : {std::vector<int64_t>{3, 4}, std::vector<int64_t>{2, 6}, std::vector<int64_t>{-1}}) {
        Tensor value = t.backend.reshape(base, target); t.close("reshape", value, sequence(12, 0, 1));
    }
    t.close("permute", t.backend.permute(host({2, 3}, {0,1,2,3,4,5}), {1,0}), {0,3,1,4,2,5});
    t.close("permute", t.backend.permute(host({1,2,2}, {0,1,2,3}), {2,0,1}), {0,2,1,3});
    t.close("permute", t.backend.permute(host({2,1,2,1}, {0,1,2,3}), {1,3,0,2}), {0,1,2,3});
    t.close("concat", t.backend.concat({host({1,2},{1,2}),host({1,1},{3})},1), {1,2,3});
    t.close("concat", t.backend.concat({host({1,1,2},{1,2}),host({2,1,2},{3,4,5,6})},0), {1,2,3,4,5,6});
    auto split1=t.backend.split(host({1,4},{1,2,3,4}),{1,3},1);t.close("split",split1[0],{1});t.close("split",split1[1],{2,3,4});
    auto split2=t.backend.split(host({3,1},{1,2,3}),{2,1},0);t.close("split",split2[0],{1,2});
    t.close("slice",t.backend.slice(host({2,4},{0,1,2,3,4,5,6,7}),1,1,3),{1,2,5,6});
    t.close("slice",t.backend.slice(host({3,2},{0,1,2,3,4,5}),0,1,3),{2,3,4,5});
    t.close("cast",t.backend.cast(t.backend.cast(host({4},{1.1f,-2.3f,0,8}),vrhino::DType::BF16),vrhino::DType::F32),{1.1015625f,-2.296875f,0,8},1e-6f);
    t.close("cast",t.backend.cast(host({2,2},{1,2,3,4}),vrhino::DType::F32),{1,2,3,4});
    {
        Tensor fp16 = Tensor::host({2}, vrhino::DType::F16);
        const uint16_t bits[2] = {0x3c00, 0xc000};
        std::memcpy(fp16.data(), bits, sizeof(bits));
        t.close("cast.f16_f32", t.backend.cast(fp16, vrhino::DType::F32), {1.0f, -2.0f});
    }
    t.close("indexed_gather", t.backend.indexed_gather(
        host({4,2},{10,11,20,21,30,31,40,41}), host_i64({2,2},{3,0,1,2})),
        {40,41,10,11,20,21,30,31});
    t.close("indexed_gather", t.backend.indexed_gather(
        host({2,3,2},{10,11,20,21,30,31,40,41,50,51,60,61}), host_i64({2,2},{2,0,1,2})),
        {30,31,10,11,50,51,60,61});

    for (auto shape : {std::vector<int64_t>{2,3},std::vector<int64_t>{1,2,4}}) {
        auto x=sequence(vrhino::shape_numel(shape)); int width=shape.back(); auto w=sequence(width,0.8f,0.1f), b=sequence(width,-0.2f,0.05f);
        Tensor wt=host({width},w),bt=host({width},b);
        t.close("layer_norm",t.backend.layer_norm(host(shape,x),&wt,&bt,1e-5f),layer_norm_ref(x,width,1e-5f,&w,&b),5e-5f);
        t.close("rms_norm",t.backend.rms_norm(host(shape,x),&wt,1e-6f,-1),rms_ref(x,x.size()/width,width,1e-6f,&w),5e-5f);
    }
#if !VRHINO_USE_METAL_BACKEND
    // Model-neutral parallel Norm coverage: tail, medium, and production-scale
    // normalized dimensions, including direct strided arbitrary-axis reduction.
    for (int64_t width : {int64_t{17}, int64_t{128}, int64_t{1536}}) {
        const std::vector<int64_t> shape = {3, width};
        std::vector<float> x(static_cast<size_t>(vrhino::shape_numel(shape)));
        for (size_t index = 0; index < x.size(); ++index)
            x[index] = std::sin(static_cast<float>(index) * 0.173f) +
                       0.1f * std::cos(static_cast<float>(index) * 0.037f);
        auto w = sequence(width, 0.75f, 0.0003f);
        auto b = sequence(width, -0.1f, 0.0002f);
        Tensor wt = host({width}, w), bt = host({width}, b);
        Tensor layer_first = t.backend.layer_norm(host(shape, x), &wt, &bt, 1e-5f);
        Tensor layer_second = t.backend.layer_norm(host(shape, x), &wt, &bt, 1e-5f);
        t.close("layer_norm.parallel_dimension." + std::to_string(width), layer_first,
                layer_norm_ref(x, width, 1e-5f, &w, &b), 5e-5f);
        t.exact_repeat("layer_norm.parallel_repeat." + std::to_string(width), layer_first, layer_second);
        Tensor rms_first = t.backend.rms_norm(host(shape, x), &wt, 1e-6f, -1);
        Tensor rms_second = t.backend.rms_norm(host(shape, x), &wt, 1e-6f, -1);
        t.close("rms_norm.parallel_dimension." + std::to_string(width), rms_first,
                rms_ref(x, 3, width, 1e-6f, &w), 5e-5f);
        t.exact_repeat("rms_norm.parallel_repeat." + std::to_string(width), rms_first, rms_second);
    }
    for (const auto& [shape, axis] : std::vector<std::pair<
             std::vector<int64_t>, int>>{
             {{2, 5, 3}, 1},
             {{2, 7, 4, 5}, 1},
             {{2, 3, 9, 5}, 2},
             {{1, 17, 2, 3, 5}, 1}}) {
        auto x = sequence(vrhino::shape_numel(shape), -0.5f, 0.004f);
        auto w = sequence(shape[axis], 0.9f, 0.003f);
        Tensor wt = host({shape[axis]}, w);
        Tensor first = t.backend.rms_norm(host(shape, x), &wt, 1e-6f, axis);
        Tensor second = t.backend.rms_norm(host(shape, x), &wt, 1e-6f, axis);
        const std::string suffix = std::to_string(shape.size()) + "d.axis" +
                                   std::to_string(axis) + ".width" +
                                   std::to_string(shape[axis]);
        t.close("rms_norm.arbitrary_axis_parallel." + suffix, first,
                rms_axis_ref(x, shape, axis, 1e-6f, &w), 5e-5f);
        t.exact_repeat("rms_norm.arbitrary_axis_repeat." + suffix,
                       first, second);
    }
#endif
#if VRHINO_USE_METAL_BACKEND
    {
        // Production-shape, model-neutral coverage for FP32 RMSNorm with an
        // exactly representable BF16 stored affine weight.
        constexpr int64_t rows = 512, width = 4096;
        std::vector<float> input(rows * width), weight(width);
        for (int64_t column = 0; column < width; ++column)
            weight[column] = 1.0f + static_cast<float>(column % 17 - 8) / 64.0f;
        for (int64_t row = 0; row < rows; ++row) {
            const float scale = 1.0f + static_cast<float>(row % 7) / 8.0f;
            for (int64_t column = 0; column < width; ++column)
                input[row * width + column] = scale *
                    static_cast<float>((column * 17 + row * 13) % 257 - 128) / 16.0f;
        }
        Tensor stored_weight = host_bf16_exact({width}, weight);
        t.close("rms_norm.fp32_bf16_affine_large",
                t.backend.rms_norm(host({1, rows, width}, input),
                                   &stored_weight, 1e-6f),
                rms_ref(input, rows, width, 1e-6f, &weight));
    }
#endif
    for (auto shape : {std::vector<int64_t>{5},std::vector<int64_t>{2,3}}) {
        auto x=sequence(vrhino::shape_numel(shape));
        for (auto kind : {vrhino::Activation::Silu,vrhino::Activation::Gelu,vrhino::Activation::GeluTanh,vrhino::Activation::Tanh,vrhino::Activation::Relu}) {
            std::vector<float> expected; for(float v:x) expected.push_back(kind==vrhino::Activation::Silu?v/(1+std::exp(-v)):kind==vrhino::Activation::Gelu?0.5f*v*(1+std::erf(v/std::sqrt(2.0f))):kind==vrhino::Activation::GeluTanh?0.5f*v*(1+std::tanh(0.7978845608f*(v+0.044715f*v*v*v))):kind==vrhino::Activation::Tanh?std::tanh(v):std::max(v,0.0f));
            t.close("activation",t.backend.activation(host(shape,x),kind),expected,2e-5f);
        }
    }
    for (auto positions : {std::vector<float>{0,1},std::vector<float>{-2,.5f,3}}) {
        int width=6,half=3;std::vector<float> expected;
        for(float p:positions){for(int i=0;i<half;++i)expected.push_back(std::cos(p*std::exp(-std::log(10000.0f)*i/half)));for(int i=0;i<half;++i)expected.push_back(std::sin(p*std::exp(-std::log(10000.0f)*i/half)));}
        t.close("sinusoidal_embedding",t.backend.sinusoidal_embedding(host({static_cast<int64_t>(positions.size())},positions),width,true,0,false),expected,2e-5f);
    }
    for (int tokens : {1,2,4}) {
        auto x=sequence(tokens*4), c=sequence(tokens*4,.7f,.01f), s=sequence(tokens*4,.2f,.015f);std::vector<float> expected(x.size());
        for(size_t i=0;i<x.size();++i){size_t pair=(i%4)^1,paired=i-i%4+pair;float rotated=(i&1)?x[paired]:-x[paired];expected[i]=x[i]*c[i]+rotated*s[i];}
        t.close("rope_nd",t.backend.rope_nd(host({1,tokens,1,4},x),host({1,tokens,1,4},c),host({1,tokens,1,4},s)),expected);
    }
    for (auto config : {std::vector<int>{1,2,2,1,2},std::vector<int>{2,3,2,2,1},std::vector<int>{1,3,3,2,2}}) {
        int b=config[0],qs=config[1],ks=config[2],heads=config[3],width=config[4];auto q=sequence(b*qs*heads*width),k=sequence(b*ks*heads*width,.1f,.07f),v=sequence(b*ks*heads*width,.3f,.09f);
        t.close("attention",t.backend.attention(host({b,qs,heads,width},q),host({b,ks,heads,width},k),host({b,ks,heads,width},v),nullptr,false,0),attention_ref(q,k,v,b,qs,ks,heads,width),5e-5f);
    }
    {
        const float e = std::exp(2.0f), denominator = 1.0f + e;
        Tensor bias = host({1,1,1,2},{0,2});
        t.close("attention.additive_bias", t.backend.attention(
            host({1,1,1,1},{0}), host({1,2,1,1},{0,0}), host({1,2,1,1},{1,3}),
            nullptr, false, 1.0f, &bias), {(1.0f + 3.0f * e) / denominator}, 5e-5f);
    }
    {
        constexpr int b = 1, qs = 4, ks = 5, heads = 2, width = 3;
        const auto q = sequence(b * qs * heads * width, -0.3f, 0.017f);
        const auto k = sequence(b * ks * heads * width, 0.2f, -0.011f);
        const auto v = sequence(b * ks * heads * width, -0.1f, 0.023f);
        t.close("attention.causal", t.backend.attention(
            host({b,qs,heads,width},q), host({b,ks,heads,width},k),
            host({b,ks,heads,width},v), nullptr, true, 0.0f),
            attention_ref(q,k,v,b,qs,ks,heads,width,nullptr,true), 5e-5f);
    }
    {
        constexpr int b = 1, qs = 3, ks = 7, heads = 1, width = 5;
        const auto q = sequence(b * qs * heads * width, -0.4f, 0.019f);
        const auto k = sequence(b * ks * heads * width, 0.3f, -0.013f);
        const auto v = sequence(b * ks * heads * width, -0.2f, 0.021f);
        const std::vector<uint8_t> mask = {
            1,1,0,1,0,0,1,
            0,1,1,0,1,0,1,
            1,0,0,1,1,1,0};
        const Tensor mask_tensor = vrhino::host_bool({b,qs,ks}, mask);
        t.close("attention.mask", t.backend.attention(
            host({b,qs,heads,width},q), host({b,ks,heads,width},k),
            host({b,ks,heads,width},v), &mask_tensor, false, 0.0f),
            attention_ref(q,k,v,b,qs,ks,heads,width,&mask,false), 5e-5f);
    }
    {
        // Exercises Q/K/head tails and the second output-dimension tile without
        // encoding any model or architecture shape.
        constexpr int b = 1, qs = 9, ks = 35, heads = 1, width = 129;
        const auto q = sequence(b * qs * heads * width, -0.08f, 0.00011f);
        const auto k = sequence(b * ks * heads * width, 0.06f, -0.00003f);
        const auto v = sequence(b * ks * heads * width, -0.04f, 0.00005f);
        t.close("attention.non_aligned_tails", t.backend.attention(
            host({b,qs,heads,width},q), host({b,ks,heads,width},k),
            host({b,ks,heads,width},v), nullptr, false, 0.0f),
            attention_ref(q,k,v,b,qs,ks,heads,width), 5e-5f);
    }

    for (auto shape : {std::vector<int64_t>{1,1,1,2,3},std::vector<int64_t>{1,1,2,2,2},std::vector<int64_t>{2,1,1,1,2}}) {
        auto x=sequence(vrhino::shape_numel(shape),1,1);Tensor weight=host({1,1,1,1,1},{2});
        Tensor bias=host({1},{.5f});auto expected=x;for(float&v:expected)v=2*v+.5f;t.close("conv3d",t.backend.conv3d(host(shape,x),weight,&bias,{1,1,1},{0,0,0},{1,1,1},1),expected,5e-5f);
    }
    t.close("conv3d",t.backend.conv3d(host({1,2,1,1,1},{2,3}),host({1,2,1,1,1},{4,5}),nullptr,{1,1,1},{0,0,0},{1,1,1},1),{23},5e-5f);
    t.close("pad",t.backend.pad(host({1,2},{1,2}),{1,1},0,vrhino::PadMode::Constant),{0,1,2,0});
    t.close("pad",t.backend.pad(host({1,1,2},{1,2}),{1,0,1,0},-1,vrhino::PadMode::Constant),{-1,-1,-1,-1,1,2});
    t.close("pad",t.backend.pad(host({1,2},{1,2}),{1,2},0,vrhino::PadMode::Replicate),{1,1,2,2,2});
    t.close("reduce_sum",t.backend.reduce_sum(host({2,3},{1,2,3,4,5,6}),1,false),{6,15});
    t.close("reduce_sum",t.backend.reduce_sum(host({2,2,2},{1,2,3,4,5,6,7,8}),0,false),{6,8,10,12});
    t.close("softmax",t.backend.softmax(host({1,3},{0,0,0}),1),{1.0f/3,1.0f/3,1.0f/3},2e-5f);
    t.close("softmax",t.backend.softmax(host({2,2},{80,81,-80,-79}),-1),{0.26894143f,0.7310586f,0.26894143f,0.7310586f},2e-5f);
    for(auto shape:{std::vector<int64_t>{1,1,2,3},std::vector<int64_t>{2,1,1,2}}){auto x=sequence(vrhino::shape_numel(shape),1,1),expected=x;Tensor weight=host({1,1,1,1},{3}),bias=host({1},{-1});for(float&v:expected)v=3*v-1;t.close("conv2d",t.backend.conv2d(host(shape,x),weight,&bias,{1,1},{0,0},{1,1},1),expected);}
    t.close("conv2d",t.backend.conv2d(host({1,2,1,1},{2,3}),host({1,2,1,1},{4,5}),nullptr,{1,1},{0,0},{1,1},1),{23},5e-5f);
    t.close("conv_transpose2d",t.backend.conv_transpose2d(
        host({1,1,2,2},{1,2,3,4}),host({1,1,2,2},{1,0,0,1}),nullptr,
        {2,2},{0,0},{0,0},{1,1},1),
        {1,0,2,0,0,1,0,2,3,0,4,0,0,3,0,4});
    Tensor transpose_bias=host({1},{0.5f});
    t.close("conv_transpose2d",t.backend.conv_transpose2d(
        host({1,1,1,2},{2,3}),host({1,1,1,1},{4}),&transpose_bias,
        {1,1},{0,0},{0,0},{1,1},1),{8.5f,12.5f});
    t.close("maximum",t.backend.maximum(host({3},{-1,4,2}),host({3},{0,3,2})),{0,4,2});
    t.close("maximum",t.backend.maximum(host({2,1},{1,5}),host({1,2},{2,4})),{2,4,5,5});
    t.close("max_pool2d",t.backend.max_pool2d(host({1,1,2,4},{1,7,2,4,3,0,9,5}),{2,2},{2,2},{0,0}),{7,9});
    t.close("max_pool2d",t.backend.max_pool2d(host({1,1,3,3},{1,2,3,4,5,6,7,8,9}),{2,2},{1,1},{0,0}),{5,6,8,9});
    for(auto shape:{std::vector<int64_t>{1,2,2},std::vector<int64_t>{2,4,1,2}}){auto x=sequence(vrhino::shape_numel(shape));Tensor actual=t.backend.group_norm(host(shape,x),shape[1],nullptr,nullptr,1e-5f);std::vector<float> expected(x.size());int spatial=vrhino::shape_numel(shape)/(shape[0]*shape[1]);for(int b=0;b<shape[0];++b)for(int c=0;c<shape[1];++c){double sum=0,sq=0;for(int i=0;i<spatial;++i){float v=x[(b*shape[1]+c)*spatial+i];sum+=v;sq+=v*v;}float mean=sum/spatial,inv=1/std::sqrt(sq/spatial-mean*mean+1e-5f);for(int i=0;i<spatial;++i)expected[(b*shape[1]+c)*spatial+i]=(x[(b*shape[1]+c)*spatial+i]-mean)*inv;}t.close("group_norm",actual,expected,5e-5f);}
    t.close("interpolate_nearest",t.backend.interpolate_nearest(host({1,1,2},{1,2}),{2}),{1,1,2,2});
    t.close("interpolate_nearest",t.backend.interpolate_nearest(host({1,1,1,2},{1,2}),{2,2}),{1,1,2,2,1,1,2,2});
    t.close("interpolate_bilinear_2d.align_corners",t.backend.interpolate_bilinear_2d(
            host({1,1,2,2},{1,2,3,4}),3,3,true),
            {1,1.5f,2,2,2.5f,3,3,3.5f,4},5e-5f);
    for(auto shape:{std::vector<int64_t>{1,2,2},std::vector<int64_t>{1,3,1,2}}){auto x=sequence(vrhino::shape_numel(shape),.2f,.2f);std::vector<float> expected(x.size());int c=shape[1],spatial=x.size()/c;for(int p=0;p<spatial;++p){float sq=0;for(int j=0;j<c;++j)sq+=x[j*spatial+p]*x[j*spatial+p];for(int j=0;j<c;++j)expected[j*spatial+p]=x[j*spatial+p]/std::sqrt(sq/c+1e-8f);}t.close("pixel_norm",t.backend.pixel_norm(host(shape,x),1,1e-8f),expected);}
    for(auto shape:{std::vector<int64_t>{1,2,2},std::vector<int64_t>{1,3,1,2}}){auto x=sequence(vrhino::shape_numel(shape),.2f,.2f);std::vector<float> expected(x.size());int c=shape[1],spatial=x.size()/c;for(int p=0;p<spatial;++p){float sq=0;for(int j=0;j<c;++j)sq+=x[j*spatial+p]*x[j*spatial+p];for(int j=0;j<c;++j)expected[j*spatial+p]=x[j*spatial+p]/(std::sqrt(sq)+1e-10f);}t.close("l2_normalize",t.backend.l2_normalize(host(shape,x),1,1e-10f),expected);}
    t.close("pixel_shuffle_nd",t.backend.pixel_shuffle_nd(host({1,8,1,1,1},{0,1,2,3,4,5,6,7}),{2,2,2}),{0,1,2,3,4,5,6,7});
    t.close("pixel_shuffle_nd",t.backend.pixel_shuffle_nd(host({1,4,1,1,2},{0,1,2,3,4,5,6,7}),{1,2,2}),{0,2,1,3,4,6,5,7});
    vrhino::RngState rng{5701,0,"pytorch_compat.v1"};t.close("rng_normal",t.backend.rng_normal(rng,{8},t.backend.execution_dtype()),{-1.1955757141f,.5111286044f,-1.0140079260f,-.4857170284f,-.2519632876f,-1.9857326746f,1.1355758905f,.4264008999f},1e-6f);vrhino::require(rng.offset==4,"RNG offset mismatch");
    vrhino::RngState rng2{5701,0,"pytorch_compat.v1"};Tensor random512=t.backend.rng_normal(rng2,{2,256},t.backend.execution_dtype());t.shape("rng_normal",random512,{2,256});
    t.close("clamp",t.backend.clamp(host({5},{-2,-1,0,1,2}),-.5f,.75f),{-.5f,-.5f,0,.75f,.75f});
    t.close("clamp",t.backend.clamp(host({2,2},{-3,.2f,4,.8f}),0,1),{0,.2f,1,.8f});
    t.close("exp",t.backend.exp(host({4},{-1,0,1,2})),
            {std::exp(-1.0f),1.0f,std::exp(1.0f),std::exp(2.0f)},2e-5f);
    t.close("exp",t.backend.exp(host({2,2},{-2,-.5f,.5f,3})),
            {std::exp(-2.0f),std::exp(-.5f),std::exp(.5f),std::exp(3.0f)},2e-5f);
    t.close("sqrt",t.backend.sqrt(host({4},{0,1,4,9})),{0,1,2,3},2e-5f);
    t.close("sqrt",t.backend.sqrt(host({2,2},{.25f,2.25f,6.25f,12.25f})),
            {.5f,1.5f,2.5f,3.5f},2e-5f);
    t.close("batched_matmul",t.backend.batched_matmul(
            host({1,2,3},{1,2,3,4,5,6}),
            host({1,3,2},{1,2,3,4,5,6})),{22,28,49,64},5e-5f);
    t.close("batched_matmul",t.backend.batched_matmul(
            host({2,1,2},{1,2,3,4}),
            host({2,2,1},{5,6,7,8})),{17,53},5e-5f);
}

void sampling_primitives(Tests& t) {
    vrhino::SamplingPrimitives p(t.backend);
    for(int n:{3,5}){auto v=p.linear_schedule(1,0,n);vrhino::require(v.front()==1&&v.back()==0,"linear schedule mismatch");++t.cases["sampling.linear_schedule"];}
    for(auto input:{std::vector<float>{1,.5f,0},std::vector<float>{.9f,.2f}}){auto v=p.rational_time_shift(input,5);for(size_t i=0;i<v.size();++i)vrhino::require(std::abs(v[i]-5*input[i]/(1+4*input[i]))<1e-6f,"rational shift mismatch");++t.cases["sampling.rational_time_shift"];}
    for(int tokens:{4,2048}){auto v=p.resolution_time_shift({1,.5f},tokens);vrhino::require(v.size()==2&&v[0]==1,"resolution shift mismatch");++t.cases["sampling.resolution_time_shift"];}
    vrhino::require(p.schedule_lookup({1,.5f,0},.52f)==1,"lookup mismatch");++t.cases["sampling.schedule_lookup"];vrhino::require(p.schedule_lookup({3,2,1},2.9f)==0,"lookup mismatch");++t.cases["sampling.schedule_lookup"];
    for(auto shape:{std::vector<int64_t>{2},std::vector<int64_t>{2,2}}){int n=vrhino::shape_numel(shape);auto a=sequence(n,1,1),b=sequence(n,4,1);std::vector<float> expected(n);for(int i=0;i<n;++i)expected[i]=.25f*a[i]+.75f*b[i];t.close("sampling.linear_combine",p.linear_combine({host(shape,a),host(shape,b)},{.25f,.75f}),expected);t.close("sampling.cfg_combine",p.cfg_combine(host(shape,a),host(shape,b),2),[&]{auto e=a;for(int i=0;i<n;++i)e[i]=a[i]+2*(b[i]-a[i]);return e;}());t.close("sampling.euler_update",p.euler_update(host(shape,a),host(shape,b),vrhino::scalar_f32(.2f),true),[&]{auto e=a;for(int i=0;i<n;++i)e[i]=a[i]-.2f*b[i];return e;}());t.close("sampling.flow_to_x0",p.flow_to_x0(host(shape,a),host(shape,b),.3f),[&]{auto e=a;for(int i=0;i<n;++i)e[i]=a[i]-.3f*b[i];return e;}());}
    for(auto shape:{std::vector<int64_t>{2},std::vector<int64_t>{2,2}}){int n=vrhino::shape_numel(shape);auto sample=sequence(n,1,.25f),v=sequence(n,.5f,.125f);std::vector<float>x0(n),next(n);for(int i=0;i<n;++i){x0[i]=.6f*sample[i]-.8f*v[i];next[i]=.25f*sample[i]+.75f*x0[i];}Tensor actual_x0=p.v_to_x0(host(shape,sample),host(shape,v),.36f);t.close("sampling.v_to_x0",actual_x0,x0);t.close("sampling.affine_first_order",p.affine_first_order(host(shape,sample),actual_x0,.25f,.75f),next);}
    for(auto shape:{std::vector<int64_t>{4},std::vector<int64_t>{2,2}}){int n=vrhino::shape_numel(shape);std::vector<uint8_t> mask(n);for(int i=0;i<n;++i)mask[i]=i%2;std::vector<float>a(n,2),b(n,5),e(n);for(int i=0;i<n;++i)e[i]=mask[i]?2:5;t.close("sampling.tensor_where",p.tensor_where(vrhino::host_bool(shape,mask),host(shape,a),host(shape,b)),e);}
    for(auto shape:{std::vector<int64_t>{2},std::vector<int64_t>{2,2}}){int n=vrhino::shape_numel(shape);auto m1=sequence(n,.2f,.1f),m0=sequence(n,.6f,.1f),sample=sequence(n,1,.2f);std::vector<float> sigmas={.9f,.7f,.4f,.0f};float st=.4f,ss=.7f,at=1-st,as=1-ss,h=std::log(at)-std::log(st)-(std::log(as)-std::log(ss)),hh=-h,phi1=std::expm1(hh),rk=(std::log(.1f)-std::log(.9f)-(std::log(as)-std::log(ss)))/h,bh=phi1;std::vector<float> expected(n);for(int i=0;i<n;++i){float base=st/ss*sample[i]-at*phi1*m0[i],diff=(m1[i]-m0[i])/rk;expected[i]=base-at*bh*.5f*diff;}t.close("sampling.multistep_predictor",p.multistep_predictor(host(shape,sample),{host(shape,m1),host(shape,m0)},host(shape,m0),sigmas,1,2),expected,1e-5f);Tensor corrected=p.multistep_corrector(host(shape,sample),host(shape,sample),{host(shape,m1)},host(shape,m0),sigmas,1,1,host(shape,sample));t.shape("sampling.multistep_corrector",corrected,shape);}
    vrhino::RngState rng{99,0,"pytorch_compat.v1"};t.shape("sampling.rng_normal",p.rng_normal(rng,{4},t.backend.execution_dtype()),{4});t.shape("sampling.rng_normal",p.rng_normal(rng,{2,2},t.backend.execution_dtype()),{2,2});
    p.state_advance();p.state_advance();t.cases["sampling.state_advance"]+=2;
}

class ZeroDenoiser final : public vrhino::Denoiser {
public:
    ZeroDenoiser(vrhino::Backend& backend, vrhino::DType raw_output_dtype)
        : backend_(backend), raw_output_dtype_(raw_output_dtype) {}
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor&) override {
        ++calls;
        return {backend_.cast(
            backend_.mul(latent, vrhino::scalar_f32(0.0f)), raw_output_dtype_)};
    }
    int calls = 0;
private:
    vrhino::Backend& backend_;
    vrhino::DType raw_output_dtype_;
};

void sampling_contract_v1(Tests& t) {
    auto expect_failure = [](auto&& operation, const std::string& message) {
        bool failed = false;
        try { operation(); } catch (const vrhino::Error&) { failed = true; }
        vrhino::require(failed, message);
    };
    expect_failure([] {
        (void)vrhino::ScheduleContract::flow_sigma({});
    }, "empty typed sampling schedule must fail");
    expect_failure([] {
        (void)vrhino::ScheduleContract::flow_sigma({
            {vrhino::scalar_i64(900), 0.9f, 0.6f},
            {vrhino::scalar_i64(600), 0.5f, 0.0f},
        });
    }, "non-contiguous typed sampling schedule must fail");
    expect_failure([] {
        (void)vrhino::ScheduleContract::alpha_cumprod({});
    }, "empty AlphaCumprod schedule must fail");
    expect_failure([] {
        (void)vrhino::ScheduleContract::alpha_cumprod({
            {vrhino::scalar_i64(900), 0.25f, 0.49f},
            {vrhino::scalar_i64(600), 0.50f, 1.00f},
        });
    }, "non-contiguous AlphaCumprod schedule must fail");
    expect_failure([] {
        (void)vrhino::ScheduleContract::alpha_cumprod({
            {vrhino::scalar_i64(900), 1.0f, 1.0f},
        });
    }, "out-of-range AlphaCumprod schedule must fail");

    std::vector<vrhino::FlowScheduleTransition> transitions = {
        {vrhino::scalar_i64(900), 0.9f, 0.6f},
        {vrhino::scalar_i64(600), 0.6f, 0.3f},
        {vrhino::scalar_i64(300), 0.3f, 0.0f},
    };
    vrhino::SamplingProgram program;
    program.latent_shape = {1, 1, 1, 2, 2};
    program.seed = 17001;
    program.steps = 3;
    program.guidance_mode = vrhino::GuidanceMode::Linear;
    program.guidance_coefficients = {1.0f};
    program.contract.emplace(
        vrhino::PredictionContract{vrhino::PredictionSemantic::Flow},
        vrhino::SolverContract{
            vrhino::SolverSemantic::MultistepPredictorCorrector, 2},
        vrhino::ScheduleContract::flow_sigma(std::move(transitions)));

    ZeroDenoiser denoiser(t.backend, t.backend.execution_dtype());
    vrhino::SamplingRuntime runtime(t.backend);
    vrhino::SamplingResult result = runtime.run(denoiser, program);
    vrhino::require(denoiser.calls == program.steps,
                    "Sampling Contract v1 denoiser call mismatch");
    vrhino::require(result.final_latent.shape() == program.latent_shape,
                    "Sampling Contract v1 final state shape mismatch");
    vrhino::require(program.model_timesteps.empty() && program.sigmas.empty(),
                    "Sampling Contract v1 must not populate legacy schedule arrays");
    vrhino::require(program.model_timestep_at(0).data_as<int64_t>()[0] == 900,
                    "Sampling Contract v1 typed timestep mismatch");
    const auto& calls = runtime.primitives().calls();
    vrhino::require(calls.at("flow_to_x0") == 3,
                    "FLOW prediction contract call count mismatch");
    vrhino::require(calls.at("multistep_predictor") == 3 &&
                    calls.at("multistep_corrector") == 2,
                    "MULTISTEP_PREDICTOR_CORRECTOR call count mismatch");
    vrhino::require(result.trace.count("step.0.prediction.0") == 1 &&
                    result.trace.count("step.1.latent") == 1 &&
                    result.trace.count("final_latent") == 1,
                    "Sampling Contract v1 trace boundary mismatch");
    ++t.cases["sampling.contract_v1"];

    ZeroDenoiser cancelled_denoiser(t.backend, t.backend.execution_dtype());
    vrhino::SamplingRuntime cancelled_runtime(t.backend);
    int cancellation_checks = 0;
    cancelled_runtime.set_cancellation_requested([&] {
        return ++cancellation_checks >= 3;
    });
    expect_failure([&] {
        (void)cancelled_runtime.run(cancelled_denoiser, program);
    }, "sampling cancellation must propagate at an iteration boundary");
    vrhino::require(cancelled_denoiser.calls == 1,
                    "sampling cancellation crossed a denoiser boundary");
    ++t.cases["sampling.cancellation"];

    vrhino::SamplingProgram affine_program;
    affine_program.latent_shape = {1, 1, 1, 2, 2};
    affine_program.seed = 17002;
    affine_program.steps = 2;
    affine_program.guidance_mode = vrhino::GuidanceMode::Linear;
    affine_program.guidance_coefficients = {1.0f};
    affine_program.contract.emplace(
        vrhino::PredictionContract{vrhino::PredictionSemantic::V},
        vrhino::SolverContract{vrhino::SolverSemantic::AffineFirstOrder, 1},
        vrhino::ScheduleContract::alpha_cumprod({
            {vrhino::scalar_i64(900), 0.25f, 0.49f},
            {vrhino::scalar_i64(600), 0.49f, 1.00f},
        }));
    ZeroDenoiser affine_denoiser(t.backend, t.backend.execution_dtype());
    vrhino::SamplingRuntime affine_runtime(t.backend);
    const std::vector<float> affine_initial_values = {1.0f, -2.0f, 0.5f, 3.0f};
    const Tensor affine_initial = t.backend.copy_to_device(
        host(affine_program.latent_shape, affine_initial_values),
        t.backend.execution_dtype());
    vrhino::SamplingResult affine_result =
        affine_runtime.run_with_external_initial_state_for_test(
            affine_denoiser, affine_program, affine_initial);
    vrhino::require(affine_denoiser.calls == affine_program.steps,
                    "V/Affine denoiser call mismatch");
    vrhino::require(affine_result.final_latent.shape() == affine_program.latent_shape,
                    "V/Affine final state shape mismatch");
    vrhino::require(affine_program.model_timestep_at(1).data_as<int64_t>()[0] == 600,
                    "AlphaCumprod typed timestep mismatch");
    const auto& affine_calls = affine_runtime.primitives().calls();
    vrhino::require(affine_calls.at("v_to_x0") == 2 &&
                    affine_calls.at("affine_first_order") == 2,
                    "V/Affine semantic call count mismatch");
    float affine_scale = 1.0f;
    for (const auto& [alpha, previous_alpha] :
            std::vector<std::pair<float, float>>{{0.25f, 0.49f}, {0.49f, 1.0f}}) {
        const float state_coefficient =
            std::sqrt((1.0f - previous_alpha) / (1.0f - alpha));
        const float prediction_coefficient = std::sqrt(previous_alpha) -
            std::sqrt(alpha) * state_coefficient;
        // ZeroDenoiser supplies v=0, so x0=sqrt(alpha)*state.
        affine_scale *= state_coefficient +
            prediction_coefficient * std::sqrt(alpha);
    }
    std::vector<float> affine_expected = affine_initial_values;
    for (float& value : affine_expected) value *= affine_scale;
    t.close("sampling.contract_v1.affine_execution",
            affine_result.final_latent, affine_expected);
    ++t.cases["sampling.contract_v1"];

    const Tensor epsilon_sample = t.backend.copy_to_device(
        host({1, 2}, {1.0f, -2.0f}), t.backend.execution_dtype());
    const Tensor epsilon_prediction = t.backend.copy_to_device(
        host({1, 2}, {0.5f, -0.25f}), t.backend.execution_dtype());
    const float epsilon_noise = std::sqrt(0.75f);
    t.close("sampling.epsilon_to_x0",
        vrhino::SamplingPrimitives(t.backend).epsilon_to_x0(
            epsilon_sample, epsilon_prediction, 0.25f),
        {(1.0f - epsilon_noise * 0.5f) / 0.5f,
         (-2.0f + epsilon_noise * 0.25f) / 0.5f});
    ++t.cases["sampling.epsilon"];

    vrhino::ScheduleContract ddim = vrhino::make_scaled_linear_ddim_schedule(
        1000, 0.00085f, 0.012f, 20, 1, false);
    const std::array<int64_t, 20> expected_ddim = {
        951, 901, 851, 801, 751, 701, 651, 601, 551, 501,
        451, 401, 351, 301, 251, 201, 151, 101, 51, 1};
    vrhino::require(ddim.size() == expected_ddim.size(),
                    "DDIM timestep count mismatch");
    for (size_t index = 0; index < expected_ddim.size(); ++index)
        vrhino::require(ddim.model_timestep_at(index).data_as<int64_t>()[0] ==
                            expected_ddim[index],
                        "DDIM timestep generation mismatch");
    vrhino::require(std::abs(
                        ddim.alpha_cumprod_at(19).previous_alpha_cumprod -
                        (1.0f - 0.00085f)) <= 1.0e-6f,
                    "DDIM set_alpha_to_one=false final alpha mismatch");
    ++t.cases["sampling.epsilon"];

    vrhino::SamplingProgram mismatch_program;
    mismatch_program.latent_shape = {1, 1, 1, 2, 2};
    mismatch_program.seed = 17003;
    mismatch_program.steps = 2;
    mismatch_program.guidance_coefficients = {1.0f};
    mismatch_program.contract.emplace(
        vrhino::PredictionContract{vrhino::PredictionSemantic::Flow},
        vrhino::SolverContract{vrhino::SolverSemantic::AffineFirstOrder, 1},
        vrhino::ScheduleContract::alpha_cumprod({
            {vrhino::scalar_i64(900), 0.25f, 0.49f},
            {vrhino::scalar_i64(600), 0.49f, 1.00f},
        }));
    ZeroDenoiser mismatch_denoiser(t.backend, t.backend.execution_dtype());
    expect_failure([&] {
        (void)vrhino::SamplingRuntime(t.backend).run(
            mismatch_denoiser, mismatch_program);
    }, "mismatched Prediction/Solver/Schedule contract must fail");
}

void sampling_runtime_step_bounds(Tests& t) {
    for (int steps : {2, 3, 4, 5}) {
        vrhino::SamplingProgram program;
        program.latent_shape = {1, 1, 1, 2, 2};
        program.seed = 14001;
        program.steps = steps;
        program.guidance_mode = vrhino::GuidanceMode::Linear;
        program.guidance_coefficients = {1.0f};
        program.scheduler = vrhino::SchedulerKind::Euler;
        program.update_deltas.assign(static_cast<size_t>(steps), 0.0f);
        for (int index = 0; index < steps; ++index)
            program.model_timesteps.push_back(vrhino::scalar_i64(steps - index));
        const vrhino::DType raw_output_dtype = t.backend.execution_dtype() == vrhino::DType::F32
            ? vrhino::DType::BF16 : vrhino::DType::F32;
        ZeroDenoiser denoiser(t.backend, raw_output_dtype);
        vrhino::SamplingResult result = vrhino::SamplingRuntime(t.backend).run(denoiser, program);
        vrhino::require(denoiser.calls == steps, "generic sampling loop step mismatch");
        vrhino::require(result.final_latent.shape() == program.latent_shape,
                        "generic sampling loop output shape mismatch");
        for (const auto& [name, tensor] : result.trace)
            vrhino::require(tensor.device().is_host(),
                            "sampling trace must not retain device residency: " + name);
        vrhino::require(result.trace.at("step.0.prediction.0").dtype() ==
                            t.backend.execution_dtype(),
                        "First-prediction capture must follow the generic "
                        "DENOISER_OUTPUT boundary conversion");
        ++t.cases["sampling.runtime_steps"];
    }
    std::cout << "generic sampling runtime steps=2,3,4,5 pass\n";
}

}  // namespace

void run_suite(vrhino::DType dtype) {
        Tests tests(dtype); tensor_primitives(tests); sampling_primitives(tests);
        sampling_contract_v1(tests);
        sampling_runtime_step_bounds(tests);
        const std::vector<std::string> tensor_names={"linear","add","mul","reshape","permute","concat","split","slice","cast","indexed_gather","layer_norm","rms_norm","activation","sinusoidal_embedding","rope_nd","attention","conv3d","pad","reduce_sum","softmax","div","maximum","conv2d","conv_transpose2d","max_pool2d","group_norm","interpolate_nearest","pixel_norm","l2_normalize","pixel_shuffle_nd","rng_normal","clamp","exp","sqrt","batched_matmul"};
        for(const auto&name:tensor_names)vrhino::require(tests.cases[name]>=2,"primitive lacks two cases: "+name);
        for(const auto&name:{"reshape","permute","rope_nd","attention","conv3d","pad"})vrhino::require(tests.cases[name]>=3,"shape-sensitive primitive lacks extra case: "+std::string(name));
        const std::vector<std::string> sampling_names={"rng_normal","linear_schedule","rational_time_shift","resolution_time_shift","schedule_lookup","linear_combine","cfg_combine","euler_update","tensor_where","flow_to_x0","v_to_x0","affine_first_order","multistep_predictor","multistep_corrector","state_advance"};
        for(const auto&name:sampling_names)vrhino::require(tests.cases["sampling."+name]>=2,"sampling primitive lacks two cases: "+name);
        tests.backend.synchronize();
        tests.report_numerical();
        int tensor_total=0,sampling_total=0;for(const auto&[name,count]:tests.cases){if(name.rfind("sampling.",0)==0)sampling_total+=count;else tensor_total+=count;}
        std::cout<<"execution_dtype="<<vrhino::dtype_name(dtype)<<" tensor_cases="<<tensor_total<<" sampling_cases="<<sampling_total<<"\n";
}

int main() {
    try {
        run_suite(vrhino::DType::F32);
#if !VRHINO_USE_METAL_BACKEND
        run_suite(vrhino::DType::BF16);
        std::cout<<"native FP32/BF16 primitive numerical tests=pass\n";
#else
        std::cout<<"native Metal FP32 primitive numerical tests=pass\n";
#endif
        return 0;
    } catch(const std::exception&error){std::cerr<<"native primitive tests: "<<error.what()<<"\n";return 1;}
}
