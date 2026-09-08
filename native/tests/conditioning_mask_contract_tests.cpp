#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/tensor_util.h"

namespace {

using vrhino::Json;
using vrhino::Tensor;

Json graph(const std::string& semantic, bool trim_to_mask = false) {
    const std::string mask_field = semantic.empty()
        ? std::string()
        : ",\"mask_semantic\":\"" + semantic + "\"";
    const std::string trim_field = trim_to_mask
        ? ",\"output_trim_to_mask\":true"
        : std::string();
    return Json::parse(
        "{\"schema_version\":1,\"kind\":\"pre_norm_transformer\","
        "\"embedding\":{\"weight\":\"embedding\"},\"blocks\":[{"
        "\"input_norm\":{\"kind\":\"layer_norm\",\"eps\":0.00001},"
        "\"attention\":{\"q\":{\"weight\":\"q\"},"
        "\"k\":{\"weight\":\"k\"},\"v\":{\"weight\":\"v\"},"
        "\"output\":{\"weight\":\"attention_output\"},"
        "\"query_heads\":1,\"kv_heads\":1,\"causal\":false,\"scale\":1.0" +
        mask_field + "},"
        "\"post_attention_norm\":{\"kind\":\"layer_norm\",\"eps\":0.00001},"
        "\"mlp\":{\"activation_input\":{\"weight\":\"mlp_input\"},"
        "\"activation\":\"gelu\",\"output\":{\"weight\":\"mlp_output\"}}}]" +
        trim_field + "}");
}

std::vector<float> values(vrhino::CudaBackend& backend, const Tensor& tensor) {
    Tensor host = tensor.device().is_host() ? tensor : backend.copy_to_host(tensor);
    if (host.dtype() == vrhino::DType::F32)
        return {host.data_as<float>(), host.data_as<float>() + host.numel()};
    vrhino::require(host.dtype() == vrhino::DType::BF16,
                    "Conditioning mask test expected FP32/BF16 output");
    std::vector<float> output(host.numel());
    for (int64_t index = 0; index < host.numel(); ++index) {
        const uint32_t bits = static_cast<uint32_t>(host.data_as<uint16_t>()[index]) << 16U;
        output[index] = std::bit_cast<float>(bits);
    }
    return output;
}

void finite(const std::string& name, const std::vector<float>& value) {
    for (size_t index = 0; index < value.size(); ++index)
        vrhino::require(std::isfinite(value[index]),
                        name + " non-finite at " + std::to_string(index));
}

void close(const std::string& name, const std::vector<float>& actual,
           const std::vector<float>& expected, float tolerance = 1e-6f) {
    vrhino::require(actual.size() == expected.size(), name + " size mismatch");
    for (size_t index = 0; index < actual.size(); ++index)
        vrhino::require(std::abs(actual[index] - expected[index]) <= tolerance,
                        name + " mismatch at " + std::to_string(index));
}

void different(const std::string& name, const std::vector<float>& left,
               const std::vector<float>& right, float minimum = 1e-3f) {
    vrhino::require(left.size() == right.size(), name + " size mismatch");
    float maximum = 0.0f;
    for (size_t index = 0; index < left.size(); ++index)
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    vrhino::require(maximum >= minimum, name + " did not exercise mask consumption");
}

void rejects(const std::string& name, const std::function<void()>& operation,
             const std::string& expected) {
    try {
        operation();
    } catch (const vrhino::Error& error) {
        vrhino::require(std::string(error.what()).find(expected) != std::string::npos,
                        name + " rejected with unexpected error: " + error.what());
        return;
    }
    throw vrhino::Error(name + " did not fail closed");
}

}  // namespace

int main() {
    try {
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);

        std::map<std::string, Tensor> storage;
        storage.emplace("embedding", vrhino::host_f32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
        storage.emplace("q", vrhino::host_f32({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
        storage.emplace("k", vrhino::host_f32({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
        storage.emplace("v", vrhino::host_f32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
        storage.emplace("attention_output",
                        vrhino::host_f32({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));
        storage.emplace("mlp_input", vrhino::host_f32({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
        storage.emplace("mlp_output", vrhino::host_f32({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f}));
        std::map<std::string, const Tensor*> pointers;
        for (const auto& [name, tensor] : storage) pointers.emplace(name, &tensor);
        const vrhino::WeightMap weights(std::move(pointers));
        vrhino::ConditioningComponentExecutor executor(backend, weights);

        const Tensor ids = vrhino::host_i64({1, 2}, {0, 1});
        const Tensor padded_mask = vrhino::host_bool({1, 2}, {1, 0});
        const Tensor valid_mask = vrhino::host_bool({1, 2}, {1, 1});

        vrhino::require(vrhino::conditioning_mask_semantic(
                            Json::parse("{}")) == vrhino::MaskSemantic::Boolean,
                        "Missing mask semantic did not preserve Boolean default");
        vrhino::require(vrhino::conditioning_mask_semantic(
                            Json::parse("{\"mask_semantic\":\"none\"}")) ==
                            vrhino::MaskSemantic::None,
                        "None mask semantic parse mismatch");
        vrhino::require(vrhino::conditioning_mask_semantic(
                            Json::parse("{\"mask_semantic\":\"boolean\"}")) ==
                            vrhino::MaskSemantic::Boolean,
                        "Boolean mask semantic parse mismatch");
        vrhino::require(vrhino::conditioning_mask_semantic(
                            Json::parse("{\"mask_semantic\":\"additive_finite\"}")) ==
                            vrhino::MaskSemantic::AdditiveFinite,
                        "AdditiveFinite mask semantic parse mismatch");

        const std::vector<float> none_padded = values(
            backend, executor.execute(graph("none"), ids, padded_mask).hidden_states);
        const std::vector<float> none_valid = values(
            backend, executor.execute(graph("none"), ids, valid_mask).hidden_states);
        close("None must not consume tokenizer mask", none_padded, none_valid);

        const std::vector<float> boolean_padded = values(
            backend, executor.execute(graph("boolean"), ids, padded_mask).hidden_states);
        const std::vector<float> boolean_valid = values(
            backend, executor.execute(graph("boolean"), ids, valid_mask).hidden_states);
        different("Boolean must consume tokenizer mask", boolean_padded, boolean_valid);
        different("None must differ from masked Boolean", none_padded, boolean_padded);

        const std::vector<float> default_padded = values(
            backend, executor.execute(graph(""), ids, padded_mask).hidden_states);
        close("Schema-v1 default must remain Boolean", default_padded, boolean_padded);

        const std::vector<float> additive_padded = values(
            backend, executor.execute(graph("additive_finite"), ids, padded_mask).hidden_states);
        close("AdditiveFinite must preserve padding-mask semantics",
              additive_padded, boolean_padded, 2e-5f);

        const Tensor all_masked = vrhino::host_bool({1, 2}, {0, 0});
        const std::vector<float> additive_all_masked_f32 = values(
            backend, executor.execute(graph("additive_finite"), ids, all_masked).hidden_states);
        finite("FP32 all-masked AdditiveFinite", additive_all_masked_f32);

        backend.set_execution_dtype(vrhino::DType::BF16);
        const std::vector<float> additive_all_masked_bf16 = values(
            backend, executor.execute(graph("additive_finite"), ids, all_masked).hidden_states);
        finite("BF16 all-masked AdditiveFinite", additive_all_masked_bf16);
        backend.set_execution_dtype(vrhino::DType::F32);

        const Tensor trimmed = executor.execute(
            graph("none", true), ids, padded_mask).hidden_states;
        vrhino::require(trimmed.shape() == std::vector<int64_t>({1, 1, 2}),
                        "None incorrectly disabled independent output trimming");
        const std::vector<float> trimmed_values = values(backend, trimmed);
        close("Output trim must use tokenizer mask independently",
              trimmed_values, {none_padded[0], none_padded[1]});

        rejects("Unknown mask semantic", [&] {
            (void)vrhino::conditioning_mask_semantic(
                Json::parse("{\"mask_semantic\":\"model_specific\"}"));
        }, "Unsupported conditioning mask semantic");
        rejects("Non-string mask semantic", [&] {
            (void)vrhino::conditioning_mask_semantic(
                Json::parse("{\"mask_semantic\":7}"));
        }, "Conditioning mask semantic must be a string");
        rejects("None still requires tokenizer mask shape", [&] {
            (void)executor.execute(graph("none"), ids, vrhino::host_bool({1, 1}, {1}));
        }, "Conditioning mask shape mismatch");

        backend.synchronize();
        std::cout << "conditioning-mask-contract-v1 cases=14 pass\n"
                  << "mask_semantics=none,boolean,additive_finite\n"
                  << "tokenizer_mask_production=independent\n"
                  << "attention_mask_consumption=independent\n"
                  << "output_trim=independent\n"
                  << "all_masked_fp32=finite\n"
                  << "all_masked_bf16=finite\n"
                  << "all_masked_attention_semantic="
                  << "uniform_over_equal_finite_biases\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conditioning mask contract tests: " << error.what() << '\n';
        return 1;
    }
}
