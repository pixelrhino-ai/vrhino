// Synthetic qualification data only; no checkpoint or captured activations.
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include "vrhino/backend/cuda_backend.h"
#include "wan_self_attention_reference.h"

using namespace vrhino;

int main() {
    try {
        std::ifstream input(VRHINO_TEST_PRECISION_POLICY);
        require(input.good(), "generic precision fixture missing");
        const std::string policy_text{std::istreambuf_iterator<char>(input), {}};
        const auto mixed = PrecisionPolicy::from_json(Json::parse(policy_text));
        CudaBackend backend;
        for (const auto& mode : {std::pair{DType::F32, DType::F32},
                                std::pair{DType::BF16, DType::F32},
                                std::pair{DType::BF16, DType::BF16}}) {
            backend.set_execution_dtype(mode.first);
            const auto policy = mode.first == DType::F32 ? PrecisionPolicy::fp32() : mixed;
            const auto graph = wan_internal::self_attention_graph(
                1, 3, mode.first, mode.second, true);
            const auto& d = graph.description();
            TensorBundle weights;
            std::map<std::string, const Tensor*> slots;
            for (size_t k = 0; k < std::size(wan_internal::parameters); ++k) {
                const std::string name = wan_internal::parameters[k];
                Tensor t = Tensor::host(d.values[d.parameters[k]].shape, DType::F32);
                const float base = name.find("norm_") != std::string::npos ? 1.0f : 0.0f;
                for (int64_t i = 0; i < t.numel(); ++i)
                    t.data_as<float>()[i] = base + 0.005f * std::sin(float((i * 13 + k * 7) % 127));
                weights.emplace(name, std::move(t));
                slots.emplace(name, &weights.at(name));
            }
            TensorBundle inputs;
            const char* names[] = {"hidden", "modulation", "cosine", "sine"};
            for (size_t k = 0; k < 4; ++k) {
                Tensor t = Tensor::host(d.values[d.inputs[k]].shape, DType::F32);
                for (int64_t i = 0; i < t.numel(); ++i) {
                    const float angle = float(i % 128) * 0.01f;
                    t.data_as<float>()[i] = k == 2 ? std::cos(angle) : k == 3 ? std::sin(angle)
                        : 0.1f * std::sin(float((i * 17 + k * 3) % 113));
                }
                inputs.emplace(names[k], backend.copy_to_device(t, k == 0 ? mode.second : DType::F32));
            }
            const WeightMap weight_map(slots);
            const auto reference = wan_test::reference(backend, policy, weight_map, inputs);
            const auto outputs = wan_internal::self_attention(backend, policy, weight_map,
                inputs.at("hidden"), inputs.at("modulation"), inputs.at("cosine"), inputs.at("sine"), true);
            for (size_t k = 0; k < outputs.size(); ++k) {
                const std::string name = wan_internal::checkpoints[k];
                const auto found = reference.find(name);
                if (found == reference.end()) continue;
                const Tensor got = backend.copy_to_host(outputs[k]);
                const Tensor expected = backend.copy_to_host(found->second);
                require(got.shape() == expected.shape() && got.dtype() == expected.dtype() &&
                        got.bytes() == expected.bytes() &&
                        std::memcmp(got.data(), expected.data(), got.bytes()) == 0,
                        "synthetic graph/reference mismatch: " + name);
            }
            const Tensor result = backend.copy_to_host(outputs.back());
            require(result.dtype() == DType::F32, "residual precision contract");
            for (int64_t i = 0; i < result.numel(); ++i)
                require(std::isfinite(result.data_as<float>()[i]), "nonfinite synthetic residual");
            std::cout << "Wan production Graph synthetic execution=" << dtype_name(mode.first)
                      << " hidden=" << dtype_name(mode.second) << " bitwise reference: PASS\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
