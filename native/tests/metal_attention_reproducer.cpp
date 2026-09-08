#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/metal_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"

namespace {

std::string dimensions(const std::vector<int64_t>& values) {
    std::ostringstream stream;
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << values[index];
    }
    return stream.str() + ']';
}

const vrhino::Tensor& tensor(
    std::map<std::string, vrhino::TensorBundle>& bundles,
    const std::string& path, const std::string& key) {
    auto [item, inserted] = bundles.try_emplace(path);
    if (inserted) item->second = vrhino::read_bundle(path);
    const auto value = item->second.find(key);
    vrhino::require(value != item->second.end(), "Missing isolated Attention tensor: " + key);
    return value->second;
}

vrhino::Tensor host(vrhino::MetalBackend& backend, const vrhino::Tensor& value) {
    return value.device().is_host() ? value : backend.copy_to_host(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 12,
            "usage: metal-attention-reproducer Q_BUNDLE Q_KEY K_BUNDLE K_KEY V_BUNDLE V_KEY BIAS_BUNDLE BIAS_KEY OUTPUT_BUNDLE SCALE CAUSAL");
        std::map<std::string, vrhino::TensorBundle> bundles;
        const vrhino::Tensor& q = tensor(bundles, argv[1], argv[2]);
        const vrhino::Tensor& k = tensor(bundles, argv[3], argv[4]);
        const vrhino::Tensor& v = tensor(bundles, argv[5], argv[6]);
        const vrhino::Tensor& bias = tensor(bundles, argv[7], argv[8]);
        const std::string output_path = argv[9];
        const float scale = std::stof(argv[10]);
        const bool causal = std::stoi(argv[11]) != 0;

        vrhino::MetalBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_profiling(true);
        vrhino::AttentionObservation observation;
        vrhino::Tensor heads = backend.attention(
            q, k, v, nullptr, causal, scale, &bias, &observation);
        vrhino::Tensor merged = vrhino::merge_heads(backend, heads);
        backend.synchronize();
        vrhino::write_bundle(output_path, {
            {"isolated.attention_score", host(backend, observation.score)},
            {"isolated.biased_score", host(backend, observation.biased_score)},
            {"isolated.softmax", host(backend, observation.softmax)},
            {"isolated.attention_heads", host(backend, heads)},
            {"isolated.attention_merged", host(backend, merged)},
        });

        std::cout << "status=PASS"
                  << ",q_shape=" << dimensions(q.shape())
                  << ",q_strides=" << dimensions(q.strides())
                  << ",k_shape=" << dimensions(k.shape())
                  << ",k_strides=" << dimensions(k.strides())
                  << ",v_shape=" << dimensions(v.shape())
                  << ",v_strides=" << dimensions(v.strides())
                  << ",bias_shape=" << dimensions(bias.shape())
                  << ",bias_strides=" << dimensions(bias.strides())
                  << ",heads_shape=" << dimensions(heads.shape())
                  << ",merged_shape=" << dimensions(merged.shape())
                  << ",dtype=float32,scale=" << scale
                  << ",causal=" << (causal ? "true" : "false")
                  << ",boolean_mask=false,additive_bias=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
