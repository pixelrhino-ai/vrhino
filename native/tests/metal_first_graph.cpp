#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/metal_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

struct Difference {
    double maximum = 0.0;
    double mean = 0.0;
    bool allclose = true;
};

Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& expected,
                   double atol, double rtol) {
    vrhino::require(actual.device().is_host() && expected.device().is_host(),
                    "first-graph comparison requires host tensors");
    vrhino::require(actual.dtype() == vrhino::DType::F32 &&
                    expected.dtype() == vrhino::DType::F32,
                    "first-graph comparison requires FP32 tensors");
    vrhino::require(actual.shape() == expected.shape(),
                    "first-graph output shape mismatch");
    const float* left = actual.data_as<float>();
    const float* right = expected.data_as<float>();
    Difference result;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double difference = std::abs(static_cast<double>(left[index]) - right[index]);
        result.maximum = std::max(result.maximum, difference);
        result.mean += difference;
        result.allclose = result.allclose &&
            difference <= atol + rtol * std::abs(static_cast<double>(right[index]));
    }
    result.mean /= static_cast<double>(actual.numel());
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: vrhino-metal-first-graph MODEL.vrm INPUT.bundle GOLDEN.bundle\n";
            return 2;
        }
        vrhino::VrmModel model(argv[1]);
        vrhino::require(model.architecture_id() == "wan", "Phase 12B first graph gate uses canonical Wan");
        vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);
        input.emplace("audit_trace", vrhino::scalar_i64(1));
        const vrhino::TensorBundle golden = vrhino::read_bundle(argv[3]);

        vrhino::MetalBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_profiling(true);
        auto architecture = vrhino::create_architecture(model);
        const vrhino::PrecisionPolicy policy =
            vrhino::PrecisionPolicy::unqualified_default(backend.execution_dtype());
        vrhino::SamplingProgram program = architecture->create_program(input);
        auto denoiser = architecture->create_denoiser(backend, policy, input);
        vrhino::RngState rng{program.seed, 0, "pytorch_compat.v1"};
        vrhino::Tensor latent = backend.rng_normal(
            rng, program.latent_shape,
            vrhino::effective_sampling_state_dtype(policy));
        std::vector<vrhino::Tensor> predictions =
            denoiser->evaluate(latent, program.model_timestep_at(0));
        vrhino::require(predictions.size() == 2, "canonical Wan first graph must produce two CFG branches");
        vrhino::TensorBundle trace = denoiser->take_trace();
        backend.synchronize();

        const std::set<std::string> required_trace = {
            "branch.0.block0.norm1",
            "branch.0.block0.self.q_linear",
            "branch.0.block0.self.q_norm",
            "branch.0.block0.self.output",
            "branch.0.block0.self_residual",
            "branch.0.block0.cross.output",
            "branch.0.block0.cross_residual",
            "branch.0.block0.ffn0",
            "branch.0.block0.ffn2",
            "branch.0.block_0",
        };
        for (const std::string& name : required_trace)
            vrhino::require(trace.contains(name), "missing first-graph trace: " + name);

        constexpr double kAtol = 0.015;
        constexpr double kRtol = 0.001;
        bool passed = true;
        for (size_t branch = 0; branch < predictions.size(); ++branch) {
            const vrhino::Tensor actual = backend.copy_to_host(predictions[branch]);
            const std::string key = "step.0.prediction." + std::to_string(branch);
            const Difference difference = compare(actual, golden.at(key), kAtol, kRtol);
            passed = passed && difference.allclose;
            std::cout << key << ",max_abs=" << difference.maximum
                      << ",mean_abs=" << difference.mean
                      << ",allclose=" << (difference.allclose ? "true" : "false") << "\n";
        }

        const auto profiles = backend.profile_stats();
        for (const std::string& primitive : {"linear", "norm.rms", "rope", "attention", "activation"})
            vrhino::require(profiles.contains(primitive) && profiles.at(primitive).calls > 0,
                            "first graph did not execute primitive: " + primitive);
        vrhino::require(passed, "Wan first native graph exceeded the FP32 numerical gate");
        std::cout << "trace_tensors=" << trace.size() << "\n"
                  << "atol=" << kAtol << "\nrtol=" << kRtol << "\n"
                  << "native Metal Wan first graph=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native Metal first graph: " << error.what() << "\n";
        return 1;
    }
}
