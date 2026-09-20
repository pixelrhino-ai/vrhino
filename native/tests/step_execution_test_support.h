#pragma once

#include <cstring>
#include <utility>

#include "neural_graph_test_backend.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"

namespace vrhino::step_test {

// Host-only qualification backend, never a product execution backend. RNG is a
// deterministic sentinel for initialization/count/offset, not a normal sampler.
class Backend final : public neural_graph::test::TinyBackend {
public:
    size_t rng_calls = 0;
    std::string name() const override { return "test-host"; }
    bool profiling_enabled() const override { return false; }
    void profile_region_begin(const std::string&) override {}
    void profile_region_end() override {}
    void synchronize() override {
        ++syncs;
        if (fail_sync) throw Error("test completion failure");
    }
    Tensor copy_to_device(const Tensor& tensor, DType dtype) override {
        require(tensor.dtype() == dtype && tensor.device().is_host(), "Test copy contract");
        return tensor;
    }
    Tensor copy_to_host(const Tensor& tensor) override { return tensor; }
    Tensor cast(const Tensor& tensor, DType dtype) override {
        begin("cast");
        require(tensor.dtype() == dtype, "Unexpected qualification cast");
        return tensor;
    }
    Tensor rng_normal(RngState& state, const std::vector<int64_t>& shape, DType dtype) override {
        begin("rng");
        ++rng_calls;
        require(dtype == DType::F32, "Test RNG dtype");
        Tensor output = Tensor::host(shape, dtype);
        for (int64_t i = 0; i < output.numel(); ++i)
            output.data_as<float>()[i] = static_cast<float>((state.seed + state.offset + i) % 17) / 32.0f;
        state.offset += static_cast<uint64_t>(output.numel());
        return output;
    }
    size_t peak_device_bytes() const override { return 0; }
    size_t weight_upload_bytes() const override { return 0; }
    double weight_upload_seconds() const override { return 0; }
};

// G: u = x*0.25 + parameter; c = u+0.125. Only its parameter varies.
struct FakeGraph {
    std::vector<Tensor> evaluate(Backend& backend, const Tensor& latent, const Tensor& parameter) const {
        Tensor u = backend.add(backend.mul(latent, scalar_f32(0.25f)), parameter);
        return {u, backend.add(u, scalar_f32(0.125f))};
    }
};

class LegacyDenoiser : public Denoiser {
public:
    LegacyDenoiser(Backend& backend, Tensor parameter)
        : backend(backend), parameter(std::move(parameter)) {}
    std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) override {
        ++evaluations;
        PreparedTensorKey key;
        key.operation = "test.constant.v1";
        key.target_backend = "test-host";
        key.input_content_hash = hash_host_tensor_content(parameter);
        auto prepared = prepared_tensors().prepare(key, [&] {
            return PreparedTensorMaterialization{{parameter}, 1, 0, 0};
        });
        const auto& values = prepared_tensors().reuse(prepared);
        auto predictions = graph.evaluate(backend, latent, values.at(0));
        trace = {{"endpoint.timestep", timestep}, {"endpoint.parameter", parameter},
                 {"endpoint.local_call", scalar_i64(evaluations)}};
        return predictions;
    }
    std::map<std::string, Tensor> take_trace() override { return std::exchange(trace, {}); }
    Backend& backend;
    Tensor parameter;
    FakeGraph graph;
    int evaluations = 0;
    std::map<std::string, Tensor> trace;
};

inline SamplingProgram program(int kind = 0) {
    SamplingProgram p;
    p.latent_shape = {1, 2}; p.seed = 11; p.steps = 3;
    p.guidance_mode = GuidanceMode::CFG;
    // Deliberately irrelevant first coefficient: legacy CFG reads only [1].
    p.guidance_coefficients = {123.0f, 2.0f};
    p.model_timesteps = {scalar_i64(9), scalar_i64(5), scalar_i64(1)};
    p.sigmas = {0.9f, 0.5f, 0.1f, 0.0f};
    p.update_deltas = {-0.4f, -0.4f, -0.1f};
    if (kind == 1) {
        std::vector<FlowScheduleTransition> steps;
        for (int i = 0; i < p.steps; ++i)
            steps.push_back({p.model_timesteps[i], p.sigmas[i], p.sigmas[i + 1]});
        p.contract.emplace(PredictionContract{PredictionSemantic::Flow},
            SolverContract{SolverSemantic::MultistepPredictorCorrector, 2},
            ScheduleContract::flow_sigma(std::move(steps)));
    } else if (kind == 2 || kind == 3) {
        p.contract.emplace(PredictionContract{kind == 2 ? PredictionSemantic::V : PredictionSemantic::Epsilon},
            SolverContract{SolverSemantic::AffineFirstOrder, 1},
            ScheduleContract::alpha_cumprod({{scalar_i64(9),0.2f,0.5f},
                                            {scalar_i64(5),0.5f,0.8f},
                                            {scalar_i64(1),0.8f,1.0f}}));
    }
    return p;
}

inline void exact(const Tensor& a, const Tensor& b) {
    require(a.shape() == b.shape() && a.dtype() == b.dtype() && a.bytes() == b.bytes() &&
                std::memcmp(a.data(), b.data(), a.bytes()) == 0, "Tensor not byte exact");
}

inline void exact(const SamplingResult& a, const SamplingResult& b) {
    exact(a.initial_noise, b.initial_noise); exact(a.final_latent, b.final_latent);
    require(a.trace.size() == b.trace.size(), "Trace key count changed");
    for (const auto& [name, value] : a.trace) exact(value, b.trace.at(name));
    require(a.rng_after_initialization.seed == b.rng_after_initialization.seed &&
                a.rng_after_initialization.offset == b.rng_after_initialization.offset &&
                a.rng_after_initialization.algorithm_id == b.rng_after_initialization.algorithm_id,
            "RNG state changed");
    const auto& x = a.prepared_tensors;
    const auto& y = b.prepared_tensors;
    require(x.prepare_hits == y.prepare_hits && x.prepare_misses == y.prepare_misses && x.reuses == y.reuses &&
            x.host_materializations == y.host_materializations && x.host_to_device_transfers == y.host_to_device_transfers &&
            x.host_to_device_bytes == y.host_to_device_bytes && x.resident_entries == y.resident_entries &&
            x.resident_tensors == y.resident_tensors && x.resident_bytes == y.resident_bytes, "Prepared stats changed");
}
}  // namespace vrhino::step_test
