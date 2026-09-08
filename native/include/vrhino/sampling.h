#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vrhino/backend.h"
#include "vrhino/prepared_tensor.h"
#include "vrhino/precision.h"
#include "vrhino/tensor.h"

namespace vrhino {

class Denoiser {
public:
    virtual ~Denoiser() = default;
    virtual std::vector<Tensor> evaluate(const Tensor& latent, const Tensor& timestep) = 0;
    // Optional architecture instrumentation consumed by the shared sampling
    // runtime. Production execution leaves this empty.
    virtual std::map<std::string, Tensor> take_trace() { return {}; }
    const PreparedTensorCacheStats& prepared_tensor_stats() const {
        return prepared_tensors_.stats();
    }

protected:
    PreparedTensorCache& prepared_tensors() { return prepared_tensors_; }

private:
    PreparedTensorCache prepared_tensors_;
};

enum class GuidanceMode { Linear, CFG };
// The Euler path remains the native declaration used by architectures that
// have not migrated to Sampling Contract v1. It is not an external scheduler
// compatibility layer.
enum class SchedulerKind { Euler };

enum class PredictionSemantic { Flow, V, Epsilon };
enum class SolverSemantic { MultistepPredictorCorrector, AffineFirstOrder };
enum class ScheduleSemantic { FlowSigma, AlphaCumprod };

struct PredictionContract {
    PredictionSemantic semantic = PredictionSemantic::Flow;
};

struct SolverContract {
    SolverSemantic semantic = SolverSemantic::MultistepPredictorCorrector;
    int maximum_order = 2;
};

struct FlowScheduleTransition {
    Tensor model_timestep;
    float sigma = 0.0f;
    float next_sigma = 0.0f;
};

struct AlphaCumprodScheduleTransition {
    Tensor model_timestep;
    float alpha_cumprod = 0.0f;
    float previous_alpha_cumprod = 0.0f;
};

// A typed step table has no mutating API after construction. The factory
// validates the complete transition chain before it can enter SamplingRuntime.
class ScheduleContract {
public:
    static ScheduleContract flow_sigma(std::vector<FlowScheduleTransition> transitions);
    static ScheduleContract alpha_cumprod(
        std::vector<AlphaCumprodScheduleTransition> transitions);
    ScheduleSemantic semantic() const { return semantic_; }
    size_t size() const;
    const Tensor& model_timestep_at(size_t index) const;
    const FlowScheduleTransition& flow_at(size_t index) const;
    const AlphaCumprodScheduleTransition& alpha_cumprod_at(size_t index) const;

private:
    ScheduleContract(ScheduleSemantic semantic,
                     std::vector<FlowScheduleTransition> transitions)
        : semantic_(semantic), transitions_(std::move(transitions)) {}
    ScheduleContract(ScheduleSemantic semantic,
                     std::vector<AlphaCumprodScheduleTransition> transitions)
        : semantic_(semantic), transitions_(std::move(transitions)) {}
    ScheduleSemantic semantic_;
    std::variant<std::vector<FlowScheduleTransition>,
                 std::vector<AlphaCumprodScheduleTransition>> transitions_;
};

struct SamplingContract {
    PredictionContract prediction;
    SolverContract solver;
    ScheduleContract schedule;
};

struct SamplingProgram {
    std::vector<int64_t> latent_shape;
    uint64_t seed = 0;
    int steps = 3;
    GuidanceMode guidance_mode = GuidanceMode::Linear;
    std::vector<float> guidance_coefficients;
    // Contract v1 is populated only by migrated architecture declarations.
    // Unmigrated Euler programs retain their existing native fields below.
    std::optional<SamplingContract> contract;
    SchedulerKind scheduler = SchedulerKind::Euler;
    std::vector<Tensor> model_timesteps;
    std::vector<float> sigmas;
    std::vector<float> update_deltas;
    bool subtract_prediction = false;
    bool zero_is_frozen = false;

    const Tensor& model_timestep_at(int step) const;
};

struct SamplingResult {
    Tensor initial_noise;
    Tensor final_latent;
    std::map<std::string, Tensor> trace;
    RngState rng_after_initialization;
    double elapsed_seconds = 0.0;
    std::vector<double> denoiser_call_seconds;
    double scheduler_seconds = 0.0;
    PreparedTensorCacheStats prepared_tensors;
};

class SamplingPrimitives {
public:
    explicit SamplingPrimitives(Backend& backend) : backend_(backend) {}
    Tensor rng_normal(RngState& state, const std::vector<int64_t>& shape, DType dtype);
    std::vector<float> linear_schedule(float start, float stop, int count);
    std::vector<float> rational_time_shift(const std::vector<float>& values, float shift);
    std::vector<float> resolution_time_shift(const std::vector<float>& values, int token_count);
    int schedule_lookup(const std::vector<float>& schedule, float timestep);
    Tensor linear_combine(const std::vector<Tensor>& values, const std::vector<float>& coefficients);
    Tensor cfg_combine(const Tensor& unconditional, const Tensor& conditional, float scale);
    Tensor euler_update(const Tensor& sample, const Tensor& prediction, const Tensor& delta,
                        bool subtract_prediction);
    Tensor tensor_where(const Tensor& condition, const Tensor& when_true, const Tensor& when_false);
    Tensor flow_to_x0(const Tensor& sample, const Tensor& model_output, float sigma);
    Tensor v_to_x0(const Tensor& sample, const Tensor& model_output,
                   float alpha_cumprod);
    Tensor epsilon_to_x0(const Tensor& sample, const Tensor& model_output,
                         float alpha_cumprod);
    Tensor affine_first_order(const Tensor& current_state, const Tensor& x0,
                              float state_coefficient, float prediction_coefficient);
    Tensor multistep_predictor(const Tensor& sample, const std::vector<Tensor>& model_outputs,
                               const Tensor& model_t, const std::vector<float>& sigmas,
                               int step_index, int order);
    Tensor multistep_corrector(const Tensor& sample, const Tensor& this_sample,
                               const std::vector<Tensor>& model_outputs, const Tensor& model_t,
                               const std::vector<float>& sigmas, int step_index, int order,
                               const Tensor& last_sample);
    void state_advance();
    const std::map<std::string, uint64_t>& calls() const { return calls_; }

private:
    Tensor unipc_update(bool corrector, const Tensor& sample, const Tensor& this_sample,
                        const std::vector<Tensor>& model_outputs, const Tensor& model_t,
                        const std::vector<float>& sigmas, int step_index, int order);
    void use(const std::string& name) { ++calls_[name]; }
    Backend& backend_;
    std::map<std::string, uint64_t> calls_;
};

class SamplingRuntime {
public:
    explicit SamplingRuntime(Backend& backend)
        : SamplingRuntime(backend,
            PrecisionPolicy::unqualified_default(backend.execution_dtype())) {}
    SamplingRuntime(Backend& backend, PrecisionPolicy policy)
        : backend_(backend), policy_(policy), primitives_(backend) {}
    // Observation-only product hook. It runs after a completed scheduler
    // transition and cannot alter tensors or sampling declarations.
    void set_step_observer(std::function<void(int, int)> observer) {
        step_observer_ = std::move(observer);
    }
    // Generic bounded cancellation hook. It is checked before initialization
    // and before every denoiser iteration; Product layers map the failure to
    // their established cancellation status.
    void set_cancellation_requested(std::function<bool()> requested) {
        cancellation_requested_ = std::move(requested);
    }
    SamplingResult run(Denoiser& denoiser, const SamplingProgram& program);
    // Executes the same generic program from an explicit caller-owned initial
    // latent. This is the deterministic boundary for structured initial state.
    SamplingResult run_with_initial_state(
        Denoiser& denoiser, const SamplingProgram& program,
        const Tensor& initial_state);
    // Test/reference harness entrypoint for controlled same-input validation.
    // Retained as a source-compatible alias for qualification tools.
    SamplingResult run_with_external_initial_state_for_test(
        Denoiser& denoiser, const SamplingProgram& program,
        const Tensor& external_initial_state);
    const SamplingPrimitives& primitives() const { return primitives_; }

private:
    SamplingResult run_impl(Denoiser& denoiser, const SamplingProgram& program,
                            const Tensor* external_initial_state);
    Backend& backend_;
    PrecisionPolicy policy_;
    SamplingPrimitives primitives_;
    std::function<void(int, int)> step_observer_;
    std::function<bool()> cancellation_requested_;
};

DType effective_sampling_state_dtype(const PrecisionPolicy& policy);
DType effective_denoiser_output_dtype(const PrecisionPolicy& policy);

// Builds the deterministic eta=0 DDIM AlphaCumprod step table used by any
// epsilon/V-prediction program with a scaled-linear beta schedule.
ScheduleContract make_scaled_linear_ddim_schedule(
    int training_timesteps, float beta_start, float beta_end,
    int inference_steps, int steps_offset, bool set_alpha_to_one);

}  // namespace vrhino
