#include "vrhino/sampling.h"
#include "vrhino/execution.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {

namespace {

void validate_schedule_timestep(const Tensor& timestep, const char* schedule_name) {
    require(timestep.defined(),
            std::string(schedule_name) + " schedule model timestep is undefined");
    require(timestep.device() == Device::CPU,
            std::string(schedule_name) + " schedule model timestep must be host-resident");
}

}  // namespace

ScheduleContract ScheduleContract::flow_sigma(
        std::vector<FlowScheduleTransition> transitions) {
    require(!transitions.empty(), "Flow schedule step table is empty");
    for (size_t index = 0; index < transitions.size(); ++index) {
        const FlowScheduleTransition& transition = transitions[index];
        validate_schedule_timestep(transition.model_timestep, "Flow");
        require(std::isfinite(transition.sigma) && std::isfinite(transition.next_sigma),
                "Flow schedule sigma must be finite");
        if (index + 1 < transitions.size())
            require(transition.next_sigma == transitions[index + 1].sigma,
                    "Flow schedule transitions are not contiguous");
    }
    return ScheduleContract(ScheduleSemantic::FlowSigma, std::move(transitions));
}

ScheduleContract ScheduleContract::alpha_cumprod(
        std::vector<AlphaCumprodScheduleTransition> transitions) {
    require(!transitions.empty(), "AlphaCumprod schedule step table is empty");
    for (size_t index = 0; index < transitions.size(); ++index) {
        const AlphaCumprodScheduleTransition& transition = transitions[index];
        validate_schedule_timestep(transition.model_timestep, "AlphaCumprod");
        require(std::isfinite(transition.alpha_cumprod) &&
                    std::isfinite(transition.previous_alpha_cumprod),
                "AlphaCumprod schedule coefficients must be finite");
        require(transition.alpha_cumprod >= 0.0f &&
                    transition.alpha_cumprod < 1.0f &&
                    transition.previous_alpha_cumprod >= transition.alpha_cumprod &&
                    transition.previous_alpha_cumprod <= 1.0f,
                "AlphaCumprod schedule coefficients are outside denoising bounds");
        if (index + 1 < transitions.size())
            require(transition.previous_alpha_cumprod ==
                        transitions[index + 1].alpha_cumprod,
                    "AlphaCumprod schedule transitions are not contiguous");
    }
    return ScheduleContract(ScheduleSemantic::AlphaCumprod, std::move(transitions));
}

size_t ScheduleContract::size() const {
    if (semantic_ == ScheduleSemantic::FlowSigma)
        return std::get<std::vector<FlowScheduleTransition>>(transitions_).size();
    return std::get<std::vector<AlphaCumprodScheduleTransition>>(transitions_).size();
}

const Tensor& ScheduleContract::model_timestep_at(size_t index) const {
    if (semantic_ == ScheduleSemantic::FlowSigma) return flow_at(index).model_timestep;
    return alpha_cumprod_at(index).model_timestep;
}

const FlowScheduleTransition& ScheduleContract::flow_at(size_t index) const {
    require(semantic_ == ScheduleSemantic::FlowSigma,
            "Schedule Contract is not a FlowSigma table");
    return std::get<std::vector<FlowScheduleTransition>>(transitions_).at(index);
}

const AlphaCumprodScheduleTransition& ScheduleContract::alpha_cumprod_at(
        size_t index) const {
    require(semantic_ == ScheduleSemantic::AlphaCumprod,
            "Schedule Contract is not an AlphaCumprod table");
    return std::get<std::vector<AlphaCumprodScheduleTransition>>(transitions_).at(index);
}

const Tensor& SamplingProgram::model_timestep_at(int step) const {
    require(step >= 0 && step < steps, "Sampling timestep index outside program");
    return contract ? contract->schedule.model_timestep_at(static_cast<size_t>(step))
                    : model_timesteps.at(static_cast<size_t>(step));
}

DType effective_sampling_state_dtype(const PrecisionPolicy& policy) {
    return policy.persistent_state_dtype(PrecisionSemantic::SamplingState);
}

DType effective_denoiser_output_dtype(const PrecisionPolicy& policy) {
    return policy.boundary_dtype(PrecisionSemantic::DenoiserOutput);
}

Tensor SamplingPrimitives::rng_normal(RngState& state, const std::vector<int64_t>& shape, DType dtype) { use("rng_normal"); return backend_.rng_normal(state, shape, dtype); }

std::vector<float> SamplingPrimitives::linear_schedule(float start, float stop, int count) {
    use("linear_schedule"); require(count > 0, "linear_schedule count must be positive");
    std::vector<float> output(count); if (count == 1) { output[0] = start; return output; }
    for (int index = 0; index < count; ++index) output[index] = start + (stop - start) * index / (count - 1);
    return output;
}

std::vector<float> SamplingPrimitives::rational_time_shift(const std::vector<float>& values, float shift) {
    use("rational_time_shift"); std::vector<float> output; output.reserve(values.size());
    for (float value : values) output.push_back(shift * value / (1.0f + (shift - 1.0f) * value));
    return output;
}

std::vector<float> SamplingPrimitives::resolution_time_shift(const std::vector<float>& values, int token_count) {
    use("resolution_time_shift"); constexpr float min_tokens = 1024, max_tokens = 4096, min_shift = 0.95f, max_shift = 2.05f;
    const float slope = (max_shift - min_shift) / (max_tokens - min_tokens);
    const float mu = slope * token_count + min_shift - slope * min_tokens, exponential = std::exp(mu);
    std::vector<float> output; output.reserve(values.size());
    for (float value : values) output.push_back(exponential / (exponential + (1.0f / value - 1.0f)));
    return output;
}

int SamplingPrimitives::schedule_lookup(const std::vector<float>& schedule, float timestep) {
    use("schedule_lookup"); require(!schedule.empty(), "Empty schedule"); int result = 0; float distance = std::abs(schedule[0] - timestep);
    for (int index = 1; index < static_cast<int>(schedule.size()); ++index) {
        if (std::abs(schedule[index] - timestep) < distance) {
            result = index; distance = std::abs(schedule[index] - timestep);
        }
    }
    return result;
}

Tensor SamplingPrimitives::multiply_coefficient(const Tensor& value, float coefficient,
        PrecisionScalarRole role) {
    return precision_scalar_binary(backend_, policy_, role, ScalarBinaryOperation::Multiply,
                                   value, scalar_f32(coefficient));
}

Tensor SamplingPrimitives::linear_combine(const std::vector<Tensor>& values, const std::vector<float>& coefficients) {
    use("linear_combine"); require(!values.empty() && values.size() == coefficients.size(), "linear combine arity mismatch");
    Tensor result = multiply_coefficient(values[0], coefficients[0], PrecisionScalarRole::GuidanceCoefficient);
    for (size_t index = 1; index < values.size(); ++index)
        result = backend_.add(result, multiply_coefficient(values[index], coefficients[index], PrecisionScalarRole::GuidanceCoefficient));
    return result;
}

Tensor SamplingPrimitives::cfg_combine(const Tensor& unconditional, const Tensor& conditional, float scale) {
    use("cfg_combine"); return backend_.add(unconditional, multiply_coefficient(backend_.add(conditional, multiply_coefficient(unconditional, -1.0f, PrecisionScalarRole::GuidanceCoefficient)), scale, PrecisionScalarRole::GuidanceCoefficient));
}

Tensor SamplingPrimitives::euler_update(const Tensor& sample, const Tensor& prediction, const Tensor& delta, bool subtract_prediction) {
    use("euler_update");
    if (policy_.has_scalar_role(PrecisionScalarRole::SolverCoefficient)) {
        const Tensor coefficient = multiply_coefficient(delta, subtract_prediction ? -1.0f : 1.0f);
        return backend_.add(sample, precision_scalar_binary(backend_, policy_,
            PrecisionScalarRole::SolverCoefficient, ScalarBinaryOperation::Multiply, prediction, coefficient));
    }
    return backend_.add(sample, backend_.mul(prediction, backend_.mul(delta, scalar_f32(subtract_prediction ? -1.0f : 1.0f))));
}

Tensor SamplingPrimitives::tensor_where(const Tensor& condition, const Tensor& when_true, const Tensor& when_false) {
    use("tensor_where"); Tensor mask = backend_.cast(condition, backend_.execution_dtype());
    return backend_.add(backend_.mul(mask, when_true), backend_.mul(backend_.add(scalar_f32(1.0f), backend_.mul(mask, scalar_f32(-1.0f))), when_false));
}

Tensor SamplingPrimitives::flow_to_x0(const Tensor& sample, const Tensor& model_output, float sigma) {
    use("flow_to_x0"); return backend_.add(sample, multiply_coefficient(model_output, -sigma, PrecisionScalarRole::Sigma));
}

Tensor SamplingPrimitives::v_to_x0(const Tensor& sample, const Tensor& model_output,
                                   float alpha_cumprod) {
    use("v_to_x0");
    require(std::isfinite(alpha_cumprod) && alpha_cumprod >= 0.0f &&
                alpha_cumprod <= 1.0f,
            "V prediction alpha_cumprod is outside [0, 1]");
    const float signal = std::sqrt(alpha_cumprod);
    const float noise = std::sqrt(1.0f - alpha_cumprod);
    return backend_.add(multiply_coefficient(sample, signal),
                        multiply_coefficient(model_output, -noise));
}

Tensor SamplingPrimitives::epsilon_to_x0(
        const Tensor& sample, const Tensor& model_output,
        float alpha_cumprod) {
    use("epsilon_to_x0");
    require(std::isfinite(alpha_cumprod) && alpha_cumprod > 0.0f &&
                alpha_cumprod <= 1.0f,
            "Epsilon prediction alpha_cumprod is outside (0, 1]");
    const float signal = std::sqrt(alpha_cumprod);
    const float noise = std::sqrt(1.0f - alpha_cumprod);
    return multiply_coefficient(
        backend_.add(sample, multiply_coefficient(model_output, -noise)), 1.0f / signal);
}

Tensor SamplingPrimitives::affine_first_order(
        const Tensor& current_state, const Tensor& x0,
        float state_coefficient, float prediction_coefficient) {
    use("affine_first_order");
    require(std::isfinite(state_coefficient) &&
                std::isfinite(prediction_coefficient),
            "Affine first-order coefficients must be finite");
    return backend_.add(
        multiply_coefficient(current_state, state_coefficient),
        multiply_coefficient(x0, prediction_coefficient));
}

Tensor SamplingPrimitives::unipc_update(bool corrector, const Tensor& sample, const Tensor& this_sample,
                                        const std::vector<Tensor>& model_outputs, const Tensor& model_t,
                                        const std::vector<float>& sigmas, int step_index, int order) {
    (void)this_sample;
    require(order >= 1 && order <= 2, "Native UniPC supports order 1/2");
    const float sigma_t_raw = corrector ? sigmas[step_index] : sigmas[step_index + 1];
    const float sigma_s0_raw = corrector ? sigmas[step_index - 1] : sigmas[step_index];
    const float alpha_t = 1.0f - sigma_t_raw, alpha_s0 = 1.0f - sigma_s0_raw;
    const float lambda_t = std::log(alpha_t) - std::log(sigma_t_raw);
    const float lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0_raw);
    const float h = lambda_t - lambda_s0, hh = -h, h_phi_1 = std::expm1(hh), bh = std::expm1(hh);
    const Tensor& m0 = model_outputs.back();
    Tensor base = backend_.add(multiply_coefficient(sample, sigma_t_raw / sigma_s0_raw), multiply_coefficient(m0, -alpha_t * h_phi_1));
    if (corrector && order == 1) {
        Tensor residual = multiply_coefficient(backend_.add(model_t, multiply_coefficient(m0, -1)), 0.5f);
        return backend_.add(base, multiply_coefficient(residual, -alpha_t * bh));
    }
    if (!corrector && order == 1) return base;
    const int history_index = corrector ? step_index - 2 : step_index - 1;
    const float sigma_si = sigmas[history_index], alpha_si = 1.0f - sigma_si;
    const float rk = (std::log(alpha_si) - std::log(sigma_si) - lambda_s0) / h;
    Tensor difference = multiply_coefficient(backend_.add(model_outputs[model_outputs.size() - 2], multiply_coefficient(m0, -1)), 1.0f / rk);
    Tensor residual;
    const float phi2 = h_phi_1 / hh - 1.0f, b1 = phi2 / bh;
    // The frozen UniPC order-2 predictor uses an explicit midpoint
    // coefficient.  It is intentionally not the first
    // element of the corrector's linear system.
    if (!corrector) residual = multiply_coefficient(difference, 0.5f);
    else {
        const float phi3 = phi2 / hh - 0.5f, b2 = phi3 * 2.0f / bh;
        const float determinant = 1.0f - rk, c0 = (b1 - b2) / determinant, c1 = (b2 - b1 * rk) / determinant;
        residual = backend_.add(multiply_coefficient(difference, c0), multiply_coefficient(backend_.add(model_t, multiply_coefficient(m0, -1)), c1));
    }
    return backend_.add(base, multiply_coefficient(residual, -alpha_t * bh));
}

Tensor SamplingPrimitives::multistep_predictor(const Tensor& sample, const std::vector<Tensor>& model_outputs,
                                               const Tensor& model_t, const std::vector<float>& sigmas,
                                               int step_index, int order) {
    use("multistep_predictor"); return unipc_update(false, sample, sample, model_outputs, model_t, sigmas, step_index, order);
}

Tensor SamplingPrimitives::multistep_corrector(const Tensor& sample, const Tensor& this_sample,
                                               const std::vector<Tensor>& model_outputs, const Tensor& model_t,
                                               const std::vector<float>& sigmas, int step_index, int order,
                                               const Tensor& last_sample) {
    use("multistep_corrector"); (void)sample; return unipc_update(true, last_sample, this_sample, model_outputs, model_t, sigmas, step_index, order);
}
void SamplingPrimitives::state_advance() { use("state_advance"); }

SamplingResult SamplingRuntime::run(Denoiser& denoiser, const SamplingProgram& program) {
    return run(ExecutionContext::legacy(denoiser), ExecutionProgram::uniform({0}), program);
}

SamplingResult SamplingRuntime::run(const ExecutionContext& context,
        const ExecutionProgram& execution, const SamplingProgram& program) {
    return run_impl(context, execution, program, nullptr);
}

SamplingResult SamplingRuntime::run_with_initial_state(const ExecutionContext& context,
        const ExecutionProgram& execution, const SamplingProgram& program, const Tensor& initial_state) {
    return run_impl(context, execution, program, &initial_state);
}

SamplingResult SamplingRuntime::run_with_initial_state(
        Denoiser& denoiser, const SamplingProgram& program,
        const Tensor& initial_state) {
    return run_with_initial_state(ExecutionContext::legacy(denoiser),
                                  ExecutionProgram::uniform({0}), program, initial_state);
}

SamplingResult SamplingRuntime::run_with_external_initial_state_for_test(
        Denoiser& denoiser, const SamplingProgram& program,
        const Tensor& external_initial_state) {
    return run_with_initial_state(denoiser, program, external_initial_state);
}

SamplingResult SamplingRuntime::run_prefix(const ExecutionContext& context,
    const ExecutionProgram& execution, const SamplingProgram& program,
    int completed_step_limit) {
    require(completed_step_limit > 0 && completed_step_limit <= program.steps,
            "Invalid sampling prefix bound");
    return run_impl(context, execution, program, nullptr, completed_step_limit);
}

SamplingResult SamplingRuntime::run_impl(const ExecutionContext& context,
                                         const ExecutionProgram& execution,
                                         const SamplingProgram& program,
                                         const Tensor* external_initial_state,
                                         int completed_step_limit) {
    backend_.admit_execution_resources();
    constexpr int kMaximumDeclaredSamplingSteps = 10000;
    const auto started = std::chrono::steady_clock::now();
    require(program.steps >= 2 && program.steps <= kMaximumDeclaredSamplingSteps,
            "Sampling steps outside generic declared bound");
    require(!(cancellation_requested_ && cancellation_requested_()),
            "sampling cancelled");
    const SamplingContract* contract = program.contract ? &*program.contract : nullptr;
    std::vector<float> solver_sigmas;
    if (contract) {
        require(contract->schedule.size() == static_cast<size_t>(program.steps),
                "Sampling Contract v1 step count mismatch");
        if (contract->solver.semantic ==
                SolverSemantic::MultistepPredictorCorrector) {
            require(contract->prediction.semantic == PredictionSemantic::Flow,
                    "Multistep predictor/corrector requires FLOW prediction");
            require(contract->schedule.semantic() == ScheduleSemantic::FlowSigma,
                    "Multistep predictor/corrector requires FlowSigma schedule");
            require(contract->solver.maximum_order >= 1 &&
                        contract->solver.maximum_order <= 2,
                    "Multistep predictor/corrector order is unsupported");
            solver_sigmas.reserve(static_cast<size_t>(program.steps + 1));
            for (int step = 0; step < program.steps; ++step)
                solver_sigmas.push_back(
                    contract->schedule.flow_at(static_cast<size_t>(step)).sigma);
            solver_sigmas.push_back(contract->schedule.flow_at(
                static_cast<size_t>(program.steps - 1)).next_sigma);
        } else {
            require(contract->solver.semantic == SolverSemantic::AffineFirstOrder,
                    "Sampling Contract v1 solver semantic is unsupported");
            require(contract->prediction.semantic == PredictionSemantic::V ||
                        contract->prediction.semantic ==
                            PredictionSemantic::Epsilon,
                    "Affine first-order solver requires V or Epsilon prediction");
            require(contract->schedule.semantic() == ScheduleSemantic::AlphaCumprod,
                    "Affine first-order solver requires AlphaCumprod schedule");
            require(contract->solver.maximum_order == 1,
                    "Affine first-order solver order must be one");
        }
    } else {
        require(program.model_timesteps.size() == static_cast<size_t>(program.steps),
                "Sampling timestep count mismatch");
        require(program.update_deltas.size() == static_cast<size_t>(program.steps),
                "Euler update count mismatch");
    }
    const auto selected = execution.admit(context, program, effective_sampling_state_dtype(policy_));
    // Publish opaque backing leases before RNG, transfer, or device evaluation.
    // The Backend session retains them even after the execution context dies.
    std::vector<std::shared_ptr<const void>> owners;
    for(const auto& instance:context.instances())
        owners.insert(owners.end(),instance.resource_owners().begin(),instance.resource_owners().end());
    backend_.retain_resource_owners(owners);
    try {
        auto trace_tensor = [&](const Tensor& tensor) {
            return tensor.device().is_host() ? tensor : backend_.copy_to_host(tensor);
        };
        RngState rng{program.seed, 0, "pytorch_compat.v1"};
        const DType state_dtype = effective_sampling_state_dtype(policy_);
        Tensor latent;
        if (external_initial_state) {
            require(external_initial_state->defined(), "External initial state is undefined");
            require(external_initial_state->shape() == program.latent_shape,
                    "External initial state shape mismatch");
            require(external_initial_state->dtype() == state_dtype,
                    "External initial state dtype mismatch");
            latent = external_initial_state->device().is_host()
                ? backend_.copy_to_device(*external_initial_state, state_dtype)
                : *external_initial_state;
        } else {
            latent = primitives_.rng_normal(rng, program.latent_shape, state_dtype);
        }
        SamplingResult result; result.initial_noise = latent; if (tensor_trace_enabled_) result.trace["initial_noise"] = trace_tensor(latent); result.rng_after_initialization = rng;
        std::vector<Tensor> model_outputs; Tensor last_sample; int lower_order = 0, previous_order = 1;
        const int execution_steps = completed_step_limit ? completed_step_limit : program.steps;
        for (int step = 0; step < execution_steps; ++step) {
            require(!(cancellation_requested_ && cancellation_requested_()),
                    "sampling cancelled");
            const Tensor& timestep = program.model_timestep_at(step);
            const ComponentInstance& instance = *selected.at(static_cast<size_t>(step));
            Denoiser& denoiser = instance.denoiser();
            const StepExecutionContext step_context{static_cast<size_t>(step), timestep, instance};
            const auto solver_prefix = solver_trace_enabled_
                ? "step." + std::to_string(step) + ".solver." : std::string{};
            if (tensor_trace_enabled_ && solver_trace_enabled_) {
                auto& trace = result.trace;
                // Invocation-local opaque identity: all steps in this result
                // belong to this one run, irrespective of selected instance.
                trace[solver_prefix+"state_id"] = scalar_i64(0);
                trace[solver_prefix+"step"] = scalar_i64(step);
                trace[solver_prefix+"instance_id"] = scalar_i64(instance.id().value);
                trace[solver_prefix+"timestep"] = trace_tensor(timestep);
                trace[solver_prefix+"rng_seed"] = scalar_i64(rng.seed);
                trace[solver_prefix+"rng_offset"] = scalar_i64(rng.offset);
                trace[solver_prefix+"history_before"] = scalar_i64(model_outputs.size());
                trace[solver_prefix+"warmup_before"] = scalar_i64(lower_order);
                trace[solver_prefix+"previous_order"] = scalar_i64(previous_order);
                for (size_t i = 0; i < model_outputs.size(); ++i)
                    trace[solver_prefix+"history_before."+std::to_string(i)] = trace_tensor(model_outputs[i]);
            }
            if (tensor_trace_enabled_) result.trace["step." + std::to_string(step) + ".input_latent"] =
                trace_tensor(latent);
            if (backend_.profiling_enabled()) backend_.synchronize();
            const auto denoiser_started = std::chrono::steady_clock::now();
            std::vector<Tensor> predictions;
            {
                BackendProfileRegion region(
                    backend_, "sampling.denoiser.step." + std::to_string(step));
                predictions = denoiser.evaluate_step(latent, step_context);
                validate_component_predictions(instance, predictions);
                const DType output_dtype = effective_denoiser_output_dtype(policy_);
                for (Tensor& prediction : predictions)
                    prediction = backend_.cast(prediction, output_dtype);
            }
            for (auto& [name, tensor] : denoiser.take_trace())
                if (tensor_trace_enabled_) result.trace["step." + std::to_string(step) + "." + name] = trace_tensor(tensor);
            if (backend_.profiling_enabled()) backend_.synchronize();
            result.denoiser_call_seconds.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - denoiser_started).count());
            const auto scheduler_started = std::chrono::steady_clock::now();
            Tensor guided, next;
            {
                BackendProfileRegion region(
                    backend_, "sampling.scheduler.step." + std::to_string(step));
                const GuidanceParameters* guidance = program.guidance_schedule
                    ? &program.guidance_schedule->at(static_cast<size_t>(step)) : nullptr;
                const GuidanceMode mode = guidance ? guidance->mode : program.guidance_mode;
                if (tensor_trace_enabled_ && solver_trace_enabled_ && mode == GuidanceMode::CFG)
                    result.trace[solver_prefix+"guidance_scale"] = scalar_f32(
                        guidance ? guidance->scale : program.guidance_coefficients.at(1));
                if (mode == GuidanceMode::CFG) {
                    require(predictions.size() == 2, "CFG requires two predictions");
                    guided = primitives_.cfg_combine(predictions[0], predictions[1],
                        guidance ? guidance->scale : program.guidance_coefficients.at(1));
                } else guided = primitives_.linear_combine(predictions,
                    guidance ? guidance->coefficients : program.guidance_coefficients);
                for (size_t branch = 0; tensor_trace_enabled_ && branch < predictions.size(); ++branch) result.trace["step." + std::to_string(step) + ".prediction." + std::to_string(branch)] = trace_tensor(predictions[branch]);
                if (tensor_trace_enabled_) result.trace["step." + std::to_string(step) + ".guidance"] = trace_tensor(guided);
                if (!contract) next = primitives_.euler_update(latent, guided, scalar_f32(program.update_deltas.at(step)), program.subtract_prediction);
                else if (contract->solver.semantic ==
                            SolverSemantic::MultistepPredictorCorrector) {
                    const FlowScheduleTransition& transition =
                        contract->schedule.flow_at(static_cast<size_t>(step));
                    Tensor converted = primitives_.flow_to_x0(
                        latent, guided, transition.sigma);
                    if (step > 0 && last_sample.defined()) latent = primitives_.multistep_corrector(latent, latent, model_outputs, converted, solver_sigmas, step, previous_order, last_sample);
                    if (model_outputs.size() == 2) model_outputs.erase(model_outputs.begin());
                    model_outputs.push_back(converted);
                    const int maximum_order = contract->solver.maximum_order;
                    const int remaining = program.steps - step;
                    const int order = std::min(
                        std::min(maximum_order, remaining), lower_order + 1);
                    last_sample = latent; next = primitives_.multistep_predictor(latent, model_outputs, converted, solver_sigmas, step, order); previous_order = order; lower_order = std::min(maximum_order, lower_order + 1);
                    if (tensor_trace_enabled_ && solver_trace_enabled_) {
                        auto& trace = result.trace;
                        trace[solver_prefix+"sigma"] = scalar_f32(transition.sigma);
                        trace[solver_prefix+"next_sigma"] = scalar_f32(transition.next_sigma);
                        trace[solver_prefix+"converted"] = trace_tensor(converted);
                        trace[solver_prefix+"last_sample"] = trace_tensor(last_sample);
                        trace[solver_prefix+"order"] = scalar_i64(order);
                        trace[solver_prefix+"warmup_after"] = scalar_i64(lower_order);
                        trace[solver_prefix+"history_after"] = scalar_i64(model_outputs.size());
                        for (size_t i = 0; i < model_outputs.size(); ++i)
                            trace[solver_prefix+"history_after."+std::to_string(i)] = trace_tensor(model_outputs[i]);
                    }
                } else {
                    const AlphaCumprodScheduleTransition& transition =
                        contract->schedule.alpha_cumprod_at(static_cast<size_t>(step));
                    Tensor converted = contract->prediction.semantic ==
                            PredictionSemantic::Epsilon
                        ? primitives_.epsilon_to_x0(
                            latent, guided, transition.alpha_cumprod)
                        : primitives_.v_to_x0(
                            latent, guided, transition.alpha_cumprod);
                    if (tensor_trace_enabled_) result.trace["step." + std::to_string(step) +
                        ".predicted_x0"] = trace_tensor(converted);
                    const float state_coefficient = std::sqrt(
                        (1.0f - transition.previous_alpha_cumprod) /
                        (1.0f - transition.alpha_cumprod));
                    const float prediction_coefficient =
                        std::sqrt(transition.previous_alpha_cumprod) -
                        std::sqrt(transition.alpha_cumprod) * state_coefficient;
                    next = primitives_.affine_first_order(
                        latent, converted, state_coefficient, prediction_coefficient);
                }
                primitives_.state_advance(); if (tensor_trace_enabled_) result.trace["step." + std::to_string(step) + ".latent"] = trace_tensor(next); latent = next;
            }
            if (backend_.profiling_enabled()) backend_.synchronize();
            result.scheduler_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - scheduler_started).count();
            result.completed_steps = step + 1;
            if (step_observer_) step_observer_(step + 1, program.steps);
        }
        result.program_complete = result.completed_steps == program.steps;
        result.final_latent = latent; if (tensor_trace_enabled_) result.trace["final_latent"] = trace_tensor(latent);
        result.prepared_tensors = context.prepared_tensor_stats();
        result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); return result;
    } catch (...) {
        // Do not replace the primary failure or dispatch a fallback component.
        // If completion itself fails, the caller must retire the Backend session
        // before releasing the source-owner leases (see execution.h).
        try { backend_.synchronize(); } catch (...) {}
        throw;
    }
}

ScheduleContract make_scaled_linear_ddim_schedule(
        int training_timesteps, float beta_start, float beta_end,
        int inference_steps, int steps_offset, bool set_alpha_to_one) {
    require(training_timesteps >= 2 && inference_steps >= 2 &&
                inference_steps <= training_timesteps,
            "DDIM training/inference timestep count is invalid");
    require(std::isfinite(beta_start) && std::isfinite(beta_end) &&
                beta_start > 0.0f && beta_start < beta_end && beta_end < 1.0f,
            "DDIM scaled-linear beta bounds are invalid");
    require(training_timesteps % inference_steps == 0,
            "DDIM leading timestep spacing requires an integral step ratio");
    const float start = std::sqrt(beta_start);
    const float end = std::sqrt(beta_end);
    std::vector<float> cumulative(static_cast<size_t>(training_timesteps));
    float product = 1.0f;
    for (int index = 0; index < training_timesteps; ++index) {
        const float fraction = static_cast<float>(index) /
            static_cast<float>(training_timesteps - 1);
        const float root = start + (end - start) * fraction;
        product *= 1.0f - root * root;
        cumulative[static_cast<size_t>(index)] = product;
    }
    const int ratio = training_timesteps / inference_steps;
    std::vector<AlphaCumprodScheduleTransition> transitions;
    transitions.reserve(static_cast<size_t>(inference_steps));
    for (int step = inference_steps - 1; step >= 0; --step) {
        const int timestep = step * ratio + steps_offset;
        require(timestep >= 0 && timestep < training_timesteps,
                "DDIM timestep offset exceeds training schedule");
        const int previous = timestep - ratio;
        const float previous_alpha = previous >= 0
            ? cumulative[static_cast<size_t>(previous)]
            : (set_alpha_to_one ? 1.0f : cumulative.front());
        transitions.push_back(AlphaCumprodScheduleTransition{
            scalar_i64(timestep), cumulative[static_cast<size_t>(timestep)],
            previous_alpha});
    }
    return ScheduleContract::alpha_cumprod(std::move(transitions));
}

}  // namespace vrhino
