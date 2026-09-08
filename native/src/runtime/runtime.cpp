#include "vrhino/runtime.h"

#include <chrono>

#include "vrhino/tensor_util.h"

namespace vrhino {

RuntimeResult NativeRuntime::execute(Architecture& architecture, const TensorBundle& input) {
    return execute_impl(architecture, input, nullptr);
}

RuntimeResult NativeRuntime::execute_with_external_initial_state_for_test(
        Architecture& architecture, const TensorBundle& input,
        const Tensor& external_initial_state) {
    return execute_impl(architecture, input, &external_initial_state);
}

Tensor NativeRuntime::decode_component(Architecture& architecture,
                                       const Tensor& latent,
                                       const TensorBundle& input) {
    ComponentGraphExecutor executor(backend_, component_execution_);
    Tensor output = executor.execute(latent, [&](const Tensor& tile) {
        return architecture.decode(backend_, policy_, tile, input);
    });
    component_execution_stats_ = executor.stats();
    return output;
}

RuntimeResult NativeRuntime::execute_impl(Architecture& architecture,
                                          const TensorBundle& input,
                                          const Tensor* external_initial_state) {
    const auto started = std::chrono::steady_clock::now();
    auto denoiser = architecture.create_denoiser(backend_, policy_, input);
    SamplingProgram program = architecture.create_program(input);
    SamplingRuntime sampling_runtime(backend_, policy_);
    sampling_runtime.set_step_observer(sampling_step_observer_);
    SamplingResult sampling;
    {
        BackendProfileRegion region(backend_, "runtime.sampling");
        sampling = external_initial_state
            ? sampling_runtime.run_with_external_initial_state_for_test(
                  *denoiser, program, *external_initial_state)
            : sampling_runtime.run(*denoiser, program);
    }
    const auto decode_started = std::chrono::steady_clock::now();
    Tensor video;
    {
        BackendProfileRegion region(backend_, "runtime.vae_decode");
        video = decode_component(architecture, sampling.final_latent, input);
    }
    TensorBundle component_trace = architecture.take_trace();
    backend_.synchronize();
    RuntimeResult result;
    for (const auto& [name, tensor] : sampling.trace)
        result.outputs.emplace(name, tensor.device().is_host() ? tensor
                                                               : backend_.copy_to_host(tensor));
    for (int step = 0; step < program.steps; ++step) {
        result.outputs.emplace("step." + std::to_string(step) + ".scheduler_output",
                               result.outputs.at("step." + std::to_string(step) + ".latent"));
        result.outputs.emplace("step." + std::to_string(step) + ".scheduler.step_index",
                               scalar_i64(step + 1));
        const Tensor& timestep = program.model_timestep_at(step);
        result.outputs.emplace("step." + std::to_string(step) + ".timestep",
            timestep.device() == Device::CPU ? timestep : backend_.copy_to_host(timestep));
    }
    result.outputs.emplace("video", backend_.copy_to_host(video));
    for (const auto& [name, tensor] : component_trace)
        result.outputs.emplace(name, backend_.copy_to_host(tensor));
    result.outputs.emplace("rng.offset", Tensor::host({}, DType::I64));
    result.outputs.at("rng.offset").data_as<int64_t>()[0] = sampling.rng_after_initialization.offset;
    result.outputs.emplace("rng.seed", scalar_i64(static_cast<int64_t>(program.seed)));
    result.outputs.emplace("test.external_initial_state",
                           scalar_i64(external_initial_state ? 1 : 0));
    result.sampling_seconds = sampling.elapsed_seconds;
    result.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_started).count();
    result.execution_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.peak_device_bytes = backend_.peak_device_bytes();
    result.upload_bytes = backend_.weight_upload_bytes();
    result.upload_seconds = backend_.weight_upload_seconds();
    result.sampling_primitive_calls = sampling_runtime.primitives().calls();
    result.denoiser_call_seconds = sampling.denoiser_call_seconds;
    result.scheduler_seconds = sampling.scheduler_seconds;
    result.prepared_tensors = sampling.prepared_tensors;
    result.component_execution = component_execution_stats_;
    result.outputs.emplace("metric.sampling_seconds", scalar_f32(static_cast<float>(result.sampling_seconds)));
    result.outputs.emplace("metric.decode_seconds", scalar_f32(static_cast<float>(result.decode_seconds)));
    result.outputs.emplace("metric.execution_seconds", scalar_f32(static_cast<float>(result.execution_seconds)));
    result.outputs.emplace("metric.weight_upload_seconds", scalar_f32(static_cast<float>(result.upload_seconds)));
    result.outputs.emplace("metric.weight_upload_bytes", scalar_i64(static_cast<int64_t>(result.upload_bytes)));
    result.outputs.emplace("metric.peak_device_bytes", scalar_i64(static_cast<int64_t>(result.peak_device_bytes)));
    return result;
}

}  // namespace vrhino
