#include "vrhino/product/prepared_execution.h"
#include "vrhino/architecture_binding.h"
#include "vrhino/error.h"
#include <cstring>

namespace vrhino::product {
namespace {
Tensor owned_host(Backend& backend, const Tensor& tensor) {
    require(tensor.defined(), "Undefined Product phase output");
    if (!tensor.device().is_host()) return backend.copy_to_host(tensor);
    validate_architecture_tensor(tensor, tensor.shape(), tensor.dtype());
    auto copy = Tensor::host(tensor.shape(), tensor.dtype());
    std::memcpy(copy.data(), tensor.data(), tensor.bytes());
    return copy;
}
Json resource_stats(Backend& backend) {
    const auto m = backend.memory_runtime_stats();
    return Json(Json::Object{{"peak_device_bytes", Json(int64_t(backend.peak_device_bytes()))},
        {"weight_upload_bytes", Json(int64_t(backend.weight_upload_bytes()))},
        {"cache_resident_bytes", Json(int64_t(backend.weight_cache_resident_bytes()))},
        {"cache_capacity_bytes", Json(int64_t(backend.weight_cache_capacity_bytes()))},
        {"cache_hits", Json(int64_t(m.cache_hits))}, {"cache_misses", Json(int64_t(m.cache_misses))},
        {"evictions", Json(int64_t(m.evictions))},
        {"unsafe_eviction_rejections", Json(int64_t(m.unsafe_eviction_rejections))},
        {"peak_resident_weight_bytes", Json(int64_t(m.accounting.peak_device_resident_weight_bytes))},
        {"source_owner_count", Json(int64_t(backend.resource_owner_count()))},
        {"total_device_memory_bound_guaranteed", Json(false)}});
}
}
PreparedProductExecution execute_prepared_product(const PreparedTextProductRequest& request,
    const PrecisionPolicy& policy, const ProductBackendFactory& factory,
    const PreparedExecutionControl& control) {
    require(request.owner && request.owner->architecture && bool(factory), "Missing prepared Product execution");
    const bool prefix = control.qualification_prefix_steps.has_value();
    const int limit = control.qualification_prefix_steps.value_or(request.sampling.steps);
    require(limit > 0 && (!prefix || limit < request.sampling.steps), "Invalid qualification prefix");
    const auto cancelled = [&] {
        if (control.cancellation_requested && control.cancellation_requested())
            throw ModelPackageError(ModelPackageErrorCode::Cancelled, "Product execution cancelled");
    };
    PreparedProductExecution output;
    const auto phase = [&](ProductExecutionPhase kind, const char* name, auto execute) {
        cancelled();
        {
            auto backend = factory();
            require(backend && backend->execution_dtype() == policy.requested_dtype(), "Product Backend/policy mismatch");
            backend->retain_resource_owners({request.owner});
            try {
                backend->admit_execution_resources();
                execute(*backend);
                backend->synchronize();
                output.resources.emplace(name, resource_stats(*backend));
            } catch (...) {
                backend->retire_resource_session();
                try { backend->synchronize(); } catch (...) {}
                throw;
            }
        }
        // Observers see completed host results after the device phase is gone.
        if (control.phase_completed) control.phase_completed(kind, output);
    };
    phase(ProductExecutionPhase::Conditioning, "conditioning", [&](Backend& backend) {
        output.conditioning = execute_text_conditioning(backend, request);
        for (auto& [name, tensor] : output.conditioning) tensor = owned_host(backend, tensor);
    });
    phase(ProductExecutionPhase::Sampling, "sampling", [&](Backend& backend) {
        output.sampling_admission = admit_conditioned_sampling(backend, policy, request, output.conditioning);
        auto setup = request.owner->architecture->create_execution_setup(backend, policy, output.conditioning);
        const auto selected = request.execution.admit(setup.context, request.sampling,
            policy.persistent_state_dtype(PrecisionSemantic::SamplingState));
        output.shared_graph_identity = true;
        for (const auto* instance : selected)
            output.shared_graph_identity &= instance->graph() == selected.front()->graph();
        SamplingRuntime runtime(backend, policy);
        runtime.set_solver_trace_enabled(control.solver_trace);
        runtime.set_tensor_trace_enabled(control.tensor_trace);
        runtime.set_cancellation_requested([&] { cancelled(); return false; });
        int observed = 0;
        runtime.set_step_observer([&](int completed, int total) {
            require(total == request.sampling.steps && completed == ++observed && completed <= limit,
                    "Product sampling completion mismatch");
            cancelled();
            if (control.step_completed) {
                const auto* instance = selected.at(static_cast<size_t>(completed - 1));
                control.step_completed(completed, total, instance->id(), instance->binding_id(), resource_stats(backend));
            }
        });
        output.sampling = prefix ? runtime.run_prefix(setup.context, request.execution, request.sampling, limit) :
            runtime.run(setup.context, request.execution, request.sampling);
        backend.synchronize();
        require(output.sampling.completed_steps == limit && observed == limit &&
                output.sampling.program_complete == !prefix, "Unexpected Product sampling completion");
        output.sampling.initial_noise = owned_host(backend, output.sampling.initial_noise);
        output.sampling.final_latent = owned_host(backend, output.sampling.final_latent);
        validate_architecture_tensor(output.sampling.final_latent, request.sampling.latent_shape,
            policy.persistent_state_dtype(PrecisionSemantic::SamplingState));
        require_finite_component_output(output.sampling.final_latent);
        for (auto& [name, tensor] : output.sampling.trace) {
            tensor = owned_host(backend, tensor);
            if (tensor.dtype() == DType::F32 || tensor.dtype() == DType::BF16) require_finite_component_output(tensor);
        }
        output.primitive_calls = runtime.primitives().calls();
    });
    if (!prefix) phase(ProductExecutionPhase::Decode, "decoder", [&](Backend& backend) {
        auto video = request.owner->architecture->decode(backend, policy, output.sampling.final_latent, request.runtime_inputs);
        output.video = owned_host(backend, video);
        backend.synchronize();
        require_finite_component_output(output.video);
        require(output.video.shape() == request.expected_video_shape, "Product decoder output shape mismatch");
    });
    return output;
}
} // namespace vrhino::product
