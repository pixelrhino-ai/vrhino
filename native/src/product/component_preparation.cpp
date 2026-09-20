#include "vrhino/product/component_preparation.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/architecture_binding.h"
#include <bit>
#include <cmath>
#include <cstring>

namespace vrhino::product {
void require_finite_component_output(const Tensor& value) {
    require(value.defined() && value.device().is_host() && value.numel()>0,
            "Component output must be nonempty host tensor");
    validate_architecture_tensor(value,value.shape(),value.dtype());
    require(value.dtype()==DType::F32 || value.dtype()==DType::BF16,
            "Unsupported component output dtype");
    for(int64_t i=0;i<value.numel();++i) {
        const float x=value.dtype()==DType::F32 ? value.data_as<float>()[i] :
            std::bit_cast<float>(uint32_t(value.data_as<uint16_t>()[i])<<16);
        require(std::isfinite(x),"Nonfinite component output");
    }
}

TensorBundle execute_text_conditioning(Backend& backend,const PreparedTextProductRequest& request) {
    require(request.owner!=nullptr && !request.conditioning.empty(),"Missing prepared component request");
    backend.retain_resource_owners({request.owner});
    backend.admit_execution_resources();
    std::vector<Tensor> hidden;
    try {
        ConditioningComponentExecutor executor(backend,request.conditioning_weights());
        for(const auto& input:request.conditioning) {
            auto result=executor.execute(input.graph,input.input_ids,input.attention_mask);
            auto value=result.hidden_states.device().is_host() ? result.hidden_states :
                backend.copy_to_host(result.hidden_states);
            backend.synchronize();
            require_finite_component_output(value);
            hidden.push_back(std::move(value));
        }
        return request.bind_conditioning_outputs(hidden);
    } catch (...) {
        backend.retire_resource_session();
        try { backend.synchronize(); } catch (...) {}
        throw;
    }
}

Json admit_conditioned_sampling(Backend& backend,const PrecisionPolicy& policy,
    const PreparedTextProductRequest& request,const TensorBundle& input) {
    require(request.owner && request.owner->architecture,"Missing product architecture");
    require(policy.requested_dtype()==backend.execution_dtype(),"Policy/backend dtype mismatch");
    require(input.size()==request.runtime_inputs.size()+request.conditioning.size(),
            "Unexpected conditioned request field set");
    for(const auto& [name, expected]:request.runtime_inputs) {
        const auto& actual=input.at(name);
        validate_architecture_tensor(actual,expected.shape(),expected.dtype());
        require(std::memcmp(actual.data(),expected.data(),expected.bytes())==0,
                "Prepared request intent cannot be changed during conditioning");
    }
    std::vector<Tensor> hidden;
    for(const auto& c:request.conditioning) {
        require_finite_component_output(input.at(c.target));
        hidden.push_back(input.at(c.target));
    }
    (void)request.bind_conditioning_outputs(hidden);
    backend.retain_resource_owners({request.owner});
    backend.admit_execution_resources();
    // The adapter constructs endpoints; only the admitted program selects them.
    // No denoiser evaluation, RNG advance, solver state or sampling loop here.
    auto setup=request.owner->architecture->create_execution_setup(backend,policy,input);
    const auto selected=request.execution.admit(setup.context,request.sampling,
        policy.persistent_state_dtype(PrecisionSemantic::SamplingState));
    require(selected.size()==static_cast<size_t>(request.sampling.steps),"Admitted step count mismatch");
    Json::Array trace;
    for(size_t i=0;i<selected.size();++i) {
        const auto* instance=selected[i];
        trace.emplace_back(Json::Object{{"step",Json(int64_t(i))},
            {"instance_id",Json(int64_t(instance->id().value))},
            {"binding_id",Json(int64_t(instance->binding_id().value))}});
    }
    backend.synchronize();
    return Json(Json::Object{{"admitted",Json(true)},{"denoiser_evaluated",Json(false)},
        {"solver_created",Json(false)},{"steps",Json(std::move(trace))}});
}
} // namespace vrhino::product
