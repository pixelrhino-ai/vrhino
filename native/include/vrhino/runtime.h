#pragma once

#include <map>
#include <functional>
#include <utility>

#include "vrhino/architecture.h"
#include "vrhino/component_tiling.h"
#include "vrhino/precision.h"

namespace vrhino {

struct RuntimeResult {
    TensorBundle outputs;
    double sampling_seconds = 0.0;
    double decode_seconds = 0.0;
    double execution_seconds = 0.0;
    size_t peak_device_bytes = 0;
    size_t upload_bytes = 0;
    double upload_seconds = 0.0;
    std::map<std::string, uint64_t> sampling_primitive_calls;
    std::vector<double> denoiser_call_seconds;
    double scheduler_seconds = 0.0;
    PreparedTensorCacheStats prepared_tensors;
    ComponentExecutionStats component_execution;
};

class NativeRuntime {
public:
    explicit NativeRuntime(Backend& backend)
        : NativeRuntime(backend,
            PrecisionPolicy::unqualified_default(backend.execution_dtype())) {}
    NativeRuntime(Backend& backend, PrecisionPolicy policy,
                  ComponentExecutionConfig component_execution = {})
        : backend_(backend), policy_(policy),
          component_execution_(std::move(component_execution)) {
        component_execution_.validate();
    }
    RuntimeResult execute(Architecture& architecture, const TensorBundle& input);
    // Product/UI observation only; the callback executes after each completed
    // sampling transition and does not participate in numerical semantics.
    void set_sampling_step_observer(std::function<void(int, int)> observer) {
        sampling_step_observer_ = std::move(observer);
    }
    // Shared production Component decode entry used by full execution and
    // bounded decode-only qualification. It owns tiled/untiled selection.
    Tensor decode_component(Architecture& architecture, const Tensor& latent,
                            const TensorBundle& input);
    const ComponentExecutionStats& component_execution_stats() const {
        return component_execution_stats_;
    }
    // Controlled validation only. Production execute() retains its declared
    // RNG initialization contract.
    RuntimeResult execute_with_external_initial_state_for_test(
        Architecture& architecture, const TensorBundle& input,
        const Tensor& external_initial_state);
private:
    RuntimeResult execute_impl(Architecture& architecture, const TensorBundle& input,
                               const Tensor* external_initial_state);
    Backend& backend_;
    PrecisionPolicy policy_;
    ComponentExecutionConfig component_execution_;
    ComponentExecutionStats component_execution_stats_;
    std::function<void(int, int)> sampling_step_observer_;
};

}  // namespace vrhino
