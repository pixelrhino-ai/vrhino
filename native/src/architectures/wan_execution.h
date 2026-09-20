#pragma once
#include "wan_family.h"
#include "vrhino/architecture.h"

namespace vrhino::wan_family {
// Executor integration, separate from the pure lowering adapter. A realization
// borrows Backend/policy for the invocation; do not cache it across sessions.
class ExecutionDefinition {
public:
    const std::shared_ptr<const Definition> family;
    const std::shared_ptr<const ComponentGraphDefinition> graph;
    ComponentInstance bind(ComponentInstanceID instance, BindingID binding,
        std::shared_ptr<const AdmittedArchitectureBinding> parameters) const;
private:
    ExecutionDefinition(std::shared_ptr<const Definition> definition,
        std::shared_ptr<const ComponentGraphDefinition> executable)
        : family(std::move(definition)), graph(std::move(executable)) {}
    friend ExecutionDefinition realize(std::shared_ptr<const Definition>, Backend&, const PrecisionPolicy&,
        const std::vector<int64_t>&, const Tensor&, const Tensor&, bool, int, int);
};
ExecutionDefinition realize(std::shared_ptr<const Definition> definition,
    Backend& backend, const PrecisionPolicy& policy, const std::vector<int64_t>& latent_shape,
    const Tensor& positive, const Tensor& negative,
    bool trace_enabled = false, int trace_step = -1, int trace_block = -1);
std::unique_ptr<Denoiser> bound_denoiser(std::shared_ptr<const Definition> definition,
    std::shared_ptr<const AdmittedArchitectureBinding> binding,
    Backend& backend, const PrecisionPolicy& policy, const std::vector<int64_t>& latent_shape,
    const Tensor& positive, const Tensor& negative,
    bool trace_enabled = false, int trace_step = -1, int trace_block = -1);
}  // namespace vrhino::wan_family
