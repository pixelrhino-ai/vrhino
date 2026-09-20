#pragma once

#include <memory>
#include <variant>

#include "vrhino/sampling.h"

namespace vrhino {

struct ComponentInstanceID {
    uint32_t value = 0;
    bool operator==(const ComponentInstanceID&) const = default;
};
struct BindingID {
    uint32_t value = 0;
    bool operator==(const BindingID&) const = default;
};

struct ExecutionTensorContract {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    bool operator==(const ExecutionTensorContract&) const = default;
};

// Architecture-declared interface, not inferred from a model identity. Branch
// order is unconditional, conditional for CFG; positional for Linear.
struct ComponentInterface {
    ExecutionTensorContract latent, timestep;
    std::vector<ExecutionTensorContract> predictions;
    GuidanceMode guidance_mode = GuidanceMode::CFG;
    std::optional<PredictionSemantic> prediction;
    uint64_t conditioning_contract = 0;
    bool operator==(const ComponentInterface&) const = default;
};

// An architecture constructs this once and supplies a factory closing over its
// graph. Every instance uses the same factory with separately validated slots.
class ComponentGraphDefinition {
public:
    // Factory may retain Tensor handles/elements, but not a reference to the
    // vector object itself (the owning instance is move-constructed).
    using Factory = std::function<std::unique_ptr<Denoiser>(const std::vector<Tensor>&)>;
    ComponentGraphDefinition(ComponentInterface interface,
                             std::vector<ExecutionTensorContract> parameters, Factory factory);
    const ComponentInterface& interface() const { return interface_; }
    const std::vector<ExecutionTensorContract>& parameters() const { return parameters_; }
    std::unique_ptr<Denoiser> instantiate(const std::vector<Tensor>& parameters) const;
private:
    ComponentInterface interface_;
    std::vector<ExecutionTensorContract> parameters_;
    Factory factory_;
};

class ComponentInstance {
public:
    // Owners keep mappings alive for this instance. SamplingRuntime promotes
    // these explicit leases to the Backend session before execution.
    // Tensor copies alone do not retain VrmModel mappings. No hot rebinding.
    static ComponentInstance bind(ComponentInstanceID id, BindingID binding,
        std::shared_ptr<const ComponentGraphDefinition> graph, std::vector<Tensor> parameters,
        std::vector<std::shared_ptr<const void>> owners = {});
    ComponentInstance(ComponentInstance&&) noexcept = default;
    ComponentInstance& operator=(ComponentInstance&&) = delete;
    ComponentInstanceID id() const { return id_; }
    BindingID binding_id() const { return binding_; }
    const auto& resource_owners() const { return owners_; }
    const ComponentGraphDefinition* graph() const { return graph_.get(); }
    const std::vector<Tensor>& parameters() const { return parameters_; }
    Denoiser& denoiser() const { return *endpoint_; }
private:
    ComponentInstance() = default;
    ComponentInstanceID id_;
    BindingID binding_;
    // Reverse destruction: endpoint before tensors and backing owners.
    std::vector<std::shared_ptr<const void>> owners_;
    std::shared_ptr<const ComponentGraphDefinition> graph_;
    std::vector<Tensor> parameters_;
    std::unique_ptr<Denoiser> owned_endpoint_;
    Denoiser* endpoint_ = nullptr;
    friend class ExecutionContext;
};

// Scoped borrowed view: no backend, policy, RNG, solver state, or mutable binding.
struct StepExecutionContext {
    size_t step_index;
    const Tensor& timestep;
    const ComponentInstance& selected_component;
};

class ExecutionContext {
public:
    explicit ExecutionContext(std::vector<ComponentInstance> instances);
    static ExecutionContext legacy(Denoiser& endpoint);
    static ExecutionContext legacy(std::unique_ptr<Denoiser> endpoint);
    const std::vector<ComponentInstance>& instances() const { return instances_; }
    const ComponentInstance& at(ComponentInstanceID id) const;
    bool is_legacy() const { return legacy_; }
    PreparedTensorCacheStats prepared_tensor_stats() const;
private:
    ExecutionContext(std::vector<ComponentInstance> instances, bool legacy);
    std::vector<ComponentInstance> instances_;
    bool legacy_ = false;
};

enum class SelectionCoordinate { ModelTimestepScalar, FlowSigma };
enum class SelectionComparison { LessThan, GreaterOrEqual };
struct ScalarPartition {
    SelectionCoordinate coordinate = SelectionCoordinate::ModelTimestepScalar;
    SelectionComparison comparison = SelectionComparison::GreaterOrEqual;
    std::variant<int64_t, float> threshold;
    ComponentInstanceID when_true, when_false;
};

class ExecutionProgram {
public:
    static ExecutionProgram uniform(ComponentInstanceID id) { return ExecutionProgram(id); }
    static ExecutionProgram per_step(std::vector<ComponentInstanceID> ids) {
        return ExecutionProgram(std::move(ids));
    }
    static ExecutionProgram partition(ScalarPartition partition) {
        return ExecutionProgram(std::move(partition));
    }
    // Re-admitted for every run, before RNG or denoiser evaluation. The returned
    // references are scoped to that run; caller must not mutate any input alias.
    std::vector<const ComponentInstance*> admit(const ExecutionContext& context,
        const SamplingProgram& sampling, DType state_dtype) const;
private:
    using Selection = std::variant<ComponentInstanceID, std::vector<ComponentInstanceID>, ScalarPartition>;
    explicit ExecutionProgram(Selection selection) : selection_(std::move(selection)) {}
    Selection selection_;
};

struct ExecutionSetup {
    ExecutionContext context;
    ExecutionProgram program;
};

void validate_component_predictions(const ComponentInstance& instance,
                                    const std::vector<Tensor>& predictions);

}  // namespace vrhino
