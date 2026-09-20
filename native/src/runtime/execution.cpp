#include "vrhino/execution.h"

#include <cmath>
#include <set>

#include "vrhino/error.h"

namespace vrhino {
namespace {

void validate_tensor(const Tensor& tensor, const ExecutionTensorContract& spec) {
    require(tensor.defined() && !tensor.is_quantized() && tensor.shape() == spec.shape &&
                tensor.dtype() == spec.dtype && tensor.data() != nullptr,
            "Execution tensor shape/dtype binding mismatch");
}

void validate_parameters(const ComponentGraphDefinition& graph,
                         const std::vector<Tensor>& parameters) {
    require(parameters.size() == graph.parameters().size(), "Execution parameter slot count mismatch");
    for (size_t i = 0; i < parameters.size(); ++i)
        validate_tensor(parameters[i], graph.parameters()[i]);
}

size_t branch_count(const SamplingProgram& program) {
    if (program.guidance_schedule) {
        const auto& guidance = program.guidance_schedule->at(0);
        return guidance.mode == GuidanceMode::CFG ? 2 : guidance.coefficients.size();
    }
    return program.guidance_mode == GuidanceMode::CFG ? 2 : program.guidance_coefficients.size();
}

ComponentInstanceID select(const ScalarPartition& partition, const SamplingProgram& program,
                           size_t step) {
    require(partition.comparison == SelectionComparison::LessThan ||
                partition.comparison == SelectionComparison::GreaterOrEqual,
            "Invalid execution comparison");
    bool less = false;
    if (partition.coordinate == SelectionCoordinate::FlowSigma) {
        require(program.contract && program.contract->schedule.semantic() == ScheduleSemantic::FlowSigma,
                "FlowSigma selection requires a FlowSigma schedule");
        require(std::holds_alternative<float>(partition.threshold) &&
                    std::isfinite(std::get<float>(partition.threshold)),
                "FlowSigma selection requires a finite float threshold");
        less = program.contract->schedule.flow_at(step).sigma < std::get<float>(partition.threshold);
    } else {
        require(partition.coordinate == SelectionCoordinate::ModelTimestepScalar,
                "Invalid execution coordinate");
        const Tensor& t = program.model_timestep_at(static_cast<int>(step));
        require(t.defined() && !t.is_quantized() && t.device().is_host() &&
                    t.numel() == 1 && t.data() != nullptr,
                "Execution selection requires a host scalar timestep");
        if (t.dtype() == DType::I64) {
            require(std::holds_alternative<int64_t>(partition.threshold), "Integer timestep threshold type mismatch");
            less = *t.data_as<int64_t>() < std::get<int64_t>(partition.threshold);
        } else {
            require(t.dtype() == DType::F32 && std::holds_alternative<float>(partition.threshold),
                    "Floating timestep threshold type mismatch");
            require(std::isfinite(*t.data_as<float>()) && std::isfinite(std::get<float>(partition.threshold)),
                    "Nonfinite execution selection coordinate");
            less = *t.data_as<float>() < std::get<float>(partition.threshold);
        }
    }
    const bool condition = partition.comparison == SelectionComparison::LessThan ? less : !less;
    return condition ? partition.when_true : partition.when_false;
}

}  // namespace

std::vector<Tensor> Denoiser::evaluate_step(const Tensor& latent, const StepExecutionContext& context) {
    return evaluate(latent, context.timestep);
}

void GuidanceSchedule::validate(size_t steps) const {
    require(!values_.empty() && values_.size() == (constant_ ? 1 : steps), "Guidance step table length mismatch");
    const auto& first = values_.front();
    for (const auto& value : values_) {
        require(value.mode == first.mode, "Guidance mode must remain constant");
        if (value.mode == GuidanceMode::CFG) {
            require(std::isfinite(value.scale) && value.coefficients.empty(), "Invalid CFG parameters");
        } else {
            require(value.mode == GuidanceMode::Linear && !value.coefficients.empty() &&
                        value.coefficients.size() == first.coefficients.size(), "Guidance branch count mismatch");
            for (float coefficient : value.coefficients)
                require(std::isfinite(coefficient), "Nonfinite guidance coefficient");
        }
    }
}

ComponentGraphDefinition::ComponentGraphDefinition(ComponentInterface component_contract,
        std::vector<ExecutionTensorContract> parameters, Factory factory)
    : interface_(std::move(component_contract)), parameters_(std::move(parameters)), factory_(std::move(factory)) {
    require(static_cast<bool>(factory_), "Missing component graph factory");
    require(!interface_.predictions.empty(), "Empty component prediction interface");
}

std::unique_ptr<Denoiser> ComponentGraphDefinition::instantiate(const std::vector<Tensor>& parameters) const {
    validate_parameters(*this, parameters);
    auto result = factory_(parameters);
    require(result != nullptr, "Component factory returned no endpoint");
    return result;
}

ComponentInstance ComponentInstance::bind(ComponentInstanceID id, BindingID binding,
        std::shared_ptr<const ComponentGraphDefinition> graph, std::vector<Tensor> parameters,
        std::vector<std::shared_ptr<const void>> owners) {
    require(graph != nullptr, "Missing component graph definition");
    for (const auto& owner : owners) require(owner != nullptr, "Null component resource owner");
    ComponentInstance instance;
    instance.id_ = id;
    instance.binding_ = binding;
    instance.owners_ = std::move(owners);
    instance.graph_ = std::move(graph);
    instance.parameters_ = std::move(parameters);
    instance.owned_endpoint_ = instance.graph_->instantiate(instance.parameters_);
    instance.endpoint_ = instance.owned_endpoint_.get();
    return instance;
}

ExecutionContext::ExecutionContext(std::vector<ComponentInstance> instances)
    : ExecutionContext(std::move(instances), false) {}

ExecutionContext::ExecutionContext(std::vector<ComponentInstance> instances, bool legacy)
    : instances_(std::move(instances)), legacy_(legacy) {
    require(!instances_.empty(), "Empty execution instance catalog");
    std::set<uint32_t> ids;
    std::map<std::pair<const ComponentGraphDefinition*, uint32_t>, const std::vector<Tensor>*> bindings;
    for (const auto& instance : instances_) {
        require(instance.endpoint_ != nullptr, "Missing component endpoint");
        require(ids.insert(instance.id().value).second, "Duplicate component instance ID");
        if (!legacy_) {
            require(instance.graph() != nullptr, "Untyped endpoint is only allowed in a legacy singleton");
            auto [found, inserted] = bindings.emplace(
                std::make_pair(instance.graph(), instance.binding_id().value), &instance.parameters());
            if (!inserted) {
                const auto& previous = *found->second;
                require(previous.size() == instance.parameters().size(), "Conflicting binding ID");
                for (size_t i = 0; i < previous.size(); ++i)
                    require(previous[i].data() == instance.parameters()[i].data() &&
                                previous[i].device() == instance.parameters()[i].device() &&
                                previous[i].shape() == instance.parameters()[i].shape() &&
                                previous[i].dtype() == instance.parameters()[i].dtype(),
                            "Conflicting binding ID within a graph definition");
            }
        }
    }
    require(!legacy_ || instances_.size() == 1, "Legacy context must be a singleton");
}

ExecutionContext ExecutionContext::legacy(Denoiser& endpoint) {
    ComponentInstance instance;
    instance.endpoint_ = &endpoint;
    std::vector<ComponentInstance> instances;
    instances.push_back(std::move(instance));
    return ExecutionContext(std::move(instances), true);
}

ExecutionContext ExecutionContext::legacy(std::unique_ptr<Denoiser> endpoint) {
    require(endpoint != nullptr, "Missing legacy denoiser");
    auto context = legacy(*endpoint);
    context.instances_[0].owned_endpoint_ = std::move(endpoint);
    return context;
}

const ComponentInstance& ExecutionContext::at(ComponentInstanceID id) const {
    for (const auto& instance : instances_) if (instance.id() == id) return instance;
    throw Error("Unknown component instance ID: " + std::to_string(id.value));
}

PreparedTensorCacheStats ExecutionContext::prepared_tensor_stats() const {
    PreparedTensorCacheStats result;
    for (const auto& instance : instances_) {
        const auto& stats = instance.denoiser().prepared_tensor_stats();
        result.prepare_hits += stats.prepare_hits;
        result.prepare_misses += stats.prepare_misses;
        result.reuses += stats.reuses;
        result.host_materializations += stats.host_materializations;
        result.host_to_device_transfers += stats.host_to_device_transfers;
        result.host_to_device_bytes += stats.host_to_device_bytes;
        result.resident_entries += stats.resident_entries;
        result.resident_tensors += stats.resident_tensors;
        result.resident_bytes += stats.resident_bytes;
    }
    return result;
}

std::vector<const ComponentInstance*> ExecutionProgram::admit(const ExecutionContext& context,
        const SamplingProgram& sampling, DType state_dtype) const {
    require(sampling.steps >= 2 && sampling.steps <= 10000, "Execution step count outside declared bound");
    const size_t steps = static_cast<size_t>(sampling.steps);
    require(sampling.contract ? sampling.contract->schedule.size() == steps
                              : sampling.model_timesteps.size() == steps,
            "Execution timestep table length mismatch");
    if (sampling.guidance_schedule) sampling.guidance_schedule->validate(steps);
    else {
        require(!sampling.guidance_coefficients.empty(), "Sampling guidance is empty");
        require(sampling.guidance_mode != GuidanceMode::CFG || sampling.guidance_coefficients.size() >= 2,
                "CFG requires a legacy scale coefficient");
    }
    // Validate the entire catalog, including presently unselected candidates.
    if (!context.is_legacy()) {
        const auto& expected = context.instances().front().graph()->component_interface();
        const auto mode = sampling.guidance_schedule ? sampling.guidance_schedule->at(0).mode : sampling.guidance_mode;
        for (const auto& instance : context.instances()) {
            const auto& component_contract = instance.graph()->component_interface();
            require(component_contract == expected, "Incompatible component interfaces");
            require(component_contract.latent.shape == sampling.latent_shape && component_contract.latent.dtype == state_dtype,
                    "Component latent shape/dtype contract mismatch");
            require(component_contract.guidance_mode == mode && component_contract.predictions.size() == branch_count(sampling),
                    "Component guidance branch contract mismatch");
            require(component_contract.prediction == (sampling.contract ? std::optional(sampling.contract->prediction.semantic)
                                                               : std::nullopt),
                    "Component prediction semantic mismatch");
            for (const auto& output : component_contract.predictions)
                require(output.shape == sampling.latent_shape &&
                            (output.dtype == DType::F32 || output.dtype == DType::BF16),
                        "Component prediction shape/dtype contract mismatch");
            validate_parameters(*instance.graph(), instance.parameters());
            for (size_t step = 0; step < steps; ++step)
                validate_tensor(sampling.model_timestep_at(static_cast<int>(step)), component_contract.timestep);
        }
    }
    std::vector<const ComponentInstance*> result;
    result.reserve(steps);
    if (const auto* uniform = std::get_if<ComponentInstanceID>(&selection_)) {
        result.assign(steps, &context.at(*uniform));
    } else if (const auto* table = std::get_if<std::vector<ComponentInstanceID>>(&selection_)) {
        require(table->size() == steps, "Execution step table length mismatch");
        for (auto id : *table) result.push_back(&context.at(id));
    } else {
        const auto& partition = std::get<ScalarPartition>(selection_);
        (void)context.at(partition.when_true);
        (void)context.at(partition.when_false);
        for (size_t step = 0; step < steps; ++step)
            result.push_back(&context.at(select(partition, sampling, step)));
    }
    return result;
}

void validate_component_predictions(const ComponentInstance& instance,
                                    const std::vector<Tensor>& predictions) {
    if (!instance.graph()) return;  // Existing legacy numerical interface.
    const auto& expected = instance.graph()->component_interface().predictions;
    require(predictions.size() == expected.size(), "Component returned wrong prediction count");
    for (size_t i = 0; i < predictions.size(); ++i) validate_tensor(predictions[i], expected[i]);
}

}  // namespace vrhino
