#pragma once
#include "vrhino/product/component_preparation.h"

namespace vrhino::product {
enum class ProductExecutionPhase { Conditioning, Sampling, Decode };

// Results own host storage and survive every phase's device/cache teardown.
struct PreparedProductExecution {
    TensorBundle conditioning;
    SamplingResult sampling;
    Tensor video;
    Json sampling_admission;
    bool shared_graph_identity = false;
    std::map<std::string, Json> resources;
    std::map<std::string, uint64_t> primitive_calls;
};

// Product controls and read-only observations, not model execution semantics.
// No declaration/precision override or external initial-state injection.
struct PreparedExecutionControl {
    std::optional<int> qualification_prefix_steps;
    bool solver_trace = false;
    bool tensor_trace = true;
    std::function<bool()> cancellation_requested;
    std::function<void(int, int, ComponentInstanceID, BindingID, const Json&)> step_completed;
    std::function<void(ProductExecutionPhase, const PreparedProductExecution&)> phase_completed;
};
using ProductBackendFactory = std::function<std::unique_ptr<Backend>()>;

// Shared numerical orchestration for admitted/prepared Product requests.
// Not an authorization API: ordinary run must first enforce numerical admission;
// explicit bounded qualification may call it without promoting that status.
PreparedProductExecution execute_prepared_product(const PreparedTextProductRequest&,
    const PrecisionPolicy&, const ProductBackendFactory&, const PreparedExecutionControl& = {});
} // namespace vrhino::product
