#pragma once

#include <map>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

// Generic attention-mask consumption contract for Conditioning Component
// Graphs. Tokenizer mask production and output trimming remain independent
// graph/input semantics; None only means that Attention does not consume the
// tokenizer mask.
enum class MaskSemantic {
    None,
    Boolean,
    AdditiveFinite,
};

// Parse the optional attention.mask_semantic graph field. Absence preserves
// the schema-v1 historical behavior (Boolean). Unsupported values fail closed.
MaskSemantic conditioning_mask_semantic(const Json& attention_descriptor);

struct ConditioningComponentResult {
    Tensor hidden_states;
    Tensor pooled_output;
    std::map<std::string, Tensor> captures;
};

// Research-only observation request. It exposes generic component boundaries
// without changing graph semantics or the arithmetic selected by the Backend.
struct ConditioningObservationRequest {
    int64_t block_index = -1;
    std::vector<int64_t> output_block_indices;
};

// Model-name-free executor for declarative pre-norm Transformer component
// graphs. Architecture differences are data: weight bindings, normalization,
// attention position semantics, MLP gates, and output selection.
class ConditioningComponentExecutor {
public:
    ConditioningComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    ConditioningComponentResult execute(
        const Json& graph, const Tensor& input_ids, const Tensor& attention_mask,
        const ConditioningObservationRequest* observation = nullptr);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
