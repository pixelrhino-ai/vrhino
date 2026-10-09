#pragma once
#include "vrhino/architecture_binding.h"
#include "vrhino/bundle.h"
#include "vrhino/json.h"

namespace vrhino::ltx_internal {
struct TokenFlowDefinition {
    DType storage_dtype;
    std::shared_ptr<const ArchitectureBindingDeclaration> parameters;
};
// Exactly the admitted 0.9.1 topology, independent of the Wan canonical parser.
TokenFlowDefinition lower_token_flow(const Json &declaration);
Json token_flow_declaration(DType storage);
std::shared_ptr<const ArchitectureBindingDeclaration> decoder_slots(DType storage);
void validate_token_flow_inputs(const TensorBundle &inputs, bool conditioning);
} // namespace vrhino::ltx_internal
