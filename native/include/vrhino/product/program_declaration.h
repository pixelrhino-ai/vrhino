#pragma once
#include "vrhino/package_declaration.h"

namespace vrhino::product {
struct DeclaredPrograms {
    ExecutionProgram execution;
    SamplingProgram sampling;
};
// Closed wire declaration -> B-3 in-memory contracts. No numerical execution,
// model identity, upstream paths or resource policy belongs to this boundary.
DeclaredPrograms admit_program_declaration(const Json& declaration,
    const PackageDeclaration& package,
    const std::vector<std::string>& supported_capabilities = {
        "execution.per_step.v1", "sampling.flow_sigma_cfg.v1"});
} // namespace vrhino::product
