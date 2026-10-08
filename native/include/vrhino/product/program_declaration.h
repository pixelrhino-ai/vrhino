#pragma once
#include "vrhino/package_declaration.h"
#include <filesystem>

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
        "execution.per_step.v1", "sampling.flow_sigma_cfg.v1", "sampling.flow_euler_cfg.v1"});

struct VerifiedDeclaredPrograms {
    DeclaredPrograms programs;
    Json evidence;
};
// Small frozen sidecar: mandatory size/SHA and equality with embedded programs.
// No execution eligibility or declaration override is granted by this function.
VerifiedDeclaredPrograms admit_verified_program_artifact(const std::filesystem::path& path,
    uint64_t bytes, const std::string& sha256, const VrmModel& model);
} // namespace vrhino::product
