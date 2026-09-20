#pragma once
#include "vrhino/json.h"

namespace vrhino {
// Closed, versioned VRhino declaration. Construction validates the wire
// contract; family lowering additionally validates realizable dimensions.
// No source names, aliases, paths, allocation, selection or precision policy.
class CanonicalArchitectureDeclaration {
public:
    static CanonicalArchitectureDeclaration parse(const Json& declaration);
    const Json& declaration() const { return declaration_; }
private:
    explicit CanonicalArchitectureDeclaration(Json declaration) : declaration_(std::move(declaration)) {}
    Json declaration_;
};

// VRM input compatibility only: never accepts a raw upstream config. New VRM
// descriptors carry canonical_architecture; legacy v1 descriptors are migrated
// here, before an Architecture sees them. This does not convert checkpoints.
CanonicalArchitectureDeclaration canonical_architecture_from_vrm_descriptor(const Json& descriptor);
}  // namespace vrhino
