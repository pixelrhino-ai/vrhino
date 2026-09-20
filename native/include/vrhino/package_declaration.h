#pragma once
#include "vrhino/architecture_binding.h"
#include "vrhino/execution.h"
#include "vrhino/json.h"

namespace vrhino {
class VrmModel;
struct PackageBindingDeclaration {
    uint32_t graph;
    std::map<std::string,std::string> parameters;
};
struct PackageInstanceDeclaration { uint32_t graph, binding; };
struct LoweredPackageGraph {
    std::shared_ptr<const ArchitectureBindingDeclaration> slots;
    std::shared_ptr<const ComponentGraphDefinition> executable;
};
// Supplied by Architecture, never selected by source names or package policy.
using PackageGraphLowerer = std::function<LoweredPackageGraph(const Json&)>;

class AdmittedPackage {
public:
    // Creates endpoints only after every binding has passed admission. This
    // does not select an instance, generate timesteps, or execute numerics.
    ExecutionContext create_context() const;
    size_t graph_count() const { return graphs_.size(); }
    size_t binding_count() const { return bindings_.size(); }
private:
    std::shared_ptr<const VrmModel> owner_;
    std::map<uint32_t,LoweredPackageGraph> graphs_;
    std::map<uint32_t,std::shared_ptr<const AdmittedArchitectureBinding>> bindings_;
    std::map<uint32_t,PackageInstanceDeclaration> instances_;
    friend class PackageDeclaration;
};

// VRM graph JSON v2: a closed inventory of what exists. No selection predicates,
// timesteps, guidance, upstream filenames, memory or precision policies.
// Graph declarations are opaque here and MUST be validated by Architecture.
class PackageDeclaration {
public:
    static PackageDeclaration parse(const Json& graph);
    void validate_tensor_references(const VrmModel& model) const;
    AdmittedPackage admit(std::shared_ptr<const VrmModel> model,
                          const PackageGraphLowerer& lower) const;
    const auto& graphs() const { return graphs_; }
    const auto& bindings() const { return bindings_; }
    const auto& instances() const { return instances_; }
private:
    PackageDeclaration() = default;
    std::map<uint32_t,Json> graphs_;
    std::map<uint32_t,PackageBindingDeclaration> bindings_;
    std::map<uint32_t,PackageInstanceDeclaration> instances_;
};
} // namespace vrhino
