#include "vrhino/package_declaration.h"
#include "vrhino/loader.h"
#include "vrhino/error.h"
#include <limits>

namespace vrhino {
namespace {
void keys(const Json& j,std::initializer_list<const char*> fields) {
    require(j.object().size()==fields.size(),"Package declaration field set mismatch");
    for (const auto* f:fields) require(j.find(f)!=nullptr,"Missing package declaration field");
}
uint32_t id(const Json& j) {
    const auto n=j.integer();
    require(n>=0 && static_cast<uint64_t>(n)<=std::numeric_limits<uint32_t>::max(),"Invalid package ID");
    return static_cast<uint32_t>(n);
}
}
PackageDeclaration PackageDeclaration::parse(const Json& root) {
    keys(root,{"schema_version","required_capabilities","graphs","bindings","instances"});
    require(root.at("schema_version").integer()==2,"Unsupported package graph schema");
    const auto& caps=root.at("required_capabilities").array();
    require(caps.size()==1 && caps.front().string()=="binding_catalog.v1",
            "Unsupported or missing required package capability");
    PackageDeclaration result;
    for (const auto& graph:root.at("graphs").array()) {
        keys(graph,{"id","declaration"});
        require(!graph.at("declaration").object().empty(),"Empty architecture declaration");
        require(result.graphs_.emplace(id(graph.at("id")),graph.at("declaration")).second,
                "Duplicate graph ID");
    }
    for (const auto& binding:root.at("bindings").array()) {
        keys(binding,{"id","graph","parameters"});
        PackageBindingDeclaration b{id(binding.at("graph")),{}};
        require(result.graphs_.contains(b.graph),"Unknown binding graph ID");
        for (const auto& [slot,tensor]:binding.at("parameters").object()) {
            require(!slot.empty() && !tensor.string().empty(),"Empty parameter reference");
            b.parameters.emplace(slot,tensor.string());
        }
        require(!b.parameters.empty(),"Empty binding declaration");
        require(result.bindings_.emplace(id(binding.at("id")),std::move(b)).second,
                "Duplicate binding ID");
    }
    for (const auto& instance:root.at("instances").array()) {
        keys(instance,{"id","graph","binding"});
        PackageInstanceDeclaration i{id(instance.at("graph")),id(instance.at("binding"))};
        require(result.graphs_.contains(i.graph),"Unknown instance graph ID");
        const auto b=result.bindings_.find(i.binding);
        require(b!=result.bindings_.end(),"Unknown instance binding ID");
        require(b->second.graph==i.graph,"Instance/binding graph mismatch");
        require(result.instances_.emplace(id(instance.at("id")),i).second,"Duplicate instance ID");
    }
    require(!result.graphs_.empty() && !result.bindings_.empty() && !result.instances_.empty(),
            "Empty package catalog");
    return result;
}
void PackageDeclaration::validate_tensor_references(const VrmModel& model) const {
    for (const auto& [id,binding]:bindings_) {
        (void)id;
        for (const auto& [slot,name]:binding.parameters) { (void)slot; (void)model.tensor(name); }
    }
}
} // namespace vrhino
