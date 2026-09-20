#include "vrhino/package_declaration.h"
#include "vrhino/loader.h"
#include "vrhino/error.h"

namespace vrhino {
AdmittedPackage PackageDeclaration::admit(std::shared_ptr<const VrmModel> model,
                                         const PackageGraphLowerer& lower) const {
    require(model!=nullptr && static_cast<bool>(lower),"Missing package admission dependency");
    validate_tensor_references(*model);
    AdmittedPackage result; result.owner_=std::move(model); result.instances_=instances_;
    for (const auto& [id,declaration]:graphs_) {
        auto g=lower(declaration);
        require(g.slots && g.executable,"Architecture did not supply a complete graph contract");
        const auto& slots=g.slots->slots(); const auto& parameters=g.executable->parameters();
        require(slots.size()==parameters.size(),"Architecture/executor parameter count mismatch");
        for (size_t n=0;n<slots.size();++n)
            require(slots[n].shape==parameters[n].shape && slots[n].dtype==parameters[n].dtype,
                    "Architecture/executor slot contract mismatch");
        result.graphs_.emplace(id,std::move(g));
    }
    for (const auto& [id,binding]:bindings_) {
        std::map<std::string,const Tensor*> tensors;
        for (const auto& [slot,name]:binding.parameters) tensors.emplace(slot,&result.owner_->tensor(name));
        result.bindings_.emplace(id,AdmittedArchitectureBinding::admit(
            result.graphs_.at(binding.graph).slots,tensors,BorrowedBindingLifetime::ExplicitOwners,{result.owner_}));
    }
    return result;
}
ExecutionContext AdmittedPackage::create_context() const {
    std::vector<ComponentInstance> instances;
    for (const auto& [id,instance]:instances_) {
        const auto& graph=graphs_.at(instance.graph);
        instances.push_back(ComponentInstance::bind({id},{instance.binding},graph.executable,
            bindings_.at(instance.binding)->parameters_for(*graph.slots),{owner_}));
    }
    return ExecutionContext(std::move(instances));
}
} // namespace vrhino
