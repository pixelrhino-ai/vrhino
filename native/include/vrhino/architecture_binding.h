#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "vrhino/tensor.h"

namespace vrhino {

enum class ParameterLayout { Contiguous };
struct ArchitectureParameterSlot {
    std::string role;
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    ParameterLayout layout = ParameterLayout::Contiguous;
};

// Pure declaration/admission boundary. No Backend, allocation of tensor payloads,
// execution factory, selection, or precision overrides belong here.
class ArchitectureBindingDeclaration {
public:
    explicit ArchitectureBindingDeclaration(std::vector<ArchitectureParameterSlot> slots);
    const std::vector<ArchitectureParameterSlot>& slots() const { return slots_; }
private:
    std::vector<ArchitectureParameterSlot> slots_;
};

// Canonical input handles are dense host tensors. Device realization happens
// later at the existing executor's consumer boundary.
void validate_architecture_tensor(const Tensor& tensor,
    const std::vector<int64_t>& shape, DType dtype);

// CallerRetained is an explicit legacy non-owning contract: the caller must
// retain actual backing until Backend teardown, or register its actual owner
// with Backend::retain_resource_owners before use. A binding descriptor itself
// cannot manufacture ownership of a borrowed mapping. New managed bindings use
// ExplicitOwners; SamplingRuntime promotes those owners to the Backend session.
enum class BorrowedBindingLifetime { CallerRetained, ExplicitOwners };
class AdmittedArchitectureBinding {
public:
    static std::shared_ptr<const AdmittedArchitectureBinding> admit(
        std::shared_ptr<const ArchitectureBindingDeclaration> declaration,
        const std::map<std::string, const Tensor*>& tensors,
        BorrowedBindingLifetime lifetime,
        std::vector<std::shared_ptr<const void>> owners = {});
    const ArchitectureBindingDeclaration& declaration() const { return *declaration_; }
    const std::vector<Tensor>& parameters_for(const ArchitectureBindingDeclaration& expected) const;
    std::map<std::string, const Tensor*> weights() const;
private:
    // Endpoint may retain this object; generic execution promotes this binding's
    // owner lease into Backend lifetime. Direct endpoint users register owners
    // explicitly. Returned borrowed aliases must also stay inside that lifetime.
    std::vector<std::shared_ptr<const void>> owners_;
    std::shared_ptr<const ArchitectureBindingDeclaration> declaration_;
    std::vector<Tensor> parameters_;
};

}  // namespace vrhino
