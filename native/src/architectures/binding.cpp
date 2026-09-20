#include "vrhino/architecture_binding.h"
#include "vrhino/error.h"
#include <limits>
#include <set>

namespace vrhino {
namespace {
size_t dense_bytes(const std::vector<int64_t>& shape, DType dtype) {
    for (auto extent : shape) require(extent > 0, "Architecture tensor extent must be positive");
    const auto count = static_cast<uint64_t>(shape_numel(shape));
    require(count <= std::numeric_limits<size_t>::max() / dtype_size(dtype), "Architecture tensor byte overflow");
    return static_cast<size_t>(count) * dtype_size(dtype);
}
}

ArchitectureBindingDeclaration::ArchitectureBindingDeclaration(std::vector<ArchitectureParameterSlot> slots)
    : slots_(std::move(slots)) {
    require(!slots_.empty() && slots_.size() <= 32768, "Architecture parameter slot count outside bound");
    std::set<std::string> roles;
    for (const auto& slot : slots_) {
        require(!slot.role.empty() && roles.insert(slot.role).second, "Duplicate or empty architecture parameter role");
        require(slot.layout == ParameterLayout::Contiguous, "Unsupported architecture parameter layout");
        (void)dense_bytes(slot.shape, slot.dtype);
    }
}

void validate_architecture_tensor(const Tensor& tensor, const std::vector<int64_t>& shape, DType dtype) {
    const size_t bytes = dense_bytes(shape, dtype);
    require(tensor.defined() && !tensor.is_quantized() && tensor.dtype() == dtype &&
        tensor.logical_dtype() == dtype && tensor.shape() == shape, "Architecture binding shape/dtype mismatch");
    require(tensor.device() == DeviceId::host() &&
        (tensor.memory_domain() == MemoryDomain::HostPageable || tensor.memory_domain() == MemoryDomain::HostStaging ||
         tensor.memory_domain() == MemoryDomain::Unified), "Architecture binding requires canonical host storage");
    require(tensor.strides() == contiguous_strides(shape) && tensor.bytes() == bytes &&
        tensor.byte_offset() <= tensor.storage_bytes() && bytes <= tensor.storage_bytes() - tensor.byte_offset() &&
        tensor.data() != nullptr, "Architecture binding layout/storage extent mismatch");
}

std::shared_ptr<const AdmittedArchitectureBinding> AdmittedArchitectureBinding::admit(
    std::shared_ptr<const ArchitectureBindingDeclaration> declaration,
    const std::map<std::string, const Tensor*>& tensors, BorrowedBindingLifetime lifetime,
    std::vector<std::shared_ptr<const void>> owners) {
    require(declaration != nullptr, "Missing architecture binding declaration");
    require(lifetime == BorrowedBindingLifetime::CallerRetained || lifetime == BorrowedBindingLifetime::ExplicitOwners,
            "Invalid architecture binding lifetime");
    for (const auto& owner : owners) require(owner != nullptr, "Null architecture binding owner");
    require(tensors.size() == declaration->slots().size(), "Architecture parameter slot set mismatch");
    auto binding = std::shared_ptr<AdmittedArchitectureBinding>(new AdmittedArchitectureBinding);
    binding->owners_ = std::move(owners);
    binding->declaration_ = std::move(declaration);
    for (const auto& slot : binding->declaration_->slots()) {
        const auto found = tensors.find(slot.role);
        require(found != tensors.end() && found->second != nullptr, "Missing architecture parameter: " + slot.role);
        const auto& t = *found->second;
        validate_architecture_tensor(t, slot.shape, slot.dtype);
        require(t.owns_storage() || lifetime == BorrowedBindingLifetime::CallerRetained || !binding->owners_.empty(),
                "Borrowed architecture parameter needs an owner lease");
        binding->parameters_.push_back(t);
    }
    return binding;
}

const std::vector<Tensor>& AdmittedArchitectureBinding::parameters_for(const ArchitectureBindingDeclaration& expected) const {
    require(&expected == declaration_.get(), "Binding belongs to a different graph declaration");
    // Tensor aliases are not deeply immutable. Revalidate at the handoff.
    for (size_t i = 0; i < parameters_.size(); ++i)
        validate_architecture_tensor(parameters_[i], expected.slots()[i].shape, expected.slots()[i].dtype);
    return parameters_;
}

std::map<std::string, const Tensor*> AdmittedArchitectureBinding::weights() const {
    (void)parameters_for(*declaration_);
    std::map<std::string, const Tensor*> result;
    for (size_t i = 0; i < parameters_.size(); ++i) result.emplace(declaration_->slots()[i].role, &parameters_[i]);
    return result;
}
}  // namespace vrhino
