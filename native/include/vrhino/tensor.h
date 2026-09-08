#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vrhino {

// Values through Bool are stable in the tensor-bundle ABI. New storage-only
// dtypes are appended so existing serialized bundles remain unchanged.
enum class DType : uint8_t { F32, F16, BF16, I64, I32, U8, Bool, I8 };
enum class DeviceType : uint8_t { CPU, Accelerator };
using Device = DeviceType;
enum class MemoryDomain : uint8_t { HostPageable, HostStaging, DeviceLocal, Unified };

struct DeviceId {
    DeviceType type = DeviceType::CPU;
    int32_t index = 0;

    static constexpr DeviceId host() { return {DeviceType::CPU, 0}; }
    static constexpr DeviceId accelerator(int32_t index = 0) {
        return {DeviceType::Accelerator, index};
    }
    constexpr bool is_host() const { return type == DeviceType::CPU; }
    constexpr bool operator==(const DeviceId& other) const {
        return type == other.type && index == other.index;
    }
    constexpr bool operator!=(const DeviceId& other) const { return !(*this == other); }
    constexpr bool operator==(DeviceType other) const { return type == other; }
    constexpr bool operator!=(DeviceType other) const { return type != other; }
};
enum class QuantType : uint8_t { None, FP8E4M3FN, INT8Symmetric, INT4Symmetric };
enum class DequantizationSemantics : uint8_t {
    ScaleThenCast,
    ScaleThenInversePreconditioner,
};
enum class PreconditionerType : uint8_t { None, RegularBlockHadamardV1 };

struct QuantizationInfo;

size_t dtype_size(DType dtype);
std::string dtype_name(DType dtype);
DType dtype_from_vrm(const std::string& code);

struct Storage {
    using Deleter = void (*)(void*);
    void* data = nullptr;
    size_t bytes = 0;
    DeviceId device = DeviceId::host();
    MemoryDomain domain = MemoryDomain::HostPageable;
    bool owner = false;
    Deleter deleter = nullptr;
    // Retains backend-native allocations without exposing their concrete type.
    // Pointer-owning backends may use data+deleter; handle-owning backends can
    // retain their allocation object here.
    std::shared_ptr<void> native_owner;
    ~Storage();
};

class Tensor {
public:
    Tensor() = default;
    Tensor(std::shared_ptr<Storage> storage, size_t byte_offset,
           std::vector<int64_t> shape, DType dtype, size_t byte_length = 0);

    static Tensor borrowed(void* data, size_t bytes, std::vector<int64_t> shape, DType dtype);
    static Tensor host(std::vector<int64_t> shape, DType dtype);

    bool defined() const { return storage_ != nullptr; }
    void* data() const;
    template <typename T> T* data_as() const { return static_cast<T*>(data()); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    DType logical_dtype() const;
    bool is_quantized() const { return quantization_ != nullptr; }
    const QuantizationInfo& quantization() const;
    void set_quantization(std::shared_ptr<const QuantizationInfo> info);
    DeviceId device() const;
    MemoryDomain memory_domain() const;
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;
    size_t bytes() const;
    size_t storage_bytes() const { return storage_ ? storage_->bytes : 0; }
    size_t byte_offset() const { return byte_offset_; }
    bool owns_storage() const { return storage_ && storage_->owner; }
    bool is_view() const { return storage_ && (byte_offset_ != 0 || bytes() != storage_->bytes); }
    size_t alignment() const;
    int64_t dim(int64_t index) const;
    Tensor reshape(const std::vector<int64_t>& shape) const;

private:
    std::shared_ptr<Storage> storage_;
    size_t byte_offset_ = 0;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::F32;
    size_t byte_length_ = 0;
    std::shared_ptr<const QuantizationInfo> quantization_;
};

struct QuantizationInfo {
    QuantType type = QuantType::None;
    DType logical_dtype = DType::BF16;
    DType compute_dtype = DType::BF16;
    DType accumulation_dtype = DType::F32;
    int64_t group_size = 0;
    int64_t block_size = 0;
    int64_t axis = -1;
    bool symmetric = true;
    std::string granularity;
    std::string zero_point_mode;
    std::string packing_layout;
    DequantizationSemantics dequantization = DequantizationSemantics::ScaleThenCast;
    // Empty for the legacy quantization contract. Executable, versioned
    // schemes use an exact bounded identity and fail closed in the loader.
    std::string scheme;
    std::vector<int64_t> stored_shape;
    PreconditionerType preconditioner = PreconditionerType::None;
    int64_t preconditioner_axis = -1;
    int64_t preconditioner_group_size = 0;
    Tensor scales;
};

std::string quant_type_name(QuantType type);

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);
int64_t shape_numel(const std::vector<int64_t>& shape);

}  // namespace vrhino
