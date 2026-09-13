#include "vrhino/tensor.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#include "vrhino/error.h"

namespace vrhino {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::I32: return 4;
        case DType::U8: return 1;
        case DType::Bool: return 1;
        case DType::I8: return 1;
    }
    throw Error("Unknown dtype");
}

std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::BF16: return "bfloat16";
        case DType::I64: return "int64";
        case DType::I32: return "int32";
        case DType::U8: return "uint8";
        case DType::Bool: return "bool";
        case DType::I8: return "int8";
    }
    throw Error("Unknown dtype");
}

DType dtype_from_vrm(const std::string& code) {
    if (code == "f32") return DType::F32;
    if (code == "f16") return DType::F16;
    if (code == "bf16") return DType::BF16;
    if (code == "i64") return DType::I64;
    if (code == "i32") return DType::I32;
    if (code == "u8") return DType::U8;
    if (code == "bool") return DType::Bool;
    if (code == "i8") return DType::I8;
    throw Error("Unsupported VRM dtype: " + code);
}

Storage::~Storage() {
    if (!owner || data == nullptr) return;
    if (native_owner) return;
    if (deleter) deleter(data);
    else std::free(data);
}

int64_t shape_numel(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (int64_t value : shape) {
        require(value >= 0, "Negative tensor dimension");
        require(value == 0 || result <= INT64_MAX / value, "Tensor numel overflow");
        result *= value;
    }
    return result;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> result(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        require(shape[index + 1] == 0 || result[index + 1] <= INT64_MAX / shape[index + 1],
                "Tensor stride overflow");
        result[index] = result[index + 1] * shape[index + 1];
    }
    return result;
}

size_t checked_dense_bytes(const std::vector<int64_t>& shape, const DType dtype) {
    const int64_t elements = shape_numel(shape);
    const size_t element_size = dtype_size(dtype);
    require(static_cast<uint64_t>(elements) <=
                std::numeric_limits<size_t>::max() / element_size,
            "Tensor byte size overflow");
    return static_cast<size_t>(elements) * element_size;
}

Tensor::Tensor(std::shared_ptr<Storage> storage, size_t byte_offset,
               std::vector<int64_t> shape, DType dtype, size_t byte_length)
    : storage_(std::move(storage)), byte_offset_(byte_offset), shape_(std::move(shape)),
      strides_(contiguous_strides(shape_)), dtype_(dtype),
      byte_length_(byte_length ? byte_length : checked_dense_bytes(shape_, dtype_)) {
    require(storage_ != nullptr, "Tensor storage is null");
    require(byte_offset_ <= storage_->bytes && bytes() <= storage_->bytes - byte_offset_,
            "Tensor exceeds storage");
}

Tensor Tensor::borrowed(void* data, size_t bytes, std::vector<int64_t> shape, DType dtype) {
    auto storage = std::make_shared<Storage>();
    storage->data = data; storage->bytes = bytes; storage->device = DeviceId::host();
    storage->domain = MemoryDomain::HostPageable; storage->owner = false;
    return Tensor(std::move(storage), 0, std::move(shape), dtype, bytes);
}

Tensor Tensor::host(std::vector<int64_t> shape, DType dtype) {
    auto storage = std::make_shared<Storage>();
    storage->bytes = checked_dense_bytes(shape, dtype);
    require(storage->bytes <= std::numeric_limits<size_t>::max() - 63,
            "Host tensor allocation size overflow");
#ifdef _MSC_VER
    // The MSVC allocator rejects zero bytes. Keep empty tensors logically
    // empty while providing a minimal aligned allocation for their storage.
    storage->data = _aligned_malloc(
        storage->bytes == 0 ? 64 : (storage->bytes + 63) / 64 * 64, 64);
    storage->deleter = _aligned_free;
#else
    storage->data = std::aligned_alloc(64, (storage->bytes + 63) / 64 * 64);
#endif
    require(storage->data != nullptr || storage->bytes == 0, "Host allocation failed");
    storage->device = DeviceId::host(); storage->domain = MemoryDomain::HostPageable;
    storage->owner = true;
    return Tensor(std::move(storage), 0, std::move(shape), dtype);
}

void* Tensor::data() const {
    require(defined(), "Undefined tensor");
    return static_cast<uint8_t*>(storage_->data) + byte_offset_;
}

DeviceId Tensor::device() const { require(defined(), "Undefined tensor"); return storage_->device; }
MemoryDomain Tensor::memory_domain() const {
    require(defined(), "Undefined tensor");
    return storage_->domain;
}
int64_t Tensor::numel() const { return shape_numel(shape_); }
size_t Tensor::bytes() const { return byte_length_; }
DType Tensor::logical_dtype() const { return quantization_ ? quantization_->logical_dtype : dtype_; }
const QuantizationInfo& Tensor::quantization() const {
    require(quantization_ != nullptr, "Tensor is not quantized");
    return *quantization_;
}
void Tensor::set_quantization(std::shared_ptr<const QuantizationInfo> info) {
    require(info && info->type != QuantType::None, "Invalid quantization metadata");
    require(dtype_ == DType::U8 || dtype_ == DType::I8,
            "Quantized storage must use an 8-bit integer dtype");
    if (info->scheme == "preconditioned_symmetric_int8.v1") {
        require(dtype_ == DType::I8 && info->type == QuantType::INT8Symmetric,
                "Preconditioned symmetric INT8 requires signed-I8 storage");
        require(info->preconditioner == PreconditionerType::RegularBlockHadamardV1,
                "Preconditioned symmetric INT8 requires its executable preconditioner");
    } else {
        require(info->scheme.empty() && dtype_ == DType::U8,
                "Unknown quantized tensor scheme or storage dtype");
    }
    require(info->scales.defined() && info->scales.dtype() == DType::F32,
            "Quantization scales must be float32");
    quantization_ = std::move(info);
}
size_t Tensor::alignment() const {
    if (!defined() || data() == nullptr) return 0;
    const uintptr_t address = reinterpret_cast<uintptr_t>(data());
    size_t result = 1;
    while (result < 4096 && address % (result * 2) == 0) result *= 2;
    return result;
}
int64_t Tensor::dim(int64_t index) const {
    if (index < 0) index += ndim();
    require(index >= 0 && index < ndim(), "Tensor dimension out of range");
    return shape_[index];
}

Tensor Tensor::reshape(const std::vector<int64_t>& requested) const {
    require(!is_quantized(), "Quantized tensors cannot be reshaped as dense tensors");
    std::vector<int64_t> resolved = requested;
    int64_t inferred = -1;
    int64_t known = 1;
    for (int64_t index = 0; index < static_cast<int64_t>(resolved.size()); ++index) {
        if (resolved[index] == -1) { require(inferred == -1, "Multiple inferred dimensions"); inferred = index; }
        else { require(resolved[index] >= 0, "Invalid reshape dimension"); known *= resolved[index]; }
    }
    if (inferred >= 0) { require(known != 0 && numel() % known == 0, "Invalid inferred reshape"); resolved[inferred] = numel() / known; }
    require(shape_numel(resolved) == numel(), "Reshape changes tensor numel");
    return Tensor(storage_, byte_offset_, std::move(resolved), dtype_);
}

std::string quant_type_name(QuantType type) {
    switch (type) {
        case QuantType::None: return "none";
        case QuantType::FP8E4M3FN: return "fp8_e4m3fn";
        case QuantType::INT8Symmetric: return "int8_symmetric";
        case QuantType::INT4Symmetric: return "int4_symmetric";
    }
    throw Error("Unknown quantization type");
}

}  // namespace vrhino
