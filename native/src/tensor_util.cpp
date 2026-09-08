#include "vrhino/tensor_util.h"

#include <cstring>

#include "vrhino/error.h"

namespace vrhino {

template <typename T>
Tensor host_values(const std::vector<int64_t>& shape, DType dtype, const std::vector<T>& values) {
    require(shape_numel(shape) == static_cast<int64_t>(values.size()), "Host tensor value count mismatch");
    Tensor result = Tensor::host(shape, dtype);
    if (!values.empty()) std::memcpy(result.data(), values.data(), values.size() * sizeof(T));
    return result;
}

Tensor host_f32(const std::vector<int64_t>& shape, const std::vector<float>& values) { return host_values(shape, DType::F32, values); }
Tensor host_i64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) { return host_values(shape, DType::I64, values); }
Tensor host_bool(const std::vector<int64_t>& shape, const std::vector<uint8_t>& values) { return host_values(shape, DType::Bool, values); }
Tensor scalar_f32(float value) { return host_f32({}, {value}); }
Tensor scalar_i64(int64_t value) { return host_i64({}, {value}); }
float read_scalar_f32(const Tensor& tensor) { require(tensor.device() == Device::CPU && tensor.dtype() == DType::F32 && tensor.numel() == 1, "Expected CPU f32 scalar"); return tensor.data_as<float>()[0]; }
int64_t read_scalar_i64(const Tensor& tensor) { require(tensor.device() == Device::CPU && tensor.dtype() == DType::I64 && tensor.numel() == 1, "Expected CPU i64 scalar"); return tensor.data_as<int64_t>()[0]; }

}  // namespace vrhino
