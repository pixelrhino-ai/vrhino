#pragma once

#include <cstdint>
#include <vector>

#include "vrhino/tensor.h"

namespace vrhino {

Tensor host_f32(const std::vector<int64_t>& shape, const std::vector<float>& values);
Tensor host_i64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values);
Tensor host_bool(const std::vector<int64_t>& shape, const std::vector<uint8_t>& values);
Tensor scalar_f32(float value);
Tensor scalar_i64(int64_t value);
float read_scalar_f32(const Tensor& tensor);
int64_t read_scalar_i64(const Tensor& tensor);

}  // namespace vrhino
