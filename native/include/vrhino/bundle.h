#pragma once

#include <map>
#include <string>

#include "vrhino/tensor.h"

namespace vrhino {

using TensorBundle = std::map<std::string, Tensor>;
TensorBundle read_bundle(const std::string& path);
void write_bundle(const std::string& path, const TensorBundle& bundle);

}  // namespace vrhino
