#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace vrhino {

using InputScalar = std::variant<bool, int64_t, double, std::string>;

struct TypedInput {
    std::string type;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<std::byte> payload;
};

struct InputRequest {
    std::string prompt;
    std::optional<std::string> negative_prompt;
    uint64_t seed = 0;
    std::map<std::string, InputScalar> generation_parameters;
    std::map<std::string, TypedInput> optional_inputs;
};

}  // namespace vrhino
