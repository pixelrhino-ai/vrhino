#pragma once

#include <stdexcept>
#include <string>

namespace vrhino {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

void require(bool condition, const std::string& message);

}  // namespace vrhino
