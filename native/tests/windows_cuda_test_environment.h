#pragma once

// POSIX environment operations used by the existing synthetic CUDA fixtures.
// Only Windows test targets include this shim; production code is unchanged.
#include <cstdlib>

inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name)) return 0;
    return _putenv_s(name, value);
}

inline int unsetenv(const char* name) {
    return _putenv_s(name, "");
}
