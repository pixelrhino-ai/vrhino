#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "vrhino/loader.h"

namespace vrhino::product {

struct VrmComponentIntegrityContract {
    std::string semantic_name;
    std::string architecture;
    uint64_t bytes = 0;
    std::string sha256;
};

struct VrmVerificationObservation {
    uint64_t sha256_bytes = 0;
    uint64_t blake2b_bytes = 0;
    double sha256_seconds = 0.0;
    double blake2b_seconds = 0.0;
    double concurrent_wall_seconds = 0.0;
    bool sha256_completed = false;
    bool blake2b_completed = false;
    size_t maximum_digest_consumers = 0;
};

// Test-only synchronization seam: invoked after the authoritative descriptor
// and identity are established, before either verifier consumes bytes. Normal
// production callers leave this empty.
using StableOpenVerificationHook = std::function<void()>;

std::unique_ptr<VrmModel> load_verified_vrm_component(
    const std::filesystem::path& path,
    const VrmComponentIntegrityContract& contract,
    VrmVerificationObservation* observation = nullptr,
    const StableOpenVerificationHook& stable_open_hook = {});

}  // namespace vrhino::product
