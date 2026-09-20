#pragma once
#include "vrhino/json.h"
#include <filesystem>
#include <functional>
#include <cstdint>

namespace vrhino::product {
struct EvaluationScope {
    bool complete = false;
    int max_steps = 0;
    uint64_t device_bytes = 0, host_bytes = 0, output_bytes = 0;
    uint64_t weight_cache_bytes = 0; // zero: legacy budget derivation
    uint64_t latent_elements = 0, video_elements = 0;
    int timeout_seconds = 0, fps = 0;
    float video_min = 0, video_max = 0;
    std::string precision_artifact;
    static EvaluationScope parse(const Json&);
    void admit(uint64_t latent, uint64_t video, int declared_steps) const;
};

// Linux CLI process isolation. It monitors process-group RSS and output bytes;
// a supplied device sensor must fail closed if measurements are unavailable.
// Sampled watchdogs are not hard allocation guarantees. No model semantics.
Json supervise_evaluation(const EvaluationScope&, const std::filesystem::path& output,
    const std::function<void()>& worker, const std::function<uint64_t()>& device_bytes);

Json evaluate_local_product(const std::filesystem::path& manifest,
    const std::filesystem::path& resources, const std::filesystem::path& request,
    const std::filesystem::path& options, const std::filesystem::path& output);
} // namespace vrhino::product
