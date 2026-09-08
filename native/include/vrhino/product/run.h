#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vrhino/product/model_package.h"
#include "vrhino/product/run_event.h"

namespace vrhino::product {

enum class PreflightStatus {
    Supported,
    SupportedWithWarning,
    InsufficientVram,
    UnsupportedGpu,
    DriverIncompatible,
};

struct HardwareSnapshot {
    std::string gpu_name;
    int compute_major = 0;
    int compute_minor = 0;
    uint64_t total_vram_bytes = 0;
    uint64_t available_vram_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

struct PreflightResult {
    PreflightStatus status = PreflightStatus::SupportedWithWarning;
    HardwareSnapshot hardware;
    std::optional<uint64_t> minimum_vram_bytes;
    std::optional<uint64_t> recommended_vram_bytes;
    std::string message;
};

struct RunOptions {
    std::string model_reference;
    std::string prompt;
    std::filesystem::path video;
    std::filesystem::path audio;
    std::filesystem::path output;
    std::string preset;
    std::optional<uint64_t> seed;
    bool overwrite = false;
    bool debug = false;
    std::filesystem::path encoder_path;
    std::function<bool()> cancellation_requested;
};

struct RunResult {
    PackageIdentity identity;
    std::string preset;
    uint64_t seed = 0;
    int64_t width = 0;
    int64_t height = 0;
    int64_t frames = 0;
    int64_t fps = 0;
    double package_validation_seconds = 0.0;
    double conditioning_seconds = 0.0;
    double media_input_seconds = 0.0;
    double face_analysis_seconds = 0.0;
    double source_preparation_seconds = 0.0;
    double sampling_seconds = 0.0;
    double decode_seconds = 0.0;
    double parser_mask_seconds = 0.0;
    double composite_seconds = 0.0;
    double encoding_seconds = 0.0;
    double total_seconds = 0.0;
    uint64_t peak_device_bytes = 0;
    uint64_t output_bytes = 0;
    float video_minimum = 0.0f;
    float video_maximum = 0.0f;
    int64_t nan_count = 0;
    int64_t inf_count = 0;
    double maximum_alpha_support_change = 0.0;
    double maximum_alpha_centroid_movement = 0.0;
    uint64_t invalid_alpha_masks = 0;
    PreflightResult preflight;
};

HardwareSnapshot query_hardware();
PreflightResult preflight_runnable_model(const ResolvedRunnableModel& model,
                                         const std::string& preset);
PreflightResult preflight_runnable_model(const ResolvedRunnableModel& model,
                                         const std::string& preset,
                                         const HardwareSnapshot& hardware);
void preflight_output_destination(const std::filesystem::path& output,
                                  bool overwrite);
void preflight_media_encoder(const std::filesystem::path& encoder);
std::filesystem::path default_media_encoder_path();
RunResult run_runnable_model(const ResolvedRunnableModel& model,
                             const RunOptions& options,
                             RunEventSink events = {});
RunResult run_lip_sync_product(const ResolvedRunnableModel& model,
                               const RunOptions& options,
                               RunEventSink events = {});
RunResult run_lip_sync_diffusion_product(const ResolvedRunnableModel& model,
                                         const RunOptions& options,
                                         RunEventSink events = {});

}  // namespace vrhino::product
