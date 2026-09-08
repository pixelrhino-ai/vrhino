#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "vrhino/product/run.h"

namespace vrhino::product {

enum class DoctorOverall {
    Ready,
    Warning,
    Failed,
};

struct DoctorOptions {
    std::string product_version;
    std::string git_head;
    std::string cuda_contract = kCudaRuntimeContract;
    std::filesystem::path cache_root;
    bool cache_root_is_default = false;
    std::filesystem::path converter_spec_root;
    std::filesystem::path encoder_path;
    std::optional<std::string> encoder_resolution_error;
    std::optional<std::string> model_reference;
    std::optional<HardwareSnapshot> hardware_override;
};

struct DoctorReport {
    DoctorOverall overall = DoctorOverall::Ready;
    std::string text;

    int exit_code() const noexcept { return overall == DoctorOverall::Failed ? 1 : 0; }
};

const char* doctor_overall_name(DoctorOverall overall);
DoctorReport run_doctor(const DoctorOptions& options);

}  // namespace vrhino::product
