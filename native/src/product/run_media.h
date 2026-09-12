#pragma once
#include <filesystem>
#include <functional>
#include <limits>
#include "vrhino/tensor.h"
namespace vrhino::product::run_media_detail {
struct EncodeResult {
    double seconds = 0.0;
    uint64_t bytes = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    int64_t nan_count = 0;
    int64_t inf_count = 0;
};

void check_output_destination(const std::filesystem::path& requested, bool overwrite);
void check_media_encoder(const std::filesystem::path& requested);
std::filesystem::path product_encoder_path();
EncodeResult encode_mp4(const Tensor&, int64_t, const std::filesystem::path&,
    const std::filesystem::path&, bool, const std::function<bool()>&, float, float);
}
