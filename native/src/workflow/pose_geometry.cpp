#include "vrhino/pose_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vrhino {
namespace {

uint8_t bilinear_pixel(const uint8_t* source, int64_t height, int64_t width,
                       int channel, double x, double y) {
    int64_t ix = static_cast<int64_t>(std::floor(x));
    int64_t iy = static_cast<int64_t>(std::floor(y));
    double fx = std::floor((x - ix) * 32.0 + 0.5) / 32.0;
    double fy = std::floor((y - iy) * 32.0 + 0.5) / 32.0;
    if (fx >= 1.0) { ++ix; fx = 0.0; }
    if (fy >= 1.0) { ++iy; fy = 0.0; }
    double value = 0.0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int64_t sx = ix + dx, sy = iy + dy;
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
            const double wx = dx ? fx : 1.0 - fx;
            const double wy = dy ? fy : 1.0 - fy;
            value += source[(sy * width + sx) * 3 + channel] * wx * wy;
        }
    }
    return static_cast<uint8_t>(std::clamp(std::floor(value + 0.5), 0.0, 255.0));
}

void require_host_f32(const Tensor& tensor, const std::vector<int64_t>& shape,
                      const char* name) {
    if (!tensor.device().is_host() || tensor.dtype() != DType::F32 ||
        tensor.shape() != shape)
        throw std::invalid_argument(std::string(name) + " tensor contract mismatch");
}

}  // namespace

PosePreprocessResult preprocess_topdown_pose_bgr_u8(
        const uint8_t* bgr, int64_t height, int64_t width) {
    if (!bgr || height <= 0 || width <= 0)
        throw std::invalid_argument("pose image input is invalid");
    PosePreprocessResult result;
    result.transform.source_height = height;
    result.transform.source_width = width;
    result.transform.center = {static_cast<float>(width) * 0.5f,
                               static_cast<float>(height) * 0.5f};
    float scale_width = static_cast<float>(width) * 1.25f;
    float scale_height = static_cast<float>(height) * 1.25f;
    constexpr float aspect = 288.0f / 384.0f;
    if (scale_width > scale_height * aspect)
        scale_height = scale_width / aspect;
    else
        scale_width = scale_height * aspect;
    result.transform.scale = {scale_width, scale_height};
    const double factor = 288.0 / static_cast<double>(scale_width);
    const double tx = 144.0 - factor * result.transform.center[0];
    const double ty = 192.0 - factor * result.transform.center[1];
    result.transform.matrix = {factor, 0.0, tx, 0.0, factor, ty};

    result.warped_bgr.resize(384 * 288 * 3);
    result.normalized_nchw = Tensor::host({1, 3, 384, 288}, DType::F32);
    float* output = result.normalized_nchw.data_as<float>();
    constexpr float mean[3] = {123.675f, 116.28f, 103.53f};
    constexpr float stddev[3] = {58.395f, 57.12f, 57.375f};
    for (int64_t y = 0; y < 384; ++y) {
        for (int64_t x = 0; x < 288; ++x) {
            const double source_x = (x - tx) / factor;
            const double source_y = (y - ty) / factor;
            const int64_t pixel = y * 288 + x;
            for (int bgr_channel = 0; bgr_channel < 3; ++bgr_channel) {
                const uint8_t value = bilinear_pixel(
                    bgr, height, width, bgr_channel, source_x, source_y);
                result.warped_bgr[pixel * 3 + bgr_channel] = value;
                const int rgb_channel = 2 - bgr_channel;
                output[rgb_channel * 384 * 288 + pixel] =
                    (static_cast<float>(value) - mean[rgb_channel]) /
                    stddev[rgb_channel];
            }
        }
    }
    return result;
}

std::pair<Tensor, Tensor> combine_simcc_flip_tta(
        const Tensor& original_x, const Tensor& original_y,
        const Tensor& flipped_x, const Tensor& flipped_y,
        const std::vector<int64_t>& flip_indices) {
    require_host_f32(original_x, {1, 133, 576}, "original SimCC-X");
    require_host_f32(original_y, {1, 133, 768}, "original SimCC-Y");
    require_host_f32(flipped_x, {1, 133, 576}, "flipped SimCC-X");
    require_host_f32(flipped_y, {1, 133, 768}, "flipped SimCC-Y");
    if (flip_indices.size() != 133)
        throw std::invalid_argument("pose flip index contract mismatch");
    Tensor x = Tensor::host({1, 133, 576}, DType::F32);
    Tensor y = Tensor::host({1, 133, 768}, DType::F32);
    for (int64_t point = 0; point < 133; ++point) {
        const int64_t partner = flip_indices[point];
        if (partner < 0 || partner >= 133)
            throw std::invalid_argument("pose flip index out of range");
        for (int64_t bin = 0; bin < 576; ++bin)
            x.data_as<float>()[point * 576 + bin] = 0.5f * (
                original_x.data_as<float>()[point * 576 + bin] +
                flipped_x.data_as<float>()[partner * 576 + (575 - bin)]);
        for (int64_t bin = 0; bin < 768; ++bin)
            y.data_as<float>()[point * 768 + bin] = 0.5f * (
                original_y.data_as<float>()[point * 768 + bin] +
                flipped_y.data_as<float>()[partner * 768 + bin]);
    }
    return {std::move(x), std::move(y)};
}

KeypointSet decode_simcc_keypoints(
        const Tensor& simcc_x, const Tensor& simcc_y,
        const PoseAffineTransform& transform, float split_ratio) {
    require_host_f32(simcc_x, {1, 133, 576}, "SimCC-X");
    require_host_f32(simcc_y, {1, 133, 768}, "SimCC-Y");
    if (!(split_ratio > 0.0f) || transform.output_width != 288 ||
        transform.output_height != 384)
        throw std::invalid_argument("pose decode contract mismatch");
    KeypointSet result;
    result.points.reserve(133);
    for (int64_t point = 0; point < 133; ++point) {
        int64_t x_index = 0, y_index = 0;
        float x_score = -std::numeric_limits<float>::infinity();
        float y_score = -std::numeric_limits<float>::infinity();
        for (int64_t bin = 0; bin < 576; ++bin) {
            const float value = simcc_x.data_as<float>()[point * 576 + bin];
            if (value > x_score) { x_score = value; x_index = bin; }
        }
        for (int64_t bin = 0; bin < 768; ++bin) {
            const float value = simcc_y.data_as<float>()[point * 768 + bin];
            if (value > y_score) { y_score = value; y_index = bin; }
        }
        const float confidence = std::min(x_score, y_score);
        double model_x = confidence > 0.0f ? x_index / split_ratio : -1.0;
        double model_y = confidence > 0.0f ? y_index / split_ratio : -1.0;
        const double source_x = model_x / transform.output_width *
            transform.scale[0] + transform.center[0] - 0.5 * transform.scale[0];
        const double source_y = model_y / transform.output_height *
            transform.scale[1] + transform.center[1] - 0.5 * transform.scale[1];
        result.points.push_back({source_x, source_y, confidence});
    }
    return result;
}

FaceGeometry select_musetalk_face_geometry(
        const KeypointSet& keypoints, const std::optional<FaceROI>& fallback,
        int64_t frame_height, int bbox_shift, int lower_margin) {
    if (keypoints.points.size() != 133 || frame_height <= 0 || lower_margin < 0)
        throw std::invalid_argument("face geometry input contract mismatch");
    FaceGeometry result;
    result.facial_landmarks.reserve(68);
    for (int index = 23; index < 91; ++index)
        result.facial_landmarks.push_back({
            static_cast<int32_t>(keypoints.points[index].x),
            static_cast<int32_t>(keypoints.points[index].y)});
    int32_t min_x = std::numeric_limits<int32_t>::max();
    int32_t max_x = std::numeric_limits<int32_t>::min();
    int32_t max_y = std::numeric_limits<int32_t>::min();
    for (const auto& point : result.facial_landmarks) {
        min_x = std::min(min_x, point[0]);
        max_x = std::max(max_x, point[0]);
        max_y = std::max(max_y, point[1]);
    }
    int32_t half_y = result.facial_landmarks[29][1] + bbox_shift;
    const int32_t half_distance = max_y - half_y;
    const int32_t upper = std::max<int32_t>(0, half_y - half_distance);
    result.base_bbox = {min_x, upper, max_x, max_y};
    result.used_fallback = max_y - upper <= 0 || max_x - min_x <= 0 || min_x < 0;
    if (result.used_fallback) {
        if (!fallback) return result;
        result.base_bbox = {
            static_cast<int32_t>(std::max(0.0f, fallback->x1)),
            static_cast<int32_t>(std::max(0.0f, fallback->y1)),
            static_cast<int32_t>(std::max(0.0f, fallback->x2)),
            static_cast<int32_t>(std::max(0.0f, fallback->y2))};
    }
    result.crop_bbox = result.base_bbox;
    result.crop_bbox[3] = std::min<int32_t>(
        result.crop_bbox[3] + lower_margin, static_cast<int32_t>(frame_height));
    result.valid = result.crop_bbox[2] > result.crop_bbox[0] &&
                   result.crop_bbox[3] > result.crop_bbox[1];
    return result;
}

}  // namespace vrhino
