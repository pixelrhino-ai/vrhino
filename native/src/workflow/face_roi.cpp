#include "vrhino/face_roi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vrhino {
namespace {

uint8_t opencv_bilinear_zero(const uint8_t* source, int64_t height,
                             int64_t width, int channel, double x, double y) {
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

float dense_iou(const FaceROI& a, const FaceROI& b) {
    const float width = std::max(0.0f, std::min(a.x2, b.x2) -
                                           std::max(a.x1, b.x1));
    const float height = std::max(0.0f, std::min(a.y2, b.y2) -
                                            std::max(a.y1, b.y1));
    const float intersection = width * height;
    const float area_a = std::max(0.0f, a.x2 - a.x1) *
                         std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) *
                         std::max(0.0f, b.y2 - b.y1);
    const float denominator = area_a + area_b - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

}  // namespace

std::vector<float> preprocess_detector_rgb_u8(const uint8_t* rgb,
                                               int64_t height, int64_t width) {
    if (!rgb || height <= 0 || width <= 0)
        throw std::invalid_argument("detector RGB input is invalid");
    const int64_t pixels = height * width;
    std::vector<float> output(static_cast<size_t>(3 * pixels));
    constexpr float means[3] = {104.0f, 117.0f, 123.0f};
    for (int64_t pixel = 0; pixel < pixels; ++pixel)
        for (int channel = 0; channel < 3; ++channel)
            output[static_cast<size_t>(channel * pixels + pixel)] =
                static_cast<float>(rgb[3 * pixel + channel]) - means[channel];
    return output;
}

float face_roi_iou_inclusive(const FaceROI& a, const FaceROI& b) {
    const float ax = std::max(0.0f, a.x2 - a.x1 + 1.0f);
    const float ay = std::max(0.0f, a.y2 - a.y1 + 1.0f);
    const float bx = std::max(0.0f, b.x2 - b.x1 + 1.0f);
    const float by = std::max(0.0f, b.y2 - b.y1 + 1.0f);
    const float width = std::max(0.0f, std::min(a.x2, b.x2) -
                                           std::max(a.x1, b.x1) + 1.0f);
    const float height = std::max(0.0f, std::min(a.y2, b.y2) -
                                            std::max(a.y1, b.y1) + 1.0f);
    const float intersection = width * height;
    const float denominator = ax * ay + bx * by - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

std::vector<FaceROI> deterministic_face_nms(const std::vector<FaceROI>& input,
                                            float iou_threshold) {
    if (iou_threshold < 0.0f) throw std::invalid_argument("NMS threshold is invalid");
    std::vector<int64_t> order(input.size());
    std::iota(order.begin(), order.end(), int64_t{0});
    // The frozen detector produces exactly saturated confidence ties. Its
    // reference NumPy ordering selects the earlier dense candidate first, so
    // make that tie contract explicit instead of depending on STL sort details.
    std::sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
        if (input[left].confidence != input[right].confidence)
            return input[left].confidence > input[right].confidence;
        return input[left].source_index < input[right].source_index;
    });
    std::vector<FaceROI> output;
    while (!order.empty()) {
        const int64_t selected = order.front();
        output.push_back(input[selected]);
        std::vector<int64_t> remaining;
        for (size_t i = 1; i < order.size(); ++i)
            if (face_roi_iou_inclusive(input[selected], input[order[i]]) <= iou_threshold)
                remaining.push_back(order[i]);
        order = std::move(remaining);
    }
    return output;
}

FaceDetectionStages decode_multiscale_face_detector(
        const std::vector<DetectorScaleHost>& scales,
        float candidate_threshold, float nms_threshold, float final_threshold) {
    if (scales.size() != 6 || candidate_threshold < 0.0f ||
        nms_threshold < 0.0f || final_threshold < 0.0f)
        throw std::invalid_argument("detector postprocess contract is invalid");
    FaceDetectionStages stages;
    int64_t source_index = 0;
    for (size_t scale = 0; scale < scales.size(); ++scale) {
        const auto& item = scales[scale];
        if (item.batch != 1 || item.height <= 0 || item.width <= 0)
            throw std::invalid_argument("detector scale shape is invalid");
        const int64_t area = item.height * item.width;
        if (item.confidence_logits.size() != static_cast<size_t>(2 * area) ||
            item.localization.size() != static_cast<size_t>(4 * area))
            throw std::invalid_argument("detector scale storage is invalid");
        const float stride = static_cast<float>(int64_t{1} << (scale + 2));
        const float anchor = stride * 4.0f;
        for (int64_t y = 0; y < item.height; ++y) {
            for (int64_t x = 0; x < item.width; ++x) {
                const int64_t index = y * item.width + x;
                const float background = item.confidence_logits[index];
                const float face = item.confidence_logits[area + index];
                const float maximum = std::max(background, face);
                const float face_score = std::exp(face - maximum) /
                    (std::exp(background - maximum) + std::exp(face - maximum));
                if (!(face_score > candidate_threshold)) continue;
                const float center_x = stride * 0.5f + x * stride;
                const float center_y = stride * 0.5f + y * stride;
                const float decoded_x = center_x + item.localization[index] * 0.1f * anchor;
                const float decoded_y = center_y + item.localization[area + index] * 0.1f * anchor;
                const float decoded_w = anchor * std::exp(item.localization[2 * area + index] * 0.2f);
                const float decoded_h = anchor * std::exp(item.localization[3 * area + index] * 0.2f);
                FaceROI roi{decoded_x - decoded_w * 0.5f,
                            decoded_y - decoded_h * 0.5f,
                            decoded_x + decoded_w * 0.5f,
                            decoded_y + decoded_h * 0.5f,
                            face_score, source_index++, static_cast<int64_t>(scale)};
                stages.decoded_candidates.push_back(roi);
                stages.thresholded_candidates.push_back(roi);
            }
        }
    }
    stages.post_nms = deterministic_face_nms(stages.thresholded_candidates,
                                             nms_threshold);
    stages.post_nms.erase(std::remove_if(stages.post_nms.begin(), stages.post_nms.end(),
        [&](const FaceROI& roi) { return !(roi.confidence > final_threshold); }),
        stages.post_nms.end());
    if (!stages.post_nms.empty()) stages.selected = stages.post_nms.front();
    return stages;
}

BlazeFacePreprocessResult preprocess_dense_face_detector_rgb_u8(
        const uint8_t* rgb, int64_t height, int64_t width) {
    if (!rgb || height <= 0 || width <= 0)
        throw std::invalid_argument("dense face detector RGB input is invalid");
    BlazeFacePreprocessResult result;
    result.source_height = height;
    result.source_width = width;
    constexpr int64_t target = 128;
    const double extent = static_cast<double>(std::max(height, width));
    const double origin_x = 0.5 * (static_cast<double>(width) - extent);
    const double origin_y = 0.5 * (static_cast<double>(height) - extent);
    result.horizontal_padding = static_cast<float>(
        width < height ? (1.0 - static_cast<double>(width) / height) * 0.5 : 0.0);
    result.vertical_padding = static_cast<float>(
        height < width ? (1.0 - static_cast<double>(height) / width) * 0.5 : 0.0);
    result.resized_width = static_cast<int64_t>(std::lround(width * target / extent));
    result.resized_height = static_cast<int64_t>(std::lround(height * target / extent));
    result.pad_left = static_cast<int64_t>(std::lround(
        result.horizontal_padding * target));
    result.pad_top = static_cast<int64_t>(std::lround(
        result.vertical_padding * target));
    result.normalized_nchw.resize(3 * target * target);
    for (int64_t y = 0; y < target; ++y) {
        for (int64_t x = 0; x < target; ++x) {
            const double source_x = origin_x + x * extent / target;
            const double source_y = origin_y + y * extent / target;
            const int64_t pixel = y * target + x;
            for (int channel = 0; channel < 3; ++channel) {
                const uint8_t value = opencv_bilinear_zero(
                    rgb, height, width, channel, source_x, source_y);
                result.normalized_nchw[channel * target * target + pixel] =
                    static_cast<float>(value) * (2.0f / 255.0f) - 1.0f;
            }
        }
    }
    return result;
}

std::vector<std::array<float, 4>> short_range_face_detector_anchors() {
    constexpr std::array<int, 4> strides{8, 16, 16, 16};
    std::vector<std::array<float, 4>> anchors;
    for (size_t layer = 0; layer < strides.size();) {
        size_t last = layer;
        while (last < strides.size() && strides[last] == strides[layer]) ++last;
        const int anchors_per_cell = static_cast<int>(2 * (last - layer));
        const int feature = static_cast<int>(std::ceil(128.0 / strides[layer]));
        for (int y = 0; y < feature; ++y)
            for (int x = 0; x < feature; ++x)
                for (int anchor = 0; anchor < anchors_per_cell; ++anchor)
                    anchors.push_back({(x + 0.5f) / feature,
                                       (y + 0.5f) / feature, 1.0f, 1.0f});
        layer = last;
    }
    if (anchors.size() != 896)
        throw std::logic_error("short-range face anchor generation drift");
    return anchors;
}

DenseFaceDetectionStages decode_dense_face_detector(
        const DenseDetectorHost& raw,
        const BlazeFacePreprocessResult& transform,
        float score_threshold, float suppression_threshold) {
    if (raw.regressors.size() != 896 * 16 ||
        raw.classification_logits.size() != 896 ||
        transform.source_height <= 0 || transform.source_width <= 0 ||
        score_threshold < 0.0f || score_threshold > 1.0f ||
        suppression_threshold < 0.0f)
        throw std::invalid_argument("dense detector postprocess contract is invalid");
    const auto anchors = short_range_face_detector_anchors();
    DenseFaceDetectionStages result;
    for (int64_t index = 0; index < 896; ++index) {
        const float* coordinates = raw.regressors.data() + index * 16;
        const auto& anchor = anchors[static_cast<size_t>(index)];
        // MediaPipe short-range metadata uses reverse_output_order=true,
        // selecting the XYWH decoder rather than the legacy YXHW order.
        const float center_x = coordinates[0] / 128.0f * anchor[2] + anchor[0];
        const float center_y = coordinates[1] / 128.0f * anchor[3] + anchor[1];
        const float width = coordinates[2] / 128.0f * anchor[2];
        const float height = coordinates[3] / 128.0f * anchor[3];
        const float logit = std::clamp(raw.classification_logits[index],
                                       -100.0f, 100.0f);
        const float score = 1.0f / (1.0f + std::exp(-logit));
        FaceROI roi{center_x - width * 0.5f, center_y - height * 0.5f,
                    center_x + width * 0.5f, center_y + height * 0.5f,
                    score, index, -1};
        result.decoded_candidates.push_back(roi);
        if (score >= score_threshold) result.thresholded_candidates.push_back(roi);
    }
    std::stable_sort(result.thresholded_candidates.begin(),
                     result.thresholded_candidates.end(),
        [](const FaceROI& left, const FaceROI& right) {
            if (left.confidence != right.confidence)
                return left.confidence > right.confidence;
            return left.source_index < right.source_index;
        });
    std::vector<bool> consumed(result.thresholded_candidates.size(), false);
    for (size_t selected = 0; selected < result.thresholded_candidates.size(); ++selected) {
        if (consumed[selected]) continue;
        const FaceROI seed = result.thresholded_candidates[selected];
        double total = 0.0, x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
        for (size_t candidate = selected;
             candidate < result.thresholded_candidates.size(); ++candidate) {
            if (consumed[candidate]) continue;
            const FaceROI& item = result.thresholded_candidates[candidate];
            if (candidate != selected && dense_iou(seed, item) <= suppression_threshold)
                continue;
            consumed[candidate] = true;
            total += item.confidence;
            x1 += item.x1 * item.confidence;
            y1 += item.y1 * item.confidence;
            x2 += item.x2 * item.confidence;
            y2 += item.y2 * item.confidence;
        }
        FaceROI merged = seed;
        if (total > 0.0) {
            merged.x1 = static_cast<float>(x1 / total);
            merged.y1 = static_cast<float>(y1 / total);
            merged.x2 = static_cast<float>(x2 / total);
            merged.y2 = static_cast<float>(y2 / total);
        }
        result.post_suppression.push_back(merged);
    }
    const float content_width = 1.0f - 2.0f * transform.horizontal_padding;
    const float content_height = 1.0f - 2.0f * transform.vertical_padding;
    for (FaceROI& roi : result.post_suppression) {
        roi.x1 = (roi.x1 - transform.horizontal_padding) / content_width *
                 transform.source_width;
        roi.x2 = (roi.x2 - transform.horizontal_padding) / content_width *
                 transform.source_width;
        roi.y1 = (roi.y1 - transform.vertical_padding) / content_height *
                 transform.source_height;
        roi.y2 = (roi.y2 - transform.vertical_padding) / content_height *
                 transform.source_height;
        roi.x1 = std::clamp(roi.x1, 0.0f,
                            static_cast<float>(transform.source_width));
        roi.x2 = std::clamp(roi.x2, 0.0f,
                            static_cast<float>(transform.source_width));
        roi.y1 = std::clamp(roi.y1, 0.0f,
                            static_cast<float>(transform.source_height));
        roi.y2 = std::clamp(roi.y2, 0.0f,
                            static_cast<float>(transform.source_height));
    }
    result.post_suppression.erase(std::remove_if(
        result.post_suppression.begin(), result.post_suppression.end(),
        [](const FaceROI& roi) { return roi.x2 <= roi.x1 || roi.y2 <= roi.y1; }),
        result.post_suppression.end());
    if (!result.post_suppression.empty())
        result.selected = result.post_suppression.front();
    return result;
}

}  // namespace vrhino
