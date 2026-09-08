#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vrhino/tensor.h"

namespace vrhino {

struct SemanticLabelMap {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> labels;
    std::string coordinate_space = "segmenter_pixels";
};

struct AlphaMask {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> values;
    std::string coordinate_space = "expanded_face_crop_pixels";
};

struct FaceParserPreparation {
    Tensor normalized_nchw;
    std::vector<uint8_t> resized_rgb;
    std::array<int32_t, 4> crop_box{};
};

struct SemanticSegmenterPreparation {
    Tensor normalized_nchw;
    std::vector<uint8_t> resized_rgb;
    std::array<int32_t, 4> crop_box{};
};

FaceParserPreparation preprocess_face_parser_bgr_u8(
    const uint8_t* source_bgr, int32_t height, int32_t width,
    const std::array<int32_t, 4>& face_box, double expand = 1.5);

// Bounded RGB preprocessing for a 256x256 semantic segmenter. The caller
// supplies the product-owned expanded crop; the neural component receives
// only normalized NCHW FP32 data.
SemanticSegmenterPreparation preprocess_semantic_segmenter_rgb_u8(
    const uint8_t* source_rgb, int32_t height, int32_t width,
    const std::array<int32_t, 4>& expanded_crop_box);

SemanticLabelMap semantic_argmax_first(const Tensor& logits);

struct JawMaskObservation {
    std::vector<uint8_t> class_one_region;
    std::vector<uint8_t> dilated;
    std::vector<uint8_t> eroded;
    std::vector<uint8_t> selected;
    std::vector<uint8_t> lower_half;
};

// Bounded host policy for constructing a lower-face alpha mask from a generic
// semantic label map and a configured 68-landmark layout.  Class and landmark
// identities are workflow configuration; neither is interpreted by the neural
// component or Backend.
struct LandmarkConstrainedMaskConfig {
    uint8_t face_skin_label = 3;
    uint8_t body_skin_label = 2;
    // The outer two landmarks on each side approach the temple.  The bounded
    // lower-face policy starts at 2 and ends at 14 to avoid eye/hair support.
    int32_t jaw_first = 2;
    int32_t jaw_last = 14;
    int32_t nose_left = 31;
    int32_t nose_center = 33;
    int32_t nose_right = 35;
    int32_t mouth_first = 48;
    int32_t mouth_last = 59;
    int32_t close_kernel = 5;
    int32_t mouth_dilation_kernel = 3;
    int32_t blur_kernel = 11;
    bool include_body_skin_inside_constraint = true;
};

struct LandmarkConstrainedMaskObservation {
    std::vector<uint8_t> geometry_constraint;
    std::vector<uint8_t> semantic_seed;
    std::vector<uint8_t> mouth_restoration;
    std::vector<uint8_t> selected;
    double geometry_ms = 0.0;
    double morphology_ms = 0.0;
    double resize_blur_ms = 0.0;
};

AlphaMask build_landmark_constrained_alpha_mask(
    const SemanticLabelMap& labels,
    const std::vector<std::array<int32_t, 2>>& facial_landmarks,
    const std::array<int32_t, 4>& expanded_crop_box,
    const LandmarkConstrainedMaskConfig& config = {},
    LandmarkConstrainedMaskObservation* observation = nullptr);

AlphaMask build_musetalk_jaw_alpha_mask(
    const SemanticLabelMap& labels,
    const std::array<int32_t, 4>& face_box,
    const std::array<int32_t, 4>& expanded_crop_box,
    int32_t left_cheek_width = 90,
    int32_t right_cheek_width = 90,
    double lower_half_ratio = 0.5,
    JawMaskObservation* observation = nullptr);

}  // namespace vrhino
