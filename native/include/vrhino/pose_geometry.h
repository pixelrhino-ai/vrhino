#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vrhino/face_roi.h"
#include "vrhino/tensor.h"

namespace vrhino {

struct Keypoint {
    double x = 0.0;
    double y = 0.0;
    float confidence = 0.0f;
};

struct KeypointSet {
    std::vector<Keypoint> points;
    std::string coordinate_space = "source_image_pixels";
};

struct PoseAffineTransform {
    std::array<float, 2> center{};
    std::array<float, 2> scale{};
    // Forward source-to-model 2x3 matrix, row-major.
    std::array<double, 6> matrix{};
    int64_t source_height = 0;
    int64_t source_width = 0;
    int64_t output_height = 384;
    int64_t output_width = 288;
};

struct PosePreprocessResult {
    Tensor normalized_nchw;
    PoseAffineTransform transform;
    std::vector<uint8_t> warped_bgr;
};

struct FaceGeometry {
    std::vector<std::array<int32_t, 2>> facial_landmarks;
    std::array<int32_t, 4> base_bbox{};
    std::array<int32_t, 4> crop_bbox{};
    bool used_fallback = false;
    bool valid = false;
};

// Bounded full-frame top-down affine preprocessing used by the fixed pose
// contract. Input is interleaved BGR uint8; output is RGB-normalized NCHW FP32.
PosePreprocessResult preprocess_topdown_pose_bgr_u8(
    const uint8_t* bgr, int64_t height, int64_t width);

// Deterministic flip-TTA combination for raw SimCC outputs. `flip_indices`
// maps each semantic keypoint to its horizontally mirrored partner.
std::pair<Tensor, Tensor> combine_simcc_flip_tta(
    const Tensor& original_x, const Tensor& original_y,
    const Tensor& flipped_x, const Tensor& flipped_y,
    const std::vector<int64_t>& flip_indices);

KeypointSet decode_simcc_keypoints(
    const Tensor& simcc_x, const Tensor& simcc_y,
    const PoseAffineTransform& transform, float split_ratio = 2.0f);

// Bounded MuseTalk v1.5 geometry policy. DWPose remains independent of S3FD;
// the already-qualified detector ROI is consumed only as a workflow fallback.
FaceGeometry select_musetalk_face_geometry(
    const KeypointSet& keypoints, const std::optional<FaceROI>& fallback,
    int64_t frame_height, int bbox_shift = 0, int lower_margin = 10);

}  // namespace vrhino
