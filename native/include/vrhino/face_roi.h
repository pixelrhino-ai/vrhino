#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace vrhino {

struct FaceROI {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int64_t source_index = -1;
    int64_t scale_index = -1;
};

struct DetectorScaleHost {
    int64_t batch = 0;
    int64_t height = 0;
    int64_t width = 0;
    // Dense NCHW tensors. Confidence has two channels (background, face),
    // localization has four channels (dx,dy,dw,dh).
    std::vector<float> confidence_logits;
    std::vector<float> localization;
};

struct FaceDetectionStages {
    std::vector<FaceROI> decoded_candidates;
    std::vector<FaceROI> thresholded_candidates;
    std::vector<FaceROI> post_nms;
    std::optional<FaceROI> selected;
};

struct BlazeFacePreprocessResult {
    std::vector<float> normalized_nchw;
    int64_t source_height = 0;
    int64_t source_width = 0;
    int64_t resized_height = 0;
    int64_t resized_width = 0;
    int64_t pad_top = 0;
    int64_t pad_left = 0;
    float horizontal_padding = 0.0f;
    float vertical_padding = 0.0f;
};

struct DenseDetectorHost {
    // Dense row-major [anchors,16] box/keypoint regressions and
    // [anchors,1] classification logits.
    std::vector<float> regressors;
    std::vector<float> classification_logits;
};

struct DenseFaceDetectionStages {
    std::vector<FaceROI> decoded_candidates;
    std::vector<FaceROI> thresholded_candidates;
    std::vector<FaceROI> post_suppression;
    std::optional<FaceROI> selected;
};

// Exact bounded RGB-u8 to detector NCHW-FP32 preprocessing. The three channel
// means follow the frozen detector contract and no resize is performed.
std::vector<float> preprocess_detector_rgb_u8(const uint8_t* rgb,
                                               int64_t height, int64_t width);

float face_roi_iou_inclusive(const FaceROI& a, const FaceROI& b);
std::vector<FaceROI> deterministic_face_nms(const std::vector<FaceROI>& input,
                                            float iou_threshold);

// Bounded S3FD-format host decode. The format is a detector-output contract,
// not MuseTalk behavior; no Neural Runtime operation performs this work.
FaceDetectionStages decode_multiscale_face_detector(
    const std::vector<DetectorScaleHost>& scales,
    float candidate_threshold = 0.05f,
    float nms_threshold = 0.3f,
    float final_threshold = 0.5f);

// MediaPipe-compatible FIT preprocessing for the generic 128x128 dense-anchor
// detector contract: RGB, bilinear resize, centered zero letterbox, [-1,1].
BlazeFacePreprocessResult preprocess_dense_face_detector_rgb_u8(
    const uint8_t* rgb, int64_t height, int64_t width);

// Exact short-range dense-anchor generation and bounded host postprocess.
// The neural component returns only raw tensors; sigmoid, decode and weighted
// suppression remain workflow operations.
std::vector<std::array<float, 4>> short_range_face_detector_anchors();
DenseFaceDetectionStages decode_dense_face_detector(
    const DenseDetectorHost& raw,
    const BlazeFacePreprocessResult& transform,
    float score_threshold = 0.5f,
    float suppression_threshold = 0.3f);

}  // namespace vrhino
