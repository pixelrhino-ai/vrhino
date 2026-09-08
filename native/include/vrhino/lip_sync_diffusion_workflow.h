#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vrhino/audio_conditioning.h"
#include "vrhino/face_roi.h"
#include "vrhino/lip_sync_workflow.h"
#include "vrhino/pose_geometry.h"

namespace vrhino {

inline constexpr const char* kLipSyncDiffusionWorkflowFamily =
    "lip_sync_diffusion_workflow_v1";

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

// Orientation-preserving 2-D similarity in the canonical
// [[a,-b,tx],[b,a,ty]] representation.
struct SimilarityAffine {
    double a = 1.0;
    double b = 0.0;
    double tx = 0.0;
    double ty = 0.0;

    std::array<double, 6> matrix() const;
    SimilarityAffine inverse() const;
};

struct LipSyncDiffusionAlignmentState {
    std::optional<SimilarityAffine> previous;
    std::optional<FaceROI> previous_roi;
    void reset() noexcept { previous.reset(); previous_roi.reset(); }
};

struct LipSyncDiffusionAlignment {
    std::array<Point2D, 3> source_anchors{};
    SimilarityAffine unsmoothed;
    SimilarityAffine smoothed;
    SimilarityAffine inverse;
    bool scene_reset = false;
    float minimum_anchor_confidence = 0.0f;
};

// DWPose COCO-WholeBody points [23,91) are interpreted in the standard
// 68-point order. Invalid geometry clears state and returns no transform;
// there is deliberately no whole-frame or BlazeFace-only alignment fallback.
std::optional<LipSyncDiffusionAlignment> align_lip_sync_diffusion_face(
    const KeypointSet& keypoints, const FaceROI& selected_roi,
    LipSyncDiffusionAlignmentState& state);

RgbFrame warp_lip_sync_diffusion_face(const RgbFrame& source,
                                      const SimilarityAffine& transform);

struct LipSyncDiffusionAudioPlan {
    int64_t retained_feature_frames = 0;
    std::vector<std::array<int64_t, 10>> indices;
};

LipSyncDiffusionAudioPlan plan_lip_sync_diffusion_audio(
    int64_t waveform_samples, int64_t output_frames,
    int64_t sample_rate = 16000, int64_t feature_rate = 50,
    int64_t video_fps = 25);

// `stacked` is [1,L,5,384]. Each video frame gathers ten clamped 50-Hz
// positions and flattens position-major retained states into [50,384].
Tensor assemble_lip_sync_diffusion_audio_windows(
    Backend& backend, const Tensor& stacked, int64_t waveform_samples,
    int64_t output_frames);

struct LipSyncDiffusionVaeInput {
    Tensor reference;
    Tensor masked;
    Tensor mask;
};

RgbFrame prepare_lip_sync_diffusion_fixed_mask(const RgbFrame& fixed_mask);
LipSyncDiffusionVaeInput prepare_lip_sync_diffusion_vae_input(
    const std::vector<RgbFrame>& aligned_frames,
    const RgbFrame& resized_fixed_mask);

enum class LipSyncDiffusionRngBranch : uint64_t {
    MaskedSource = 0x6d61736b65645f73ULL,
    ReferenceSource = 0x7265666572656e63ULL,
    InitialDiffusionNoise = 0x696e69745f6e6f69ULL,
};

// Derives a call-order-independent stream from a bounded semantic index. VAE
// branches pass the source-frame identity; chunk noise passes the chunk start.
RngState lip_sync_diffusion_rng(uint64_t workflow_seed,
                                int64_t semantic_index,
                                LipSyncDiffusionRngBranch branch);

struct TemporalFrameChunk {
    int64_t start = 0;
    int64_t frames = 0;
};

std::vector<TemporalFrameChunk> plan_temporal_frame_chunks(
    int64_t frames, int64_t maximum_chunk = 16);

// Exact forward/reverse source sequence used by the bounded workflow.
std::vector<int64_t> plan_lip_sync_diffusion_source_frames(
    int64_t source_frames, int64_t output_frames);

Tensor assemble_lip_sync_diffusion_unet_input(
    Backend& backend, const Tensor& noisy_latent,
    const Tensor& fixed_mask_512, const Tensor& masked_source_latent,
    const Tensor& reference_source_latent);

// Generic temporal-latent to frame-batch lowering used at VAE boundaries:
// [B,C,F,H,W] -> [B*F,C,H,W].
Tensor temporal_latents_to_frame_batch(Backend& backend,
                                       const Tensor& temporal_latents);

Tensor reconstruct_lip_sync_diffusion_crop(
    Backend& backend, const Tensor& decoded_normalized,
    const Tensor& reference_normalized, const Tensor& fixed_mask_512);

struct LipSyncDiffusionCompositeObservation {
    std::vector<float> inverse_warp_rgb;
    std::vector<float> soft_mask;
    int32_t feather_edge = 0;
    int32_t feather_radius = 0;
};

RgbFrame composite_lip_sync_diffusion_face(
    const RgbFrame& source, const Tensor& reconstructed_face_chw,
    const SimilarityAffine& source_to_alignment,
    LipSyncDiffusionCompositeObservation* observation = nullptr);

}  // namespace vrhino
