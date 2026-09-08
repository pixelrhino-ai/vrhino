#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vrhino/lip_sync_diffusion_workflow.h"

namespace {

void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

bool close(double left, double right, double tolerance = 1.0e-9) {
    return std::abs(left - right) <= tolerance;
}

bool same_affine(const vrhino::SimilarityAffine& left,
                 const vrhino::SimilarityAffine& right) {
    return close(left.a, right.a) && close(left.b, right.b) &&
           close(left.tx, right.tx) && close(left.ty, right.ty);
}

vrhino::KeypointSet face_points(double shift_x = 0.0, float confidence = 0.99f) {
    vrhino::KeypointSet result;
    result.points.assign(133, {200.0 + shift_x, 200.0, confidence});
    for (int index = 17; index < 22; ++index)
        result.points[static_cast<size_t>(23 + index)] =
            {145.0 + 4.0 * (index - 17) + shift_x, 150.0, confidence};
    for (int index = 22; index < 27; ++index)
        result.points[static_cast<size_t>(23 + index)] =
            {239.0 + 4.0 * (index - 22) + shift_x, 150.0, confidence};
    for (int index : {31, 32, 34, 35})
        result.points[static_cast<size_t>(23 + index)] =
            {200.0 + shift_x, 215.0, confidence};
    return result;
}

vrhino::RgbFrame solid(int width, int height, uint8_t value) {
    return {width, height,
            std::vector<uint8_t>(static_cast<size_t>(width) * height * 3,
                                 value)};
}

}  // namespace

int main() {
    try {
        vrhino::LipSyncDiffusionAlignmentState state;
        const vrhino::FaceROI roi{100, 100, 300, 300, 0.95f, 0, -1};
        const auto first = vrhino::align_lip_sync_diffusion_face(
            face_points(), roi, state);
        check(first && first->scene_reset,
              "first valid alignment did not reset bounded state");
        check(close(first->source_anchors[0].x, 153.0) &&
              close(first->source_anchors[1].x, 247.0) &&
              close(first->source_anchors[2].x, 200.0),
              "DWPose three-point derivation mismatch");
        const auto matrix = first->smoothed.matrix();
        const auto inverse = first->inverse.matrix();
        const double source_x = 173.25, source_y = 188.75;
        const double aligned_x = matrix[0] * source_x + matrix[1] * source_y +
                                 matrix[2];
        const double aligned_y = matrix[3] * source_x + matrix[4] * source_y +
                                 matrix[5];
        check(close(inverse[0] * aligned_x + inverse[1] * aligned_y + inverse[2],
                    source_x, 1.0e-8) &&
              close(inverse[3] * aligned_x + inverse[4] * aligned_y + inverse[5],
                    source_y, 1.0e-8),
              "similarity inverse mismatch");

        const auto previous = first->smoothed;
        const auto second = vrhino::align_lip_sync_diffusion_face(
            face_points(10.0), {110, 100, 310, 300, 0.95f, 1, -1}, state);
        check(second && !second->scene_reset &&
              close(second->smoothed.tx,
                    0.8 * second->unsmoothed.tx + 0.2 * previous.tx),
              "0.8/0.2 affine smoothing mismatch");

        // A demanded frame's EMA state includes every predecessor. Preparing
        // the exact prefix reproduces full-source transforms; sparse skipping
        // does not and is deliberately not a supported demand semantic.
        const std::array<double, 5> stateful_shifts{0.0, 6.0, 15.0, 27.0, 42.0};
        std::vector<vrhino::SimilarityAffine> full_transforms;
        vrhino::LipSyncDiffusionAlignmentState full_state;
        for (size_t index = 0; index < stateful_shifts.size(); ++index) {
            const double shift = stateful_shifts[index];
            const auto aligned = vrhino::align_lip_sync_diffusion_face(
                face_points(shift),
                {static_cast<float>(100.0 + shift), 100,
                 static_cast<float>(300.0 + shift), 300, .95f,
                 static_cast<int64_t>(index), -1}, full_state);
            check(aligned.has_value(),
                  "stateful full-source alignment fixture failed");
            full_transforms.push_back(aligned->smoothed);
        }
        vrhino::LipSyncDiffusionAlignmentState prefix_state;
        for (size_t index = 0; index < 3; ++index) {
            const double shift = stateful_shifts[index];
            const auto aligned = vrhino::align_lip_sync_diffusion_face(
                face_points(shift),
                {static_cast<float>(100.0 + shift), 100,
                 static_cast<float>(300.0 + shift), 300, .95f,
                 static_cast<int64_t>(index), -1}, prefix_state);
            check(aligned && same_affine(aligned->smoothed,
                                        full_transforms[index]),
                  "contiguous prefix changed stateful alignment semantics");
        }
        vrhino::LipSyncDiffusionAlignmentState sparse_state;
        check(vrhino::align_lip_sync_diffusion_face(
                  face_points(stateful_shifts[0]),
                  {100,100,300,300,.95f,0,-1}, sparse_state).has_value(),
              "sparse-state fixture first frame failed");
        const auto sparse_third = vrhino::align_lip_sync_diffusion_face(
            face_points(stateful_shifts[2]),
            {115,100,315,300,.95f,2,-1}, sparse_state);
        check(sparse_third &&
                  !same_affine(sparse_third->smoothed, full_transforms[2]),
              "sparse skipping unexpectedly preserved predecessor EMA state");

        const auto reset = vrhino::align_lip_sync_diffusion_face(
            face_points(600.0), {700, 100, 900, 300, 0.95f, 2, -1}, state);
        check(reset && reset->scene_reset &&
              close(reset->smoothed.tx, reset->unsmoothed.tx),
              "low-IoU scene reset retained stale affine state");
        auto invalid = face_points(600.0);
        invalid.points[23 + 17].confidence = 0.1f;
        check(!vrhino::align_lip_sync_diffusion_face(
                  invalid, {700,100,900,300,.95f,3,-1}, state) &&
              !state.previous && !state.previous_roi,
              "invalid DWPose geometry did not clear state");

        // Bounded workflow-safety fixtures. These exercise geometry policy,
        // never neural quality or a whole-frame fallback.
        vrhino::LipSyncDiffusionAlignmentState fixture_state;
        auto profile = face_points();
        for (int index : {31, 32, 34, 35})
            profile.points[static_cast<size_t>(23 + index)].x += 18.0;
        check(vrhino::align_lip_sync_diffusion_face(
                  profile, roi, fixture_state).has_value(),
              "profile fixture rejected usable geometry");
        check(vrhino::align_lip_sync_diffusion_face(
                  face_points(35.0), {135,100,335,300,.95f,1,-1},
                  fixture_state).has_value(),
              "large-head-motion fixture rejected usable geometry");
        auto occluded = face_points();
        occluded.points[23 + 31].confidence = 0.05f;
        check(!vrhino::align_lip_sync_diffusion_face(
                  occluded, roi, fixture_state) && !fixture_state.previous,
              "occlusion fixture reused stale geometry");
        fixture_state = {};
        check(!vrhino::align_lip_sync_diffusion_face(
                  face_points(), {600,600,800,800,.95f,1,-1}, fixture_state),
              "multiple-person fixture accepted points outside selected ROI");
        fixture_state = {};
        check(vrhino::align_lip_sync_diffusion_face(
                  face_points(), roi, fixture_state).has_value() &&
              vrhino::align_lip_sync_diffusion_face(
                  face_points(600.0), {700,100,900,300,.95f,2,-1},
                  fixture_state)->scene_reset,
              "scene-cut fixture did not reset smoothing");
        fixture_state = {};
        check(!vrhino::align_lip_sync_diffusion_face(
                  face_points(), {-300,-300,-100,-100,0.0f,-1,-1},
                  fixture_state),
              "no-face fixture did not fail closed");
        fixture_state = {};
        auto invalid_pose = face_points();
        invalid_pose.points.resize(90);
        check(!vrhino::align_lip_sync_diffusion_face(
                  invalid_pose, roi, fixture_state),
              "DWPose-invalid fixture did not fail closed");
        fixture_state = {};
        check(vrhino::align_lip_sync_diffusion_face(
                  face_points(-90.0), {0,90,220,310,.95f,0,-1},
                  fixture_state).has_value(),
              "edge-face fixture rejected bounded visible geometry");

        const auto audio = vrhino::plan_lip_sync_diffusion_audio(10880, 17);
        check(audio.retained_feature_frames == 34 &&
              audio.indices.front() ==
                  std::array<int64_t,10>({0,0,0,0,0,1,2,3,4,5}) &&
              audio.indices.back() ==
                  std::array<int64_t,10>({28,29,30,31,32,33,33,33,33,33}),
              "50-Hz audio window edge semantics mismatch");

        const auto chunks = vrhino::plan_temporal_frame_chunks(17);
        check(chunks.size() == 2 && chunks[0].start == 0 &&
              chunks[0].frames == 16 && chunks[1].start == 16 &&
              chunks[1].frames == 1,
              "non-overlapping temporal chunk plan mismatch");
        check(vrhino::plan_lip_sync_diffusion_source_frames(1, 4) ==
                  std::vector<int64_t>({0,0,0,0}) &&
              vrhino::plan_lip_sync_diffusion_source_frames(2, 8) ==
                  std::vector<int64_t>({0,1,1,0,0,1,1,0}) &&
              vrhino::plan_lip_sync_diffusion_source_frames(4, 10) ==
                  std::vector<int64_t>({0,1,2,3,3,2,1,0,0,1}),
              "forward/reverse frame cycling mismatch");

        const auto masked_a = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        const auto masked_b = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        const auto reference = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::ReferenceSource);
        const auto noise = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
        const auto masked_frame_9 = vrhino::lip_sync_diffusion_rng(
            1247, 9, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        const auto chunk_noise_0 = vrhino::lip_sync_diffusion_rng(
            1247, 0, vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
        const auto chunk_noise_16 = vrhino::lip_sync_diffusion_rng(
            1247, 16, vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
        const auto reordered_reference = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::ReferenceSource);
        const auto reordered_masked = vrhino::lip_sync_diffusion_rng(
            1247, 8, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        check(masked_a.seed == masked_b.seed && masked_a.offset == 0 &&
              masked_a.seed != reference.seed && masked_a.seed != noise.seed &&
              masked_a.seed != masked_frame_9.seed &&
              chunk_noise_0.seed != chunk_noise_16.seed &&
              reordered_reference.seed == reference.seed &&
              reordered_masked.seed == masked_a.seed,
              "explicit branch-local RNG assignment mismatch");

        auto mask = solid(256, 256, 255);
        std::fill(mask.pixels.begin(), mask.pixels.begin() + 3, 0);
        const auto resized_mask =
            vrhino::prepare_lip_sync_diffusion_fixed_mask(mask);
        check(resized_mask.width == 512 && resized_mask.height == 512,
              "fixed-mask resize contract mismatch");
        const auto vae = vrhino::prepare_lip_sync_diffusion_vae_input(
            {solid(512, 512, 255)}, resized_mask);
        check(vae.reference.shape() == std::vector<int64_t>({1,3,512,512}) &&
              vae.masked.shape() == vae.reference.shape() &&
              vae.mask.shape() == std::vector<int64_t>({1,1,512,512}),
              "full/masked VAE input shape mismatch");
        check(vae.reference.data_as<float>()[512 * 512 + 100] == 1.0f,
              "VAE RGB normalization mismatch");

        bool failed = false;
        try { (void)vrhino::plan_temporal_frame_chunks(0); }
        catch (const std::exception&) { failed = true; }
        check(failed, "empty temporal input did not fail closed");

        std::cout << "bounded lip-sync diffusion workflow CPU tests: PASS\n"
                  << "profile=PASS head_motion=PASS occlusion=PASS "
                     "multiple_person=PASS scene_cut=PASS no_face=PASS "
                     "dwpose_invalid=PASS edge_face=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bounded lip-sync diffusion workflow CPU tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
