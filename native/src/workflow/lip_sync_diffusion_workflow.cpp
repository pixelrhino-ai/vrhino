#include "vrhino/lip_sync_diffusion_workflow.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "vrhino/error.h"
#include "vrhino/image_filter.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

constexpr std::array<Point2D, 3> kTargetAnchors{{
    {95.2, 112.0}, {324.8, 112.0}, {210.0, 224.0}}};

double roi_iou(const FaceROI& left, const FaceROI& right) {
    const double x1 = std::max<double>(left.x1, right.x1);
    const double y1 = std::max<double>(left.y1, right.y1);
    const double x2 = std::min<double>(left.x2, right.x2);
    const double y2 = std::min<double>(left.y2, right.y2);
    const double intersection = std::max(0.0, x2 - x1) *
                                std::max(0.0, y2 - y1);
    const double left_area = std::max<double>(0.0, left.x2 - left.x1) *
                             std::max<double>(0.0, left.y2 - left.y1);
    const double right_area = std::max<double>(0.0, right.x2 - right.x1) *
                              std::max<double>(0.0, right.y2 - right.y1);
    return intersection / std::max(1.0e-12,
        left_area + right_area - intersection);
}

Point2D mean_points(const std::vector<Keypoint>& points, int first, int last) {
    Point2D result;
    for (int index = first; index < last; ++index) {
        result.x += points[static_cast<size_t>(23 + index)].x;
        result.y += points[static_cast<size_t>(23 + index)].y;
    }
    const double count = last - first;
    result.x /= count;
    result.y /= count;
    return result;
}

SimilarityAffine fit_similarity(const std::array<Point2D, 3>& source) {
    Point2D source_mean, target_mean;
    for (size_t index = 0; index < source.size(); ++index) {
        source_mean.x += source[index].x;
        source_mean.y += source[index].y;
        target_mean.x += kTargetAnchors[index].x;
        target_mean.y += kTargetAnchors[index].y;
    }
    source_mean.x /= source.size(); source_mean.y /= source.size();
    target_mean.x /= source.size(); target_mean.y /= source.size();
    double dot = 0.0, cross = 0.0, denominator = 0.0;
    for (size_t index = 0; index < source.size(); ++index) {
        const double sx = source[index].x - source_mean.x;
        const double sy = source[index].y - source_mean.y;
        const double tx = kTargetAnchors[index].x - target_mean.x;
        const double ty = kTargetAnchors[index].y - target_mean.y;
        dot += sx * tx + sy * ty;
        cross += sx * ty - sy * tx;
        denominator += sx * sx + sy * sy;
    }
    require(denominator > 1.0e-12, "degenerate similarity-transform anchors");
    SimilarityAffine result;
    result.a = dot / denominator;
    result.b = cross / denominator;
    result.tx = target_mean.x - result.a * source_mean.x +
                result.b * source_mean.y;
    result.ty = target_mean.y - result.b * source_mean.x -
                result.a * source_mean.y;
    return result;
}

uint8_t bilinear_u8(const RgbFrame& source, int channel, double x, double y,
                    uint8_t border) {
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
            const double sample = sx >= 0 && sx < source.width && sy >= 0 &&
                    sy < source.height
                ? source.pixels[(static_cast<size_t>(sy) * source.width + sx) *
                    3 + channel]
                : border;
            value += sample * (dx ? fx : 1.0 - fx) *
                              (dy ? fy : 1.0 - fy);
        }
    }
    return static_cast<uint8_t>(std::clamp(
        std::floor(value + 0.5), 0.0, 255.0));
}

double cubic(double value) {
    constexpr double coefficient = -0.75;
    value = std::abs(value);
    if (value < 1.0)
        return (coefficient + 2.0) * value * value * value -
               (coefficient + 3.0) * value * value + 1.0;
    if (value < 2.0)
        return coefficient * value * value * value -
               5.0 * coefficient * value * value +
               8.0 * coefficient * value - 4.0 * coefficient;
    return 0.0;
}

std::vector<std::vector<std::pair<int, double>>> resize_weights(
        int input, int output) {
    const double scale = static_cast<double>(input) / output;
    const double support_scale = std::max(1.0, scale);
    const double support = 2.0 * support_scale;
    std::vector<std::vector<std::pair<int, double>>> result(
        static_cast<size_t>(output));
    for (int destination = 0; destination < output; ++destination) {
        const double center = (destination + 0.5) * scale - 0.5;
        const int begin = static_cast<int>(std::ceil(center - support));
        const int end = static_cast<int>(std::floor(center + support));
        double sum = 0.0;
        for (int source = begin; source <= end; ++source) {
            if (source < 0 || source >= input) continue;
            const double weight = cubic((center - source) / support_scale);
            if (weight == 0.0) continue;
            result[static_cast<size_t>(destination)].push_back({source, weight});
            sum += weight;
        }
        require(std::abs(sum) > 1.0e-12, "bicubic resize has empty support");
        for (auto& entry : result[static_cast<size_t>(destination)])
            entry.second /= sum;
    }
    return result;
}

std::vector<float> resize_bicubic_antialias_chw(
        const float* source, int channels, int input_height, int input_width,
        int output_height, int output_width) {
    const auto horizontal = resize_weights(input_width, output_width);
    const auto vertical = resize_weights(input_height, output_height);
    std::vector<float> intermediate(static_cast<size_t>(channels) *
        input_height * output_width);
    for (int channel = 0; channel < channels; ++channel)
        for (int y = 0; y < input_height; ++y)
            for (int x = 0; x < output_width; ++x) {
                double value = 0.0;
                for (const auto& [sx, weight] : horizontal[static_cast<size_t>(x)])
                    value += source[(static_cast<size_t>(channel) * input_height + y) *
                                    input_width + sx] * weight;
                intermediate[(static_cast<size_t>(channel) * input_height + y) *
                             output_width + x] = static_cast<float>(value);
            }
    std::vector<float> output(static_cast<size_t>(channels) * output_height *
                              output_width);
    for (int channel = 0; channel < channels; ++channel)
        for (int y = 0; y < output_height; ++y)
            for (int x = 0; x < output_width; ++x) {
                double value = 0.0;
                for (const auto& [sy, weight] : vertical[static_cast<size_t>(y)])
                    value += intermediate[(static_cast<size_t>(channel) * input_height +
                        sy) * output_width + x] * weight;
                output[(static_cast<size_t>(channel) * output_height + y) *
                       output_width + x] = static_cast<float>(value);
            }
    return output;
}

float bilinear_f32(const std::vector<float>& source, int height, int width,
                   int channel, double x, double y, float border) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const double fx = x - ix, fy = y - iy;
    double value = 0.0;
    for (int dy = 0; dy <= 1; ++dy)
        for (int dx = 0; dx <= 1; ++dx) {
            const int sx = ix + dx, sy = iy + dy;
            const float sample = sx >= 0 && sx < width && sy >= 0 && sy < height
                ? source[(static_cast<size_t>(channel) * height + sy) * width + sx]
                : border;
            value += sample * (dx ? fx : 1.0 - fx) *
                              (dy ? fy : 1.0 - fy);
        }
    return static_cast<float>(value);
}

int reflect101(int value, int size) {
    while (value < 0 || value >= size) {
        if (value < 0) value = -value;
        else value = 2 * size - value - 2;
    }
    return value;
}

std::vector<float> gaussian_blur(const std::vector<float>& source, int height,
                                 int width, int kernel, double sigma) {
    const int radius = kernel / 2;
    std::vector<double> weights(static_cast<size_t>(kernel));
    double total = 0.0;
    for (int index = -radius; index <= radius; ++index) {
        const double value = std::exp(-0.5 * index * index / (sigma * sigma));
        weights[static_cast<size_t>(index + radius)] = value;
        total += value;
    }
    for (double& value : weights) value /= total;
    std::vector<float> horizontal(source.size()), output(source.size());
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double value = 0.0;
            for (int index = -radius; index <= radius; ++index)
                value += source[static_cast<size_t>(y) * width +
                    reflect101(x + index, width)] *
                    weights[static_cast<size_t>(index + radius)];
            horizontal[static_cast<size_t>(y) * width + x] =
                static_cast<float>(value);
        }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double value = 0.0;
            for (int index = -radius; index <= radius; ++index)
                value += horizontal[static_cast<size_t>(reflect101(y + index,
                    height)) * width + x] *
                    weights[static_cast<size_t>(index + radius)];
            output[static_cast<size_t>(y) * width + x] =
                static_cast<float>(value);
        }
    return output;
}

}  // namespace

std::array<double, 6> SimilarityAffine::matrix() const {
    return {a, -b, tx, b, a, ty};
}

SimilarityAffine SimilarityAffine::inverse() const {
    const double determinant = a * a + b * b;
    require(determinant > 1.0e-18, "singular similarity transform");
    SimilarityAffine result;
    result.a = a / determinant;
    result.b = -b / determinant;
    result.tx = -(result.a * tx - result.b * ty);
    result.ty = -(result.b * tx + result.a * ty);
    return result;
}

std::optional<LipSyncDiffusionAlignment> align_lip_sync_diffusion_face(
        const KeypointSet& keypoints, const FaceROI& selected_roi,
        LipSyncDiffusionAlignmentState& state) {
    auto invalid = [&]() -> std::optional<LipSyncDiffusionAlignment> {
        state.reset();
        return std::nullopt;
    };
    if (keypoints.coordinate_space != "source_image_pixels" ||
        keypoints.points.size() != 133 || selected_roi.x2 <= selected_roi.x1 ||
        selected_roi.y2 <= selected_roi.y1)
        return invalid();
    float minimum = std::numeric_limits<float>::infinity();
    for (int index = 17; index < 27; ++index)
        minimum = std::min(minimum,
            keypoints.points[static_cast<size_t>(23 + index)].confidence);
    for (int index : {31, 32, 34, 35})
        minimum = std::min(minimum,
            keypoints.points[static_cast<size_t>(23 + index)].confidence);
    if (!(minimum >= 0.30f)) return invalid();

    Point2D centroid;
    for (int index = 23; index < 91; ++index) {
        centroid.x += keypoints.points[static_cast<size_t>(index)].x;
        centroid.y += keypoints.points[static_cast<size_t>(index)].y;
    }
    centroid.x /= 68.0; centroid.y /= 68.0;
    if (centroid.x < selected_roi.x1 || centroid.x > selected_roi.x2 ||
        centroid.y < selected_roi.y1 || centroid.y > selected_roi.y2)
        return invalid();

    LipSyncDiffusionAlignment result;
    result.source_anchors[0] = mean_points(keypoints.points, 17, 22);
    result.source_anchors[1] = mean_points(keypoints.points, 22, 27);
    if (result.source_anchors[0].x > result.source_anchors[1].x)
        std::swap(result.source_anchors[0], result.source_anchors[1]);
    for (int index : {31, 32, 34, 35}) {
        result.source_anchors[2].x +=
            keypoints.points[static_cast<size_t>(23 + index)].x;
        result.source_anchors[2].y +=
            keypoints.points[static_cast<size_t>(23 + index)].y;
    }
    result.source_anchors[2].x /= 4.0;
    result.source_anchors[2].y /= 4.0;
    result.unsmoothed = fit_similarity(result.source_anchors);
    result.scene_reset = !state.previous_roi ||
        roi_iou(*state.previous_roi, selected_roi) < 0.10;
    result.smoothed = result.unsmoothed;
    if (state.previous && !result.scene_reset) {
        result.smoothed.a = 0.8 * result.unsmoothed.a + 0.2 * state.previous->a;
        result.smoothed.b = 0.8 * result.unsmoothed.b + 0.2 * state.previous->b;
        result.smoothed.tx = 0.8 * result.unsmoothed.tx + 0.2 * state.previous->tx;
        result.smoothed.ty = 0.8 * result.unsmoothed.ty + 0.2 * state.previous->ty;
    }
    result.inverse = result.smoothed.inverse();
    result.minimum_anchor_confidence = minimum;
    state.previous = result.smoothed;
    state.previous_roi = selected_roi;
    return result;
}

RgbFrame warp_lip_sync_diffusion_face(const RgbFrame& source,
                                       const SimilarityAffine& transform) {
    require(source.width > 0 && source.height > 0 &&
                source.pixels.size() == static_cast<size_t>(source.width) *
                    source.height * 3,
            "invalid alignment source frame");
    const SimilarityAffine inverse = transform.inverse();
    RgbFrame intermediate{420, 560, std::vector<uint8_t>(420 * 560 * 3)};
    for (int y = 0; y < intermediate.height; ++y)
        for (int x = 0; x < intermediate.width; ++x) {
            const double sx = inverse.a * x - inverse.b * y + inverse.tx;
            const double sy = inverse.b * x + inverse.a * y + inverse.ty;
            for (int channel = 0; channel < 3; ++channel)
                intermediate.pixels[(static_cast<size_t>(y) * 420 + x) * 3 +
                    channel] = bilinear_u8(source, channel, sx, sy, 127);
        }
    return crop_resize_lanczos4(intermediate, {0, 0, 420, 560}, 512, 512);
}

LipSyncDiffusionAudioPlan plan_lip_sync_diffusion_audio(
        int64_t waveform_samples, int64_t output_frames, int64_t sample_rate,
        int64_t feature_rate, int64_t video_fps) {
    require(waveform_samples > 0 && output_frames > 0 && sample_rate > 0 &&
                feature_rate > 0 && video_fps > 0,
            "invalid lip-sync diffusion audio plan");
    LipSyncDiffusionAudioPlan result;
    result.retained_feature_frames = waveform_samples * feature_rate / sample_rate;
    require(result.retained_feature_frames > 0,
            "lip-sync diffusion audio has no retained features");
    result.indices.reserve(static_cast<size_t>(output_frames));
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        const int64_t center = frame * feature_rate / video_fps;
        std::array<int64_t, 10> indices{};
        for (int offset = -4; offset <= 5; ++offset)
            indices[static_cast<size_t>(offset + 4)] = std::clamp<int64_t>(
                center + offset, 0, result.retained_feature_frames - 1);
        result.indices.push_back(indices);
    }
    return result;
}

Tensor assemble_lip_sync_diffusion_audio_windows(
        Backend& backend, const Tensor& stacked, int64_t waveform_samples,
        int64_t output_frames) {
    require(stacked.ndim() == 4 && stacked.dtype() == DType::F32 &&
                stacked.dim(0) == 1 && stacked.dim(2) == 5 &&
                stacked.dim(3) == 384,
            "invalid lip-sync diffusion retained-state tensor");
    const auto plan = plan_lip_sync_diffusion_audio(
        waveform_samples, output_frames);
    require(plan.retained_feature_frames <= stacked.dim(1),
            "retained audio extent exceeds encoder output");
    std::vector<Tensor> frames;
    frames.reserve(static_cast<size_t>(output_frames));
    for (const auto& indices : plan.indices) {
        std::vector<Tensor> positions;
        positions.reserve(indices.size());
        for (int64_t index : indices)
            positions.push_back(backend.reshape(
                backend.slice(stacked, 1, index, index + 1), {1, 5, 384}));
        frames.push_back(backend.concat(positions, 1));
    }
    return backend.concat(frames, 0);
}

RgbFrame prepare_lip_sync_diffusion_fixed_mask(const RgbFrame& fixed_mask) {
    require(fixed_mask.width == 256 && fixed_mask.height == 256,
            "fixed mask must be 256x256 RGB");
    return crop_resize_lanczos4(fixed_mask, {0, 0, 256, 256}, 512, 512);
}

LipSyncDiffusionVaeInput prepare_lip_sync_diffusion_vae_input(
        const std::vector<RgbFrame>& aligned_frames,
        const RgbFrame& resized_fixed_mask) {
    require(!aligned_frames.empty() && resized_fixed_mask.width == 512 &&
                resized_fixed_mask.height == 512,
            "invalid fixed-mask VAE input");
    const int64_t frames = static_cast<int64_t>(aligned_frames.size());
    Tensor reference = Tensor::host({frames, 3, 512, 512}, DType::F32);
    Tensor masked = Tensor::host({frames, 3, 512, 512}, DType::F32);
    Tensor mask = Tensor::host({frames, 1, 512, 512}, DType::F32);
    float* reference_values = reference.data_as<float>();
    float* masked_values = masked.data_as<float>();
    float* mask_values = mask.data_as<float>();
    for (int64_t frame = 0; frame < frames; ++frame) {
        const RgbFrame& image = aligned_frames[static_cast<size_t>(frame)];
        require(image.width == 512 && image.height == 512,
                "aligned VAE frame must be 512x512");
        for (int64_t y = 0; y < 512; ++y)
            for (int64_t x = 0; x < 512; ++x) {
                const int64_t pixel = y * 512 + x;
                const float alpha = resized_fixed_mask.pixels[
                    static_cast<size_t>(pixel) * 3] / 255.0f;
                mask_values[(frame * 512 + y) * 512 + x] = alpha;
                for (int channel = 0; channel < 3; ++channel) {
                    const float value = image.pixels[
                        static_cast<size_t>(pixel) * 3 + channel] / 127.5f - 1.0f;
                    const int64_t offset = ((frame * 3 + channel) * 512 + y) *
                        512 + x;
                    reference_values[offset] = value;
                    masked_values[offset] = value * alpha;
                }
            }
    }
    return {std::move(reference), std::move(masked), std::move(mask)};
}

RngState lip_sync_diffusion_rng(uint64_t workflow_seed,
        int64_t semantic_index, LipSyncDiffusionRngBranch branch) {
    return lip_sync_component_rng(workflow_seed, semantic_index,
                                  static_cast<uint64_t>(branch));
}

std::vector<TemporalFrameChunk> plan_temporal_frame_chunks(
        int64_t frames, int64_t maximum_chunk) {
    require(frames > 0 && maximum_chunk > 0 && maximum_chunk <= 24,
            "invalid temporal frame chunk plan");
    std::vector<TemporalFrameChunk> result;
    for (int64_t start = 0; start < frames; start += maximum_chunk)
        result.push_back({start, std::min(maximum_chunk, frames - start)});
    return result;
}

std::vector<int64_t> plan_lip_sync_diffusion_source_frames(
        int64_t source_frames, int64_t output_frames) {
    return ping_pong_frame_cycle(source_frames, output_frames);
}

Tensor assemble_lip_sync_diffusion_unet_input(
        Backend& backend, const Tensor& noisy_latent,
        const Tensor& fixed_mask_512, const Tensor& masked_source_latent,
        const Tensor& reference_source_latent) {
    require(noisy_latent.ndim() == 5 && noisy_latent.dim(0) == 1 &&
                noisy_latent.dim(1) == 4 && noisy_latent.dim(3) == 64 &&
                noisy_latent.dim(4) == 64 && fixed_mask_512.ndim() == 4 &&
                fixed_mask_512.dim(0) == noisy_latent.dim(2) &&
                fixed_mask_512.dim(1) == 1 && fixed_mask_512.dim(2) == 512 &&
                fixed_mask_512.dim(3) == 512 &&
                masked_source_latent.shape() == reference_source_latent.shape() &&
                masked_source_latent.ndim() == 4 &&
                masked_source_latent.dim(0) == noisy_latent.dim(2) &&
                masked_source_latent.dim(1) == 4 &&
                masked_source_latent.dim(2) == 64 &&
                masked_source_latent.dim(3) == 64,
            "invalid lip-sync diffusion UNet assembly contract");
    const int64_t frames = noisy_latent.dim(2);
    Tensor mask = backend.interpolate_nearest(fixed_mask_512, {0.125, 0.125});
    mask = backend.reshape(backend.permute(mask, {1, 0, 2, 3}),
                           {1, 1, frames, 64, 64});
    Tensor masked = backend.reshape(backend.permute(masked_source_latent,
        {1, 0, 2, 3}), {1, 4, frames, 64, 64});
    Tensor reference = backend.reshape(backend.permute(reference_source_latent,
        {1, 0, 2, 3}), {1, 4, frames, 64, 64});
    return backend.concat({noisy_latent, mask, masked, reference}, 1);
}

Tensor temporal_latents_to_frame_batch(Backend& backend,
                                       const Tensor& temporal_latents) {
    require(temporal_latents.ndim() == 5 &&
                temporal_latents.dtype() == DType::F32,
            "temporal latent frame-batch lowering requires FP32 BCFHW");
    Tensor frames_first = backend.permute(temporal_latents, {0, 2, 1, 3, 4});
    return backend.reshape(frames_first,
        {temporal_latents.dim(0) * temporal_latents.dim(2),
         temporal_latents.dim(1), temporal_latents.dim(3),
         temporal_latents.dim(4)});
}

Tensor reconstruct_lip_sync_diffusion_crop(
        Backend& backend, const Tensor& decoded_normalized,
        const Tensor& reference_normalized, const Tensor& fixed_mask_512) {
    require(decoded_normalized.shape() == reference_normalized.shape() &&
                decoded_normalized.ndim() == 4 &&
                decoded_normalized.dim(1) == 3 &&
                fixed_mask_512.ndim() == 4 &&
                fixed_mask_512.dim(0) == decoded_normalized.dim(0) &&
                fixed_mask_512.dim(1) == 1 &&
                fixed_mask_512.dim(2) == decoded_normalized.dim(2) &&
                fixed_mask_512.dim(3) == decoded_normalized.dim(3),
            "invalid lip-sync diffusion crop reconstruction contract");
    Tensor mask3 = backend.concat(
        {fixed_mask_512, fixed_mask_512, fixed_mask_512}, 1);
    Tensor inverse = backend.add(
        scalar_f32(1.0f), backend.mul(mask3, scalar_f32(-1.0f)));
    return backend.add(backend.mul(decoded_normalized, inverse),
                       backend.mul(reference_normalized, mask3));
}

RgbFrame composite_lip_sync_diffusion_face(
        const RgbFrame& source, const Tensor& reconstructed_face_chw,
        const SimilarityAffine& transform,
        LipSyncDiffusionCompositeObservation* observation) {
    require(reconstructed_face_chw.device().is_host() &&
                reconstructed_face_chw.dtype() == DType::F32 &&
                reconstructed_face_chw.shape() == std::vector<int64_t>({3, 512, 512}) &&
                source.width > 0 && source.height > 0,
            "invalid lip-sync diffusion composite contract");
    const std::vector<float> resized = resize_bicubic_antialias_chw(
        reconstructed_face_chw.data_as<float>(), 3, 512, 512, 560, 420);
    const size_t source_pixels = static_cast<size_t>(source.width) * source.height;
    std::vector<float> inverse_face(source_pixels * 3);
    std::vector<float> inverse_mask(source_pixels);
    const std::vector<float> aligned_ones(560 * 420, 1.0f);
    for (int y = 0; y < source.height; ++y)
        for (int x = 0; x < source.width; ++x) {
            const double face_x = transform.a * x - transform.b * y + transform.tx;
            const double face_y = transform.b * x + transform.a * y + transform.ty;
            const size_t pixel = static_cast<size_t>(y) * source.width + x;
            inverse_mask[pixel] = bilinear_f32(
                aligned_ones, 560, 420, 0, face_x, face_y, 0.0f);
            for (int channel = 0; channel < 3; ++channel) {
                const float normalized = bilinear_f32(resized, 560, 420,
                    channel, face_x, face_y, 127.0f);
                inverse_face[pixel * 3 + channel] = std::clamp(
                    (normalized * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
            }
        }
    std::vector<float> eroded = minimum_filter_square_zero_border_f32(
        inverse_mask, source.height, source.width, 2, 1);
    long double total_area = 0.0;
    for (float value : eroded) total_area += value;
    const int edge = static_cast<int>(std::sqrt(total_area)) / 20;
    const int radius = std::max(1, edge * 2);
    std::vector<float> center = minimum_filter_square_zero_border_f32(
        eroded, source.height, source.width, radius, radius / 2);
    const int blur_size = edge * 2 + 1;
    const double sigma = 0.3 * ((blur_size - 1) * 0.5 - 1.0) + 0.8;
    std::vector<float> soft = gaussian_blur(
        center, source.height, source.width, blur_size, sigma);
    RgbFrame result = source;
    for (size_t pixel = 0; pixel < source_pixels; ++pixel)
        for (int channel = 0; channel < 3; ++channel) {
            const float value = soft[pixel] * inverse_face[pixel * 3 + channel] +
                (1.0f - soft[pixel]) * source.pixels[pixel * 3 + channel];
            result.pixels[pixel * 3 + channel] = static_cast<uint8_t>(
                std::clamp(value, 0.0f, 255.0f));
        }
    if (observation) {
        observation->inverse_warp_rgb = std::move(inverse_face);
        observation->soft_mask = std::move(soft);
        observation->feather_edge = edge;
        observation->feather_radius = radius;
    }
    return result;
}

}  // namespace vrhino
