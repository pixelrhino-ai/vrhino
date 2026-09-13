#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/backend.h"
#include "vrhino/face_mask.h"
#include "vrhino/loader.h"
#include "vrhino/tensor.h"

namespace vrhino {

enum class LipSyncStage {
    MediaInput,
    AudioConditioning,
    FaceAnalysis,
    SourcePreparation,
    ComponentExecution,
    Composite,
    MediaOutput,
};

class LipSyncWorkflowError final : public std::runtime_error {
public:
    LipSyncWorkflowError(LipSyncStage stage, const std::string& message);
    LipSyncStage stage() const noexcept { return stage_; }
private:
    LipSyncStage stage_;
};

struct RgbFrame {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> pixels;
};

struct VideoInput {
    int32_t width = 0;
    int32_t height = 0;
    int32_t fps_numerator = 0;
    int32_t fps_denominator = 1;
    std::vector<RgbFrame> frames;
};

struct AudioInput {
    int32_t sample_rate = 16000;
    int32_t channels = 1;
    std::vector<float> samples;
};

struct LipSyncWorkflowConfig {
    std::string family = "lip_sync_workflow_v1";
    int32_t fps = 25;
    int32_t output_frames = 0;
    int32_t crop_size = 256;
    int32_t lower_mask_rows = 128;
    int32_t bbox_shift = 0;
    int32_t lower_margin = 10;
    int32_t audio_padding_left = 2;
    int32_t audio_padding_right = 2;
    std::array<int32_t, 2> facial_keypoint_range{23, 91};
    std::array<int32_t, 2> cheek_widths{90, 90};
    uint64_t seed = 11001;
};

struct ComponentIdentityContract {
    std::string semantic_name;
    std::string architecture;
    uint64_t bytes = 0;
    std::string sha256;
};

struct MediaProbe {
    int32_t width = 0;
    int32_t height = 0;
    int32_t fps_numerator = 0;
    int32_t fps_denominator = 1;
    std::string video_codec;
    std::string pixel_format;
    bool has_audio = false;
    std::string audio_codec;
    int32_t audio_sample_rate = 0;
    int32_t audio_channels = 0;
};

struct MediaEncodeResult {
    uint64_t bytes = 0;
    std::filesystem::path output;
};

struct MediaEncodeContract {
    int32_t audio_sample_rate = 48000;
    int32_t audio_channels = 2;
    // A negative value retains the encoder default used by existing products.
    int32_t h264_crf = -1;
};

int64_t lip_sync_output_frame_count(int64_t audio_samples,
                                    int32_t sample_rate, int32_t fps);
std::vector<int64_t> ping_pong_frame_cycle(int64_t source_frames,
                                           int64_t output_frames);

// Conservative bounded demand for stateful source preparation. Every source
// frame in [0, required_prefix_frame_count) must be prepared so predecessor
// state remains identical for all referenced frames. A maximum of -1 denotes
// an empty output-frame plan.
struct SourceFrameDemandPlan {
    int64_t source_frame_count = 0;
    int64_t output_frame_count = 0;
    int64_t maximum_referenced_source_index = -1;
    int64_t required_prefix_frame_count = 0;
};

SourceFrameDemandPlan plan_source_frame_prefix_demand(
    int64_t source_frame_count,
    const std::vector<int64_t>& referenced_source_indices);

RngState lip_sync_component_rng(uint64_t workflow_seed, int64_t source_index,
                                uint64_t branch_identity);

VideoInput decode_video_rgb24(const std::filesystem::path& helper,
                              const std::filesystem::path& input,
                              int64_t max_frames,
                              const std::function<bool()>& cancelled = {});
AudioInput decode_audio_mono_f32_16khz(
    const std::filesystem::path& helper, const std::filesystem::path& input,
    int64_t max_samples, const std::function<bool()>& cancelled = {});
MediaProbe probe_media_bounded(const std::filesystem::path& helper,
                               const std::filesystem::path& input
#ifdef _WIN32
                               , const std::function<bool()>& cancelled = {}
#endif
                               );
MediaEncodeResult encode_mux_mp4_atomic(
    const std::filesystem::path& helper, const std::vector<RgbFrame>& frames,
    int32_t fps, const std::filesystem::path& driving_audio,
    const std::filesystem::path& output,
    const std::function<bool()>& cancelled = {}, bool overwrite = false,
    const MediaEncodeContract& contract = {});

RgbFrame crop_resize_lanczos4(const RgbFrame& source,
                              const std::array<int32_t, 4>& box,
                              int32_t output_width, int32_t output_height);
RgbFrame resize_bilinear_rgb(const RgbFrame& source, int32_t output_width,
                             int32_t output_height);
Tensor normalize_vae_rgb(const RgbFrame& source, bool mask_lower_half);
RgbFrame vae_rgb_tensor_to_frame(Backend& backend, const Tensor& rgb_0_1);
Tensor concatenate_latent_branches(Backend& backend, const Tensor& masked,
                                   const Tensor& full);
RgbFrame place_generated_crop(const RgbFrame& source,
                              const RgbFrame& generated,
                              const std::array<int32_t, 4>& face_box);
RgbFrame composite_rgb_alpha(const RgbFrame& source,
                             const RgbFrame& generated_region,
                             const AlphaMask& alpha,
                             const std::array<int32_t, 4>& expanded_box);

}  // namespace vrhino
