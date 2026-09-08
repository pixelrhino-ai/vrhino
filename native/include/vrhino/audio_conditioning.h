#pragma once

#include <cstdint>
#include <vector>

#include "vrhino/backend.h"
#include "vrhino/json.h"

namespace vrhino {

struct AudioFeatureWindowConfig {
    int64_t sample_rate = 16000;
    int64_t feature_rate = 50;
    int64_t video_fps = 25;
    int64_t left_context_frames = 2;
    int64_t right_context_frames = 2;
};

struct AudioFeatureWindowPlan {
    int64_t actual_feature_frames = 0;
    int64_t hidden_frames_per_video_frame = 0;
    int64_t left_padding = 0;
    int64_t right_padding = 0;
    int64_t window_feature_frames = 0;
    std::vector<int64_t> starts;
};

// Bounded product/audio preprocessing. Input is mono FP32 waveform at 16 kHz;
// output is the fixed 30-second [1,80,3000] Whisper log-mel contract.
Tensor whisper_log_mel_80(const Tensor& waveform,
                          const Json& preprocessor_config);

// OpenAI Whisper's variable-duration frontend computes the centered STFT over
// only the decoded waveform, drops the final STFT column, normalizes that
// actual extent, and then right-pads mel columns to the encoder contract.
Tensor whisper_log_mel_80_variable_audio(
    const Tensor& waveform, const Json& preprocessor_config);

// Pure planning boundary used by both product execution and CPU validation.
// It makes the asymmetric MuseTalk audio-window padding explicit without
// introducing model identity into Runtime or Backend.
AudioFeatureWindowPlan plan_audio_feature_windows(
    int64_t waveform_samples, int64_t encoder_feature_frames,
    int64_t output_frames, const AudioFeatureWindowConfig& config = {});

// Deterministic host-side generic positional encoding. The Backend-facing
// operation below only adds this bounded constant to the component output.
Tensor sinusoidal_position_encoding_f32(int64_t tokens, int64_t width,
                                        double max_timescale = 10000.0);

// Bounded, typed audio-conditioning workflow operations. Neural execution is
// owned by Shared Runtime/Backend; these operations only assemble component
// outputs and align them to video frames.
Tensor stack_audio_encoder_states(Backend& backend,
                                  const std::vector<Tensor>& states);
Tensor frame_audio_feature_windows(Backend& backend, const Tensor& stacked,
                                   int64_t waveform_samples,
                                   int64_t output_frames,
                                   const AudioFeatureWindowConfig& config = {});
Tensor add_sinusoidal_position_encoding(Backend& backend,
                                        const Tensor& conditioning,
                                        double max_timescale = 10000.0);

}  // namespace vrhino
