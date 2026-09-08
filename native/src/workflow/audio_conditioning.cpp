#include "vrhino/audio_conditioning.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

constexpr int64_t kSampleRate = 16000;
constexpr int64_t kFft = 400;
constexpr int64_t kHop = 160;
constexpr int64_t kMelBins = 80;
constexpr int64_t kSamples = 30 * kSampleRate;
constexpr int64_t kFrames = kSamples / kHop;

std::vector<float> mel_filter_values(const Json& config) {
    require(config.at("feature_size").integer() == kMelBins &&
            config.at("hop_length").integer() == kHop &&
            config.at("n_fft").integer() == kFft &&
            config.at("sampling_rate").integer() == kSampleRate &&
            config.at("n_samples").integer() == kSamples,
            "Unsupported Whisper feature-extractor contract");
    const auto& rows = config.at("mel_filters").array();
    // Transformers serializes this derived field transposed as [mel,freq] in
    // this frozen preprocessor config, while the runtime extractor rebuilds
    // [freq,mel]. Accept only those two exact orientations and canonicalize.
    const bool transposed = rows.size() == static_cast<size_t>(kMelBins);
    require(transposed || rows.size() == static_cast<size_t>(kFft / 2 + 1),
            "Whisper mel-filter dimension mismatch");
    std::vector<float> values(static_cast<size_t>((kFft / 2 + 1) * kMelBins));
    for (size_t outer = 0; outer < rows.size(); ++outer) {
        const auto& row = rows[outer].array();
        require(row.size() == static_cast<size_t>(transposed ? kFft / 2 + 1 : kMelBins),
                "Whisper mel-filter row width mismatch");
        for (size_t inner = 0; inner < row.size(); ++inner) {
            const size_t frequency = transposed ? inner : outer;
            const size_t mel = transposed ? outer : inner;
            values[frequency * kMelBins + mel] =
                static_cast<float>(row[inner].number());
        }
    }
    return values;
}

std::vector<float> centered_reflect_pad(const float* waveform, int64_t count) {
    require(count > kFft / 2, "Whisper waveform is too short for reflect padding");
    std::vector<float> padded(static_cast<size_t>(count + kFft), 0.0f);
    const int64_t half = kFft / 2;
    std::copy(waveform, waveform + count, padded.begin() + half);
    for (int64_t index = 0; index < half; ++index) {
        padded[static_cast<size_t>(half - 1 - index)] = waveform[index + 1];
        padded[static_cast<size_t>(half + count + index)] = waveform[count - 2 - index];
    }
    return padded;
}

}  // namespace

Tensor whisper_log_mel_80(const Tensor& waveform, const Json& config) {
    require(waveform.device().is_host() && waveform.dtype() == DType::F32 &&
            waveform.ndim() == 1 && waveform.numel() > 0 &&
            waveform.numel() <= kSamples,
            "Whisper waveform must be mono FP32 with at most 30 seconds at 16 kHz");
    const std::vector<float> filters = mel_filter_values(config);
    std::vector<float> fixed(static_cast<size_t>(kSamples), 0.0f);
    std::copy(waveform.data_as<float>(),
              waveform.data_as<float>() + waveform.numel(), fixed.begin());
    const std::vector<float> padded = centered_reflect_pad(fixed.data(), kSamples);

    std::vector<double> window(kFft);
    for (int64_t sample = 0; sample < kFft; ++sample)
        window[static_cast<size_t>(sample)] = 0.5 - 0.5 *
            std::cos(2.0 * std::numbers::pi * sample / static_cast<double>(kFft));
    std::vector<double> cosine(static_cast<size_t>((kFft / 2 + 1) * kFft));
    std::vector<double> sine(cosine.size());
    for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency) {
        for (int64_t sample = 0; sample < kFft; ++sample) {
            const double phase = 2.0 * std::numbers::pi * frequency * sample /
                                 static_cast<double>(kFft);
            const size_t offset = static_cast<size_t>(frequency * kFft + sample);
            cosine[offset] = std::cos(phase);
            sine[offset] = -std::sin(phase);
        }
    }

    // Center padding creates 3001 STFT frames. The reference drops the final
    // frame, leaving exactly 3000 feature columns.
    std::vector<float> power(static_cast<size_t>((kFft / 2 + 1) * kFrames));
    for (int64_t frame = 0; frame < kFrames; ++frame) {
        const int64_t base = frame * kHop;
        for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency) {
            double real = 0.0, imaginary = 0.0;
            const size_t twiddle = static_cast<size_t>(frequency * kFft);
            for (int64_t sample = 0; sample < kFft; ++sample) {
                const double value = static_cast<double>(padded[static_cast<size_t>(base + sample)]) *
                                     window[static_cast<size_t>(sample)];
                real += value * cosine[twiddle + static_cast<size_t>(sample)];
                imaginary += value * sine[twiddle + static_cast<size_t>(sample)];
            }
            const float real32 = static_cast<float>(real);
            const float imaginary32 = static_cast<float>(imaginary);
            power[static_cast<size_t>(frequency * kFrames + frame)] =
                real32 * real32 + imaginary32 * imaginary32;
        }
    }

    std::vector<float> output(static_cast<size_t>(kMelBins * kFrames));
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t mel = 0; mel < kMelBins; ++mel) {
        for (int64_t frame = 0; frame < kFrames; ++frame) {
            float sum = 0.0f;
            for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency)
                sum += filters[static_cast<size_t>(frequency * kMelBins + mel)] *
                       power[static_cast<size_t>(frequency * kFrames + frame)];
            const float value = std::log10(std::max(1.0e-10f, sum));
            output[static_cast<size_t>(mel * kFrames + frame)] = value;
            maximum = std::max(maximum, value);
        }
    }
    for (float& value : output)
        value = (std::max(value, maximum - 8.0f) + 4.0f) / 4.0f;
    return host_f32({1, kMelBins, kFrames}, output);
}

Tensor whisper_log_mel_80_variable_audio(const Tensor& waveform,
                                         const Json& config) {
    require(waveform.device().is_host() && waveform.dtype() == DType::F32 &&
            waveform.ndim() == 1 && waveform.numel() > kFft / 2 &&
            waveform.numel() <= kSamples,
            "Whisper variable waveform must be mono FP32 at 16 kHz");
    const std::vector<float> filters = mel_filter_values(config);
    const int64_t actual_frames = waveform.numel() / kHop;
    require(actual_frames > 0 && actual_frames <= kFrames,
            "Whisper variable waveform feature extent is invalid");
    const std::vector<float> padded = centered_reflect_pad(
        waveform.data_as<float>(), waveform.numel());
    std::vector<double> window(kFft);
    for (int64_t sample = 0; sample < kFft; ++sample)
        window[static_cast<size_t>(sample)] = 0.5 - 0.5 *
            std::cos(2.0 * std::numbers::pi * sample / static_cast<double>(kFft));
    std::vector<double> cosine(static_cast<size_t>((kFft / 2 + 1) * kFft));
    std::vector<double> sine(cosine.size());
    for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency)
        for (int64_t sample = 0; sample < kFft; ++sample) {
            const double phase = 2.0 * std::numbers::pi * frequency * sample /
                                 static_cast<double>(kFft);
            const size_t offset = static_cast<size_t>(frequency * kFft + sample);
            cosine[offset] = std::cos(phase);
            sine[offset] = -std::sin(phase);
        }
    std::vector<float> power(
        static_cast<size_t>((kFft / 2 + 1) * actual_frames));
    for (int64_t frame = 0; frame < actual_frames; ++frame)
        for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency) {
            double real = 0.0, imaginary = 0.0;
            for (int64_t sample = 0; sample < kFft; ++sample) {
                const double value = padded[static_cast<size_t>(frame * kHop + sample)] *
                                     window[static_cast<size_t>(sample)];
                const size_t offset = static_cast<size_t>(frequency * kFft + sample);
                real += value * cosine[offset];
                imaginary += value * sine[offset];
            }
            const float real32 = static_cast<float>(real);
            const float imaginary32 = static_cast<float>(imaginary);
            power[static_cast<size_t>(frequency * actual_frames + frame)] =
                real32 * real32 + imaginary32 * imaginary32;
        }
    std::vector<float> output(static_cast<size_t>(kMelBins * kFrames), 0.0f);
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t mel = 0; mel < kMelBins; ++mel)
        for (int64_t frame = 0; frame < actual_frames; ++frame) {
            float sum = 0.0f;
            for (int64_t frequency = 0; frequency <= kFft / 2; ++frequency)
                sum += filters[static_cast<size_t>(frequency * kMelBins + mel)] *
                       power[static_cast<size_t>(frequency * actual_frames + frame)];
            const float value = std::log10(std::max(1.0e-10f, sum));
            output[static_cast<size_t>(mel * kFrames + frame)] = value;
            maximum = std::max(maximum, value);
        }
    for (int64_t mel = 0; mel < kMelBins; ++mel)
        for (int64_t frame = 0; frame < actual_frames; ++frame) {
            float& value = output[static_cast<size_t>(mel * kFrames + frame)];
            value = (std::max(value, maximum - 8.0f) + 4.0f) / 4.0f;
        }
    return host_f32({1, kMelBins, kFrames}, output);
}

AudioFeatureWindowPlan plan_audio_feature_windows(
        const int64_t waveform_samples, const int64_t encoder_feature_frames,
        const int64_t output_frames, const AudioFeatureWindowConfig& config) {
    require(waveform_samples >= 0 && encoder_feature_frames > 0 &&
            output_frames > 0 && config.sample_rate > 0 &&
            config.feature_rate > 0 && config.video_fps > 0 &&
            config.left_context_frames >= 0 && config.right_context_frames >= 0,
            "Invalid AudioFeatureWindow planning contract");
    AudioFeatureWindowPlan plan;
    plan.actual_feature_frames =
        waveform_samples * config.feature_rate / config.sample_rate;
    require(plan.actual_feature_frames > 0 &&
            plan.actual_feature_frames <= encoder_feature_frames,
            "AudioFeatureWindow source duration exceeds encoder output");
    plan.hidden_frames_per_video_frame = config.feature_rate / config.video_fps;
    require(plan.hidden_frames_per_video_frame > 0 &&
            plan.hidden_frames_per_video_frame * config.video_fps ==
                config.feature_rate,
            "AudioFeatureWindow rates must have an integral ratio");
    plan.left_padding =
        config.left_context_frames * plan.hidden_frames_per_video_frame;
    // The frozen audio processor pads three right-context spans so tail
    // windows remain defined, while each selected window contains left +
    // current + right spans. These are bounded workflow semantics, not
    // Whisper graph behavior.
    plan.right_padding = 3 * config.right_context_frames *
        plan.hidden_frames_per_video_frame;
    plan.window_feature_frames = plan.hidden_frames_per_video_frame *
        (config.left_context_frames + config.right_context_frames + 1);
    const int64_t padded_frames = plan.left_padding +
        plan.actual_feature_frames + plan.right_padding;
    plan.starts.reserve(static_cast<size_t>(output_frames));
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        const int64_t start = frame * config.feature_rate / config.video_fps;
        require(start + plan.window_feature_frames <= padded_frames,
                "AudioFeatureWindow output extends beyond declared padding");
        plan.starts.push_back(start);
    }
    return plan;
}

Tensor sinusoidal_position_encoding_f32(const int64_t tokens,
                                        const int64_t width,
                                        const double max_timescale) {
    require(tokens > 0 && width > 0 && width % 2 == 0 &&
            max_timescale > 1.0,
            "Invalid sinusoidal position-encoding contract");
    std::vector<float> values(static_cast<size_t>(tokens * width));
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t pair = 0; pair < width / 2; ++pair) {
            const float exponent = static_cast<float>(2 * pair) *
                static_cast<float>(-std::log(max_timescale) / width);
            const float phase = static_cast<float>(token) * std::exp(exponent);
            values[static_cast<size_t>(token * width + 2 * pair)] = std::sin(phase);
            values[static_cast<size_t>(token * width + 2 * pair + 1)] = std::cos(phase);
        }
    }
    return host_f32({1, tokens, width}, values);
}

Tensor stack_audio_encoder_states(Backend& backend,
                                  const std::vector<Tensor>& states) {
    require(!states.empty(), "Audio state stack cannot be empty");
    const std::vector<int64_t> shape = states.front().shape();
    require(shape.size() == 3, "Audio encoder states must be BLC");
    std::vector<Tensor> expanded;
    expanded.reserve(states.size());
    for (const Tensor& state : states) {
        require(state.shape() == shape && state.dtype() == DType::F32,
                "Audio encoder state stack shape/dtype mismatch");
        expanded.push_back(backend.reshape(state,
            {shape[0], shape[1], 1, shape[2]}));
    }
    return backend.concat(expanded, 2);
}

Tensor frame_audio_feature_windows(Backend& backend, const Tensor& stacked,
                                   const int64_t waveform_samples,
                                   const int64_t output_frames,
                                   const AudioFeatureWindowConfig& config) {
    require(stacked.ndim() == 4 && stacked.dtype() == DType::F32 &&
            stacked.dim(0) == 1,
            "Invalid AudioFeatureWindow contract");
    const AudioFeatureWindowPlan plan = plan_audio_feature_windows(
        waveform_samples, stacked.dim(1), output_frames, config);
    const int64_t state_count = stacked.dim(2), width = stacked.dim(3);
    Tensor trimmed = backend.slice(stacked, 1, 0, plan.actual_feature_frames);
    std::vector<float> left_zeros(
        static_cast<size_t>(plan.left_padding * state_count * width), 0.0f);
    std::vector<float> right_zeros(
        static_cast<size_t>(plan.right_padding * state_count * width), 0.0f);
    Tensor padded = backend.concat({
        host_f32({1, plan.left_padding, state_count, width}, left_zeros),
        trimmed,
        host_f32({1, plan.right_padding, state_count, width}, right_zeros)}, 1);
    std::vector<Tensor> windows;
    windows.reserve(static_cast<size_t>(output_frames));
    for (const int64_t start : plan.starts)
        windows.push_back(backend.slice(
            padded, 1, start, start + plan.window_feature_frames));
    return backend.reshape(backend.concat(windows, 0),
        {output_frames, plan.window_feature_frames * state_count, width});
}

Tensor add_sinusoidal_position_encoding(Backend& backend,
                                        const Tensor& conditioning,
                                        const double max_timescale) {
    require(conditioning.ndim() == 3 && conditioning.dtype() == DType::F32 &&
            conditioning.dim(2) > 0 && conditioning.dim(2) % 2 == 0 &&
            max_timescale > 1.0,
            "Invalid sinusoidal position-encoding contract");
    return backend.add(conditioning, sinusoidal_position_encoding_f32(
        conditioning.dim(1), conditioning.dim(2), max_timescale));
}

}  // namespace vrhino
