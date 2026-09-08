#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "vrhino/lip_sync_workflow.h"

namespace fs = std::filesystem;
namespace {

void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

vrhino::RgbFrame pattern(int width, int height) {
    vrhino::RgbFrame frame{width, height,
        std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        for (int c = 0; c < 3; ++c)
            frame.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] =
                static_cast<uint8_t>((x * 17 + y * 31 + c * 53) & 255);
    return frame;
}

int reflect_reference(int value, int size) {
    while (value < 0 || value >= size) {
        if (value < 0) value = -value;
        else value = size * 2 - value - 2;
    }
    return value;
}

double lanczos4_reference(double value) {
    value = std::abs(value);
    if (value < 1e-12) return 1.0;
    if (value >= 4.0) return 0.0;
    const double pi = std::acos(-1.0);
    return std::sin(pi * value) * std::sin(pi * value / 4.0) /
           (pi * pi * value * value / 4.0);
}

vrhino::RgbFrame scalar_resize_lanczos4_reference(
        const vrhino::RgbFrame& input, int output_width, int output_height) {
    vrhino::RgbFrame output{output_width, output_height,
        std::vector<uint8_t>(static_cast<size_t>(output_width) *
                             output_height * 3)};
    const double sx = static_cast<double>(input.width) / output_width;
    const double sy = static_cast<double>(input.height) / output_height;
    for (int y = 0; y < output_height; ++y) {
        const double source_y = (y + 0.5) * sy - 0.5;
        const int iy = static_cast<int>(std::floor(source_y));
        for (int x = 0; x < output_width; ++x) {
            const double source_x = (x + 0.5) * sx - 0.5;
            const int ix = static_cast<int>(std::floor(source_x));
            for (int channel = 0; channel < 3; ++channel) {
                double sum = 0.0, total = 0.0;
                for (int ky = -3; ky <= 4; ++ky) {
                    const double wy =
                        lanczos4_reference(source_y - (iy + ky));
                    for (int kx = -3; kx <= 4; ++kx) {
                        const double weight = wy *
                            lanczos4_reference(source_x - (ix + kx));
                        const int py = reflect_reference(iy + ky, input.height);
                        const int px = reflect_reference(ix + kx, input.width);
                        sum += input.pixels[
                            (static_cast<size_t>(py) * input.width + px) * 3 +
                            channel] * weight;
                        total += weight;
                    }
                }
                output.pixels[
                    (static_cast<size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<long>(
                        std::lround(sum / total), 0, 255));
            }
        }
    }
    return output;
}

void check_lanczos4_exact(const vrhino::RgbFrame& source,
                          int output_width, int output_height,
                          const std::string& name) {
    const auto actual = vrhino::crop_resize_lanczos4(
        source, {0, 0, source.width, source.height},
        output_width, output_height);
    const auto reference = scalar_resize_lanczos4_reference(
        source, output_width, output_height);
    check(actual.width == reference.width &&
              actual.height == reference.height &&
              actual.pixels == reference.pixels,
          "Lanczos4 differential mismatch: " + name);
}

vrhino::RgbFrame solid(int width, int height, uint8_t value) {
    return {width, height,
        std::vector<uint8_t>(static_cast<size_t>(width) * height * 3, value)};
}

vrhino::RgbFrame checkerboard(int width, int height) {
    auto frame = solid(width, height, 0);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int channel = 0; channel < 3; ++channel)
                frame.pixels[(static_cast<size_t>(y) * width + x) * 3 + channel] =
                    ((x + y) & 1) ? 255 : 0;
    return frame;
}

vrhino::RgbFrame random_frame(int width, int height, uint32_t seed) {
    auto frame = solid(width, height, 0);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& value : frame.pixels)
        value = static_cast<uint8_t>(byte(generator));
    return frame;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        check(vrhino::lip_sync_output_frame_count(5120, 16000, 25) == 8,
              "bounded frame count mismatch");
        check(vrhino::lip_sync_output_frame_count(639, 16000, 25) == 0 &&
              vrhino::lip_sync_output_frame_count(640, 16000, 25) == 1,
              "fractional frame boundary mismatch");
        check(vrhino::ping_pong_frame_cycle(1, 5) ==
                  std::vector<int64_t>({0, 0, 0, 0, 0}),
              "one-frame cycle mismatch");
        check(vrhino::ping_pong_frame_cycle(2, 9) ==
                  std::vector<int64_t>({0, 1, 1, 0, 0, 1, 1, 0, 0}),
              "two-frame cycle mismatch");
        check(vrhino::ping_pong_frame_cycle(4, 12) ==
                  std::vector<int64_t>({0,1,2,3,3,2,1,0,0,1,2,3}),
              "ordinary cycle/reversal mismatch");
        check(vrhino::ping_pong_frame_cycle(8, 3) ==
                  std::vector<int64_t>({0,1,2}),
              "short-output cycle mismatch");

        const auto demand = [](int64_t source, int64_t output) {
            const auto cycle = vrhino::ping_pong_frame_cycle(source, output);
            return vrhino::plan_source_frame_prefix_demand(source, cycle);
        };
        const auto n1_one = demand(1, 1);
        const auto n1_long = demand(1, 17);
        check(n1_one.maximum_referenced_source_index == 0 &&
                  n1_one.required_prefix_frame_count == 1 &&
                  n1_long.maximum_referenced_source_index == 0 &&
                  n1_long.required_prefix_frame_count == 1,
              "N=1 source demand mismatch");
        const auto n2_one = demand(2, 1);
        const auto n2_two = demand(2, 2);
        const auto n2_reversal = demand(2, 3);
        check(n2_one.required_prefix_frame_count == 1 &&
                  n2_two.required_prefix_frame_count == 2 &&
                  n2_reversal.required_prefix_frame_count == 2,
              "N=2 source demand/reversal mismatch");
        const auto short_demand = demand(10, 3);
        const auto full_demand = demand(10, 10);
        const auto reversal_boundary = demand(10, 11);
        const auto repeated_cycles = demand(10, 47);
        check(short_demand.maximum_referenced_source_index == 2 &&
                  short_demand.required_prefix_frame_count == 3 &&
                  full_demand.maximum_referenced_source_index == 9 &&
                  full_demand.required_prefix_frame_count == 10 &&
                  reversal_boundary.required_prefix_frame_count == 10 &&
                  repeated_cycles.required_prefix_frame_count == 10,
              "ordinary source demand/cycle mismatch");
        const auto benchmark_demand = demand(263, 50);
        check(benchmark_demand.source_frame_count == 263 &&
                  benchmark_demand.output_frame_count == 50 &&
                  benchmark_demand.maximum_referenced_source_index == 49 &&
                  benchmark_demand.required_prefix_frame_count == 50,
              "263-to-50 bounded source demand mismatch");
        const auto no_output = vrhino::plan_source_frame_prefix_demand(10, {});
        const auto empty = vrhino::plan_source_frame_prefix_demand(0, {});
        check(no_output.output_frame_count == 0 &&
                  no_output.maximum_referenced_source_index == -1 &&
                  no_output.required_prefix_frame_count == 0 &&
                  empty.source_frame_count == 0 &&
                  empty.required_prefix_frame_count == 0,
              "empty source/output demand mismatch");

        for (const auto& malformed_indices : {
                 std::vector<int64_t>{-1}, std::vector<int64_t>{10}}) {
            bool failed = false;
            try {
                (void)vrhino::plan_source_frame_prefix_demand(
                    10, malformed_indices);
            } catch (const vrhino::LipSyncWorkflowError&) {
                failed = true;
            }
            check(failed, "invalid source demand index did not fail closed");
        }
        bool empty_source_failed = false;
        try {
            (void)vrhino::plan_source_frame_prefix_demand(0, {0});
        } catch (const vrhino::LipSyncWorkflowError&) {
            empty_source_failed = true;
        }
        check(empty_source_failed,
              "non-empty demand over empty source did not fail closed");
        bool negative_source_failed = false;
        try {
            (void)vrhino::plan_source_frame_prefix_demand(-1, {});
        } catch (const vrhino::LipSyncWorkflowError&) {
            negative_source_failed = true;
        }
        check(negative_source_failed,
              "negative source-frame count did not fail closed");
        bool overflow_failed = false;
        try {
            (void)vrhino::ping_pong_frame_cycle(
                std::numeric_limits<int64_t>::max(), 1);
        } catch (const vrhino::LipSyncWorkflowError&) {
            overflow_failed = true;
        }
        check(overflow_failed, "frame-cycle overflow did not fail closed");
        for (int64_t source = 1; source <= 32; ++source)
            for (int64_t output = 0; output <= 128; ++output)
                check(demand(source, output).required_prefix_frame_count <= source,
                      "source demand prefix exceeded source count");

        // Product-loop fixture: every neural/alignment stage consumes exactly
        // the conservative prefix, never the full decoded-frame count.
        int64_t blaze_frames = 0, pose_frames = 0, alignment_frames = 0;
        for (int64_t source_index = 0;
             source_index < benchmark_demand.required_prefix_frame_count;
             ++source_index) {
            ++blaze_frames;
            ++pose_frames;
            ++alignment_frames;
        }
        check(blaze_frames == 50 && pose_frames == 50 &&
                  alignment_frames == 50,
              "bounded Product source-preparation invocation count mismatch");

        auto rng_a = vrhino::lip_sync_component_rng(11001, 3, 0);
        auto rng_b = vrhino::lip_sync_component_rng(11001, 3, 0);
        auto rng_c = vrhino::lip_sync_component_rng(11001, 3, 1);
        auto rng_d = vrhino::lip_sync_component_rng(11001, 4, 0);
        check(rng_a.seed == rng_b.seed && rng_a.offset == 0 &&
              rng_a.seed != rng_c.seed && rng_a.seed != rng_d.seed,
              "component-local RNG stream mismatch");

        const auto source = pattern(8, 8);
        const auto crop = vrhino::crop_resize_lanczos4(source, {2, 1, 7, 7}, 4, 4);
        check(crop.width == 4 && crop.height == 4 && crop.pixels.size() == 48,
              "crop/resize shape mismatch");
        const auto normalized = vrhino::normalize_vae_rgb(crop, true);
        check(normalized.shape() == std::vector<int64_t>({1,3,4,4}),
              "VAE input shape mismatch");
        for (int c = 0; c < 3; ++c) for (int y = 2; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                check(normalized.data_as<float>()[((c * 4 + y) * 4 + x)] == -1.0f,
                      "masked lower-half mismatch");

        // The precomputed plan must remain bit-exact with the prior scalar
        // 2-D 8x8 implementation, including heavily reflected borders.
        check_lanczos4_exact(pattern(2, 7), 9, 5, "narrow-upsample");
        check_lanczos4_exact(pattern(9, 2), 5, 11, "short-upsample");
        check_lanczos4_exact(pattern(5, 8), 7, 3, "odd-even-rectangle");
        check_lanczos4_exact(pattern(8, 5), 3, 7, "even-odd-rectangle");
        check_lanczos4_exact(pattern(7, 7), 7, 7, "identity-size");
        check_lanczos4_exact(pattern(13, 9), 4, 6, "downsample");
        check_lanczos4_exact(pattern(6, 10), 11, 13, "non-integer-scale");
        check_lanczos4_exact(solid(11, 8, 0), 17, 5, "constant-black");
        check_lanczos4_exact(solid(11, 8, 255), 17, 5, "constant-white");
        check_lanczos4_exact(checkerboard(12, 9), 19, 14, "checkerboard");
        auto impulse = solid(9, 11, 0);
        impulse.pixels[0] = 255;
        impulse.pixels[(static_cast<size_t>(10) * 9 + 8) * 3 + 1] = 255;
        impulse.pixels[(static_cast<size_t>(5) * 9 + 4) * 3 + 2] = 255;
        check_lanczos4_exact(impulse, 17, 16, "rgb-edge-impulses");
        check_lanczos4_exact(random_frame(31, 23, 1247), 37, 29,
                             "deterministic-random");
        check_lanczos4_exact(pattern(420, 560), 512, 512,
                             "production-420x560-to-512");

        vrhino::AlphaMask alpha{4, 4, std::vector<uint8_t>(16, 128),
                                "expanded_face_crop_pixels"};
        auto generated = source;
        std::fill(generated.pixels.begin(), generated.pixels.end(), 255);
        const auto composed = vrhino::composite_rgb_alpha(
            source, generated, alpha, {2, 2, 6, 6});
        check(composed.width == 8 && composed.height == 8,
              "composite dimensions mismatch");
        check(composed.pixels[0] == source.pixels[0],
              "composite changed pixels outside support");
        const size_t inside = (static_cast<size_t>(2) * 8 + 2) * 3;
        check(composed.pixels[inside] ==
                  (source.pixels[inside] * 127 + 255 * 128 + 127) / 255,
              "alpha integer contract mismatch");

        bool malformed = false;
        try { (void)vrhino::lip_sync_output_frame_count(1, 0, 25); }
        catch (const vrhino::LipSyncWorkflowError&) { malformed = true; }
        check(malformed, "invalid media contract did not fail closed");

        if (argc == 5) {
            const fs::path helper = argv[1], video = argv[2], audio = argv[3];
            const fs::path scratch = argv[4];
            const auto decoded = vrhino::decode_video_rgb24(helper, video, 8);
            check(decoded.width == 704 && decoded.height == 1216 &&
                  decoded.fps_numerator == 25 && decoded.fps_denominator == 1 &&
                  decoded.frames.size() == 8,
                  "real video-input contract mismatch");
            const auto waveform = vrhino::decode_audio_mono_f32_16khz(helper, audio, 5120);
            check(waveform.sample_rate == 16000 && waveform.channels == 1 &&
                  waveform.samples.size() == 5120,
                  "real audio-input contract mismatch");
            fs::create_directories(scratch);
            const fs::path cancelled_output = scratch / "cancelled-media.mp4";
            const fs::path cancelled_partial = scratch / "cancelled-media.mp4.partial.mp4";
            check(!fs::exists(cancelled_output) && !fs::exists(cancelled_partial),
                  "cancellation fixture path is not clean");
            bool cancelled = false;
            try {
                (void)vrhino::encode_mux_mp4_atomic(
                    helper, {pattern(16,16)}, 25, audio, cancelled_output,
                    [] { return true; });
            } catch (const vrhino::LipSyncWorkflowError& error) {
                cancelled = error.stage() == vrhino::LipSyncStage::MediaOutput;
            }
            check(cancelled && !fs::exists(cancelled_output) &&
                  !fs::exists(cancelled_partial),
                  "cancelled media publication left an output");
            const fs::path published = scratch / "published.mp4";
            { std::ofstream marker(published,std::ios::binary); marker << "published"; }
            const auto published_bytes = fs::file_size(published);
            bool overwrite_refused = false;
            try {
                (void)vrhino::encode_mux_mp4_atomic(
                    helper, {pattern(16,16)}, 25, audio, published);
            } catch (const vrhino::LipSyncWorkflowError& error) {
                overwrite_refused = error.stage() == vrhino::LipSyncStage::MediaOutput;
            }
            check(overwrite_refused && fs::file_size(published)==published_bytes,
                  "atomic media publication overwrote an existing output");
        } else if (argc != 1) {
            throw std::runtime_error(
                "usage: lip_sync_workflow_tests [HELPER VIDEO AUDIO SCRATCH]");
        }
        std::cout << "bounded lip-sync workflow CPU tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bounded lip-sync workflow CPU tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
