#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/face_mask.h"
#include "vrhino/json.h"
#include "vrhino/lip_sync_workflow.h"
#include "vrhino/loader.h"
#include "vrhino/semantic_segmenter.h"

namespace fs = std::filesystem;

namespace {

void check(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot read successor oracle metadata");
    return {std::istreambuf_iterator<char>(input), {}};
}

struct NpyBytes {
    std::vector<int64_t> shape;
    std::vector<uint8_t> values;
};

NpyBytes read_u8(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot open u8 NPY fixture");
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    check(input && std::memcmp(prefix, "\x93NUMPY", 6) == 0 &&
              prefix[6] == 1 && prefix[7] == 0,
          "unsupported u8 NPY header");
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
        (static_cast<uint16_t>(prefix[9]) << 8);
    std::string header(header_size, '\0');
    input.read(header.data(), header.size());
    check(input && (header.find("'descr': '|u1'") != std::string::npos ||
                    header.find("'descr': '<u1'") != std::string::npos),
          "u8 NPY dtype mismatch");
    const size_t begin = header.find('(', header.find("'shape':"));
    const size_t end = header.find(')', begin);
    check(begin != std::string::npos && end != std::string::npos,
          "u8 NPY shape is missing");
    NpyBytes result;
    std::istringstream fields(header.substr(begin + 1, end - begin - 1));
    std::string field;
    size_t count = 1;
    while (std::getline(fields, field, ',')) {
        field.erase(std::remove_if(field.begin(), field.end(),
            [](unsigned char value) { return std::isspace(value); }), field.end());
        if (!field.empty()) {
            result.shape.push_back(std::stoll(field));
            count *= static_cast<size_t>(result.shape.back());
        }
    }
    result.values.resize(count);
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(result.values.size()));
    check(input && input.peek() == std::char_traits<char>::eof(),
          "u8 NPY payload mismatch");
    return result;
}

std::vector<double> read_f64(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot open f64 NPY fixture");
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
        (static_cast<uint16_t>(prefix[9]) << 8);
    std::string header(header_size, '\0');
    input.read(header.data(), header.size());
    check(input && header.find("'descr': '<f8'") != std::string::npos,
          "f64 NPY dtype mismatch");
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    const std::streamoff data_offset = 10 + header_size;
    check(end >= data_offset && (end - data_offset) % 8 == 0,
          "f64 NPY payload mismatch");
    std::vector<double> result(static_cast<size_t>((end - data_offset) / 8));
    input.seekg(data_offset);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size() * sizeof(double)));
    check(static_cast<bool>(input), "cannot read f64 NPY payload");
    return result;
}

std::string frame_name(const int frame) {
    std::ostringstream output;
    output << "frame_" << std::setw(2) << std::setfill('0') << frame;
    return output.str();
}

std::array<int32_t, 4> box(const vrhino::Json& value) {
    std::array<int32_t, 4> result{};
    for (size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<int32_t>(value.array()[index].integer());
    return result;
}

vrhino::RgbFrame source_frame(const vrhino::Tensor& fixture) {
    check(fixture.shape() == std::vector<int64_t>({1216, 704, 3}),
          "successor source RGB shape mismatch");
    vrhino::RgbFrame result{704, 1216,
        std::vector<uint8_t>(static_cast<size_t>(fixture.numel()))};
    for (int64_t index = 0; index < fixture.numel(); ++index) {
        const float value = fixture.data_as<float>()[index];
        check(value >= 0 && value <= 255 && value == std::floor(value),
              "source RGB is not uint8-valued");
        result.pixels[static_cast<size_t>(index)] = static_cast<uint8_t>(value);
    }
    return result;
}

vrhino::RgbFrame generated_frame(const vrhino::Tensor& fixture) {
    check(fixture.shape() == std::vector<int64_t>({1, 3, 256, 256}),
          "successor generated crop shape mismatch");
    vrhino::RgbFrame result{256, 256, std::vector<uint8_t>(256 * 256 * 3)};
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 256; ++x)
            for (int channel = 0; channel < 3; ++channel) {
                const float value = fixture.data_as<float>()[
                    (channel * 256 + y) * 256 + x];
                result.pixels[(static_cast<size_t>(y) * 256 + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<double>(std::nearbyint(
                        std::clamp(value, 0.0f, 1.0f) * 255.0f), 0, 255));
            }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: musetalk_successor_oracle_tests SELFIE_VRM "
                     "SUCCESSOR_ORACLE BLAZEFACE_ORACLE SELFIE_ORACLE "
                     "POSE_ORACLE PHASE2F_ORACLE PHASE1_OFFICIAL\n";
        return 2;
    }
    try {
        const fs::path successor = argv[2], blaze = argv[3], selfie = argv[4];
        const fs::path pose = argv[5], phase2f = argv[6], official = argv[7];
        const vrhino::Json metadata = vrhino::Json::parse(
            read_text(phase2f / "metadata.json"));
        vrhino::VrmModel component(argv[1], true);
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::SemanticSegmenter2DComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())));

        uint64_t label_mismatches = 0, alpha_differences = 0;
        uint64_t rgb_pixels = 0, rgb_channels = 0, unexplained = 0;
        uint64_t semantic_boundary_pixels = 0, mask_rounding_pixels = 0;
        uint64_t upstream_quantization_pixels = 0, boundary_mismatches = 0;
        uint8_t worst_delta = 0;
        long double rgb_sum = 0.0;
        double worst_raw_max = 0.0, worst_raw_mean = 0.0;
        double minimum_raw_cosine = 1.0;
        double maximum_support_step = 0.0;
        uint64_t previous_support = 0;
        for (int frame = 0; frame < 8; ++frame) {
            const std::string name = frame_name(frame);
            const vrhino::RgbFrame source = source_frame(
                vrhino::test::read_npy_f32((blaze / name / "rgb.npy").string()));
            const auto crop = box(metadata.at("records").array()[frame].at("crop_box"));
            const auto bbox = box(metadata.at("records").array()[frame].at("bbox"));
            const std::vector<uint8_t> bgr = [&] {
                std::vector<uint8_t> result = source.pixels;
                for (size_t index = 0; index < result.size(); index += 3)
                    std::swap(result[index], result[index + 2]);
                return result;
            }();
            const auto legacy_preparation = vrhino::preprocess_face_parser_bgr_u8(
                bgr.data(), source.height, source.width, bbox);
            check(legacy_preparation.crop_box == crop,
                  "successor expanded crop changed from qualified geometry");
            const auto preparation = vrhino::preprocess_semantic_segmenter_rgb_u8(
                legacy_preparation.resized_rgb.data(), 512, 512,
                {0, 0, 512, 512});
            const vrhino::Tensor reference_input = vrhino::test::read_npy_f32(
                (successor / name / "parser_input.npy").string());
            uint64_t input_mismatches = 0;
            double input_max = 0.0;
            for (int channel = 0; channel < 3; ++channel)
                for (int y = 0; y < 256; ++y)
                    for (int x = 0; x < 256; ++x) {
                        const float actual = preparation.normalized_nchw.data_as<float>()[
                            (channel * 256 + y) * 256 + x];
                        const float expected = reference_input.data_as<float>()[
                            (y * 256 + x) * 3 + channel];
                        const double delta = std::abs(actual - expected);
                        input_mismatches += delta != 0;
                        input_max = std::max(input_max, delta);
                    }
            std::cout << name << " input_probe mismatches=" << input_mismatches
                      << " max=" << input_max << '\n';
            const auto raw = executor.execute(
                component.graph(), preparation.normalized_nchw);
            const vrhino::Tensor native_raw = backend.copy_to_host(raw.logits);
            const vrhino::Tensor reference_raw = vrhino::test::read_npy_f32(
                (successor / name / "probabilities.npy").string());
            check(reference_raw.shape() ==
                      std::vector<int64_t>({1, 256, 256, 6}),
                  "successor reference raw schema mismatch");
            long double raw_sum = 0.0, raw_dot = 0.0;
            long double native_square = 0.0, reference_square = 0.0;
            double raw_max = 0.0;
            uint64_t nonfinite = 0;
            for (int channel = 0; channel < 6; ++channel)
                for (int y = 0; y < 256; ++y)
                    for (int x = 0; x < 256; ++x) {
                        const float actual = native_raw.data_as<float>()[
                            (channel * 256 + y) * 256 + x];
                        const float expected = reference_raw.data_as<float>()[
                            (y * 256 + x) * 6 + channel];
                        nonfinite += !std::isfinite(actual);
                        const double delta = std::abs(actual - expected);
                        raw_max = std::max(raw_max, delta);
                        raw_sum += delta;
                        raw_dot += static_cast<long double>(actual) * expected;
                        native_square += static_cast<long double>(actual) * actual;
                        reference_square += static_cast<long double>(expected) * expected;
                    }
            const double raw_mean = static_cast<double>(raw_sum / native_raw.numel());
            const double raw_cosine = static_cast<double>(raw_dot /
                std::sqrt(native_square * reference_square));
            std::cout << name << " raw_probe max=" << raw_max
                      << " mean=" << raw_mean << " cosine=" << raw_cosine
                      << " nonfinite=" << nonfinite << '\n';
            check(nonfinite == 0 && raw_max <= 0.1 && raw_cosine >= 0.99999,
                  "successor raw semantic FP32 gate failed");
            if (raw_max > worst_raw_max) {
                worst_raw_max = raw_max;
                worst_raw_mean = raw_mean;
            }
            minimum_raw_cosine = std::min(minimum_raw_cosine, raw_cosine);
            vrhino::SemanticLabelMap labels = vrhino::semantic_argmax_first(
                native_raw);
            labels.coordinate_space = "successor_segmenter_pixels";
            const NpyBytes reference_labels = read_u8(selfie / name / "labels.npy");
            uint64_t frame_label_mismatch = 0;
            uint64_t frame_boundary_mismatch = 0;
            for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) {
                const size_t index = static_cast<size_t>(y) * 256 + x;
                if (labels.labels[index] == reference_labels.values[index]) continue;
                ++frame_label_mismatch;
                bool at_boundary = false;
                for (int dy = -2; dy <= 2 && !at_boundary; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < 256 && ny >= 0 && ny < 256 &&
                            reference_labels.values[static_cast<size_t>(ny) * 256 + nx] ==
                                labels.labels[index]) {
                            at_boundary = true;
                            break;
                        }
                    }
                frame_boundary_mismatch += at_boundary;
            }
            check(frame_label_mismatch <= 256 &&
                      frame_boundary_mismatch == frame_label_mismatch,
                  "successor labels diverged outside a bounded class edge");
            label_mismatches += frame_label_mismatch;
            boundary_mismatches += frame_boundary_mismatch;

            const auto keypoints = read_f64(pose / name / "decoded_keypoints.npy");
            std::vector<std::array<int32_t, 2>> face(68);
            for (size_t index = 0; index < face.size(); ++index)
                face[index] = {
                    static_cast<int32_t>(keypoints[(23 + index) * 2]),
                    static_cast<int32_t>(keypoints[(23 + index) * 2 + 1])};
            const vrhino::AlphaMask alpha =
                vrhino::build_landmark_constrained_alpha_mask(
                    labels, face, crop);
            vrhino::SemanticLabelMap counterfactual_labels{
                256, 256, reference_labels.values,
                "successor_segmenter_pixels"};
            const vrhino::AlphaMask counterfactual_alpha =
                vrhino::build_landmark_constrained_alpha_mask(
                    counterfactual_labels, face, crop);
            const NpyBytes reference_alpha = read_u8(successor / name / "alpha.npy");
            uint64_t frame_alpha_difference = 0, support = 0;
            for (size_t index = 0; index < alpha.values.size(); ++index) {
                frame_alpha_difference += alpha.values[index] !=
                    reference_alpha.values[index];
                support += alpha.values[index] > 127;
            }
            alpha_differences += frame_alpha_difference;
            if (previous_support != 0)
                maximum_support_step = std::max(maximum_support_step,
                    std::abs(static_cast<double>(support) - previous_support) /
                    previous_support);
            previous_support = support;

            const vrhino::RgbFrame generated = generated_frame(
                vrhino::test::read_npy_f32((official / "tensors/vae_decode" /
                    (name + ".rgb_0_1.npy")).string()));
            const vrhino::RgbFrame placed =
                vrhino::place_generated_crop(source, generated, bbox);
            const vrhino::RgbFrame final =
                vrhino::composite_rgb_alpha(source, placed, alpha, crop);
            const NpyBytes reference_rgb = read_u8(
                successor / name / "final_rgb.npy");
            uint64_t frame_pixels = 0, frame_channels = 0, frame_unexplained = 0;
            uint64_t frame_semantic = 0, frame_mask_rounding = 0;
            uint64_t frame_upstream = 0;
            long double frame_sum = 0.0;
            for (int y = 0; y < final.height; ++y) {
                for (int x = 0; x < final.width; ++x) {
                    bool pixel_differs = false;
                    bool pixel_unexplained = false;
                    const bool in_crop = x >= crop[0] && x < crop[2] &&
                                         y >= crop[1] && y < crop[3];
                    const size_t alpha_index = in_crop
                        ? static_cast<size_t>(y - crop[1]) * alpha.width + x - crop[0]
                        : 0;
                    for (int channel = 0; channel < 3; ++channel) {
                        const size_t index =
                            (static_cast<size_t>(y) * final.width + x) * 3 + channel;
                        const uint8_t delta = static_cast<uint8_t>(std::abs(
                            static_cast<int>(final.pixels[index]) -
                            reference_rgb.values[index]));
                        frame_sum += delta;
                        if (delta != 0) {
                            pixel_differs = true;
                            ++frame_channels;
                            worst_delta = std::max(worst_delta, delta);
                            if (!in_crop || alpha.values[alpha_index] ==
                                    reference_alpha.values[alpha_index])
                                pixel_unexplained = true;
                        }
                    }
                    if (pixel_differs) {
                        ++frame_pixels;
                        if (pixel_unexplained) {
                            ++frame_unexplained;
                        } else if (alpha.values[alpha_index] !=
                                   counterfactual_alpha.values[alpha_index]) {
                            ++frame_semantic;
                        } else if (counterfactual_alpha.values[alpha_index] !=
                                   reference_alpha.values[alpha_index]) {
                            ++frame_mask_rounding;
                        } else {
                            ++frame_upstream;
                        }
                    }
                }
            }
            check(frame_unexplained == 0,
                  "successor RGB attribution has unexplained differences");
            rgb_pixels += frame_pixels;
            rgb_channels += frame_channels;
            unexplained += frame_unexplained;
            semantic_boundary_pixels += frame_semantic;
            mask_rounding_pixels += frame_mask_rounding;
            upstream_quantization_pixels += frame_upstream;
            rgb_sum += frame_sum;
            std::cout << name << " labels=" << frame_label_mismatch
                      << " alpha_pixels=" << frame_alpha_difference
                      << " rgb_pixels=" << frame_pixels
                      << " rgb_channels=" << frame_channels
                      << " semantic_boundary=" << frame_semantic
                      << " mask_rounding=" << frame_mask_rounding
                      << " upstream_quantization=" << frame_upstream
                      << " unexplained=" << frame_unexplained
                      << " max=" << static_cast<int>(worst_delta)
                      << " mean=" << static_cast<double>(frame_sum /
                          final.pixels.size()) << '\n';
        }
        check(unexplained == 0 && maximum_support_step < 0.03,
              "successor composed workflow gate failed");
        std::cout << "MuseTalk successor Native/reference oracle: PASS\n"
                  << "label_mismatches=" << label_mismatches << '\n'
                  << "boundary_mismatches=" << boundary_mismatches << '\n'
                  << "worst_raw_max_abs=" << worst_raw_max << '\n'
                  << "worst_raw_mean_abs=" << worst_raw_mean << '\n'
                  << "minimum_raw_cosine=" << minimum_raw_cosine << '\n'
                  << "alpha_difference_pixels=" << alpha_differences << '\n'
                  << "rgb_difference_pixels=" << rgb_pixels << '\n'
                  << "rgb_difference_channels=" << rgb_channels << '\n'
                  << "rgb_max_abs=" << static_cast<int>(worst_delta) << '\n'
                  << "rgb_mean_abs=" << static_cast<double>(rgb_sum /
                       (8.0 * 1216 * 704 * 3)) << '\n'
                  << "unexplained=" << unexplained << '\n'
                  << "semantic_boundary_pixels=" << semantic_boundary_pixels << '\n'
                  << "mask_rounding_pixels=" << mask_rounding_pixels << '\n'
                  << "upstream_quantization_pixels="
                  << upstream_quantization_pixels << '\n'
                  << "maximum_support_step=" << maximum_support_step << '\n'
                  << "peak_device_bytes=" << backend.peak_device_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MuseTalk successor oracle tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
