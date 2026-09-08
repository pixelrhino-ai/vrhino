#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/tensor.h"

namespace {

struct Statistics {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sum_squares = 0.0;
    uint64_t count = 0;

    void add(double value) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        sum += value;
        sum_squares += value * value;
        ++count;
    }
    double mean() const { return sum / static_cast<double>(count); }
    double standard_deviation() const {
        const double variance = sum_squares / static_cast<double>(count) - mean() * mean();
        return std::sqrt(std::max(0.0, variance));
    }
};

uint8_t to_u8(float value) {
    const float normalized = std::clamp(value, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lround(normalized * 255.0f));
}

float load_video_value(const vrhino::Tensor& video, int64_t index) {
    if (video.dtype() == vrhino::DType::F32) return video.data_as<float>()[index];
    const uint32_t bits = static_cast<uint32_t>(video.data_as<uint16_t>()[index]) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void print_f32_metric(const vrhino::TensorBundle& bundle, const std::string& name) {
    const auto found = bundle.find(name);
    if (found == bundle.end()) return;
    const vrhino::Tensor& value = found->second;
    vrhino::require(value.dtype() == vrhino::DType::F32 && value.numel() == 1,
                    "Invalid scalar metric: " + name);
    std::cout << '\n' << name << '=' << value.data_as<float>()[0];
}

void print_channel_statistics(const std::string& prefix,
                              const std::array<Statistics, 3>& statistics) {
    constexpr std::array<const char*, 3> names = {"R", "G", "B"};
    for (size_t channel = 0; channel < statistics.size(); ++channel) {
        const Statistics& values = statistics[channel];
        std::cout << '\n' << prefix << '.' << names[channel] << ".min=" << values.minimum
                  << '\n' << prefix << '.' << names[channel] << ".max=" << values.maximum
                  << '\n' << prefix << '.' << names[channel] << ".mean=" << values.mean()
                  << '\n' << prefix << '.' << names[channel] << ".std="
                  << values.standard_deviation();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 3 || argc == 4,
            "usage: vrhino-export-bcthw-f32-rgb24 INPUT.bundle OUTPUT.rgb24 "
            "[FIRST_FRAME.ppm]");
        const vrhino::TensorBundle bundle = vrhino::read_bundle(argv[1]);
        const auto found = bundle.find("video");
        vrhino::require(found != bundle.end(), "Bundle does not contain tensor: video");
        const vrhino::Tensor& video = found->second;
        vrhino::require(video.device().is_host(), "Video tensor must be on the host");
        vrhino::require(video.dtype() == vrhino::DType::F32 ||
                            video.dtype() == vrhino::DType::BF16,
                        "Video tensor must use FP32 or BF16 storage");
        vrhino::require(video.ndim() == 5, "Video tensor must use BCTHW rank-5 layout");
        vrhino::require(video.dim(0) == 1, "Exporter currently requires batch size 1");
        vrhino::require(video.dim(1) == 3, "Exporter requires three RGB channels");

        const int64_t frames = video.dim(2);
        const int64_t height = video.dim(3);
        const int64_t width = video.dim(4);
        vrhino::require(frames > 0 && height > 0 && width > 0,
                        "Video tensor dimensions must be positive");

        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        std::array<Statistics, 3> source_statistics;
        for (int64_t channel = 0; channel < 3; ++channel) {
            for (int64_t frame = 0; frame < frames; ++frame) {
                for (int64_t row = 0; row < height; ++row) {
                    for (int64_t column = 0; column < width; ++column) {
                        const int64_t index = channel * frames * height * width
                            + frame * height * width + row * width + column;
                        const float value = load_video_value(video, index);
                        vrhino::require(std::isfinite(value),
                                        "Video tensor contains non-finite data");
                        minimum = std::min(minimum, value);
                        maximum = std::max(maximum, value);
                        source_statistics[static_cast<size_t>(channel)].add(value);
                    }
                }
            }
        }
        constexpr float kRangeEpsilon = 1.0e-6f;
        vrhino::require(minimum >= -kRangeEpsilon && maximum <= 1.0f + kRangeEpsilon,
                        "Video tensor violates the declared [0, 1] range");

        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        vrhino::require(output.good(), "Cannot create RGB24 output");
        std::array<Statistics, 3> rgb24_statistics;
        std::vector<uint8_t> first_frame;
        first_frame.reserve(static_cast<size_t>(height * width * 3));
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t row = 0; row < height; ++row) {
                for (int64_t column = 0; column < width; ++column) {
                    for (int64_t channel = 0; channel < 3; ++channel) {
                        const int64_t index = channel * frames * height * width
                            + frame * height * width + row * width + column;
                        const uint8_t converted = to_u8(load_video_value(video, index));
                        rgb24_statistics[static_cast<size_t>(channel)].add(converted);
                        if (frame == 0) first_frame.push_back(converted);
                        output.write(reinterpret_cast<const char*>(&converted), 1);
                    }
                }
            }
        }
        vrhino::require(output.good(), "RGB24 output write failed");

        if (argc == 4) {
            std::ofstream ppm(argv[3], std::ios::binary | std::ios::trunc);
            vrhino::require(ppm.good(), "Cannot create PPM output");
            ppm << "P6\n" << width << ' ' << height << "\n255\n";
            ppm.write(reinterpret_cast<const char*>(first_frame.data()),
                      static_cast<std::streamsize>(first_frame.size()));
            vrhino::require(ppm.good(), "PPM output write failed");
        }

        std::cout << std::setprecision(10)
                  << "status=PASS"
                  << "\ntensor=video"
                  << "\ndtype=" << vrhino::dtype_name(video.dtype())
                  << "\nlayout=BCTHW"
                  << "\nchannel_order=RGB"
                  << "\nshape=[" << video.dim(0) << ',' << video.dim(1) << ','
                  << frames << ',' << height << ',' << width << ']'
                  << "\nsource_min=" << minimum
                  << "\nsource_max=" << maximum
                  << "\nsource_contract=[0,1]"
                  << "\nconversion=clamp(x,0,1)*255"
                  << "\nindexing=src[c*T*H*W+t*H*W+y*W+x]"
                  << "\nframes=" << frames
                  << "\nwidth=" << width
                  << "\nheight=" << height
                  << "\noutput_bytes=" << frames * height * width * 3
                  << "\noutput=" << argv[2];
        if (argc == 4) std::cout << "\nfirst_frame_ppm=" << argv[3];
        print_channel_statistics("source", source_statistics);
        print_channel_statistics("rgb24", rgb24_statistics);
        print_f32_metric(bundle, "metric.sampling_seconds");
        print_f32_metric(bundle, "metric.decode_seconds");
        print_f32_metric(bundle, "metric.execution_seconds");
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-export-bcthw-f32-rgb24: " << error.what() << '\n';
        return 1;
    }
}
