#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/image_filter.h"

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> scalar_oracle(const std::vector<float>& source,
                                 int64_t height, int64_t width,
                                 int64_t kernel, int64_t anchor) {
    std::vector<float> output(source.size());
    for (int64_t y = 0; y < height; ++y)
        for (int64_t x = 0; x < width; ++x) {
            float value = 1.0f;
            for (int64_t ky = 0; ky < kernel; ++ky)
                for (int64_t kx = 0; kx < kernel; ++kx) {
                    const int64_t sx = x + kx - anchor;
                    const int64_t sy = y + ky - anchor;
                    const float sample = sx >= 0 && sx < width && sy >= 0 &&
                            sy < height
                        ? source[static_cast<size_t>(sy * width + sx)]
                        : 0.0f;
                    value = std::min(value, sample);
                }
            output[static_cast<size_t>(y * width + x)] = value;
        }
    return output;
}

void exact_case(const std::vector<float>& source, int64_t height, int64_t width,
                int64_t kernel, int64_t anchor, const std::string& name) {
    const auto expected = scalar_oracle(source, height, width, kernel, anchor);
    const auto actual = vrhino::minimum_filter_square_zero_border_f32(
        source, height, width, kernel, anchor);
    check(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size() * sizeof(float)) == 0,
          name + " minimum-filter bytes differ");
}

std::vector<float> random_mask(int64_t height, int64_t width, uint32_t seed) {
    std::vector<float> result(static_cast<size_t>(height * width));
    uint32_t state = seed;
    for (float& value : result) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(state & 0xffffU) / 65535.0f;
    }
    return result;
}

template <class Operation>
void rejects(const Operation& operation, const std::string& message) {
    bool failed = false;
    try { operation(); }
    catch (const vrhino::Error&) { failed = true; }
    check(failed, message);
}

}  // namespace

int main() {
    try {
        for (const auto& [height, width] : {
                 std::pair<int64_t,int64_t>{1,1}, {1,7}, {9,1}, {2,2},
                 {3,5}, {4,6}, {7,8}, {11,13}}) {
            const auto source = random_mask(height, width,
                static_cast<uint32_t>(height * 100 + width));
            for (int64_t kernel : {1,2,3,4,5,7})
                exact_case(source, height, width, kernel, kernel / 2,
                           "tiny/rectangular/random");
        }

        const auto medium = random_mask(43, 47, 1247);
        for (int64_t kernel : {1,2,3,4,36,37,38,39,40,41})
            exact_case(medium, 43, 47, kernel, kernel / 2,
                       "qualified kernel matrix");
        for (int64_t anchor = 0; anchor < 4; ++anchor)
            exact_case(medium, 43, 47, 4, anchor,
                       "even-kernel anchor matrix");

        std::vector<float> zeros(35, 0.0f), ones(35, 1.0f);
        exact_case(zeros, 5, 7, 4, 2, "all zeros");
        exact_case(ones, 5, 7, 4, 2, "all ones");
        std::vector<float> impulse(35, 0.0f);
        impulse[17] = 1.0f;
        exact_case(impulse, 5, 7, 3, 1, "impulse");
        std::vector<float> rectangle(99, 0.0f);
        for (int y = 2; y < 7; ++y)
            for (int x = 3; x < 8; ++x) rectangle[y * 11 + x] = 1.0f;
        exact_case(rectangle, 9, 11, 4, 2, "rectangle");
        std::vector<float> gradient(99);
        for (size_t index = 0; index < gradient.size(); ++index)
            gradient[index] = static_cast<float>(index) / gradient.size();
        exact_case(gradient, 9, 11, 5, 2, "gradient");

        std::vector<float> special = {
            0.0f, -0.0f, std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(), 2.0f, -2.0f,
            std::bit_cast<float>(uint32_t{0x7fc01234U})};
        exact_case(special, 2, 4, 2, 1, "special floating values");
        exact_case(random_mask(5, 7, 91), 5, 7, 7, 3,
                   "kernel equals width");
        exact_case(random_mask(3, 4, 92), 3, 4, 9, 4,
                   "kernel exceeds image");
        exact_case({}, 0, 0, 1, 0, "empty image");
        exact_case({}, 0, 5, 3, 1, "empty-height image");
        exact_case({}, 5, 0, 3, 1, "empty-width image");

        const auto production_like = random_mask(560, 420, 1248);
        exact_case(production_like, 560, 420, 40, 20,
                   "production-like aligned mask");
        const auto full_size = random_mask(1920, 1080, 1249);
        exact_case(full_size, 1920, 1080, 2, 1,
                   "production-sized mask");

        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32({}, -1, 0, 1, 0);
        }, "negative dimensions were accepted");
        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32({0}, 1, 2, 1, 0);
        }, "mismatched source size was accepted");
        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32({0}, 1, 1, 0, 0);
        }, "zero kernel was accepted");
        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32({0}, 1, 1, 2, 2);
        }, "out-of-range anchor was accepted");
        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32(
                {}, std::numeric_limits<int64_t>::max(), 2, 1, 0);
        }, "image-size overflow was accepted");
        rejects([] {
            (void)vrhino::minimum_filter_square_zero_border_f32(
                {0}, 1, 1, std::numeric_limits<int64_t>::max(), 0);
        }, "workspace-size overflow was accepted");

        const auto benchmark_source = random_mask(1920, 1080, 1250);
        const auto scalar_started = std::chrono::steady_clock::now();
        const auto expected = scalar_oracle(benchmark_source, 1920, 1080, 38, 19);
        const double scalar_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - scalar_started).count();
        const auto optimized_started = std::chrono::steady_clock::now();
        const auto actual = vrhino::minimum_filter_square_zero_border_f32(
            benchmark_source, 1920, 1080, 38, 19);
        const double optimized_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - optimized_started).count();
        check(std::memcmp(expected.data(), actual.data(),
                          expected.size() * sizeof(float)) == 0,
              "production-sized kernel-38 differential mismatch");
        std::cout << "minimum-filter differential tests: PASS\n"
                  << "scalar_seconds=" << scalar_seconds
                  << " optimized_seconds=" << optimized_seconds
                  << " speedup=" << scalar_seconds / optimized_seconds
                  << " bit_exact=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "minimum-filter differential tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
