#pragma once

#include <cstdint>
#include <vector>

namespace vrhino {

// Exact flat square minimum filter for a row-major FP32 image. The window at
// (x, y) covers [x - anchor, x + kernel - anchor) on each axis. Samples outside
// the image are +0.0f. Reduction order preserves the scalar row-major oracle,
// including its 1.0f initial value and first-equal-value behavior.
std::vector<float> minimum_filter_square_zero_border_f32(
    const std::vector<float>& source, int64_t height, int64_t width,
    int64_t kernel, int64_t anchor);

}  // namespace vrhino
