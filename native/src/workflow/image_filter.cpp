#include "vrhino/image_filter.h"

#include <algorithm>
#include <limits>

#include "vrhino/error.h"

namespace vrhino {
namespace {

size_t checked_pixels(int64_t height, int64_t width) {
    require(height >= 0 && width >= 0,
            "minimum-filter dimensions must be non-negative");
    const uint64_t unsigned_height = static_cast<uint64_t>(height);
    const uint64_t unsigned_width = static_cast<uint64_t>(width);
    require(unsigned_width == 0 ||
                unsigned_height <=
                    std::numeric_limits<size_t>::max() / unsigned_width,
            "minimum-filter image size overflow");
    return static_cast<size_t>(unsigned_height * unsigned_width);
}

int64_t checked_extended_length(int64_t length, int64_t kernel) {
    require(length > 0 && kernel > 0 &&
                kernel - 1 <= std::numeric_limits<int64_t>::max() - length,
            "minimum-filter window range overflow");
    return length + kernel - 1;
}

template <class Read, class Write>
void filter_line(int64_t length, int64_t kernel, int64_t anchor,
                 const Read& read, const Write& write,
                 std::vector<int64_t>& positions,
                 std::vector<float>& values) {
    const int64_t right_extent = kernel - anchor - 1;
    const int64_t first = -anchor;
    const int64_t last = length - 1 + right_extent;
    size_t head = 0;
    size_t tail = 0;
    for (int64_t position = first;; ++position) {
        const float sample = position < 0 || position >= length
            ? 0.0f
            : std::min(1.0f, read(position));
        while (head < tail && values[tail - 1] > sample) --tail;
        positions[tail] = position;
        values[tail] = sample;
        ++tail;
        while (head < tail && position - positions[head] >= kernel) ++head;
        if (position >= right_extent) write(position - right_extent, values[head]);
        if (position == last) break;
    }
}

}  // namespace

std::vector<float> minimum_filter_square_zero_border_f32(
        const std::vector<float>& source, int64_t height, int64_t width,
        int64_t kernel, int64_t anchor) {
    require(kernel > 0 && anchor >= 0 && anchor < kernel,
            "invalid minimum-filter kernel/anchor");
    const size_t pixels = checked_pixels(height, width);
    require(source.size() == pixels,
            "minimum-filter source size mismatch");
    if (pixels == 0) return {};

    const int64_t horizontal_length = checked_extended_length(width, kernel);
    const int64_t vertical_length = checked_extended_length(height, kernel);
    const int64_t workspace_length =
        std::max(horizontal_length, vertical_length);
    require(static_cast<uint64_t>(workspace_length) <=
                std::numeric_limits<size_t>::max() /
                    (sizeof(int64_t) + sizeof(float)) &&
                pixels <= std::numeric_limits<size_t>::max() /
                    (2 * sizeof(float)),
            "minimum-filter workspace size overflow");
    std::vector<int64_t> positions(static_cast<size_t>(workspace_length));
    std::vector<float> values(static_cast<size_t>(workspace_length));
    std::vector<float> horizontal(pixels);
    std::vector<float> output(pixels);

    for (int64_t y = 0; y < height; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        filter_line(width, kernel, anchor,
            [&](int64_t x) { return source[row + static_cast<size_t>(x)]; },
            [&](int64_t x, float value) {
                horizontal[row + static_cast<size_t>(x)] = value;
            }, positions, values);
    }
    for (int64_t x = 0; x < width; ++x) {
        filter_line(height, kernel, anchor,
            [&](int64_t y) {
                return horizontal[static_cast<size_t>(y) *
                    static_cast<size_t>(width) + static_cast<size_t>(x)];
            },
            [&](int64_t y, float value) {
                output[static_cast<size_t>(y) * static_cast<size_t>(width) +
                       static_cast<size_t>(x)] = value;
            }, positions, values);
    }
    return output;
}

}  // namespace vrhino
