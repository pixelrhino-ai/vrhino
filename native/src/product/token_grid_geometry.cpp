#include "vrhino/product/token_grid_geometry.h"
#include "vrhino/architecture_binding.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"
#include <algorithm>
#include <bit>
#include <limits>

namespace vrhino::product {
namespace {
uint64_t multiply(uint64_t a, uint64_t b) {
    require(b && a <= uint64_t(INT64_MAX) / b, "Token grid integer multiplication overflow");
    return a * b;
}
std::vector<int64_t> grid(int64_t w, int64_t h, int64_t f, uint64_t bound) {
    require(w > 0 && h > 0 && f >= 1 && w <= 16384 && h <= 16384 && f <= 4096,
            "Token grid output dimensions outside bound");
    require(w % 32 == 0 && h % 32 == 0 && (f - 1) % 8 == 0,
            "Token grid output not divisible by scales");
    const int64_t t = (f - 1) / 8 + 1, y = h / 32, x = w / 32;
    const auto n = multiply(multiply(t, y), x);
    const auto latent = multiply(multiply(n, 128), sizeof(float));
    const auto coordinates = multiply(multiply(n, 3), sizeof(float));
    const auto video = multiply(multiply(multiply(multiply(f, h), w), 3), sizeof(float));
    require(bound > 0 && bound <= std::numeric_limits<size_t>::max() && latent <= bound &&
                video <= bound && coordinates <= bound && bound >= 64 && coordinates <= bound - 64,
            "Token grid tensor byte/resource bound exceeded");
    return {t, y, x};
}
} // namespace
TokenGridGeometry admit_token_grid_geometry(const Json &d, int64_t w, int64_t h, int64_t f,
                                            uint64_t seed, uint64_t bound) {
    require(d.object().size() == 6 && d.at("kind").string() == "token_grid_thw.v1" &&
                d.at("spatial_scale").integer() == 32 && d.at("temporal_scale").integer() == 8 &&
                d.at("temporal_origin").integer() == 1 && d.at("channels").integer() == 128 &&
                d.at("flatten_order").string() == "THW_W_FASTEST",
            "Unsupported token grid declaration");
    const auto g = grid(w, h, f, bound);
    const int64_t n = multiply(multiply(g[0], g[1]), g[2]);
    auto coordinates = Tensor::host({1, 3, n}, DType::F32);
    for (int64_t t = 0; t < g[0]; ++t)
        for (int64_t y = 0; y < g[1]; ++y)
            for (int64_t x = 0; x < g[2]; ++x) {
                const int64_t i = (t * g[1] + y) * g[2] + x;
                coordinates.data_as<float>()[i] = float(t);
                coordinates.data_as<float>()[n + i] = float(y);
                coordinates.data_as<float>()[2 * n + i] = float(x);
            }
    TokenGridGeometry out{
        w,
        h,
        f,
        {1, n, 128},
        {1, 3, f, h, w},
        {{"seed", scalar_i64(std::bit_cast<int64_t>(seed))},
         {"decode_seed", scalar_i64(std::bit_cast<int64_t>(seed + uint64_t(100)))},
         {"latent_grid", host_i64({3}, g)},
         {"latent_shape", host_i64({3}, {1, n, 128})},
         {"coordinates", std::move(coordinates)}},
        bound};
    validate_token_grid_geometry(out);
    return out;
}
void validate_token_grid_geometry(const TokenGridGeometry &out) {
    const auto g = grid(out.width, out.height, out.frames, out.tensor_bytes_bound);
    const int64_t n = multiply(multiply(g[0], g[1]), g[2]);
    require(out.latent_shape == std::vector<int64_t>({1, n, 128}) &&
                out.expected_video_shape ==
                    std::vector<int64_t>({1, 3, out.frames, out.height, out.width}),
            "Token grid decoder/latent shape mismatch");
    const auto &inputs = out.runtime_inputs;
    require(inputs.size() == 5, "Token grid input field set mismatch");
    for (auto name : {"seed", "decode_seed"})
        validate_architecture_tensor(inputs.at(name), {}, DType::I64);
    require(std::bit_cast<uint64_t>(read_scalar_i64(inputs.at("decode_seed"))) ==
                std::bit_cast<uint64_t>(read_scalar_i64(inputs.at("seed"))) + uint64_t(100),
            "Token grid decode seed mismatch");
    for (auto name : {"latent_grid", "latent_shape"})
        validate_architecture_tensor(inputs.at(name), {3}, DType::I64);
    for (int i = 0; i < 3; ++i)
        require(inputs.at("latent_grid").data_as<int64_t>()[i] == g[i] &&
                    inputs.at("latent_shape").data_as<int64_t>()[i] == out.latent_shape[i],
                "Token grid shape/grid inconsistency");
    const auto &c = inputs.at("coordinates");
    validate_architecture_tensor(c, {1, 3, n}, DType::F32);
    for (int64_t i = 0; i < n; ++i)
        require(c.data_as<float>()[i] == float(i / (g[1] * g[2])) &&
                    c.data_as<float>()[n + i] == float((i / g[2]) % g[1]) &&
                    c.data_as<float>()[2 * n + i] == float(i % g[2]),
                "Token grid coordinate/order mismatch");
}
} // namespace vrhino::product
