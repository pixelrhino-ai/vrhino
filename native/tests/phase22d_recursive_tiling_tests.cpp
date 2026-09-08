#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/component_tiling.h"
#include "vrhino/error.h"

namespace {

using vrhino::ComponentSpatialTileBox;
using vrhino::ComponentSpatialTilePlanNode;
using vrhino::DType;
using vrhino::Tensor;

uint16_t bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>(
        (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

float as_float(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

Tensor filled(const std::vector<int64_t>& shape, DType dtype, float value) {
    Tensor result = Tensor::host(shape, dtype);
    if (dtype == DType::F32) {
        std::fill(result.data_as<float>(),
                  result.data_as<float>() + result.numel(), value);
    } else {
        vrhino::require(dtype == DType::BF16, "test fill dtype mismatch");
        std::fill(result.data_as<uint16_t>(),
                  result.data_as<uint16_t>() + result.numel(), bf16(value));
    }
    return result;
}

std::vector<ComponentSpatialTileBox> leaves(
        const ComponentSpatialTilePlanNode& root) {
    std::vector<ComponentSpatialTileBox> result;
    const auto visit = [&](const auto& self,
                           const ComponentSpatialTilePlanNode& node) -> void {
        if (node.leaf()) {
            result.push_back(node.region);
            return;
        }
        vrhino::require(node.first && node.second,
                        "golden planner tree is incomplete");
        self(self, *node.first);
        self(self, *node.second);
    };
    visit(visit, root);
    return result;
}

vrhino::RecursiveSpatialTilingConfig recursive_config(
        int64_t tiles_h, int64_t tiles_w, int64_t overlap_h,
        int64_t overlap_w, int64_t minimum_h = 1,
        int64_t minimum_w = 1) {
    vrhino::RecursiveSpatialTilingConfig result;
    result.axis_priority = {
        vrhino::SpatialTileAxis::Width,
        vrhino::SpatialTileAxis::Height,
    };
    result.tile_counts = {tiles_h, tiles_w};
    result.input_overlap = {overlap_h, overlap_w};
    result.minimum_block = {minimum_h, minimum_w};
    return result;
}

vrhino::ComponentTilingConfig executor_config(
        int64_t tiles_h, int64_t tiles_w, int64_t overlap_h,
        int64_t overlap_w) {
    vrhino::ComponentTilingConfig result;
    result.planner = vrhino::TilePlannerPolicy::RecursiveBisection;
    result.merge = vrhino::TileMergePolicy::RecursiveOverlapReplace;
    result.temporal_policy = vrhino::TemporalTilingPolicy::Full;
    result.output_scale = {1, 1, 1};
    result.temporal_boundary = vrhino::TemporalBoundaryPolicy::Independent;
    result.blend = vrhino::TileBlendPolicy::InteriorLinspace;
    result.recursive = recursive_config(
        tiles_h, tiles_w, overlap_h, overlap_w);
    return result;
}

void require_boxes(const std::string& name, int64_t height, int64_t width,
                   const vrhino::RecursiveSpatialTilingConfig& config,
                   const std::vector<ComponentSpatialTileBox>& expected) {
    auto root = vrhino::TiledComponentExecutor::plan_recursive_spatial(
        height, width, config);
    const auto actual = leaves(*root);
    vrhino::require(actual.size() == expected.size(),
                    name + " leaf count mismatch");
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto& a = actual[index];
        const auto& e = expected[index];
        vrhino::require(a.height_start == e.height_start &&
                            a.height_stop == e.height_stop &&
                            a.width_start == e.width_start &&
                            a.width_stop == e.width_stop,
                        name + " interval mismatch at leaf " +
                            std::to_string(index));
    }
    std::cout << "golden." << name << "=PASS\n";
}

std::vector<ComponentSpatialTileBox> product(
        const std::vector<std::pair<int64_t, int64_t>>& height,
        const std::vector<std::pair<int64_t, int64_t>>& width) {
    // Width-priority recursion visits every vertical subtree inside a width
    // leaf before moving to the next width leaf.
    std::vector<ComponentSpatialTileBox> result;
    for (const auto& w : width)
        for (const auto& h : height)
            result.push_back({h.first, h.second, w.first, w.second});
    return result;
}

Tensor constant_leaf_execution(vrhino::CudaBackend& backend, DType dtype,
                               int64_t height, int64_t width,
                               const vrhino::ComponentTilingConfig& config,
                               const std::vector<float>& leaf_values) {
    backend.set_execution_dtype(dtype);
    Tensor input = backend.copy_to_device(
        filled({1, 1, 1, height, width}, dtype, 0.0f), dtype);
    size_t call = 0;
    vrhino::TiledComponentExecutor executor(backend, config);
    Tensor result = executor.execute(input, [&](const Tensor& tile) {
        vrhino::require(call < leaf_values.size(),
                        "unexpected recursive decoder call");
        return backend.copy_to_device(
            filled(tile.shape(), dtype, leaf_values[call++]), dtype);
    });
    backend.synchronize();
    vrhino::require(call == leaf_values.size(),
                    "recursive decoder call count mismatch");
    return backend.copy_to_host(result);
}

float value(const Tensor& tensor, int64_t row, int64_t column) {
    const int64_t index = row * tensor.dim(4) + column;
    return tensor.dtype() == DType::F32
        ? tensor.data_as<float>()[index]
        : as_float(tensor.data_as<uint16_t>()[index]);
}

void require_near(float actual, float expected, float tolerance,
                  const std::string& name) {
    vrhino::require(std::abs(actual - expected) <= tolerance,
                    name + " mismatch: " + std::to_string(actual) +
                        " vs " + std::to_string(expected));
}

void blend_tests(vrhino::CudaBackend& backend, DType dtype) {
    const std::string label = dtype == DType::F32 ? "fp32" : "bf16";
    const float tolerance = dtype == DType::F32 ? 0.0f : 0.0f;

    const Tensor horizontal = constant_leaf_execution(
        backend, dtype, 1, 12, executor_config(1, 2, 0, 8), {0.0f, 1.0f});
    const std::vector<float> expected_f32{
        0.1111111119389534f, 0.2222222238779068f,
        0.3333333432674408f, 0.4444444477558136f,
        0.5555555820465088f, 0.6666666865348816f,
        0.7777777910232544f, 0.8888888955116272f,
    };
    const std::vector<float> expected_bf16{
        0.111328125f, 0.22265625f, 0.3359375f, 0.447265625f,
        0.5546875f, 0.66796875f, 0.77734375f, 0.890625f,
    };
    const auto& expected = dtype == DType::F32 ? expected_f32 : expected_bf16;
    for (int64_t index = 0; index < 8; ++index)
        require_near(value(horizontal, 0, index + 2), expected[index],
                     tolerance, label + " horizontal interior linspace");

    const Tensor vertical = constant_leaf_execution(
        backend, dtype, 12, 1, executor_config(2, 1, 8, 0), {0.0f, 1.0f});
    for (int64_t index = 0; index < 8; ++index)
        require_near(value(vertical, index + 2, 0), expected[index],
                     tolerance, label + " vertical interior linspace");

    const Tensor tree = constant_leaf_execution(
        backend, dtype, 6, 6, executor_config(2, 2, 2, 2),
        {1.0f, 3.0f, 2.0f, 4.0f});
    const std::vector<float> official = dtype == DType::F32
        ? std::vector<float>{1.9999998807907104f, 2.3333332538604736f,
                             2.6666665077209473f, 3.0f}
        : std::vector<float>{1.984375f, 2.328125f, 2.65625f, 3.0f};
    const std::vector<float> alternate = dtype == DType::F32
        ? std::vector<float>{2.0f, 2.3333334922790527f,
                             2.6666667461395264f, 3.000000238418579f}
        : std::vector<float>{1.9921875f, 2.34375f, 2.65625f, 3.0f};
    float order_difference = 0.0f;
    size_t corner = 0;
    for (int64_t row = 2; row < 4; ++row) {
        for (int64_t column = 2; column < 4; ++column) {
            const float actual = value(tree, row, column);
            require_near(actual, official[corner], tolerance,
                         label + " planner-tree corner");
            order_difference = std::max(
                order_difference, std::abs(actual - alternate[corner]));
            ++corner;
        }
    }
    const float expected_order_difference = dtype == DType::F32
        ? 2.384185791015625e-7f : 0.015625f;
    require_near(order_difference, expected_order_difference, 0.0f,
                 label + " merge-order evidence");
    std::cout << "blend." << label << "=PASS\n"
              << "merge_order_difference." << label << '='
              << order_difference << '\n';
}

}  // namespace

int main() {
    try {
        const auto parsed = vrhino::component_execution_config_from_json(
            vrhino::Json::parse(R"({
                "mode":"TILED",
                "tiling":{
                    "planner_policy":"RECURSIVE_BISECTION",
                    "axis_priority":["WIDTH","HEIGHT"],
                    "tile_counts":[2,4],
                    "input_overlap":[8,8],
                    "minimum_block":[1,1],
                    "rounding_policy":"TIES_TO_EVEN",
                    "temporal_policy":"FULL",
                    "temporal_boundary":"PRESERVE_FIRST_FRAME_CAUSAL",
                    "output_scale":[6,8,8],
                    "merge_policy":"RECURSIVE_OVERLAP_REPLACE",
                    "blend_weights":"INTERIOR_LINSPACE",
                    "merge_order":"PLANNER_TREE"
                }
            })"));
        vrhino::require(parsed.mode == vrhino::ComponentExecutionMode::Tiled &&
                            parsed.tiling.planner ==
                                vrhino::TilePlannerPolicy::RecursiveBisection &&
                            parsed.tiling.recursive.tile_counts ==
                                std::array<int64_t, 2>{2, 4} &&
                            parsed.tiling.output_scale ==
                                std::array<int64_t, 3>{6, 8, 8},
                        "recursive Component config binding mismatch");
        std::cout << "config.binding=PASS\n";

        require_boxes("production", 60, 106,
            recursive_config(2, 4, 8, 8),
            product({{0, 34}, {26, 60}},
                    {{0, 32}, {24, 57}, {49, 81}, {73, 106}}));
        require_boxes("even", 64, 128,
            recursive_config(2, 4, 8, 8),
            product({{0, 36}, {28, 64}},
                    {{0, 38}, {30, 68}, {60, 98}, {90, 128}}));
        require_boxes("odd", 61, 107,
            recursive_config(2, 4, 8, 8),
            product({{0, 34}, {26, 61}},
                    {{0, 32}, {24, 57}, {49, 82}, {74, 107}}));
        require_boxes("non_divisible", 59, 102,
            recursive_config(2, 4, 8, 8),
            product({{0, 33}, {25, 59}},
                    {{0, 31}, {23, 55}, {47, 78}, {70, 102}}));
        require_boxes("small_boundary", 16, 24,
            recursive_config(2, 4, 8, 8),
            product({{0, 12}, {4, 16}},
                    {{0, 12}, {4, 16}, {8, 20}, {12, 24}}));
        require_boxes("minimum_2", 62, 110,
            recursive_config(2, 4, 8, 8, 2, 2),
            product({{0, 36}, {28, 62}},
                    {{0, 34}, {26, 60}, {52, 84}, {76, 110}}));
        require_boxes("minimum_4", 64, 112,
            recursive_config(2, 4, 8, 8, 4, 4),
            product({{0, 36}, {28, 64}},
                    {{0, 36}, {28, 60}, {52, 88}, {80, 112}}));

        vrhino::CudaBackend backend;
        blend_tests(backend, DType::F32);
        blend_tests(backend, DType::BF16);
        std::cout << "planner_tree=PASS\ncorner=PASS\nstatus=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase22d_recursive_tiling_tests: " << error.what()
                  << '\n';
        return 1;
    }
}
