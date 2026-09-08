#include "vrhino/component_tiling.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

void validate_axis(int64_t tile, int64_t stride, double overlap,
                   const char* name) {
    require(tile > 0 && stride > 0 && stride <= tile,
            std::string(name) + " tile/stride contract violation");
    require(std::isfinite(overlap) && overlap >= 0.0 && overlap < 1.0,
            std::string(name) + " overlap must be in [0,1)");
    const double declared_stride = static_cast<double>(tile) * (1.0 - overlap);
    require(std::abs(declared_stride - static_cast<double>(stride)) < 1e-9,
            std::string(name) + " overlap and stride disagree");
}

int64_t overlap_output_extent(int64_t tile, int64_t stride,
                              int64_t scale) {
    return (tile - stride) * scale;
}

bool is_power_of_two(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

size_t spatial_axis_index(SpatialTileAxis axis) {
    require(axis == SpatialTileAxis::Height || axis == SpatialTileAxis::Width,
            "Recursive Component tile axis must be height or width");
    return axis == SpatialTileAxis::Height ? 0U : 1U;
}

int64_t nearest_multiple_ties_to_even(int64_t value, int64_t multiple) {
    require(value >= 0 && multiple > 0,
            "Invalid recursive Component split rounding input");
    const int64_t quotient = value / multiple;
    const int64_t remainder = value % multiple;
    if (remainder * 2 < multiple) return quotient * multiple;
    if (remainder * 2 > multiple) return (quotient + 1) * multiple;
    return (quotient + (quotient & 1)) * multiple;
}

uint16_t float_to_bf16_rne(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + bias) >> 16U);
}

float bf16_to_float(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

float round_to_bf16(float value) {
    return bf16_to_float(float_to_bf16_rne(value));
}

float bf16_add(float left, float right) {
    return round_to_bf16(left + right);
}

float bf16_sub(float left, float right) {
    return round_to_bf16(left - right);
}

float bf16_mul(float left, float right) {
    return round_to_bf16(left * right);
}

float bf16_div(float left, float right) {
    return round_to_bf16(left / right);
}

// Matches output-dtype torch.linspace construction, including its midpoint
// formulation that anchors the second half at the right endpoint.
Tensor interior_linspace_host(const std::vector<int64_t>& shape,
                              int64_t extent, DType dtype,
                              bool complement) {
    require(extent > 0 && shape_numel(shape) == extent &&
                (dtype == DType::F32 || dtype == DType::BF16),
            "Invalid recursive Component blend ramp declaration");
    Tensor result = Tensor::host(shape, dtype);
    if (dtype == DType::F32) {
        float* values = result.data_as<float>();
        const float denominator = static_cast<float>(extent + 1);
        const float start = 1.0f / denominator;
        const float end = static_cast<float>(extent) / denominator;
        const float step = extent == 1
            ? 0.0f : (end - start) / static_cast<float>(extent - 1);
        for (int64_t index = 0; index < extent; ++index) {
            const float current = index < extent / 2
                ? start + step * static_cast<float>(index)
                : end - step * static_cast<float>(extent - index - 1);
            values[index] = complement ? 1.0f - current : current;
        }
        return result;
    }

    uint16_t* values = result.data_as<uint16_t>();
    const float denominator = round_to_bf16(static_cast<float>(extent + 1));
    const float start = bf16_div(round_to_bf16(1.0f), denominator);
    const float end = bf16_div(
        round_to_bf16(static_cast<float>(extent)), denominator);
    const float step = extent == 1 ? round_to_bf16(0.0f) : bf16_div(
        bf16_sub(end, start), round_to_bf16(static_cast<float>(extent - 1)));
    for (int64_t index = 0; index < extent; ++index) {
        const float distance = round_to_bf16(static_cast<float>(
            index < extent / 2 ? index : extent - index - 1));
        const float offset = bf16_mul(distance, step);
        const float current = index < extent / 2
            ? bf16_add(start, offset) : bf16_sub(end, offset);
        const float value = complement
            ? bf16_sub(round_to_bf16(1.0f), current) : current;
        values[index] = float_to_bf16_rne(value);
    }
    return result;
}

SpatialTileAxis spatial_axis_from_json(const Json& value) {
    if (value.string() == "HEIGHT") return SpatialTileAxis::Height;
    require(value.string() == "WIDTH",
            "Unknown recursive Component tile axis: " + value.string());
    return SpatialTileAxis::Width;
}

TemporalBoundaryPolicy temporal_boundary_from_json(const Json& value) {
    if (value.string() == "INDEPENDENT")
        return TemporalBoundaryPolicy::Independent;
    require(value.string() == "PRESERVE_FIRST_FRAME_CAUSAL",
            "Unknown Component temporal boundary policy: " + value.string());
    return TemporalBoundaryPolicy::PreserveFirstFrameCausal;
}

std::array<int64_t, 2> spatial_pair_from_json(const Json& value,
                                               const char* name) {
    require(value.is_array() && value.array().size() == 2,
            std::string(name) + " must contain H/W values");
    return {value.array()[0].integer(), value.array()[1].integer()};
}

}  // namespace

void ComponentExecutionConfig::validate() const {
    if (mode == ComponentExecutionMode::Untiled) return;
    require(mode == ComponentExecutionMode::Tiled,
            "Unsupported Component execution mode");
    tiling.validate();
}

const char* component_execution_mode_name(ComponentExecutionMode mode) {
    if (mode == ComponentExecutionMode::Untiled) return "UNTILED";
    if (mode == ComponentExecutionMode::Tiled) return "TILED";
    throw Error("Unsupported Component execution mode");
}

ComponentExecutionConfig component_execution_config_from_json(
        const Json& descriptor) {
    require(descriptor.is_object(),
            "Component execution config must be an object");
    const std::string& mode = descriptor.at("mode").string();
    ComponentExecutionConfig result;
    if (mode == "UNTILED") {
        require(descriptor.object().size() == 1,
                "UNTILED Component execution config has unexpected fields");
        result.mode = ComponentExecutionMode::Untiled;
        return result;
    }
    require(mode == "TILED", "Unknown Component execution mode: " + mode);
    require(descriptor.object().size() == 2,
            "TILED Component execution config must contain mode and tiling");
    const Json& tiling = descriptor.at("tiling");
    result.mode = ComponentExecutionMode::Tiled;
    require(tiling.is_object(), "TILED Component tiling config must be an object");

    if (tiling.find("planner_policy")) {
        require(tiling.object().size() == 12,
                "Recursive Component tiling config is incomplete or has unexpected fields");
        require(tiling.at("planner_policy").string() ==
                    "RECURSIVE_BISECTION",
                "Unknown Component tile planner policy");
        result.tiling.planner = TilePlannerPolicy::RecursiveBisection;
        require(tiling.at("merge_policy").string() ==
                    "RECURSIVE_OVERLAP_REPLACE",
                "Recursive planner requires recursive overlap replacement");
        result.tiling.merge = TileMergePolicy::RecursiveOverlapReplace;
        require(tiling.at("blend_weights").string() ==
                    "INTERIOR_LINSPACE",
                "Recursive overlap replacement requires interior linspace weights");
        result.tiling.blend = TileBlendPolicy::InteriorLinspace;
        require(tiling.at("merge_order").string() == "PLANNER_TREE",
                "Recursive overlap replacement requires planner-tree order");
        require(tiling.at("rounding_policy").string() == "TIES_TO_EVEN",
                "Unknown recursive Component split rounding policy");
        result.tiling.recursive.rounding = TileRoundingPolicy::TiesToEven;
        require(tiling.at("temporal_policy").string() == "FULL",
                "Recursive spatial Component tiling requires full temporal policy");
        result.tiling.temporal_policy = TemporalTilingPolicy::Full;
        result.tiling.temporal_boundary = temporal_boundary_from_json(
            tiling.at("temporal_boundary"));

        const auto& priority = tiling.at("axis_priority").array();
        require(priority.size() == 2,
                "Recursive Component axis_priority must contain two axes");
        result.tiling.recursive.axis_priority = {
            spatial_axis_from_json(priority[0]),
            spatial_axis_from_json(priority[1]),
        };
        result.tiling.recursive.tile_counts = spatial_pair_from_json(
            tiling.at("tile_counts"), "Recursive Component tile_counts");
        result.tiling.recursive.input_overlap = spatial_pair_from_json(
            tiling.at("input_overlap"), "Recursive Component input_overlap");
        result.tiling.recursive.minimum_block = spatial_pair_from_json(
            tiling.at("minimum_block"), "Recursive Component minimum_block");
        const auto& scale = tiling.at("output_scale").array();
        require(scale.size() == 3,
                "Component tiling output_scale must contain T/H/W values");
        for (size_t axis = 0; axis < scale.size(); ++axis)
            result.tiling.output_scale[axis] = scale[axis].integer();
        result.validate();
        return result;
    }

    require(tiling.object().size() == 14,
            "Fixed-stride Component tiling config is incomplete or has unexpected fields");
    result.tiling.temporal_tile = tiling.at("temporal_tile").integer();
    result.tiling.temporal_stride = tiling.at("temporal_stride").integer();
    result.tiling.spatial_tile_height =
        tiling.at("spatial_tile_height").integer();
    result.tiling.spatial_tile_width =
        tiling.at("spatial_tile_width").integer();
    result.tiling.spatial_stride_height =
        tiling.at("spatial_stride_height").integer();
    result.tiling.spatial_stride_width =
        tiling.at("spatial_stride_width").integer();
    result.tiling.temporal_overlap = tiling.at("temporal_overlap").number();
    result.tiling.spatial_overlap_height =
        tiling.at("spatial_overlap_height").number();
    result.tiling.spatial_overlap_width =
        tiling.at("spatial_overlap_width").number();
    const auto& scale = tiling.at("output_scale").array();
    require(scale.size() == 3,
            "Component tiling output_scale must contain T/H/W values");
    for (size_t axis = 0; axis < scale.size(); ++axis)
        result.tiling.output_scale[axis] = scale[axis].integer();
    result.tiling.temporal_context =
        tiling.at("temporal_context").integer();
    result.tiling.temporal_boundary = temporal_boundary_from_json(
        tiling.at("temporal_boundary"));
    require(tiling.at("crop_policy").string() == "STRIDE_CORE",
            "Unknown Component tile crop policy");
    result.tiling.crop = TileCropPolicy::StrideCore;
    require(tiling.at("blend_policy").string() == "LINEAR",
            "Unknown Component tile blend policy");
    result.tiling.blend = TileBlendPolicy::Linear;
    result.validate();
    return result;
}

void ComponentTilingConfig::validate() const {
    require(output_scale[0] > 0 && output_scale[1] > 0 &&
                output_scale[2] > 0,
            "Component tile output scales must be positive");
    if (planner == TilePlannerPolicy::RecursiveBisection) {
        require(merge == TileMergePolicy::RecursiveOverlapReplace &&
                    blend == TileBlendPolicy::InteriorLinspace,
                "Recursive Component planner/merge contract mismatch");
        require(temporal_policy == TemporalTilingPolicy::Full,
                "Recursive spatial Component planner requires full time");
        require(recursive.rounding == TileRoundingPolicy::TiesToEven,
                "Unsupported recursive Component split rounding policy");
        require(recursive.axis_priority[0] != recursive.axis_priority[1] &&
                    recursive.axis_priority[0] != SpatialTileAxis::None &&
                    recursive.axis_priority[1] != SpatialTileAxis::None,
                "Recursive Component axis priority must contain H and W once");
        for (size_t axis = 0; axis < 2; ++axis) {
            require(is_power_of_two(recursive.tile_counts[axis]),
                    "Recursive Component tile count must be a power of two");
            require(recursive.input_overlap[axis] >= 0 &&
                        recursive.input_overlap[axis] % 2 == 0,
                    "Recursive Component overlap must be non-negative and even");
            require(recursive.minimum_block[axis] > 0 &&
                        (recursive.input_overlap[axis] / 2) %
                                recursive.minimum_block[axis] == 0,
                    "Recursive Component overlap/minimum-block mismatch");
        }
        return;
    }

    require(planner == TilePlannerPolicy::FixedStride &&
                merge == TileMergePolicy::StrideCoreLinear &&
                temporal_policy == TemporalTilingPolicy::Configured,
            "Unsupported fixed-stride Component planner contract");
    validate_axis(temporal_tile, temporal_stride, temporal_overlap,
                  "temporal");
    validate_axis(spatial_tile_height, spatial_stride_height,
                  spatial_overlap_height, "spatial height");
    validate_axis(spatial_tile_width, spatial_stride_width,
                  spatial_overlap_width, "spatial width");
    require(temporal_context >= 0,
            "Component tile temporal context must be non-negative");
    require(crop == TileCropPolicy::StrideCore,
            "Unsupported Component tile crop policy");
    require(blend == TileBlendPolicy::Linear,
            "Unsupported Component tile blend policy");
    if (temporal_boundary ==
        TemporalBoundaryPolicy::PreserveFirstFrameCausal)
        require(temporal_context > 0,
                "First-frame causal tiling requires temporal context");
}

std::vector<ComponentTileRegion> TiledComponentExecutor::plan_axis(
    int64_t length, int64_t tile, int64_t stride, int64_t trailing_context,
    bool drop_empty_causal_tail) {
    require(length > 0 && tile > 0 && stride > 0 && stride <= tile &&
                trailing_context >= 0,
            "Invalid Component tile axis declaration");
    std::vector<ComponentTileRegion> result;
    for (int64_t start = 0; start < length; start += stride) {
        if (drop_empty_causal_tail && start > 0 && start >= length - 1)
            break;
        const int64_t stop = std::min(length, start + tile + trailing_context);
        result.push_back({start, stop, start == 0, stop == length});
        if (stop == length && start == 0) break;
    }
    require(!result.empty(), "Component tile planner produced no regions");
    result.back().last = true;
    return result;
}

std::unique_ptr<ComponentSpatialTilePlanNode>
TiledComponentExecutor::plan_recursive_spatial(
        int64_t height, int64_t width,
        const RecursiveSpatialTilingConfig& config) {
    require(height > 0 && width > 0,
            "Recursive Component planner requires positive H/W");
    for (size_t axis = 0; axis < 2; ++axis) {
        require(is_power_of_two(config.tile_counts[axis]),
                "Recursive Component tile count must be a power of two");
        require(config.minimum_block[axis] > 0 &&
                    config.input_overlap[axis] >= 0 &&
                    config.input_overlap[axis] % 2 == 0 &&
                    (config.input_overlap[axis] / 2) %
                            config.minimum_block[axis] == 0,
                "Invalid recursive Component overlap/minimum block");
    }
    require(config.axis_priority[0] != config.axis_priority[1] &&
                config.axis_priority[0] != SpatialTileAxis::None &&
                config.axis_priority[1] != SpatialTileAxis::None,
            "Invalid recursive Component axis priority");

    using Node = ComponentSpatialTilePlanNode;
    std::function<std::unique_ptr<Node>(
        ComponentSpatialTileBox, std::array<int64_t, 2>)> visit;
    visit = [&](ComponentSpatialTileBox region,
                std::array<int64_t, 2> counts) -> std::unique_ptr<Node> {
        auto node = std::make_unique<Node>();
        node->region = region;
        const std::array<int64_t, 2> extents{
            region.height_stop - region.height_start,
            region.width_stop - region.width_start,
        };
        for (size_t axis = 0; axis < 2; ++axis)
            require(extents[axis] > 0 &&
                        extents[axis] % config.minimum_block[axis] == 0,
                    "Recursive Component parent violates minimum block");

        SpatialTileAxis split = SpatialTileAxis::None;
        for (SpatialTileAxis candidate : config.axis_priority) {
            if (counts[spatial_axis_index(candidate)] >= 2) {
                split = candidate;
                break;
            }
        }
        if (split == SpatialTileAxis::None) return node;

        const size_t axis_index = spatial_axis_index(split);
        const int64_t minimum = config.minimum_block[axis_index];
        const int64_t half = nearest_multiple_ties_to_even(
            extents[axis_index] / 2, minimum);
        const int64_t overlap = config.input_overlap[axis_index] / 2;
        const int64_t parent_start = split == SpatialTileAxis::Height
            ? region.height_start : region.width_start;
        const int64_t parent_stop = split == SpatialTileAxis::Height
            ? region.height_stop : region.width_stop;
        const int64_t middle = parent_start + half;
        require(middle > parent_start && middle < parent_stop,
                "Recursive Component split produced an empty core");

        ComponentSpatialTileBox first = region;
        ComponentSpatialTileBox second = region;
        const int64_t first_stop = std::min(parent_stop, middle + overlap);
        const int64_t second_start = std::max(parent_start, middle - overlap);
        if (split == SpatialTileAxis::Height) {
            first.height_stop = first_stop;
            second.height_start = second_start;
        } else {
            first.width_stop = first_stop;
            second.width_start = second_start;
        }
        require(first_stop > parent_start && second_start < parent_stop &&
                    first_stop - second_start ==
                        config.input_overlap[axis_index],
                "Recursive Component split overlap was clamped inconsistently");

        node->split_axis = split;
        counts[axis_index] /= 2;
        node->first = visit(first, counts);
        node->second = visit(second, counts);
        return node;
    };

    return visit({0, height, 0, width}, config.tile_counts);
}

Tensor TiledComponentExecutor::crop_prefix(const Tensor& input, int64_t axis,
                                           int64_t extent) {
    extent = std::clamp<int64_t>(extent, 0, input.dim(axis));
    require(extent > 0, "Component tile crop produced an empty region");
    if (extent == input.dim(axis)) return input;
    return backend_.slice(input, axis, 0, extent);
}

Tensor TiledComponentExecutor::blend_prefix(
    const Tensor& previous, const Tensor& current, int64_t axis,
    int64_t extent) {
    extent = std::min({extent, previous.dim(axis), current.dim(axis)});
    if (extent <= 0) return current;
    Tensor previous_tail = backend_.slice(
        previous, axis, previous.dim(axis) - extent, previous.dim(axis));
    Tensor current_head = backend_.slice(current, axis, 0, extent);
    std::vector<int64_t> ramp_shape(static_cast<size_t>(current.ndim()), 1);
    ramp_shape[static_cast<size_t>(axis)] = extent;
    std::vector<float> current_weight(static_cast<size_t>(extent));
    std::vector<float> previous_weight(static_cast<size_t>(extent));
    for (int64_t index = 0; index < extent; ++index) {
        const float alpha = static_cast<float>(index) /
                            static_cast<float>(extent);
        current_weight[static_cast<size_t>(index)] = alpha;
        previous_weight[static_cast<size_t>(index)] = 1.0f - alpha;
    }
    Tensor blended = backend_.add(
        backend_.mul(previous_tail, host_f32(ramp_shape, previous_weight)),
        backend_.mul(current_head, host_f32(ramp_shape, current_weight)));
    if (extent == current.dim(axis)) return blended;
    Tensor tail = backend_.slice(current, axis, extent, current.dim(axis));
    return backend_.concat({blended, tail}, axis);
}

void TiledComponentExecutor::validate_tile_output(
    const Tensor& input, const Tensor& output) const {
    require(input.ndim() == 5 && output.ndim() == 5 &&
                input.dim(0) == output.dim(0),
            "Tiled Component graph must preserve BCTHW batch rank");
    const int64_t expected_t = config_.temporal_boundary ==
            TemporalBoundaryPolicy::PreserveFirstFrameCausal
        ? (input.dim(2) - 1) * config_.output_scale[0] + 1
        : input.dim(2) * config_.output_scale[0];
    require(output.dim(2) == expected_t &&
                output.dim(3) == input.dim(3) * config_.output_scale[1] &&
                output.dim(4) == input.dim(4) * config_.output_scale[2],
            "Tiled Component graph output scale contract mismatch");
}

Tensor TiledComponentExecutor::merge_recursive_overlap(
        const Tensor& first, const Tensor& second, int64_t axis,
        int64_t extent) {
    require(first.ndim() == second.ndim() && first.dtype() == second.dtype() &&
                (first.dtype() == DType::F32 || first.dtype() == DType::BF16),
            "Recursive Component merge tensor contract mismatch");
    if (extent == 0) return backend_.concat({first, second}, axis);
    require(extent > 0 && first.dim(axis) >= extent &&
                second.dim(axis) >= extent,
            "Recursive Component merge overlap exceeds child output");

    Tensor first_overlap = backend_.slice(
        first, axis, first.dim(axis) - extent, first.dim(axis));
    Tensor second_overlap = backend_.slice(second, axis, 0, extent);
    std::vector<int64_t> ramp_shape(static_cast<size_t>(first.ndim()), 1);
    ramp_shape[static_cast<size_t>(axis)] = extent;
    Tensor first_weight = interior_linspace_host(
        ramp_shape, extent, first.dtype(), true);
    Tensor second_weight = interior_linspace_host(
        ramp_shape, extent, first.dtype(), false);
    Tensor blended = backend_.add(
        backend_.mul(first_overlap, first_weight),
        backend_.mul(second_overlap, second_weight));

    std::vector<Tensor> pieces;
    if (first.dim(axis) > extent)
        pieces.push_back(backend_.slice(
            first, axis, 0, first.dim(axis) - extent));
    pieces.push_back(std::move(blended));
    if (second.dim(axis) > extent)
        pieces.push_back(backend_.slice(
            second, axis, extent, second.dim(axis)));
    return pieces.size() == 1 ? pieces.front() : backend_.concat(pieces, axis);
}

Tensor TiledComponentExecutor::execute_recursive_node(
        const Tensor& input, const ComponentTileDecode& decode,
        const ComponentSpatialTilePlanNode& node) {
    if (node.leaf()) {
        Tensor tile = backend_.slice(input, 3, node.region.height_start,
                                     node.region.height_stop);
        tile = backend_.slice(tile, 4, node.region.width_start,
                              node.region.width_stop);
        stats_.maximum_input_tile_bytes =
            std::max(stats_.maximum_input_tile_bytes, tile.bytes());
        Tensor output = decode(tile);
        validate_tile_output(tile, output);
        ++stats_.graph_executions;
        stats_.maximum_output_tile_bytes =
            std::max(stats_.maximum_output_tile_bytes, output.bytes());
        return output;
    }

    require(node.first && node.second,
            "Recursive Component planner produced an incomplete tree");
    Tensor first = execute_recursive_node(input, decode, *node.first);
    Tensor second = execute_recursive_node(input, decode, *node.second);
    const size_t spatial_index = spatial_axis_index(node.split_axis);
    const int64_t tensor_axis = node.split_axis == SpatialTileAxis::Height
        ? 3 : 4;
    const int64_t first_input_extent = node.split_axis == SpatialTileAxis::Height
        ? node.first->region.height_stop - node.first->region.height_start
        : node.first->region.width_stop - node.first->region.width_start;
    require(first_input_extent > 0,
            "Recursive Component child has empty input extent");
    const double child_scale = static_cast<double>(first.dim(tensor_axis)) /
                               static_cast<double>(first_input_extent);
    const int64_t output_overlap = static_cast<int64_t>(
        static_cast<double>(config_.recursive.input_overlap[spatial_index]) *
        child_scale);
    return merge_recursive_overlap(
        first, second, tensor_axis, output_overlap);
}

Tensor TiledComponentExecutor::execute_recursive_spatial(
        const Tensor& input, const ComponentTileDecode& decode) {
    auto tree = plan_recursive_spatial(
        input.dim(3), input.dim(4), config_.recursive);
    std::vector<ComponentSpatialTileBox> leaves;
    std::function<void(const ComponentSpatialTilePlanNode&)> collect =
        [&](const ComponentSpatialTilePlanNode& node) {
            if (node.leaf()) {
                leaves.push_back(node.region);
                return;
            }
            require(node.first && node.second,
                    "Recursive Component planner tree is incomplete");
            collect(*node.first);
            collect(*node.second);
        };
    collect(*tree);
    stats_.spatial_tiles_per_temporal = leaves.size();

    auto axis_plan = [&](bool height) {
        std::vector<std::pair<int64_t, int64_t>> intervals;
        for (const auto& leaf : leaves)
            intervals.emplace_back(
                height ? leaf.height_start : leaf.width_start,
                height ? leaf.height_stop : leaf.width_stop);
        std::sort(intervals.begin(), intervals.end());
        intervals.erase(std::unique(intervals.begin(), intervals.end()),
                        intervals.end());
        std::vector<ComponentTileRegion> plan;
        for (size_t index = 0; index < intervals.size(); ++index)
            plan.push_back({intervals[index].first, intervals[index].second,
                            index == 0, index + 1 == intervals.size()});
        return plan;
    };
    stats_.height_plan = axis_plan(true);
    stats_.width_plan = axis_plan(false);
    return execute_recursive_node(input, decode, *tree);
}

Tensor TiledComponentExecutor::execute_spatial(
    const Tensor& input, const ComponentTileDecode& decode) {
    const bool tile_height = input.dim(3) > config_.spatial_tile_height;
    const bool tile_width = input.dim(4) > config_.spatial_tile_width;
    if (!tile_height && !tile_width) {
        Tensor output = decode(input);
        validate_tile_output(input, output);
        ++stats_.graph_executions;
        stats_.maximum_input_tile_bytes =
            std::max(stats_.maximum_input_tile_bytes, input.bytes());
        stats_.maximum_output_tile_bytes =
            std::max(stats_.maximum_output_tile_bytes, output.bytes());
        return output;
    }

    const auto height = plan_axis(input.dim(3), config_.spatial_tile_height,
                                  config_.spatial_stride_height);
    const auto width = plan_axis(input.dim(4), config_.spatial_tile_width,
                                 config_.spatial_stride_width);
    if (stats_.height_plan.empty()) stats_.height_plan = height;
    if (stats_.width_plan.empty()) stats_.width_plan = width;
    stats_.spatial_tiles_per_temporal = std::max<uint64_t>(
        stats_.spatial_tiles_per_temporal, height.size() * width.size());

    const int64_t blend_h = overlap_output_extent(
        config_.spatial_tile_height, config_.spatial_stride_height,
        config_.output_scale[1]);
    const int64_t blend_w = overlap_output_extent(
        config_.spatial_tile_width, config_.spatial_stride_width,
        config_.output_scale[2]);
    const int64_t keep_h = config_.spatial_stride_height *
                           config_.output_scale[1];
    const int64_t keep_w = config_.spatial_stride_width *
                           config_.output_scale[2];

    std::vector<Tensor> previous_row;
    std::vector<Tensor> output_rows;
    for (size_t row_index = 0; row_index < height.size(); ++row_index) {
        std::vector<Tensor> current_row;
        std::vector<Tensor> output_pieces;
        current_row.reserve(width.size());
        output_pieces.reserve(width.size());
        for (size_t column = 0; column < width.size(); ++column) {
            Tensor tile = backend_.slice(input, 3, height[row_index].start,
                                         height[row_index].stop);
            tile = backend_.slice(tile, 4, width[column].start,
                                  width[column].stop);
            stats_.maximum_input_tile_bytes =
                std::max(stats_.maximum_input_tile_bytes, tile.bytes());
            Tensor decoded = decode(tile);
            validate_tile_output(tile, decoded);
            ++stats_.graph_executions;
            stats_.maximum_output_tile_bytes =
                std::max(stats_.maximum_output_tile_bytes, decoded.bytes());
            if (!previous_row.empty())
                decoded = blend_prefix(previous_row[column], decoded, 3,
                                       blend_h);
            if (!current_row.empty())
                decoded = blend_prefix(current_row.back(), decoded, 4,
                                       blend_w);
            current_row.push_back(decoded);
            Tensor piece = crop_prefix(decoded, 3, keep_h);
            piece = crop_prefix(piece, 4, keep_w);
            output_pieces.push_back(std::move(piece));
        }
        output_rows.push_back(backend_.concat(output_pieces, 4));
        previous_row = std::move(current_row);
    }
    return backend_.concat(output_rows, 3);
}

Tensor TiledComponentExecutor::execute(
    const Tensor& input, const ComponentTileDecode& decode) {
    require(input.ndim() == 5 && input.dim(0) > 0 && input.dim(1) > 0 &&
                input.dim(2) > 0 && input.dim(3) > 0 && input.dim(4) > 0,
            "Tiled Component executor expects non-empty BCTHW input");
    require(static_cast<bool>(decode),
            "Tiled Component executor requires a graph callback");
    stats_ = {};

    if (config_.planner == TilePlannerPolicy::RecursiveBisection) {
        require(config_.temporal_policy == TemporalTilingPolicy::Full,
                "Recursive spatial Component executor requires full time");
        stats_.temporal_tiles = 1;
        stats_.temporal_plan = {{0, input.dim(2), true, true}};
        Tensor output = execute_recursive_spatial(input, decode);
        validate_tile_output(input, output);
        return output;
    }

    const bool tile_temporal = input.dim(2) > config_.temporal_tile;
    if (!tile_temporal) {
        stats_.temporal_tiles = 1;
        stats_.temporal_plan = {{0, input.dim(2), true, true}};
        Tensor output = execute_spatial(input, decode);
        validate_tile_output(input, output);
        return output;
    }

    const bool causal = config_.temporal_boundary ==
                        TemporalBoundaryPolicy::PreserveFirstFrameCausal;
    stats_.temporal_plan = plan_axis(
        input.dim(2), config_.temporal_tile, config_.temporal_stride,
        config_.temporal_context, causal);
    stats_.temporal_tiles = stats_.temporal_plan.size();
    const int64_t blend_t = overlap_output_extent(
        config_.temporal_tile, config_.temporal_stride,
        config_.output_scale[0]);
    const int64_t keep_t = config_.temporal_stride *
                           config_.output_scale[0];

    std::vector<Tensor> output_pieces;
    Tensor previous;
    for (size_t index = 0; index < stats_.temporal_plan.size(); ++index) {
        const ComponentTileRegion& region = stats_.temporal_plan[index];
        Tensor tile = backend_.slice(input, 2, region.start, region.stop);
        Tensor decoded = execute_spatial(tile, decode);
        if (causal && index > 0) {
            require(decoded.dim(2) > 1,
                    "Causal temporal tile has no output after first-frame crop");
            decoded = backend_.slice(decoded, 2, 1, decoded.dim(2));
        }
        if (index > 0) decoded = blend_prefix(previous, decoded, 2, blend_t);
        previous = decoded;
        const int64_t wanted = keep_t + (causal && index == 0 ? 1 : 0);
        output_pieces.push_back(crop_prefix(decoded, 2, wanted));
    }
    Tensor output = backend_.concat(output_pieces, 2);
    const int64_t expected_t = causal
        ? (input.dim(2) - 1) * config_.output_scale[0] + 1
        : input.dim(2) * config_.output_scale[0];
    require(output.dim(2) == expected_t &&
                output.dim(3) == input.dim(3) * config_.output_scale[1] &&
                output.dim(4) == input.dim(4) * config_.output_scale[2],
            "Tiled Component stitched output shape mismatch");
    return output;
}

Tensor ComponentGraphExecutor::execute(const Tensor& input,
                                  const ComponentTileDecode& decode) {
    require(static_cast<bool>(decode),
            "Component executor requires a graph callback");
    stats_ = {};
    stats_.mode = config_.mode;
    if (config_.mode == ComponentExecutionMode::Untiled) {
        Tensor output = decode(input);
        stats_.tiling.graph_executions = 1;
        stats_.tiling.temporal_tiles = 1;
        stats_.tiling.spatial_tiles_per_temporal = 1;
        stats_.tiling.maximum_input_tile_bytes = input.bytes();
        stats_.tiling.maximum_output_tile_bytes = output.bytes();
        return output;
    }
    TiledComponentExecutor executor(backend_, config_.tiling);
    Tensor output = executor.execute(input, decode);
    stats_.tiling = executor.stats();
    return output;
}

}  // namespace vrhino
