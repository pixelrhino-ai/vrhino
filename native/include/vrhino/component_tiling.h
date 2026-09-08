#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "vrhino/backend.h"
#include "vrhino/json.h"

namespace vrhino {

enum class TileCropPolicy : uint8_t { StrideCore };
enum class TileBlendPolicy : uint8_t { Linear, InteriorLinspace };
enum class TilePlannerPolicy : uint8_t { FixedStride, RecursiveBisection };
enum class TileMergePolicy : uint8_t {
    StrideCoreLinear,
    RecursiveOverlapReplace,
};
enum class TileRoundingPolicy : uint8_t { TiesToEven };
enum class TemporalTilingPolicy : uint8_t { Configured, Full };
enum class SpatialTileAxis : uint8_t { None, Height, Width };
enum class TemporalBoundaryPolicy : uint8_t {
    Independent,
    PreserveFirstFrameCausal,
};

// Shape-only declaration for a recursive spatial partition. Arrays containing
// geometry use H/W order; axis_priority declares the recursive split order.
struct RecursiveSpatialTilingConfig {
    std::array<SpatialTileAxis, 2> axis_priority{
        SpatialTileAxis::Width, SpatialTileAxis::Height};
    std::array<int64_t, 2> tile_counts{1, 1};
    std::array<int64_t, 2> input_overlap{0, 0};
    std::array<int64_t, 2> minimum_block{1, 1};
    TileRoundingPolicy rounding = TileRoundingPolicy::TiesToEven;
};

// Model-neutral declaration for executing an arbitrary BCTHW Component graph
// over bounded overlapping input regions. Values are expressed in input
// coordinates; output_scale maps them into decoded coordinates.
struct ComponentTilingConfig {
    TilePlannerPolicy planner = TilePlannerPolicy::FixedStride;
    TileMergePolicy merge = TileMergePolicy::StrideCoreLinear;
    TemporalTilingPolicy temporal_policy =
        TemporalTilingPolicy::Configured;
    int64_t temporal_tile = 0;
    int64_t temporal_stride = 0;
    int64_t spatial_tile_height = 0;
    int64_t spatial_tile_width = 0;
    int64_t spatial_stride_height = 0;
    int64_t spatial_stride_width = 0;
    double temporal_overlap = 0.0;
    double spatial_overlap_height = 0.0;
    double spatial_overlap_width = 0.0;
    std::array<int64_t, 3> output_scale{1, 1, 1};
    int64_t temporal_context = 0;
    TemporalBoundaryPolicy temporal_boundary =
        TemporalBoundaryPolicy::Independent;
    TileCropPolicy crop = TileCropPolicy::StrideCore;
    TileBlendPolicy blend = TileBlendPolicy::Linear;
    RecursiveSpatialTilingConfig recursive;

    void validate() const;
};

enum class ComponentExecutionMode : uint8_t { Untiled, Tiled };

// Model-neutral execution policy for a Component graph. Untiled is the legal
// compatibility default; a declared Tiled mode must carry a complete, valid
// tiling contract and never silently falls back.
struct ComponentExecutionConfig {
    ComponentExecutionMode mode = ComponentExecutionMode::Untiled;
    ComponentTilingConfig tiling;

    void validate() const;
};

ComponentExecutionConfig component_execution_config_from_json(
    const Json& descriptor);
const char* component_execution_mode_name(ComponentExecutionMode mode);

struct ComponentTileRegion {
    int64_t start = 0;
    int64_t stop = 0;
    bool first = false;
    bool last = false;
};

struct ComponentSpatialTileBox {
    int64_t height_start = 0;
    int64_t height_stop = 0;
    int64_t width_start = 0;
    int64_t width_stop = 0;
};

// The tree, rather than only its leaf list, is part of the execution plan:
// recursive overlap replacement is deliberately evaluated bottom-up in this
// topology so floating-point merge order remains deterministic.
struct ComponentSpatialTilePlanNode {
    ComponentSpatialTileBox region;
    SpatialTileAxis split_axis = SpatialTileAxis::None;
    std::unique_ptr<ComponentSpatialTilePlanNode> first;
    std::unique_ptr<ComponentSpatialTilePlanNode> second;

    bool leaf() const { return split_axis == SpatialTileAxis::None; }
};

struct ComponentTilingStats {
    uint64_t graph_executions = 0;
    uint64_t temporal_tiles = 0;
    uint64_t spatial_tiles_per_temporal = 0;
    size_t maximum_input_tile_bytes = 0;
    size_t maximum_output_tile_bytes = 0;
    std::vector<ComponentTileRegion> temporal_plan;
    std::vector<ComponentTileRegion> height_plan;
    std::vector<ComponentTileRegion> width_plan;
};

using ComponentTileDecode = std::function<Tensor(const Tensor&)>;

class TiledComponentExecutor {
public:
    TiledComponentExecutor(Backend& backend, ComponentTilingConfig config)
        : backend_(backend), config_(std::move(config)) {
        config_.validate();
    }

    static std::vector<ComponentTileRegion> plan_axis(
        int64_t length, int64_t tile, int64_t stride,
        int64_t trailing_context = 0, bool drop_empty_causal_tail = false);
    static std::unique_ptr<ComponentSpatialTilePlanNode>
    plan_recursive_spatial(int64_t height, int64_t width,
                           const RecursiveSpatialTilingConfig& config);

    Tensor execute(const Tensor& input, const ComponentTileDecode& decode);
    const ComponentTilingStats& stats() const { return stats_; }

private:
    Tensor execute_spatial(const Tensor& input,
                           const ComponentTileDecode& decode);
    Tensor execute_recursive_spatial(const Tensor& input,
                                     const ComponentTileDecode& decode);
    Tensor execute_recursive_node(
        const Tensor& input, const ComponentTileDecode& decode,
        const ComponentSpatialTilePlanNode& node);
    Tensor merge_recursive_overlap(const Tensor& first, const Tensor& second,
                                   int64_t axis, int64_t extent);
    Tensor blend_prefix(const Tensor& previous, const Tensor& current,
                        int64_t axis, int64_t extent);
    Tensor crop_prefix(const Tensor& input, int64_t axis, int64_t extent);
    void validate_tile_output(const Tensor& input, const Tensor& output) const;

    Backend& backend_;
    ComponentTilingConfig config_;
    ComponentTilingStats stats_;
};

struct ComponentExecutionStats {
    ComponentExecutionMode mode = ComponentExecutionMode::Untiled;
    ComponentTilingStats tiling;
};

// Executes a component graph callback under the declared tiling policy.
// Distinct from the weighted numerical ComponentExecutor in components.h.
class ComponentGraphExecutor {
public:
    ComponentGraphExecutor(Backend& backend, ComponentExecutionConfig config)
        : backend_(backend), config_(std::move(config)) {
        config_.validate();
    }

    Tensor execute(const Tensor& input, const ComponentTileDecode& decode);
    const ComponentExecutionStats& stats() const { return stats_; }

private:
    Backend& backend_;
    ComponentExecutionConfig config_;
    ComponentExecutionStats stats_;
};

}  // namespace vrhino
