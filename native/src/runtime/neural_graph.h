#pragma once

#include <optional>
#include <limits>
#include <variant>

#include "vrhino/backend.h"
#include "vrhino/error.h"

// Internal, independently admitted executable contract. No package/API routing.
namespace vrhino::neural_graph {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

inline constexpr char schema_v1[] = "vrhino.neural-graph.v1";
inline constexpr size_t max_values = 65536, max_nodes = 65536;
inline constexpr size_t max_outputs = 64, max_constants = 64, max_rank = 8;
inline constexpr size_t max_detail = 256;
using ValueId = uint32_t;
using Location = std::optional<size_t>;

enum class Code {
    UnsupportedSchema, UnsupportedPrimitive, InvalidOperation, LimitExceeded,
    InvalidShape, InvalidDType, InvalidValue, InvalidDependency,
    InvalidInputBinding, InvalidParameterBinding, InvalidContext,
    BackendFailure, InvalidNodeOutput, CompletionFailure, InvalidOutput
};
enum class Phase { Admit, Bind, Run, Drain, Capture };

class GraphError : public Error {
public:
    GraphError(Code code, Phase phase, const std::string& detail,
               Location node = {}, Location value = {}, Location primitive = {})
        : Error(detail.substr(0, max_detail)), code(code), phase(phase),
          node(node), value(value), primitive(primitive) {}
    Code code;
    Phase phase;
    Location node, value, primitive;
    std::string secondary_completion;
};

struct TensorSpec {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
};
struct Constant { ValueId value; float scalar; };
struct Add { ValueId a, b; };
struct Mul { ValueId a, b; };
struct Reshape { ValueId x; std::vector<int64_t> shape; };
struct Linear {
    ValueId x, weight; std::optional<ValueId> bias;
    DType compute_dtype = DType::F32;
};
struct RmsNorm { ValueId x; std::optional<ValueId> weight; int64_t axis; float epsilon; };
struct Activate { ValueId x; Activation kind; };
struct Permute { ValueId x; std::vector<int64_t> axes; };
struct Slice { ValueId x; int64_t axis, start, stop; };
struct Concat { std::vector<ValueId> tensors; int64_t axis; };
// Rank-2 F32/BF16 table; external I32/I64 selectors, shape + table width.
// Selector payload bounds are checked by Backend before indexed dispatch.
struct Gather { ValueId table, indices; };
// Adjacent pairs on the even final axis: x*cos + [-x_odd, x_even]*sin.
// Equal-shape supplied frequencies broadcast to x without expanding it.
struct Rope { ValueId x, cosine, sine; };
// Frozen subset: contiguous F32/BF16 BSHD, equal B/H/D, K.shape == V.shape;
// Q/K sequence lengths may differ. Explicit finite positive scale only.
// Unmasked, noncausal, no additive bias or observation; no implicit variants.
struct Attention { ValueId q, k, v; float scale; };
// Cast v1 admits only F32 <-> BF16; source dtype is the referenced TensorSpec.
struct Cast { ValueId x; DType destination; };
// Final contiguous axis only; either no affine or a homogeneous [D] pair.
// Explicit positive finite epsilon; result dtype equals input dtype.
struct LayerNorm {
    ValueId x; std::optional<ValueId> weight, bias; float epsilon;
};
// Closed v1 subset: explicit F32/BF16 arithmetic, external integer selectors only.
// Absent/unknown forms reject before dispatch.
using Primitive = std::variant<std::monostate, Add, Mul, Reshape, Linear, RmsNorm,
                               Activate, Permute, Slice, Concat, Gather, Rope, Attention, Cast,
                               LayerNorm>;
struct Node { ValueId result; Primitive op; };
struct Description {
    std::string schema;  // Must be explicitly supplied; never inferred.
    std::vector<TensorSpec> values;
    std::vector<ValueId> inputs, parameters;
    std::vector<Constant> constants;
    std::vector<Node> nodes;
    std::vector<ValueId> outputs;
    // Equality precondition on Backend; F32 retains the frozen F32-only contract.
    DType execution_dtype = DType::F32;
};

class Graph {
public:
    const Description& description() const { return data_; }
private:
    explicit Graph(const Description& data) : data_(data) {}
    const Description data_;
    friend Graph admit(const Description&);
};

// Admission copies validated data; caller mutation cannot change the graph.
Graph admit(const Description& description);
// Caller grants exclusive Backend use, supplies host or same-context handles,
// and retains borrowed backing owners through evaluation AND returned aliases.
// No unique allocator identity exists in Tensor; provenance is a precondition.
std::vector<Tensor> evaluate(Backend& backend, const Graph& graph,
                             const std::vector<Tensor>& inputs,
                             const std::vector<Tensor>& parameters);

}  // namespace vrhino::neural_graph
