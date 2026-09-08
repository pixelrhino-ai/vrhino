#include "neural_graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vrhino::neural_graph {
namespace {

int64_t elements(const std::vector<int64_t>& shape, Location value = {},
                 Location node = {}) {
    auto fail = [&] { throw GraphError(Code::InvalidShape, Phase::Admit,
        "Shape must be positive rank 1..8 with checked extent", node, value); };
    if (shape.empty() || shape.size() > max_rank) fail();
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0 || d > INT32_MAX / n) fail();
        n *= d;
    }
    if (static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max() / sizeof(int64_t)) fail();
    return n;
}

void validate(const Description& d) {
    auto fail = [](Code code, const char* detail, Location node = {},
                   Location value = {}, Location primitive = {}) {
        throw GraphError(code, Phase::Admit, detail, node, value, primitive);
    };
    if (d.schema != schema_v1) fail(Code::UnsupportedSchema, "Unsupported executable schema");
    if (d.values.size() > max_values || d.nodes.size() > max_nodes ||
        d.inputs.size() > d.values.size() || d.parameters.size() > d.values.size() ||
        d.constants.size() > max_constants || d.constants.size() > d.values.size() ||
        d.outputs.size() > max_outputs)
        fail(Code::LimitExceeded, "Graph record limit exceeded");
    const auto numeric = [](DType type) { return type == DType::F32 || type == DType::BF16; };
    if (!numeric(d.execution_dtype)) fail(Code::InvalidDType, "Unsupported execution dtype");
    for (size_t i = 0; i < d.values.size(); ++i) {
        if (!(d.values[i].dtype == DType::F32 ||
              (d.execution_dtype == DType::BF16 && d.values[i].dtype == DType::BF16)) &&
            d.values[i].dtype != DType::I32 &&
            d.values[i].dtype != DType::I64)
            fail(Code::InvalidDType, "Unsupported value dtype for execution context", {}, i);
        elements(d.values[i].shape, i);
    }
    // Whole-graph support scan occurs before any binding or Backend invocation.
    for (size_t i = 0; i < d.nodes.size(); ++i) {
        if (d.nodes[i].op.valueless_by_exception() ||
            std::holds_alternative<std::monostate>(d.nodes[i].op))
            fail(Code::UnsupportedPrimitive, "Unsupported primitive", i, {}, d.nodes[i].op.index());
        if (const auto* op = std::get_if<Activate>(&d.nodes[i].op))
            if (op->kind != Activation::GeluTanh && op->kind != Activation::Silu)
                fail(Code::UnsupportedPrimitive, "Unsupported activation kind", i, {}, d.nodes[i].op.index());
    }
    std::vector<bool> defined(d.values.size(), false);
    std::vector<bool> external(d.values.size(), false), selectors(d.values.size(), false);
    auto floating = [&](ValueId id, Location node = {}) {
        if (!numeric(d.values[id].dtype))
            fail(Code::InvalidDType, "Numerical values must be F32/BF16", node, id);
    };
    auto define = [&](ValueId id, Location node = {}) {
        if (id >= defined.size() || defined[id])
            fail(Code::InvalidValue, "Invalid or duplicate value origin", node, id);
        defined[id] = true;
    };
    for (ValueId id : d.inputs) { define(id); external[id] = true; }
    for (ValueId id : d.parameters) { define(id); floating(id); }
    for (const auto& c : d.constants) {
        define(c.value);
        floating(c.value);
        if (!std::isfinite(c.scalar) || d.values[c.value].shape != std::vector<int64_t>{1})
            fail(Code::InvalidValue, "Constant must be finite scalar [1]", {}, c.value);
        uint32_t bits; std::memcpy(&bits, &c.scalar, sizeof(bits));
        if (d.values[c.value].dtype == DType::BF16 && (bits & 0xffffu))
            fail(Code::InvalidDType, "BF16 constant must be exactly representable", {}, c.value);
    }
    for (size_t i = 0; i < d.nodes.size(); ++i) {
        const auto& node = d.nodes[i];
        auto reference = [&](ValueId id) -> const std::vector<int64_t>& {
            if (id >= defined.size() || !defined[id])
                fail(Code::InvalidDependency, "Operand must have a strictly earlier definition",
                     i, id, node.op.index());
            return d.values[id].shape;
        };
        auto operand = [&](ValueId id) -> const std::vector<int64_t>& {
            const auto& shape = reference(id); floating(id, i); return shape;
        };
        auto selector = [&](ValueId id) -> const std::vector<int64_t>& {
            const auto& shape = reference(id);
            if (!external[id] || (d.values[id].dtype != DType::I32 && d.values[id].dtype != DType::I64))
                fail(Code::InvalidDType, "Gather selectors must be external I32/I64 inputs", i, id);
            selectors[id] = true;
            return shape;
        };
        auto binary = [&](ValueId a, ValueId b) {
            const auto& sa = operand(a);
            const auto& sb = operand(b);
            std::vector<int64_t> result(std::max(sa.size(), sb.size()), 1);
            for (size_t k = 0; k < result.size(); ++k) {
                const int64_t da = k < sa.size() ? sa[sa.size() - 1 - k] : 1;
                const int64_t db = k < sb.size() ? sb[sb.size() - 1 - k] : 1;
                if (da != db && da != 1 && db != 1)
                    fail(Code::InvalidShape, "Incompatible broadcast dimensions", i, node.result);
                result[result.size() - 1 - k] = std::max(da, db);
            }
            return result;
        };
        struct Infer {
            decltype(binary)& bin;
            decltype(operand)& arg;
            decltype(selector)& select;
            size_t index;
            std::vector<int64_t> operator()(std::monostate) const {
                throw GraphError(Code::UnsupportedPrimitive, Phase::Admit, "Unsupported primitive", index);
            }
            std::vector<int64_t> operator()(const Add& op) const { return bin(op.a, op.b); }
            std::vector<int64_t> operator()(const Mul& op) const { return bin(op.a, op.b); }
            std::vector<int64_t> operator()(const Reshape& op) const {
                const auto& source = arg(op.x);
                if (elements(op.shape, {}, index) != elements(source))
                    throw GraphError(Code::InvalidShape, Phase::Admit,
                                     "Reshape changes element count", index);
                return op.shape;
            }
            std::vector<int64_t> operator()(const Linear& op) const {
                auto shape = arg(op.x);
                const auto& weight = arg(op.weight);
                if (weight.size() != 2 || weight[1] != shape.back())
                    throw GraphError(Code::InvalidShape, Phase::Admit,
                                     "Linear requires [out,in] weight matching last input axis", index);
                if (op.bias && arg(*op.bias) != std::vector<int64_t>{weight[0]})
                    throw GraphError(Code::InvalidShape, Phase::Admit,
                                     "Linear bias must be [out]", index);
                shape.back() = weight[0];
                return shape;
            }
            std::vector<int64_t> operator()(const RmsNorm& op) const {
                const auto& shape = arg(op.x);
                if (op.axis < 0 || static_cast<size_t>(op.axis) >= shape.size() ||
                    !std::isfinite(op.epsilon) || op.epsilon <= 0)
                    throw GraphError(Code::InvalidOperation, Phase::Admit,
                                     "RmsNorm requires canonical axis and finite positive epsilon", index);
                if (op.weight && arg(*op.weight) != std::vector<int64_t>{shape[op.axis]})
                    throw GraphError(Code::InvalidShape, Phase::Admit,
                                     "RmsNorm weight must match axis width", index);
                return shape;
            }
            std::vector<int64_t> operator()(const LayerNorm& op) const {
                const auto& shape = arg(op.x);
                if (!std::isfinite(op.epsilon) || op.epsilon <= 0 ||
                    op.weight.has_value() != op.bias.has_value())
                    throw GraphError(Code::InvalidOperation, Phase::Admit,
                                     "LayerNorm requires positive finite epsilon and an affine pair", index);
                if (op.weight) {
                    const std::vector<int64_t> affine_shape{shape.back()};
                    if (arg(*op.weight) != affine_shape || arg(*op.bias) != affine_shape)
                        throw GraphError(Code::InvalidShape, Phase::Admit,
                                         "LayerNorm weight and bias must match final width", index);
                }
                return shape;
            }
            std::vector<int64_t> operator()(const Cast& op) const { return arg(op.x); }
            std::vector<int64_t> operator()(const Activate& op) const { return arg(op.x); }
            std::vector<int64_t> operator()(const Permute& op) const {
                const auto& source = arg(op.x);
                if (op.axes.size() != source.size()) invalid("Permutation rank mismatch");
                std::vector<bool> seen(source.size(), false);
                std::vector<int64_t> shape;
                for (int64_t axis : op.axes) {
                    if (axis < 0 || static_cast<size_t>(axis) >= source.size() || seen[axis])
                        invalid("Permutation axes must be a bijection");
                    seen[axis] = true; shape.push_back(source[axis]);
                }
                return shape;
            }
            [[noreturn]] void invalid(const char* detail) const {
                throw GraphError(Code::InvalidOperation, Phase::Admit, detail, index);
            }
            void axis_check(int64_t axis, size_t rank) const {
                if (axis < 0 || static_cast<size_t>(axis) >= rank) invalid("Axis must be canonical");
            }
            std::vector<int64_t> operator()(const Slice& op) const {
                auto shape = arg(op.x); axis_check(op.axis, shape.size());
                if (op.start < 0 || op.start >= op.stop || op.stop > shape[op.axis])
                    invalid("Slice requires a positive in-bounds interval");
                shape[op.axis] = op.stop - op.start;
                return shape;
            }
            std::vector<int64_t> operator()(const Concat& op) const {
                if (op.tensors.empty() || op.tensors.size() > 16) invalid("Concat requires 1..16 tensors");
                auto shape = arg(op.tensors.front()); axis_check(op.axis, shape.size());
                shape[op.axis] = 0;
                for (ValueId id : op.tensors) {
                    const auto& s = arg(id);
                    if (s.size() != shape.size()) invalid("Concat rank mismatch");
                    for (size_t k = 0; k < shape.size(); ++k)
                        if (static_cast<int64_t>(k) != op.axis && shape[k] != s[k])
                            invalid("Concat non-axis dimension mismatch");
                    if (s[op.axis] > INT32_MAX - shape[op.axis]) invalid("Concat extent overflow");
                    shape[op.axis] += s[op.axis];
                }
                return shape;
            }
            std::vector<int64_t> operator()(const Gather& op) const {
                const auto& table = arg(op.table);
                if (table.size() != 2) invalid("Gather requires rank-2 table");
                auto shape = select(op.indices);
                shape.push_back(table[1]);
                return shape;
            }
            std::vector<int64_t> operator()(const Rope& op) const {
                const auto& shape = arg(op.x);
                if (shape.back() % 2 || arg(op.cosine) != arg(op.sine) || bin(op.x, op.cosine) != shape)
                    invalid("RoPE requires even width and matching frequencies broadcasting to input");
                return shape;
            }
            std::vector<int64_t> operator()(const Attention& op) const {
                const auto& q = arg(op.q); const auto& k = arg(op.k); const auto& v = arg(op.v);
                if (q.size() != 4 || k.size() != 4 || k != v ||
                    q[0] != k[0] || q[2] != k[2] || q[3] != k[3])
                    invalid("Attention requires matching BSHD operands");
                if (!std::isfinite(op.scale) || op.scale <= 0) invalid("Attention requires positive finite scale");
                return q;
            }
        };
        const auto expected = std::visit(Infer{binary, operand, selector, i}, node.op);
        elements(expected, node.result, i);
        define(node.result, i);  // Only after all operand checks, including self edges.
        floating(node.result, i);
        // Shape/dependency checks above make every referenced ID safe here.
        const DType output = d.values[node.result].dtype;
        const auto type = [&](ValueId id) { return d.values[id].dtype; };
        const bool allowed = std::visit([&](const auto& op) -> bool {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, std::monostate>) return false;
            else if constexpr (std::is_same_v<T, Add> || std::is_same_v<T, Mul>)
                return type(op.a) == type(op.b) && output == type(op.a);
            else if constexpr (std::is_same_v<T, Concat>) {
                for (ValueId id : op.tensors) if (type(id) != output) return false;
                return true;
            } else if constexpr (std::is_same_v<T, Linear>)
                return (op.compute_dtype == DType::F32 && output == DType::F32) ||
                    (d.execution_dtype == DType::BF16 && op.compute_dtype == DType::BF16 && numeric(output));
            else if constexpr (std::is_same_v<T, RmsNorm>)
                return output == type(op.x) || (type(op.x) == DType::BF16 &&
                    output == DType::F32 && op.weight &&
                    static_cast<size_t>(op.axis) + 1 == expected.size());
            else if constexpr (std::is_same_v<T, LayerNorm>)
                return output == type(op.x) &&
                    (!op.weight || type(*op.weight) == type(*op.bias));
            else if constexpr (std::is_same_v<T, Rope>)
                return type(op.x) == DType::F32 && output == DType::F32 &&
                    type(op.cosine) == type(op.sine);
            else if constexpr (std::is_same_v<T, Attention> || std::is_same_v<T, Gather>)
                return output == d.execution_dtype;
            else if constexpr (std::is_same_v<T, Cast>)
                return numeric(op.destination) && op.destination != type(op.x) && output == op.destination;
            else if constexpr (std::is_same_v<T, Reshape> || std::is_same_v<T, Activate> ||
                               std::is_same_v<T, Permute> || std::is_same_v<T, Slice>)
                return output == type(op.x);
            else return false;
        }, node.op);
        if (!allowed) fail(Code::InvalidDType, "Unsupported primitive dtype combination",
                           i, node.result, node.op.index());
        if (d.values[node.result].shape != expected)
            fail(Code::InvalidShape, "Declared result shape differs from operation", i, node.result);
    }
    for (size_t id = 0; id < defined.size(); ++id)
        if (!defined[id]) fail(Code::InvalidValue, "Value has no origin", {}, id);
    for (size_t id = 0; id < d.values.size(); ++id)
        if (!numeric(d.values[id].dtype) && !selectors[id])
            fail(Code::InvalidDType, "Integer inputs are restricted to gather selectors", {}, id);
    if (d.outputs.empty()) fail(Code::InvalidOutput, "At least one output is required");
    std::vector<bool> selected(d.values.size(), false);
    for (ValueId id : d.outputs) {
        if (id >= defined.size() || !defined[id])
            fail(Code::InvalidOutput, "Invalid output reference", {}, id);
        if (selected[id]) fail(Code::InvalidValue, "Duplicate output reference", {}, id);
        selected[id] = true;
    }
}
}  // namespace

Graph admit(const Description& description) {
    try {
        validate(description);
        return Graph(description);
    } catch (const GraphError&) { throw; }
      catch (const std::bad_alloc&) { throw; }
      catch (const std::exception& e) {
        throw GraphError(Code::InvalidOperation, Phase::Admit, e.what());
    }
}

}  // namespace vrhino::neural_graph
