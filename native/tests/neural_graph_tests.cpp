#include "neural_graph.h"
#include "neural_graph_test_backend.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace vrhino;
using namespace vrhino::neural_graph;
using vrhino::neural_graph::test::TinyBackend;

namespace {
size_t cases = 0;
void pass(const std::string& name) {
    ++cases;
    std::cout << "PASS\t" << name << '\n';
}
Tensor tensor(std::vector<int64_t> shape, std::initializer_list<float> values) {
    Tensor t = Tensor::host(std::move(shape), DType::F32);
    require(t.numel() == static_cast<int64_t>(values.size()), "Bad test data");
    std::copy(values.begin(), values.end(), t.data_as<float>());
    return t;
}
Description fixture() {
    return {schema_v1,
        {{{2, 2}}, {{2, 2}}, {{1}}, {{2, 2}}, {{2, 2}}, {{4}}, {{2, 2}}},
        {0}, {1}, {{2, 1.0f}},
        {{3, Add{0, 2}}, {4, Mul{3, 1}}, {5, Reshape{4, {4}}}, {6, Add{4, 0}}},
        {5, 6}};
}
std::vector<Tensor> inputs() { return {tensor({2, 2}, {1, 2, 3, 4})}; }
std::vector<Tensor> parameters() { return {tensor({2, 2}, {2, 3, 4, 5})}; }
void exact(const Tensor& actual, const std::vector<int64_t>& shape,
           const std::vector<float>& expected) {
    require(actual.shape() == shape && actual.dtype() == DType::F32 &&
            actual.numel() == static_cast<int64_t>(expected.size()), "Output metadata mismatch");
    for (size_t i = 0; i < expected.size(); ++i)
        require(std::isfinite(actual.data_as<float>()[i]) &&
                actual.data_as<float>()[i] == expected[i], "Output value mismatch");
}
template<class F>
GraphError rejects(const std::string& name, Code code, Phase phase, F&& run) {
    try { run(); }
    catch (const GraphError& e) {
        require(e.code == code && e.phase == phase, "Wrong rejection code/phase: " + name);
        require(std::string(e.what()).size() <= max_detail, "Unbounded detail");
        pass(name);
        return e;
    }
    throw Error("Accepted malformed case: " + name);
}
void reject_graph(const std::string& name, Description d, Code code) {
    TinyBackend backend;
    rejects(name, code, Phase::Admit, [&] { evaluate(backend, admit(d), inputs(), parameters()); });
    require(backend.calls.empty() && backend.syncs == 0, "Dispatch during admission failure");
}
template<class F>
void malformed(const std::string& name, Code code, F&& mutate) {
    auto d = fixture(); mutate(d); reject_graph(name, d, code);
}

void validation() {
    static_assert(!std::is_default_constructible_v<Graph>);
    static_assert(!std::is_copy_assignable_v<Graph>);
    static_assert(!std::is_constructible_v<Primitive, std::string>);
    Description identity{schema_v1, {{{1}}}, {0}, {}, {}, {}, {0}};
    admit(identity); pass("E0_zero_node_identity");
    admit(fixture()); pass("E0_full_fixture");
    auto d = fixture(); d.nodes.resize(1); d.values.resize(4); d.outputs = {3};
    admit(d); pass("E0_single_add");
    d = fixture(); d.nodes.resize(2); d.values.resize(5); d.outputs = {4};
    admit(d); pass("E0_add_mul");
    d = identity; d.values.push_back({{1, 1}}); d.values.push_back({{1}});
    d.nodes = {{1, Reshape{0, {1, 1}}}, {2, Reshape{1, {1}}}}; d.outputs = {2};
    admit(d); pass("E0_reshape_chain");
    reject_graph("E0_completely_empty", {schema_v1, {}, {}, {}, {}, {}, {}}, Code::InvalidOutput);
    malformed("E0_unknown_schema", Code::UnsupportedSchema, [](auto& g) { g.schema = "vrhino.neural-graph.v2"; });
    malformed("E0_missing_schema", Code::UnsupportedSchema, [](auto& g) { g.schema.clear(); });
    malformed("E0_legacy_identity", Code::UnsupportedSchema, [](auto& g) { g.schema = "schema_version=1"; });
    malformed("E0_unknown_primitive_first", Code::UnsupportedPrimitive, [](auto& g) { g.nodes[0].op = std::monostate{}; });
    malformed("E0_unknown_primitive_late", Code::UnsupportedPrimitive, [](auto& g) { g.nodes.back().op = std::monostate{}; });
    malformed("E0_missing_input_origin", Code::InvalidDependency, [](auto& g) { g.inputs.clear(); });
    malformed("E0_missing_parameter_origin", Code::InvalidDependency, [](auto& g) { g.parameters.clear(); });
    malformed("E0_invalid_input_slot", Code::InvalidValue, [](auto& g) { g.inputs[0] = UINT32_MAX; });
    malformed("E0_invalid_parameter_slot", Code::InvalidValue, [](auto& g) { g.parameters[0] = UINT32_MAX; });
    malformed("E0_invalid_parameter_operand", Code::InvalidDependency, [](auto& g) { std::get<Mul>(g.nodes[1].op).b = UINT32_MAX; });
    malformed("E0_duplicate_input", Code::InvalidValue, [](auto& g) { g.inputs.push_back(0); });
    malformed("E0_duplicate_parameter", Code::InvalidValue, [](auto& g) { g.parameters.push_back(1); });
    malformed("E0_cross_category_origin", Code::InvalidValue, [](auto& g) { g.parameters[0] = 0; });
    malformed("E0_duplicate_constant", Code::InvalidValue, [](auto& g) { g.constants.push_back(g.constants[0]); });
    malformed("E0_constant_input_collision", Code::InvalidValue, [](auto& g) { g.constants[0].value = 0; });
    malformed("E0_duplicate_result", Code::InvalidValue, [](auto& g) { g.nodes.back().result = 4; });
    malformed("E0_result_input_collision", Code::InvalidValue, [](auto& g) { g.nodes[0].result = 0; });
    malformed("E0_invalid_result", Code::InvalidValue, [](auto& g) { g.nodes[0].result = UINT32_MAX; });
    malformed("E0_hole", Code::InvalidValue, [](auto& g) { g.values.push_back({{1}}); });
    malformed("E0_self", Code::InvalidDependency, [](auto& g) { std::get<Add>(g.nodes[0].op).a = 3; });
    malformed("E0_future_cycle", Code::InvalidDependency, [](auto& g) { std::get<Add>(g.nodes[0].op).a = 4; });
    malformed("E0_unsorted_dag", Code::InvalidDependency, [](auto& g) { std::swap(g.nodes[0], g.nodes[1]); });
    malformed("E0_undefined_operand", Code::InvalidDependency, [](auto& g) { std::get<Add>(g.nodes[0].op).a = UINT32_MAX; });
    for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {0}, {-1}, {2, -1},
            std::vector<int64_t>(9, 1), {INT64_MAX, INT64_MAX}, {INT32_MAX, 2}}) {
        d = fixture(); d.values[0].shape = shape;
        reject_graph("E0_invalid_shape_" + std::to_string(cases), d, Code::InvalidShape);
    }
    for (DType dtype : {DType::F16, DType::BF16, DType::I8, DType::I32, DType::I64, static_cast<DType>(255)}) {
        d = fixture(); d.values[0].dtype = dtype;
        reject_graph("E0_dtype_" + std::to_string(static_cast<int>(dtype)), d, Code::InvalidDType);
    }
    malformed("E0_broadcast_incompatible", Code::InvalidShape, [](auto& g) { g.values[1].shape = {3}; });
    malformed("E0_result_shape", Code::InvalidShape, [](auto& g) { g.values[3].shape = {4}; });
    malformed("E0_reshape_count", Code::InvalidShape, [](auto& g) { std::get<Reshape>(g.nodes[2].op).shape = {3}; });
    malformed("E0_reshape_inference", Code::InvalidShape, [](auto& g) { std::get<Reshape>(g.nodes[2].op).shape = {-1}; });
    malformed("E0_reshape_rank", Code::InvalidShape, [](auto& g) { std::get<Reshape>(g.nodes[2].op).shape.assign(9, 1); });
    malformed("E0_missing_output", Code::InvalidOutput, [](auto& g) { g.outputs.clear(); });
    malformed("E0_invalid_output", Code::InvalidOutput, [](auto& g) { g.outputs = {UINT32_MAX}; });
    malformed("E0_duplicate_output", Code::InvalidValue, [](auto& g) { g.outputs = {5, 5}; });
    malformed("E0_constant_shape", Code::InvalidValue, [](auto& g) { g.values[2].shape = {1, 1}; });
    malformed("E0_constant_infinite", Code::InvalidValue, [](auto& g) { g.constants[0].scalar = std::numeric_limits<float>::infinity(); });
    malformed("E0_constant_nan", Code::InvalidValue, [](auto& g) { g.constants[0].scalar = std::numeric_limits<float>::quiet_NaN(); });
    malformed("E0_constant_invalid_id", Code::InvalidValue, [](auto& g) { g.constants[0].value = UINT32_MAX; });
    malformed("E0_values_limit", Code::LimitExceeded, [](auto& g) { g.values.resize(max_values + 1); });
    malformed("E0_nodes_limit", Code::LimitExceeded, [](auto& g) { g.nodes.resize(max_nodes + 1); });
    malformed("E0_inputs_limit", Code::LimitExceeded, [](auto& g) { g.inputs.resize(max_values + 1); });
    malformed("E0_parameters_limit", Code::LimitExceeded, [](auto& g) { g.parameters.resize(max_values + 1); });
    malformed("E0_constants_limit", Code::LimitExceeded, [](auto& g) { g.constants.resize(max_constants + 1); });
    malformed("E0_outputs_limit", Code::LimitExceeded, [](auto& g) { g.outputs.resize(max_outputs + 1); });
    d = {schema_v1, {{{46341, 1}}, {{1, 46341}}, {{1}}}, {0, 1}, {}, {}, {{2, Add{0, 1}}}, {2}};
    reject_graph("E0_broadcast_product_limit", d, Code::InvalidShape);
    d = identity; d.values[0].shape = {INT32_MAX}; admit(d); pass("E0_element_limit_exact_no_allocation");
    d.values[0].shape.assign(8, 1); admit(d); pass("E0_rank_limit_exact");
    d = {schema_v1, {}, {}, {}, {}, {}, {}};
    for (ValueId id = 0; id < max_constants; ++id) {
        d.values.push_back({{1}}); d.constants.push_back({id, static_cast<float>(id)}); d.outputs.push_back(id);
    }
    admit(d); pass("E0_constants_outputs_limits_exact");
    d = identity;
    for (ValueId id = 1; id < max_values; ++id) {
        d.values.push_back({{1}}); d.nodes.push_back({id, Reshape{id - 1, {1}}});
    }
    d.outputs = {max_values - 1}; admit(d); pass("E0_values_limit_exact_serial_chain");
}

void extended_validation() {
    const Description linear{schema_v1, {{{2, 3}}, {{4, 3}}, {{4}}, {{2, 4}}},
        {0}, {1, 2}, {}, {{3, Linear{0, 1, 2}}}, {3}};
    const Description norm{schema_v1, {{{2, 3}}, {{3}}, {{2, 3}}},
        {0}, {1}, {}, {{2, RmsNorm{0, 1, 1, 1e-6f}}}, {2}};
    const Description activation{schema_v1, {{{2, 3}}, {{2, 3}}},
        {0}, {}, {}, {{1, Activate{0, Activation::GeluTanh}}}, {1}};
    for (const auto& base : {linear, norm, activation}) {
        admit(base); pass("E3_valid_primitive_" + std::to_string(base.nodes[0].op.index()));
        for (DType dtype : {DType::BF16, DType::F16, DType::I32, static_cast<DType>(255)}) {
            auto d = base; d.values[0].dtype = dtype;
            reject_graph("E3_wrong_dtype_" + std::to_string(cases), d, Code::InvalidDType);
        }
        for (auto shape : std::vector<std::vector<int64_t>>{{}, std::vector<int64_t>(9, 1)}) {
            auto d = base; d.values[0].shape = shape;
            reject_graph("E3_wrong_input_rank_" + std::to_string(cases), d, Code::InvalidShape);
        }
        auto d = base; d.values.back().shape = {1};
        reject_graph("E3_wrong_output_shape_" + std::to_string(cases), d, Code::InvalidShape);
        d = base; d.inputs.clear();
        reject_graph("E3_missing_required_operand_" + std::to_string(cases), d, Code::InvalidDependency);
        d = base; d.nodes[0].op = std::monostate{};
        reject_graph("E3_unknown_variant_" + std::to_string(cases), d, Code::UnsupportedPrimitive);
    }
    for (auto shape : std::vector<std::vector<int64_t>>{{12}, {1, 4, 3}, {4, 2}}) {
        auto d = linear; d.values[1].shape = shape;
        reject_graph("E3_linear_weight_rank_or_width_" + std::to_string(cases), d, Code::InvalidShape);
    }
    for (auto shape : std::vector<std::vector<int64_t>>{{1, 4}, {3}}) {
        auto d = linear; d.values[2].shape = shape;
        reject_graph("E3_linear_bias_shape_" + std::to_string(cases), d, Code::InvalidShape);
    }
    for (bool bias : {false, true}) {
        auto d = linear; auto& op = std::get<Linear>(d.nodes[0].op);
        if (bias) op.bias = UINT32_MAX; else op.weight = UINT32_MAX;
        reject_graph("E3_linear_missing_weight_bias_" + std::to_string(cases), d, Code::InvalidDependency);
    }
    for (float eps : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        auto d = norm; std::get<RmsNorm>(d.nodes[0].op).epsilon = eps;
        reject_graph("E3_norm_epsilon_" + std::to_string(cases), d, Code::InvalidOperation);
    }
    for (int64_t axis : {-1, 2, 100}) {
        auto d = norm; std::get<RmsNorm>(d.nodes[0].op).axis = axis;
        reject_graph("E3_norm_axis_" + std::to_string(cases), d, Code::InvalidOperation);
    }
    for (auto shape : std::vector<std::vector<int64_t>>{{1, 3}, {4}}) {
        auto d = norm; d.values[1].shape = shape;
        reject_graph("E3_norm_weight_shape_" + std::to_string(cases), d, Code::InvalidShape);
    }
    auto d = norm; std::get<RmsNorm>(d.nodes[0].op).weight = UINT32_MAX;
    reject_graph("E3_norm_missing_weight", d, Code::InvalidDependency);
    d = activation; std::get<Activate>(d.nodes[0].op).kind = Activation::Silu;
    admit(d); pass("E0_activation_silu_admitted");
    for (Activation kind : {static_cast<Activation>(255)}) {
        d = activation; std::get<Activate>(d.nodes[0].op).kind = kind;
        reject_graph("E3_activation_unsupported_or_unknown_" + std::to_string(cases), d, Code::UnsupportedPrimitive);
    }
    d = linear; std::get<Linear>(d.nodes[0].op).bias.reset();
    admit(d); pass("E3_linear_no_bias");
    d = norm; std::get<RmsNorm>(d.nodes[0].op).weight.reset();
    admit(d); pass("E3_norm_no_weight");
    d = norm; std::get<RmsNorm>(d.nodes[0].op).axis = 0; d.values[1].shape = {2};
    admit(d); pass("E3_norm_canonical_nonfinal_axis");
    // Fixed native operand fields: no variadic arity, string attrs or custom
    // discriminant can be supplied. Missing/out-of-range refs reject above.
    pass("E3_fixed_typed_arity_no_unchecked_attributes");
}

void bindings() {
    const auto g = admit(fixture());
    auto reject = [&](const std::string& name, std::vector<Tensor> x, std::vector<Tensor> p, Code code) {
        TinyBackend b;
        rejects(name, code, Phase::Bind, [&] { evaluate(b, g, x, p); });
        require(b.calls.empty() && b.syncs == 0, "Dispatch during bad binding");
    };
    reject("E0_missing_input", {}, parameters(), Code::InvalidInputBinding);
    reject("E0_extra_input", {inputs()[0], inputs()[0]}, parameters(), Code::InvalidInputBinding);
    reject("E0_missing_parameter", inputs(), {}, Code::InvalidParameterBinding);
    reject("E0_extra_parameter", inputs(), {parameters()[0], parameters()[0]}, Code::InvalidParameterBinding);
    for (bool parameter : {false, true}) {
        for (int fault = 0; fault < 7; ++fault) {
            Tensor t;
            switch (fault) {
                case 0: break;
                case 1: t = Tensor::host({4}, DType::F32); break;
                case 2: t = Tensor::host({2, 3}, DType::F32); break;
                case 3: t = Tensor::host({2, 2}, DType::F16); break;
                case 4: t = Tensor::borrowed(nullptr, 16, {2, 2}, DType::F32); break;
                case 5: {
                    static float small[3] = {};
                    t = Tensor::borrowed(small, sizeof(small), {2, 2}, DType::F32); break;
                }
                case 6: {
                    t = Tensor::host({2, 2}, DType::U8);
                    auto q = std::make_shared<QuantizationInfo>();
                    q->type = QuantType::INT8Symmetric; q->logical_dtype = DType::F32;
                    q->scales = tensor({1}, {1}); t.set_quantization(q); break;
                }
            }
            reject("E0_bound_" + std::to_string(parameter) + "_" + std::to_string(fault),
                   parameter ? inputs() : std::vector<Tensor>{t},
                   parameter ? std::vector<Tensor>{t} : parameters(),
                   parameter ? Code::InvalidParameterBinding : Code::InvalidInputBinding);
        }
    }
    TinyBackend b; b.dtype = DType::BF16;
    rejects("E0_context_precision", Code::InvalidContext, Phase::Bind, [&] { evaluate(b, g, inputs(), parameters()); });
    require(b.calls.empty() && b.syncs == 0 && b.dtype == DType::BF16, "Context modified");
    auto s = std::make_shared<Storage>(); float payload[4] = {};
    s->data = payload; s->bytes = sizeof(payload); s->device = DeviceId::accelerator(5);
    s->domain = MemoryDomain::DeviceLocal;
    reject("E0_foreign_device_metadata", {Tensor(s, 0, {2, 2}, DType::F32)}, parameters(), Code::InvalidContext);
    s->device = DeviceId::host();
    reject("E0_invalid_memory_domain", {Tensor(s, 0, {2, 2}, DType::F32)}, parameters(), Code::InvalidContext);
}

void execution() {
    auto d = fixture(); const auto graph = admit(d);
    d.nodes.clear(); d.constants[0].scalar = 99; d.schema.clear();
    TinyBackend backend;
    for (int repeat = 0; repeat < 10; ++repeat) {
        const auto x = inputs(), p = parameters();
        auto out = evaluate(backend, graph, x, p);
        require(out.size() == 2, "Output count");
        exact(out[0], {4}, {4, 9, 16, 25}); exact(out[1], {2, 2}, {5, 11, 19, 29});
        exact(x[0], {2, 2}, {1, 2, 3, 4}); exact(p[0], {2, 2}, {2, 3, 4, 5});
        require(backend.calls == std::vector<std::string>{"add", "mul", "reshape", "add"}, "Dispatch order");
        require(backend.syncs == static_cast<size_t>(repeat + 1), "Completion count");
        require(backend.produced[0].expired() && !backend.produced[1].expired(), "Post-drain alias lifetime");
        out.clear();
        for (const auto& weak : backend.produced) require(weak.expired(), "Temporary leak");
        backend.calls.clear(); backend.produced.clear();
    }
    pass("E1_exact_repeat_10_immutable_fanout_skip_order_lifetime");
    d = fixture(); d.outputs = {6, 5, 4, 0, 1, 2};
    TinyBackend b;
    auto out = evaluate(b, admit(d), inputs(), parameters());
    exact(out[0], {2, 2}, {5, 11, 19, 29}); exact(out[1], {4}, {4, 9, 16, 25});
    exact(out[2], {2, 2}, {4, 9, 16, 25}); exact(out[3], {2, 2}, {1, 2, 3, 4});
    exact(out[4], {2, 2}, {2, 3, 4, 5}); exact(out[5], {1}, {1});
    require(out[1].data() == out[2].data(), "Reshape must preserve mock alias");
    pass("E1_ordered_outputs_all_categories_alias");
    d = {schema_v1, {{{2, 1}}, {{3}}, {{2, 3}}, {{2, 3}}}, {1, 0}, {}, {},
        {{2, Add{0, 1}}, {3, Mul{2, 0}}}, {3, 2}};
    TinyBackend broad;
    out = evaluate(broad, admit(d), {tensor({3}, {1, 2, 3}), tensor({2, 1}, {10, 20})}, {});
    exact(out[0], {2, 3}, {110, 120, 130, 420, 440, 460});
    exact(out[1], {2, 3}, {11, 12, 13, 21, 22, 23});
    pass("E1_right_aligned_broadcast_positional_inputs");
    d = {schema_v1, {{{2, 2}}, {{4}}, {{1, 4, 1}}}, {0}, {}, {},
        {{1, Reshape{0, {4}}}, {2, Reshape{1, {1, 4, 1}}}}, {2}};
    TinyBackend views; Tensor owner = inputs()[0];
    out = evaluate(views, admit(d), {owner}, {}); owner = {};
    exact(out[0], {1, 4, 1}, {1, 2, 3, 4}); pass("E1_reshape_chain_owned_alias_survives_input");
    d = {schema_v1, {{{1}}}, {}, {}, {{0, -0.0f}}, {}, {0}};
    TinyBackend constant; out = evaluate(constant, admit(d), {}, {});
    require(std::signbit(out[0].data_as<float>()[0]) && constant.syncs == 1, "Scalar representation");
    pass("E1_constant_only_zero_nodes");
    d = {schema_v1, {{{1}}}, {0}, {}, {}, {}, {0}};
    float external = 7;
    TinyBackend identity;
    out = evaluate(identity, admit(d), {Tensor::borrowed(&external, sizeof(float), {1}, DType::F32)}, {});
    exact(out[0], {1}, {7}); require(out[0].data() == &external, "Borrowed alias");
    pass("E1_identity_borrowed_owner_kept_alive");
    d = fixture(); d.outputs = {0};
    TinyBackend unused; evaluate(unused, admit(d), inputs(), parameters());
    require(unused.calls.size() == 4 && unused.syncs == 1, "Unused nodes skipped");
    pass("E1_every_unused_node_runs_once");
    d = fixture();
    const auto original = d;
    auto remap = [](ValueId id) { return ValueId{6} - id; };
    for (ValueId id = 0; id < d.values.size(); ++id) d.values[remap(id)] = original.values[id];
    for (auto& id : d.inputs) id = remap(id);
    for (auto& id : d.parameters) id = remap(id);
    for (auto& c : d.constants) c.value = remap(c.value);
    for (auto& n : d.nodes) {
        n.result = remap(n.result);
        std::visit([&](auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, Add> || std::is_same_v<T, Mul>) {
                op.a = remap(op.a); op.b = remap(op.b);
            } else if constexpr (std::is_same_v<T, Reshape>) op.x = remap(op.x);
        }, n.op);
    }
    for (auto& id : d.outputs) id = remap(id);
    TinyBackend reindexed; out = evaluate(reindexed, admit(d), inputs(), parameters());
    exact(out[0], {4}, {4, 9, 16, 25}); exact(out[1], {2, 2}, {5, 11, 19, 29});
    pass("E1_reindexed_values_preserve_supplied_node_order");
    d = {schema_v1, {{{1}}, {{1}}, {{1}}, {{1}}, {{1}}}, {0}, {2, 1}, {},
        {{3, Add{0, 1}}, {4, Mul{3, 2}}}, {4}};
    TinyBackend slots;
    out = evaluate(slots, admit(d), {tensor({1}, {2})}, {tensor({1}, {5}), tensor({1}, {3})});
    exact(out[0], {1}, {25}); pass("E1_parameter_slot_order");
}

void failures() {
    const auto graph = admit(fixture());
    for (auto fault : {TinyBackend::Fault::Undefined, TinyBackend::Fault::Shape, TinyBackend::Fault::Dtype,
                       TinyBackend::Fault::Extent, TinyBackend::Fault::NullData, TinyBackend::Fault::Device}) {
        TinyBackend b; b.fault = fault;
        auto e = rejects("E1_bad_result_" + std::to_string(static_cast<int>(fault)), Code::InvalidNodeOutput,
                         Phase::Run, [&] { evaluate(b, graph, inputs(), parameters()); });
        require(e.node == 0 && e.value == 3 && e.primitive == 1 && b.syncs == 1 && b.calls.size() == 1 &&
                e.secondary_completion.empty(), "Invalid-result drain/location failed");
        for (const auto& weak : b.produced) require(weak.expired(), "Failed result leaked");
    }
    for (bool secondary : {false, true}) {
        TinyBackend b; b.fail_call = 2; b.fail_sync = secondary;
        std::vector<Tensor> result{tensor({1}, {91})};
        auto e = rejects("E1_primitive_failure_" + std::to_string(secondary), Code::BackendFailure,
            Phase::Run, [&] { result = evaluate(b, graph, inputs(), parameters()); });
        require(e.node == 1 && e.value == 4 && b.calls.size() == 2 && b.syncs == 1 &&
                e.secondary_completion.empty() != secondary, "Failure drain/context lost");
        exact(result[0], {1}, {91});
        require(std::string(e.what()) == "injected primitive failure", "Primary failure overwritten");
    }
    TinyBackend sync; sync.fail_sync = true;
    rejects("E1_completion_failure", Code::CompletionFailure, Phase::Drain,
            [&] { evaluate(sync, graph, inputs(), parameters()); });
    require(sync.calls.size() == 4 && sync.syncs == 1, "Sync retry or partial execution");
    TinyBackend late; late.fault = TinyBackend::Fault::Shape; late.fault_call = 2;
    auto invalid = rejects("E1_late_invalid_result_retained_with_prior_values", Code::InvalidNodeOutput,
        Phase::Run, [&] { evaluate(late, graph, inputs(), parameters()); });
    require(late.syncs == 1 && late.calls.size() == 2 && invalid.secondary_completion.empty(),
            "Prior or pending value lost during drain");
    TinyBackend capture; capture.corrupt_capture = true;
    rejects("E1_capture_revalidation", Code::InvalidOutput, Phase::Capture,
            [&] { evaluate(capture, graph, inputs(), parameters()); });
    require(capture.syncs == 1, "Capture repeated synchronization");
    TinyBackend unusual; unusual.fail_call = 2; unusual.throw_nonstandard = true;
    rejects("E1_nonstandard_backend_exception", Code::BackendFailure, Phase::Run,
            [&] { evaluate(unusual, graph, inputs(), parameters()); });
    require(unusual.syncs == 1, "Nonstandard failure did not drain");
    GraphError bounded(Code::BackendFailure, Phase::Run, std::string(1000, 'x'));
    require(std::string(bounded.what()).size() == max_detail, "Detail bound failed");
    pass("E1_bounded_error_detail");
}
}  // namespace

int main() {
    try {
        validation(); extended_validation(); bindings(); execution(); failures();
        std::cout << "TOTAL\t" << cases << "\tPASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL\t" << e.what() << '\n';
        return 1;
    }
}
