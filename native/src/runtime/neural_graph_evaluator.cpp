#include "neural_graph.h"

#include <cstring>

namespace vrhino::neural_graph {
namespace {

bool matches(const Tensor& t, const TensorSpec& spec) {
    if (!t.defined() || t.is_quantized() || t.dtype() != spec.dtype ||
        t.logical_dtype() != spec.dtype || t.shape() != spec.shape) return false;
    const size_t bytes = static_cast<size_t>(shape_numel(spec.shape)) * dtype_size(spec.dtype);
    return t.strides() == contiguous_strides(spec.shape) && t.bytes() == bytes &&
        t.byte_offset() <= t.storage_bytes() && bytes <= t.storage_bytes() - t.byte_offset() &&
        t.data() != nullptr;
}

bool valid_device(const Tensor& t, int device_count) {
    const auto device = t.device();
    const auto domain = t.memory_domain();
    if (device.type == DeviceType::CPU)
        return device.index == 0 && (domain == MemoryDomain::HostPageable ||
            domain == MemoryDomain::HostStaging || domain == MemoryDomain::Unified);
    return device.type == DeviceType::Accelerator && device.index >= 0 &&
        device.index < device_count &&
        (domain == MemoryDomain::DeviceLocal || domain == MemoryDomain::Unified);
}

struct Dispatch {
    Backend& backend;
    const std::vector<Tensor>& values;
    DType output_dtype;
    Tensor operator()(std::monostate) const {
        throw GraphError(Code::UnsupportedPrimitive, Phase::Run, "Unsupported primitive");
    }
    Tensor operator()(const Add& op) const { return backend.add(values[op.a], values[op.b]); }
    Tensor operator()(const Mul& op) const { return backend.mul(values[op.a], values[op.b]); }
    Tensor operator()(const Reshape& op) const { return backend.reshape(values[op.x], op.shape); }
    Tensor operator()(const Linear& op) const {
        return backend.linear(values[op.x], values[op.weight],
                              op.bias ? &values[*op.bias] : nullptr, op.compute_dtype, output_dtype);
    }
    Tensor operator()(const RmsNorm& op) const {
        return backend.rms_norm(values[op.x], op.weight ? &values[*op.weight] : nullptr,
                                op.epsilon, op.axis, output_dtype);
    }
    Tensor operator()(const LayerNorm& op) const {
        return backend.layer_norm(values[op.x], op.weight ? &values[*op.weight] : nullptr,
                                  op.bias ? &values[*op.bias] : nullptr, op.epsilon);
    }
    Tensor operator()(const Cast& op) const { return backend.cast(values[op.x], op.destination); }
    Tensor operator()(const Activate& op) const { return backend.activation(values[op.x], op.kind); }
    Tensor operator()(const Permute& op) const { return backend.permute(values[op.x], op.axes); }
    Tensor operator()(const Slice& op) const {
        return backend.slice(values[op.x], op.axis, op.start, op.stop);
    }
    Tensor operator()(const Concat& op) const {
        std::vector<Tensor> tensors;
        for (ValueId id : op.tensors) tensors.push_back(values[id]);
        return backend.concat(tensors, op.axis);
    }
    Tensor operator()(const Gather& op) const {
        return backend.indexed_gather(values[op.table], values[op.indices]);
    }
    Tensor operator()(const Rope& op) const {
        return backend.rope_nd(values[op.x], values[op.cosine], values[op.sine]);
    }
    Tensor operator()(const Attention& op) const {
        return backend.attention(values[op.q], values[op.k], values[op.v],
                                  nullptr, false, op.scale, nullptr, nullptr);
    }
};
}  // namespace

std::vector<Tensor> evaluate(Backend& backend, const Graph& graph,
                             const std::vector<Tensor>& inputs,
                             const std::vector<Tensor>& parameters) {
    const auto& d = graph.description();
    std::vector<Tensor> values;
    std::vector<Tensor> sources;  // Uploaded host owners survive failure drain.
    Tensor pending;  // Even an invalid Backend result stays alive through failure drain.
    Phase phase = Phase::Bind;
    Location node, value, primitive;
    bool submitted = false, completion_attempted = false;
    auto fail = [&](Code code, const char* detail) {
        throw GraphError(code, phase, detail, node, value, primitive);
    };
    auto drain_failure = [&](GraphError& error) {
        if (!submitted || completion_attempted) return;
        completion_attempted = true;
        try { backend.synchronize(); }
        catch (const std::exception& e) { error.secondary_completion = std::string(e.what()).substr(0, max_detail); }
        catch (...) { error.secondary_completion = "Non-standard completion failure"; }
    };
    try {
        if (inputs.size() != d.inputs.size()) fail(Code::InvalidInputBinding, "Input slot count mismatch");
        if (parameters.size() != d.parameters.size()) fail(Code::InvalidParameterBinding, "Parameter slot count mismatch");
        if (backend.execution_dtype() != d.execution_dtype)
            fail(Code::InvalidContext, "Backend execution dtype must match graph");
        if (d.execution_dtype == DType::BF16 && !backend.supports(DType::BF16))
            fail(Code::InvalidContext, "Backend does not support BF16");
        const int devices = backend.device_count();
        if (devices < 0) fail(Code::InvalidContext, "Invalid Backend device count");
        values.resize(d.values.size());
        auto bind = [&](const std::vector<ValueId>& slots, const std::vector<Tensor>& handles, Code code) {
            for (size_t i = 0; i < slots.size(); ++i) {
                value = slots[i];
                if (!matches(handles[i], d.values[*value])) fail(code, "Invalid bound tensor metadata");
                if (!valid_device(handles[i], devices)) fail(Code::InvalidContext, "Invalid tensor device metadata");
                values[*value] = handles[i];
            }
        };
        bind(d.inputs, inputs, Code::InvalidInputBinding);
        bind(d.parameters, parameters, Code::InvalidParameterBinding);
        for (const auto& c : d.constants) {
            value = c.value;
            Tensor t = Tensor::host({1}, d.values[c.value].dtype);
            if (t.dtype() == DType::F32) t.data_as<float>()[0] = c.scalar;
            else {
                uint32_t bits; std::memcpy(&bits, &c.scalar, sizeof(bits));
                t.data_as<uint16_t>()[0] = static_cast<uint16_t>(bits >> 16);
            }
            values[c.value] = std::move(t);
        }
        // Validate all bindings before any upload. Retain source owners until drain.
        if (d.execution_dtype == DType::BF16 && devices > 0) {
            for (size_t id = 0; id < values.size(); ++id) {
                if (!values[id].defined() || values[id].device().type != DeviceType::CPU ||
                    (d.values[id].dtype != DType::F32 && d.values[id].dtype != DType::BF16)) continue;
                value = id; submitted = true;
                pending = backend.copy_to_device(values[id], d.values[id].dtype);
                if (!matches(pending, d.values[id]) || !valid_device(pending, devices) ||
                    pending.device().type != DeviceType::Accelerator)
                    fail(Code::InvalidContext, "Invalid dtype-preserving binding placement");
                sources.push_back(values[id]);
                values[id] = std::move(pending);
            }
        }
        phase = Phase::Run;
        for (size_t i = 0; i < d.nodes.size(); ++i) {
            const auto& n = d.nodes[i];
            node = i; value = n.result; primitive = n.op.index();
            submitted = true;  // A throwing call may have submitted work.
            pending = std::visit(Dispatch{backend, values, d.values[n.result].dtype}, n.op);
            if (!matches(pending, d.values[n.result]) || !valid_device(pending, devices))
                fail(Code::InvalidNodeOutput, "Backend returned invalid tensor metadata");
            values[n.result] = std::move(pending);
        }
        phase = Phase::Drain;
        node.reset(); value.reset(); primitive.reset();
        completion_attempted = true;
        backend.synchronize();
        phase = Phase::Capture;
        std::vector<Tensor> outputs;
        outputs.reserve(d.outputs.size());
        for (ValueId id : d.outputs) {
            value = id;
            if (!matches(values[id], d.values[id]) || !valid_device(values[id], devices))
                fail(Code::InvalidOutput, "Invalid output at capture");
            outputs.push_back(values[id]);
        }
        return outputs;
    } catch (GraphError& e) {
        drain_failure(e);
        throw;
    } catch (const std::exception& e) {
        GraphError error(phase == Phase::Drain ? Code::CompletionFailure : Code::BackendFailure,
                         phase, e.what(), node, value, primitive);
        drain_failure(error);
        throw error;
    } catch (...) {
        GraphError error(phase == Phase::Drain ? Code::CompletionFailure : Code::BackendFailure,
                         phase, "Non-standard Backend failure", node, value, primitive);
        drain_failure(error);
        throw error;
    }
}

}  // namespace vrhino::neural_graph
