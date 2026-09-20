#include "vrhino/precision.h"

#include <array>
#include <cmath>

#include "vrhino/backend.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/tensor_util.h"

namespace vrhino {
namespace {

constexpr std::array<const char*, 7> kScalarRoles = {
    "GUIDANCE_COEFFICIENT", "SOLVER_COEFFICIENT", "TIMESTEP_SCALE", "SIGMA",
    "MODULATION_COEFFICIENT", "NORMALIZATION_PARAMETER", "DATA_OPERAND"};

size_t scalar_index(PrecisionScalarRole role) {
    const size_t index = static_cast<size_t>(role);
    require(index < kScalarRoles.size(), "Unknown precision scalar role");
    return index;
}

size_t operation_index(PrecisionOperation operation) {
    require(static_cast<size_t>(operation) < 13,
            "Composed precision operation has no legacy table slot");
    return static_cast<size_t>(operation);
}

PrecisionOperation base_operation(PrecisionOperation operation) {
    switch (operation) {
        case PrecisionOperation::Modulation:
        case PrecisionOperation::Embedding:
            return PrecisionOperation::Conditioning;
        case PrecisionOperation::Attention:
            return PrecisionOperation::AttentionPV;
        case PrecisionOperation::Rope:
            return PrecisionOperation::Activation;
        case PrecisionOperation::Pooling:
            return PrecisionOperation::AttentionPV;
        case PrecisionOperation::SamplingArithmetic:
            return PrecisionOperation::Scheduler;
        case PrecisionOperation::VaePrimitive:
            return PrecisionOperation::Vae;
        default:
            return operation;
    }
}

size_t semantic_index(PrecisionSemantic semantic) {
    return static_cast<size_t>(semantic);
}

DType policy_dtype(const Json& value) {
    require(value.is_string(), "Precision policy dtype must be a string");
    if (value.string() == "fp32") return DType::F32;
    if (value.string() == "bf16") return DType::BF16;
    throw Error("Precision policy dtype must be fp32 or bf16");
}

const std::array<std::pair<PrecisionOperation, const char*>, 13> kOperations = {{
    {PrecisionOperation::Linear, "LINEAR"},
    {PrecisionOperation::Convolution, "CONVOLUTION"},
    {PrecisionOperation::Normalization, "NORMALIZATION"},
    {PrecisionOperation::AttentionProjection, "ATTENTION_PROJECTION"},
    {PrecisionOperation::AttentionQK, "ATTENTION_QK"},
    {PrecisionOperation::AttentionPV, "ATTENTION_PV"},
    {PrecisionOperation::Activation, "ACTIVATION"},
    {PrecisionOperation::ResidualAdd, "RESIDUAL_ADD"},
    {PrecisionOperation::Guidance, "GUIDANCE"},
    {PrecisionOperation::Scheduler, "SCHEDULER"},
    {PrecisionOperation::Conditioning, "CONDITIONING"},
    {PrecisionOperation::Denoiser, "DENOISER"},
    {PrecisionOperation::Vae, "VAE"},
}};

const std::array<std::pair<PrecisionSemantic, const char*>, 9> kFloatingSemantics = {{
    {PrecisionSemantic::Conditioning, "CONDITIONING"},
    {PrecisionSemantic::TemporaryCompute, "TEMPORARY_COMPUTE"},
    {PrecisionSemantic::NormalizationStatistic, "NORMALIZATION_STATISTIC"},
    {PrecisionSemantic::AttentionStatistic, "ATTENTION_STATISTIC"},
    {PrecisionSemantic::ResidualState, "RESIDUAL_STATE"},
    {PrecisionSemantic::SamplingState, "SAMPLING_STATE"},
    {PrecisionSemantic::DenoiserOutput, "DENOISER_OUTPUT"},
    {PrecisionSemantic::VaeInput, "VAE_INPUT"},
    {PrecisionSemantic::VideoOutput, "VIDEO_OUTPUT"},
}};

}  // namespace

PrecisionPolicy::PrecisionPolicy(PrecisionMode requested_mode)
    : requested_mode_(requested_mode) {
    const DType requested = requested_mode == PrecisionMode::FP32
        ? DType::F32 : DType::BF16;
    operation_compute_.fill(requested);
    operation_accumulator_.fill(DType::F32);
    operation_output_.fill(requested);
    semantic_dtype_.fill(requested);
    producer_output_.fill(requested);
    producer_output_override_.fill(false);
    semantic_dtype_[semantic_index(PrecisionSemantic::DiscreteInput)] = DType::I64;
    semantic_dtype_[semantic_index(PrecisionSemantic::NormalizationStatistic)] = DType::F32;
    semantic_dtype_[semantic_index(PrecisionSemantic::AttentionStatistic)] = DType::F32;
}

PrecisionPolicy PrecisionPolicy::unqualified_default(DType requested_dtype) {
    require(requested_dtype == DType::F32 || requested_dtype == DType::BF16,
            "PrecisionPolicy requested dtype must be float32 or bfloat16");
    return PrecisionPolicy(requested_dtype == DType::F32
        ? PrecisionMode::FP32 : PrecisionMode::BF16);
}

PrecisionPolicy PrecisionPolicy::fp32() {
    return PrecisionPolicy(PrecisionMode::FP32);
}

PrecisionPolicy PrecisionPolicy::from_json(const Json& document) {
    const auto& schema = document.at("schema").string();
    require(schema == "vrhino.precision.policy.v1" || schema == "vrhino.precision.policy.v2",
            "Precision policy schema mismatch");
    require(document.at("requested_mode").string() == "bf16",
            "Qualification policy requested mode must be bf16");
    const Json& effective = document.at("effective");
    require(effective.at("requested_mode").string() == "bf16",
            "Effective precision mode mismatch");
    PrecisionPolicy policy(PrecisionMode::BF16);
    const Json& operations = effective.at("operations");
    for (const auto& [operation, name] : kOperations) {
        const Json& entry = operations.at(name);
        const size_t index = operation_index(operation);
        policy.operation_compute_[index] = policy_dtype(entry.at("compute_dtype"));
        policy.operation_accumulator_[index] = policy_dtype(entry.at("accumulator_dtype"));
        policy.operation_output_[index] = policy_dtype(entry.at("output_dtype"));
    }
    const Json& roles = effective.at("roles");
    for (const auto& [semantic, name] : kFloatingSemantics)
        policy.semantic_dtype_[semantic_index(semantic)] = policy_dtype(roles.at(name));
    if (const Json* producer_outputs = effective.find("producer_outputs")) {
        require(producer_outputs->is_object(),
                "Precision policy producer_outputs must be an object");
        for (const auto& [semantic, name] : kFloatingSemantics) {
            if (const Json* value = producer_outputs->find(name)) {
                const size_t index = semantic_index(semantic);
                policy.producer_output_[index] = policy_dtype(*value);
                policy.producer_output_override_[index] = true;
            }
        }
    }
    const Json* scalar_roles = effective.find("scalar_roles");
    if (schema == "vrhino.precision.policy.v1") {
        require(!scalar_roles, "Semantic scalar declarations require precision policy v2");
    } else {
        require(scalar_roles && scalar_roles->is_object(), "Missing scalar contract declaration");
        require(scalar_roles->object().size() == 2 &&
                scalar_roles->at("contract").string() == "semantic-scalar.v1",
                "Unsupported scalar contract");
        const auto& names = scalar_roles->at("roles").array();
        require(!names.empty(), "Empty scalar role declaration");
        std::vector<PrecisionScalarRole> roles;
        for (const auto& name : names) {
            bool found = false;
            for (size_t i = 0; i < kScalarRoles.size(); ++i)
                if (name.string() == kScalarRoles[i]) {
                    roles.push_back(static_cast<PrecisionScalarRole>(i)); found = true; break;
                }
            require(found, "Unknown precision scalar role declaration");
        }
        policy = policy.with_scalar_roles(roles);
    }
    return policy;
}

PrecisionPolicy PrecisionPolicy::with_scalar_roles(const std::vector<PrecisionScalarRole>& roles) const {
    PrecisionPolicy result = *this;
    for (const auto role : roles) {
        const auto index = scalar_index(role);
        require(!result.scalar_roles_[index], "Duplicate precision scalar role");
        result.scalar_roles_[index] = true;
    }
    return result;
}

bool PrecisionPolicy::has_scalar_role(PrecisionScalarRole role) const {
    return scalar_roles_[scalar_index(role)];
}

PrecisionScalarContract PrecisionPolicy::scalar_contract(PrecisionScalarRole role, DType storage) const {
    require(storage == DType::F32 || storage == DType::BF16, "Scalar arithmetic requires floating storage");
    const bool data = role == PrecisionScalarRole::DataOperand;
    const DType output = role == PrecisionScalarRole::TimestepScale ||
        role == PrecisionScalarRole::ModulationCoefficient ? DType::F32 : storage;
    return {has_scalar_role(role), data ? storage : DType::F32,
            data ? storage : DType::F32, output};
}

Tensor precision_scalar_binary(Backend& backend, const PrecisionPolicy& policy,
        PrecisionScalarRole role, ScalarBinaryOperation operation,
        const Tensor& value, const Tensor& scalar) {
    auto apply = [&](const Tensor& a, const Tensor& b) {
        switch (operation) {
            case ScalarBinaryOperation::Add: return backend.add(a, b);
            case ScalarBinaryOperation::Multiply: return backend.mul(a, b);
            case ScalarBinaryOperation::Divide: return backend.div(a, b);
        }
        throw Error("Unknown scalar arithmetic operation");
    };
    if (!policy.has_scalar_role(role)) return apply(value, scalar);
    const auto contract = policy.scalar_contract(role, value.dtype());
    require(role != PrecisionScalarRole::TimestepScale || value.dtype() == DType::F32,
            "Timestep coordinates must reach scalar lowering as F32");
    require(scalar.numel() == 1 && scalar.ndim() <= 1 && scalar.dtype() == contract.source_dtype,
            "Semantic scalar source shape/dtype mismatch");
    const Tensor host = scalar.device() == Device::CPU ? scalar : backend.copy_to_host(scalar);
    const Tensor host_f32_value = host.dtype() == DType::F32 ? host : backend.copy_to_host(backend.cast(host, DType::F32));
    const float coefficient = *host_f32_value.data_as<float>();
    require(std::isfinite(coefficient), "Nonfinite semantic scalar");
    require(operation != ScalarBinaryOperation::Divide || coefficient != 0.0f,
            "Zero semantic scalar divisor");
    const Tensor input = backend.copy_to_device(value, contract.compute_dtype);
    const Tensor operand = backend.copy_to_device(scalar, contract.compute_dtype);
    Tensor result = apply(input, operand);
    return result.dtype() == contract.output_dtype ? result : backend.cast(result, contract.output_dtype);
}

DType PrecisionPolicy::requested_dtype() const {
    return requested_mode_ == PrecisionMode::FP32 ? DType::F32 : DType::BF16;
}

DType PrecisionPolicy::operation_compute_dtype(PrecisionOperation operation) const {
    return operation_compute_.at(operation_index(operation));
}

DType PrecisionPolicy::operation_accumulator_dtype(PrecisionOperation operation) const {
    return operation_accumulator_.at(operation_index(operation));
}

DType PrecisionPolicy::operation_output_dtype(
        PrecisionOperation operation, PrecisionSemantic semantic) const {
    require(semantic != PrecisionSemantic::DiscreteInput,
            "Discrete input has no floating operation output dtype");
    return operation_output_.at(operation_index(operation));
}

DType PrecisionPolicy::producer_output_dtype(
        PrecisionOperation operation, PrecisionSemantic semantic) const {
    require(semantic != PrecisionSemantic::DiscreteInput,
            "Discrete input has no floating producer output dtype");
    const size_t semantic_position = semantic_index(semantic);
    return producer_output_override_.at(semantic_position)
        ? producer_output_.at(semantic_position)
        : operation_output_.at(operation_index(operation));
}

DType PrecisionPolicy::persistent_state_dtype(PrecisionSemantic semantic) const {
    require(semantic == PrecisionSemantic::ResidualState ||
            semantic == PrecisionSemantic::SamplingState,
            "Persistent-state dtype requires a persistent semantic role");
    return semantic_dtype_.at(semantic_index(semantic));
}

DType PrecisionPolicy::reduction_dtype(PrecisionSemantic semantic) const {
    require(semantic == PrecisionSemantic::NormalizationStatistic ||
            semantic == PrecisionSemantic::AttentionStatistic,
            "Reduction dtype requires a statistic semantic role");
    return semantic_dtype_.at(semantic_index(semantic));
}

DType PrecisionPolicy::boundary_dtype(PrecisionSemantic semantic) const {
    require(semantic == PrecisionSemantic::Conditioning ||
            semantic == PrecisionSemantic::DenoiserOutput ||
            semantic == PrecisionSemantic::VaeInput ||
            semantic == PrecisionSemantic::VideoOutput,
            "Boundary dtype requires a boundary semantic role");
    return semantic_dtype_.at(semantic_index(semantic));
}

PrecisionOperationContract PrecisionPolicy::operation_contract(
        PrecisionOperation operation, PrecisionSemantic output_semantic) const {
    require(output_semantic != PrecisionSemantic::DiscreteInput,
            "Floating operation contract cannot target discrete input");
    const PrecisionOperation base = base_operation(operation);
    const DType requested = requested_dtype();
    DType operand = operation_compute_dtype(base);
    DType compute = operation_compute_dtype(base);
    DType weight = compute;
    const DType accumulator = operation_accumulator_dtype(base);
    DType temporary = accumulator;
    DType output = producer_output_dtype(base, output_semantic);
    PrecisionReductionSemantics reduction = PrecisionReductionSemantics::None;
    PrecisionCastBoundary boundary = PrecisionCastBoundary::None;

    if (operation == PrecisionOperation::Normalization) {
        temporary = reduction_dtype(PrecisionSemantic::NormalizationStatistic);
        reduction = PrecisionReductionSemantics::FP32Stable;
        if (output_semantic == PrecisionSemantic::AttentionStatistic)
            output = DType::F32;
    } else if (operation == PrecisionOperation::Attention ||
               operation == PrecisionOperation::AttentionQK ||
               operation == PrecisionOperation::AttentionPV ||
               operation == PrecisionOperation::Pooling) {
        temporary = reduction_dtype(PrecisionSemantic::AttentionStatistic);
        reduction = operation == PrecisionOperation::Attention
            ? PrecisionReductionSemantics::FP32StableOrderedOnline
            : PrecisionReductionSemantics::FP32Stable;
    }

    // RoPE is a numerically sensitive temporary transform between normalized
    // Q/K production and the heavy Attention consumer.  Preserve the FP32
    // temporary through the rotation; operation_attention performs the single
    // generic consumer-operand conversion requested by its own contract.
    if (operation == PrecisionOperation::Rope) {
        operand = DType::F32;
        compute = DType::F32;
        temporary = DType::F32;
        output = DType::F32;
    }

    // Official-BF16 calibration demonstrates that a generic modulation
    // producer may retain an FP32 temporary until the next heavy consumer.
    // This is an operation-semantic rule, independent of graph identity.
    if (operation == PrecisionOperation::Modulation) {
        operand = DType::F32;
        weight = DType::F32;
        compute = DType::F32;
        temporary = DType::F32;
        output = DType::F32;
    }

    const bool heavy_consumer = operation == PrecisionOperation::Linear ||
        operation == PrecisionOperation::AttentionProjection ||
        operation == PrecisionOperation::Convolution ||
        operation == PrecisionOperation::Embedding;
    if (requested_mode_ == PrecisionMode::BF16 && heavy_consumer)
        boundary = PrecisionCastBoundary::ConsumerOperand;

    return PrecisionOperationContract{
        requested,
        operand,
        weight,
        compute,
        accumulator,
        temporary,
        output,
        reduction,
        boundary,
    };
}

std::string precision_semantic_name(PrecisionSemantic semantic) {
    switch (semantic) {
        case PrecisionSemantic::DiscreteInput: return "DISCRETE_INPUT";
        case PrecisionSemantic::Conditioning: return "CONDITIONING";
        case PrecisionSemantic::TemporaryCompute: return "TEMPORARY_COMPUTE";
        case PrecisionSemantic::NormalizationStatistic: return "NORMALIZATION_STATISTIC";
        case PrecisionSemantic::AttentionStatistic: return "ATTENTION_STATISTIC";
        case PrecisionSemantic::ResidualState: return "RESIDUAL_STATE";
        case PrecisionSemantic::SamplingState: return "SAMPLING_STATE";
        case PrecisionSemantic::DenoiserOutput: return "DENOISER_OUTPUT";
        case PrecisionSemantic::VaeInput: return "VAE_INPUT";
        case PrecisionSemantic::VideoOutput: return "VIDEO_OUTPUT";
    }
    throw Error("Unknown precision semantic role");
}

}  // namespace vrhino
