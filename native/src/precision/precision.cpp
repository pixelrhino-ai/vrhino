#include "vrhino/precision.h"

#include <array>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino {
namespace {

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
    require(document.at("schema").string() == "vrhino.precision.policy.v1",
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
    return policy;
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
