#pragma once

#include <array>
#include <string>

#include "vrhino/tensor.h"

namespace vrhino {

class Json;

// Internal names intentionally map one-to-one to the canonical vocabulary in
// tests/precision/roles.py. Architecture/model identity is not representable.
enum class PrecisionSemantic {
    DiscreteInput,
    Conditioning,
    TemporaryCompute,
    NormalizationStatistic,
    AttentionStatistic,
    ResidualState,
    SamplingState,
    DenoiserOutput,
    VaeInput,
    VideoOutput,
};

enum class PrecisionOperation {
    Linear,
    Convolution,
    Normalization,
    AttentionProjection,
    AttentionQK,
    AttentionPV,
    Activation,
    ResidualAdd,
    Guidance,
    Scheduler,
    Conditioning,
    Denoiser,
    Vae,
    // Operation-semantic families that compose existing primitive contracts.
    // They are not architecture dispatch keys.
    Modulation,
    Attention,
    Rope,
    Pooling,
    Embedding,
    SamplingArithmetic,
    VaePrimitive,
};

enum class PrecisionMode { FP32, BF16 };

enum class PrecisionCastBoundary {
    None,
    ConsumerOperand,
};

enum class PrecisionReductionSemantics {
    None,
    FP32Stable,
    FP32StableOrderedOnline,
};

// Complete, model-neutral precision contract for one operation/dataflow role.
// Storage and boundary decisions are deliberately separate from compute and
// accumulator precision.
struct PrecisionOperationContract {
    DType input_storage_dtype;
    DType operand_dtype;
    DType weight_dtype;
    DType compute_dtype;
    DType accumulator_dtype;
    DType temporary_dtype;
    DType output_dtype;
    PrecisionReductionSemantics reduction_semantics;
    PrecisionCastBoundary cast_boundary;
};

class PrecisionPolicy {
public:
    // Provides deterministic generic plumbing from the requested mode. FP32
    // preserves the known-correct default. BF16 remains unqualified and is not
    // the frozen BF16 Policy v0.
    static PrecisionPolicy unqualified_default(DType requested_dtype);
    static PrecisionPolicy fp32();
    // Generic policy-data loader used by qualification/reference harnesses.
    // The document contains only operation and semantic-role dtype decisions;
    // architecture/model identity is neither accepted nor representable.
    static PrecisionPolicy from_json(const Json& document);

    PrecisionMode requested_mode() const { return requested_mode_; }
    DType requested_dtype() const;

    DType operation_compute_dtype(PrecisionOperation operation) const;
    DType operation_accumulator_dtype(PrecisionOperation operation) const;
    DType operation_output_dtype(
        PrecisionOperation operation, PrecisionSemantic semantic) const;
    // Producer output is the dtype written directly by an operation from its
    // accumulator/result. It is distinct from a later boundary storage cast.
    DType producer_output_dtype(
        PrecisionOperation operation, PrecisionSemantic semantic) const;
    DType persistent_state_dtype(PrecisionSemantic semantic) const;
    DType reduction_dtype(PrecisionSemantic semantic) const;
    DType boundary_dtype(PrecisionSemantic semantic) const;
    PrecisionOperationContract operation_contract(
        PrecisionOperation operation, PrecisionSemantic output_semantic) const;

private:
    explicit PrecisionPolicy(PrecisionMode requested_mode);
    static constexpr size_t kOperationCount = 13;
    static constexpr size_t kSemanticCount = 10;
    PrecisionMode requested_mode_;
    std::array<DType, kOperationCount> operation_compute_;
    std::array<DType, kOperationCount> operation_accumulator_;
    std::array<DType, kOperationCount> operation_output_;
    std::array<DType, kSemanticCount> semantic_dtype_;
    std::array<DType, kSemanticCount> producer_output_;
    std::array<bool, kSemanticCount> producer_output_override_;
};

std::string precision_semantic_name(PrecisionSemantic semantic);

}  // namespace vrhino
