#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/precision.h"
#include "vrhino/sampling.h"

namespace {

const char* policy_dtype_name(vrhino::DType dtype) {
    if (dtype == vrhino::DType::F32) return "fp32";
    if (dtype == vrhino::DType::BF16) return "bf16";
    throw vrhino::Error("Precision policy dump encountered unsupported dtype");
}

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open precision policy: " + path);
    std::ostringstream output;
    output << stream.rdbuf();
    return output.str();
}

void dump_bf16_policy(const vrhino::PrecisionPolicy& policy) {
    using namespace vrhino;
    const std::array<std::pair<PrecisionOperation, std::string_view>, 13> operations = {{
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
    std::cout << "{\"requested_mode\":\"bf16\",\"operations\":{";
    bool first = true;
    for (const auto& [operation, name] : operations) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '\"' << name << "\":{\"compute_dtype\":\""
                  << policy_dtype_name(policy.operation_compute_dtype(operation))
                  << "\",\"accumulator_dtype\":\""
                  << policy_dtype_name(policy.operation_accumulator_dtype(operation))
                  << "\",\"output_dtype\":\""
                  << policy_dtype_name(policy.operation_output_dtype(
                         operation, PrecisionSemantic::TemporaryCompute))
                  << "\"}";
    }
    std::cout << "},\"roles\":{"
              << "\"CONDITIONING\":\"" << policy_dtype_name(policy.boundary_dtype(
                     PrecisionSemantic::Conditioning)) << "\","
              << "\"TEMPORARY_COMPUTE\":\"" << policy_dtype_name(policy.operation_output_dtype(
                     PrecisionOperation::Activation, PrecisionSemantic::TemporaryCompute)) << "\","
              << "\"NORMALIZATION_STATISTIC\":\"" << policy_dtype_name(policy.reduction_dtype(
                     PrecisionSemantic::NormalizationStatistic)) << "\","
              << "\"ATTENTION_STATISTIC\":\"" << policy_dtype_name(policy.reduction_dtype(
                     PrecisionSemantic::AttentionStatistic)) << "\","
              << "\"RESIDUAL_STATE\":\"" << policy_dtype_name(policy.persistent_state_dtype(
                     PrecisionSemantic::ResidualState)) << "\","
              << "\"SAMPLING_STATE\":\"" << policy_dtype_name(policy.persistent_state_dtype(
                     PrecisionSemantic::SamplingState)) << "\","
              << "\"DENOISER_OUTPUT\":\"" << policy_dtype_name(policy.boundary_dtype(
                     PrecisionSemantic::DenoiserOutput)) << "\","
              << "\"VAE_INPUT\":\"" << policy_dtype_name(policy.boundary_dtype(
                     PrecisionSemantic::VaeInput)) << "\","
              << "\"VIDEO_OUTPUT\":\"" << policy_dtype_name(policy.boundary_dtype(
                     PrecisionSemantic::VideoOutput)) << "\"},"
              << "\"producer_outputs\":{"
              << "\"DENOISER_OUTPUT\":\"" << policy_dtype_name(
                     policy.producer_output_dtype(PrecisionOperation::Linear,
                                                  PrecisionSemantic::DenoiserOutput))
              << "\"}}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        using namespace vrhino;
        if (argc == 2 && std::string_view(argv[1]) == "--dump-bf16") {
            dump_bf16_policy(PrecisionPolicy::unqualified_default(DType::BF16));
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--dump-policy") {
            const Json document = Json::parse(read_text(argv[2]));
            dump_bf16_policy(PrecisionPolicy::from_json(document));
            return 0;
        }
        if (argc == 3 &&
            std::string_view(argv[1]) == "--verify-heavy-consumer-policy") {
            const Json document = Json::parse(read_text(argv[2]));
            require(document.at("id").string() ==
                        "bf16-heavy-consumer-fp32-state-v1",
                    "Heavy-consumer policy identity mismatch");
            const PrecisionPolicy policy = PrecisionPolicy::from_json(document);
            require(policy.requested_dtype() == DType::BF16 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::Linear) == DType::BF16 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::Convolution) == DType::BF16 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::AttentionQK) == DType::BF16 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::AttentionPV) == DType::BF16,
                    "Heavy-consumer BF16 operation contract mismatch");
            require(policy.persistent_state_dtype(
                        PrecisionSemantic::ResidualState) == DType::F32 &&
                        policy.persistent_state_dtype(
                            PrecisionSemantic::SamplingState) == DType::F32 &&
                        policy.boundary_dtype(
                            PrecisionSemantic::DenoiserOutput) == DType::F32 &&
                        policy.boundary_dtype(
                            PrecisionSemantic::VaeInput) == DType::F32 &&
                        policy.reduction_dtype(
                            PrecisionSemantic::NormalizationStatistic) == DType::F32 &&
                        policy.reduction_dtype(
                            PrecisionSemantic::AttentionStatistic) == DType::F32 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::Scheduler) == DType::F32 &&
                        policy.operation_compute_dtype(
                            PrecisionOperation::Vae) == DType::F32,
                    "Heavy-consumer FP32-island contract mismatch");
            require(policy.producer_output_dtype(
                        PrecisionOperation::Convolution,
                        PrecisionSemantic::DenoiserOutput) == DType::BF16,
                    "Heavy-consumer denoiser producer contract mismatch");
            std::cout << "bf16_heavy_consumer_fp32_state_policy=PASS\n";
            return 0;
        }
        require(argc == 1,
                "usage: vrhino-precision-policy-tests "
                "[--dump-bf16|--dump-policy POLICY.json|"
                "--verify-heavy-consumer-policy POLICY.json]");
        static_assert(!std::is_constructible_v<PrecisionPolicy, std::string>,
                      "PrecisionPolicy must not accept model/architecture identity");
        const PrecisionPolicy fp32 = PrecisionPolicy::fp32();
        require(fp32.requested_dtype() == DType::F32,
                "FP32 requested dtype mismatch");
        require(fp32.operation_compute_dtype(PrecisionOperation::Linear) == DType::F32,
                "FP32 Linear compute mismatch");
        require(fp32.operation_accumulator_dtype(PrecisionOperation::Linear) == DType::F32,
                "FP32 Linear accumulator mismatch");
        require(fp32.reduction_dtype(PrecisionSemantic::NormalizationStatistic) == DType::F32,
                "FP32 normalization statistic mismatch");

        const PrecisionPolicy bf16 = PrecisionPolicy::unqualified_default(DType::BF16);
        require(bf16.operation_compute_dtype(PrecisionOperation::Linear) == DType::BF16,
                "Unqualified-default BF16 Linear compute mismatch");
        require(bf16.operation_accumulator_dtype(PrecisionOperation::Linear) == DType::F32,
                "Unqualified-default BF16 Linear accumulator mismatch");
        require(bf16.reduction_dtype(PrecisionSemantic::AttentionStatistic) == DType::F32,
                "Unqualified-default BF16 Attention statistic mismatch");
        require(bf16.producer_output_dtype(
                    PrecisionOperation::Linear,
                    PrecisionSemantic::DenoiserOutput) == DType::BF16,
                "Unqualified-default BF16 producer output mismatch");

        const PrecisionOperationContract linear_contract = bf16.operation_contract(
            PrecisionOperation::Linear, PrecisionSemantic::TemporaryCompute);
        require(linear_contract.input_storage_dtype == DType::BF16 &&
                linear_contract.operand_dtype == DType::BF16 &&
                linear_contract.weight_dtype == DType::BF16 &&
                linear_contract.compute_dtype == DType::BF16 &&
                linear_contract.accumulator_dtype == DType::F32 &&
                linear_contract.output_dtype == DType::BF16 &&
                linear_contract.cast_boundary == PrecisionCastBoundary::ConsumerOperand,
                "BF16 Linear operation contract mismatch");
        const PrecisionOperationContract modulation_contract = bf16.operation_contract(
            PrecisionOperation::Modulation, PrecisionSemantic::TemporaryCompute);
        require(modulation_contract.input_storage_dtype == DType::BF16 &&
                modulation_contract.temporary_dtype == DType::F32 &&
                modulation_contract.output_dtype == DType::F32 &&
                modulation_contract.cast_boundary == PrecisionCastBoundary::None,
                "BF16 Modulation operation contract mismatch");
        const PrecisionOperationContract attention_contract = bf16.operation_contract(
            PrecisionOperation::Attention, PrecisionSemantic::TemporaryCompute);
        require(attention_contract.operand_dtype == DType::BF16 &&
                attention_contract.temporary_dtype == DType::F32 &&
                attention_contract.output_dtype == DType::BF16 &&
                attention_contract.reduction_semantics ==
                    PrecisionReductionSemantics::FP32StableOrderedOnline,
                "BF16 Attention operation contract mismatch");
        const PrecisionOperationContract fp32_modulation = fp32.operation_contract(
            PrecisionOperation::Modulation, PrecisionSemantic::TemporaryCompute);
        require(fp32_modulation.operand_dtype == DType::F32 &&
                fp32_modulation.weight_dtype == DType::F32 &&
                fp32_modulation.compute_dtype == DType::F32 &&
                fp32_modulation.temporary_dtype == DType::F32 &&
                fp32_modulation.output_dtype == DType::F32,
                "FP32 Modulation behavior changed");
        const PrecisionOperationContract bf16_modulation = bf16.operation_contract(
            PrecisionOperation::Modulation, PrecisionSemantic::TemporaryCompute);
        require(bf16_modulation.operand_dtype == DType::F32 &&
                bf16_modulation.weight_dtype == DType::F32 &&
                bf16_modulation.compute_dtype == DType::F32 &&
                bf16_modulation.temporary_dtype == DType::F32 &&
                bf16_modulation.output_dtype == DType::F32,
                "BF16 Modulation operation contract mismatch");
        const PrecisionOperationContract bf16_rope = bf16.operation_contract(
            PrecisionOperation::Rope, PrecisionSemantic::TemporaryCompute);
        require(bf16_rope.operand_dtype == DType::F32 &&
                bf16_rope.compute_dtype == DType::F32 &&
                bf16_rope.temporary_dtype == DType::F32 &&
                bf16_rope.output_dtype == DType::F32,
                "BF16 RoPE temporary contract mismatch");
        const PrecisionOperationContract bf16_attention_norm = bf16.operation_contract(
            PrecisionOperation::Normalization, PrecisionSemantic::AttentionStatistic);
        require(bf16_attention_norm.temporary_dtype == DType::F32 &&
                bf16_attention_norm.output_dtype == DType::F32 &&
                bf16_attention_norm.reduction_semantics ==
                    PrecisionReductionSemantics::FP32Stable,
                "BF16 attention-normalization contract mismatch");

        require(effective_sampling_state_dtype(fp32) == DType::F32,
                "FP32 sampling-state semantics changed");
        require(effective_sampling_state_dtype(bf16) == DType::BF16,
                "Unqualified-default BF16 sampling state mismatch");
        require(effective_denoiser_output_dtype(fp32) == DType::F32,
                "FP32 denoiser-output boundary semantics changed");
        require(effective_denoiser_output_dtype(bf16) == DType::BF16,
                "Unqualified-default BF16 denoiser-output boundary mismatch");

        // Labels are deliberately test data only. They cannot be passed to the
        // policy API, so identical semantic input has one identical decision.
        constexpr std::array<std::string_view, 4> architecture_labels = {
            "Wan", "HunyuanVideo", "LTX", "Mochi",
        };
        for (const std::string_view ignored : architecture_labels) {
            (void)ignored;
            require(effective_sampling_state_dtype(bf16) == DType::BF16,
                    "Architecture label influenced sampling-state policy");
            require(effective_denoiser_output_dtype(bf16) == DType::BF16,
                    "Architecture label influenced denoiser-output policy");
        }

        const std::array<std::pair<PrecisionSemantic, std::string_view>, 10> roles = {{
            {PrecisionSemantic::DiscreteInput, "DISCRETE_INPUT"},
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
        for (const auto& [role, expected] : roles)
            require(precision_semantic_name(role) == expected,
                    "Canonical semantic-role mapping mismatch");

        std::cout << "precision_policy_tests=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "precision policy tests: " << error.what() << '\n';
        return 1;
    }
}
