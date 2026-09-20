#pragma once
#include "vrhino/product/package_preflight.h"
#include "vrhino/tokenizer.h"

namespace vrhino::product {
struct ConditioningInputRequest {
    std::string target;
    Json graph;
    Tensor input_ids, attention_mask;
    std::vector<int64_t> expected_hidden_shape;
};
// Retains all borrowed component/decoder backing. It is a prepared request,
// not sampling state, a synthetic hidden tensor, or permission to execute.
struct PreparedTextProductRequest {
    std::shared_ptr<AdmittedLocalProduct> owner;
    TensorBundle runtime_inputs;
    std::vector<ConditioningInputRequest> conditioning;
    SamplingProgram sampling;
    ExecutionProgram execution = ExecutionProgram::uniform({0});
    std::vector<int64_t> expected_video_shape;
    Json evidence;
    WeightMap conditioning_weights() const;
    WeightMap decoder_weights() const;
    TensorBundle bind_conditioning_outputs(const std::vector<Tensor>& hidden_states) const;
};
bool native_text_request_available();
PreparedTextProductRequest prepare_text_product_request(
    std::shared_ptr<AdmittedLocalProduct> admitted, const Json& request);
PreparedTextProductRequest dry_run_local_product(const std::filesystem::path& manifest,
    const std::filesystem::path& local_resources, const std::filesystem::path& request);
PreparedTextProductRequest dry_run_resolved_product(ResolvedRunnableModel resources,
    const std::filesystem::path& request);
} // namespace vrhino::product
