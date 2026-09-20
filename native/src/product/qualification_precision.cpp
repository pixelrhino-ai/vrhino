#include "vrhino/product/qualification_precision.h"
#include "vrhino/error.h"
#include <fstream>
#include <cstdlib>

namespace vrhino::product {
QualificationPrecision load_declared_product_precision(const ResolvedRunnableModel& model,
    const std::string& artifact_id) {
    require(!artifact_id.empty(), "Missing qualification precision artifact ID");
    const auto it = model.artifacts.find(artifact_id);
    require(it != model.artifacts.end(), "Missing qualification precision artifact");
    const auto& artifact = it->second;
    const auto& declaration = artifact.declaration;
    require(declaration.id == artifact_id && declaration.required &&
            declaration.role == "precision.policy", "Invalid qualification precision artifact declaration");
    require(declaration.size > 0 && declaration.size <= 65536,
            "Invalid qualification precision artifact size");
    std::ifstream input(artifact.path, std::ios::binary);
    require(input.good(), "Cannot read qualification precision artifact");
    // Bounded read and same-buffer hash/parse prevent a reopen race or an
    // oversized replacement document from changing the admitted policy.
    std::string bytes(static_cast<size_t>(declaration.size) + 1, '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(input.eof() && input.gcount() == static_cast<std::streamsize>(declaration.size),
            "Qualification precision artifact size mismatch");
    bytes.resize(static_cast<size_t>(declaration.size));
    const auto digest = sha256_bytes(bytes);
    require(digest == declaration.sha256, "Qualification precision artifact SHA256 mismatch");
    const auto document = Json::parse(bytes, JsonParseLimits{});
    // The existing immutable policy document contract currently supports BF16
    // only. Keep its admission and legacy error contract unchanged.
    require(document.at("requested_mode").string() == "bf16", "Qualification precision mode mismatch");
    // Qualification observes the default numerical implementation. No
    // environment-selected research implementation may contaminate evidence.
    for (const auto* name : {"VRHINO_BF16_LINEAR_OUTPUT", "VRHINO_BF16_NORM_OUTPUT",
            "VRHINO_BF16_ATTENTION_OUTPUT", "VRHINO_BF16_CONV_COMPUTE",
            "VRHINO_CUDA_ATTENTION_VARIANT", "VRHINO_CUDA_ATTENTION_BF16_QK",
            "VRHINO_CUDA_ATTENTION_SDPA_ADMISSION", "VRHINO_CUDA_ATTENTION_KEY_TILE",
            "VRHINO_ATTENTION_CAPTURE_PATH", "VRHINO_ATTENTION_CAPTURE_MIN_TOKENS",
            "VRHINO_ATTENTION_CAPTURE_EXIT"})
        require(std::getenv(name) == nullptr, std::string("Research switch is not allowed in Product qualification: ") + name);
    auto policy = PrecisionPolicy::from_json(document);
    return {std::move(policy), Json(Json::Object{{"artifact_id", Json(artifact_id)},
        {"sha256", Json(digest)}, {"declaration", document},
        {"numerical_admission_granted", Json(false)}})};
}
QualificationPrecision load_qualification_precision(const ResolvedRunnableModel& model,
    const std::string& artifact_id, const std::string& requested_mode) {
    require(requested_mode == "bf16", "Declared qualification policy requires BF16 mode");
    auto declared = load_declared_product_precision(model, artifact_id);
    require(declared.policy.requested_dtype() == DType::BF16, "Qualification precision mode mismatch");
    return declared;
}
} // namespace vrhino::product
