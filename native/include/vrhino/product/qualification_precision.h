#pragma once
#include "vrhino/product/model_package.h"
#include "vrhino/precision.h"

namespace vrhino::product {
// A bounded qualification document, not numerical admission for ordinary run.
// The policy is an immutable package artifact selected by logical resource ID.
struct QualificationPrecision {
    PrecisionPolicy policy;
    Json evidence;
};
QualificationPrecision load_qualification_precision(const ResolvedRunnableModel&,
    const std::string& artifact_id, const std::string& requested_mode);
// Verifies and parses a declaration only; admission is enforced separately.
QualificationPrecision load_declared_product_precision(const ResolvedRunnableModel&,
    const std::string& artifact_id);
} // namespace vrhino::product
