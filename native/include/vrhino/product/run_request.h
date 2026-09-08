#pragma once

#include <stdexcept>
#include <string>

#include "vrhino/json.h"
#include "vrhino/product/run.h"

namespace vrhino::product {

enum class RunRequestErrorCode {
    InvalidRequest,
    ModelUnavailable,
};

class RunRequestError : public std::runtime_error {
public:
    RunRequestError(RunRequestErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    RunRequestErrorCode code() const noexcept { return code_; }

private:
    RunRequestErrorCode code_;
};

struct ProductRunDocument {
    std::string model_reference;
    Json::Object inputs;
    Json::Object parameters;
    Json::Object outputs;
};

// Parses only the stable Product request envelope. Model-specific field
// admission and defaults are driven exclusively by the resolved package's
// ProductInputSchema v1 in map_product_run_options().
ProductRunDocument parse_product_run_document(const Json& value);

RunOptions map_product_run_options(const ResolvedRunnableModel& model,
                                   const ProductRunDocument& request);

}  // namespace vrhino::product
