#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "vrhino/json.h"

namespace vrhino::product {

inline constexpr const char* kProductInputSchemaV1 =
    "vrhino.product.input-schema.v1";

enum class ProductValueType {
    Text,
    Integer,
    MediaVideo,
    MediaAudio,
    MediaMp4,
};

struct ProductRational {
    uint64_t numerator = 0;
    uint64_t denominator = 0;

    bool operator==(const ProductRational&) const = default;
};

struct ProductValidation {
    std::optional<uint64_t> min_length;
    std::optional<uint64_t> minimum;
    std::optional<uint64_t> maximum;
    std::optional<bool> regular_file;
    std::optional<bool> decodable;
    std::optional<ProductRational> fps;
    std::optional<uint64_t> minimum_duration_ms;
    std::optional<bool> parent_creatable_and_writable;
};

using ProductDefaultValue =
    std::variant<std::monostate, std::string, uint64_t>;

struct ProductFieldDeclaration {
    std::string name;
    ProductValueType type = ProductValueType::Text;
    bool required = false;
    ProductDefaultValue default_value;
    ProductValidation validation;

    bool has_default() const noexcept {
        return !std::holds_alternative<std::monostate>(default_value);
    }
};

struct ProductInputSchema {
    std::string identity;
    std::vector<ProductFieldDeclaration> inputs;
    std::vector<ProductFieldDeclaration> parameters;
    std::vector<ProductFieldDeclaration> outputs;

    const ProductFieldDeclaration* find_input(const std::string& name) const;
    const ProductFieldDeclaration* find_parameter(const std::string& name) const;
    const ProductFieldDeclaration* find_output(const std::string& name) const;
};

struct ProductFrozenOutput {
    std::optional<uint64_t> width;
    std::optional<uint64_t> height;
    std::optional<uint64_t> frames;
    ProductRational fps;
    std::string duration;
    std::string audio;
};

struct ProductFrozenSampling {
    std::optional<std::string> method;
    std::optional<std::string> prediction;
    std::optional<uint64_t> steps;
    std::optional<double> guidance_scale;
    std::optional<double> eta;
};

struct ProductFrozenTemporal {
    std::optional<uint64_t> chunk_frames;
};

struct ProductFrozenProfile {
    ProductFrozenOutput output;
    std::optional<ProductFrozenSampling> sampling;
    std::optional<ProductFrozenTemporal> temporal;
};

const char* product_value_type_name(ProductValueType type);
ProductInputSchema parse_product_input_schema(const Json& value);
ProductFrozenProfile parse_product_frozen_profile(const Json& value);

void validate_product_contract_for_family(
    const std::string& family,
    const std::string& workflow_identity,
    const ProductInputSchema& schema,
    const ProductFrozenProfile& frozen_profile);

void validate_product_execution_consistency(
    const std::string& family,
    const std::string& workflow_identity,
    const ProductInputSchema& schema,
    const ProductFrozenProfile& frozen_profile,
    const Json& execution,
    const Json* workflow = nullptr);

uint64_t resolve_product_seed(
    const ProductInputSchema* schema,
    std::optional<uint64_t> requested,
    uint64_t legacy_execution_default);

std::string resolve_product_output(
    const ProductInputSchema* schema,
    const std::string& requested,
    const std::string& legacy_default = "output.mp4");

}  // namespace vrhino::product
