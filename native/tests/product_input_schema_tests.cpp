#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "vrhino/json.h"
#include "vrhino/product/input_schema.h"
#include "vrhino/product/model_package.h"

namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void expect_invalid(Operation&& operation, const std::string& context) {
    try {
        operation();
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == product::ModelPackageErrorCode::PackageInvalid,
                     context + ": wrong error code");
        return;
    }
    throw std::runtime_error(context + ": invalid schema was accepted");
}

std::string replace_once(std::string value, const std::string& from,
                         const std::string& to) {
    const size_t position = value.find(from);
    require_test(position != std::string::npos,
                 "test fixture replacement source not found: " + from);
    value.replace(position, from.size(), to);
    return value;
}

const std::string& all_types_schema() {
    static const std::string value = R"JSON({
      "schema":"vrhino.product.input-schema.v1",
      "inputs":[
        {"name":"prompt","type":"text","required":true,"validation":{"min_length":1}},
        {"name":"video","type":"media.video","required":true,"validation":{"regular_file":true,"decodable":true,"fps":{"numerator":25,"denominator":1}}},
        {"name":"audio","type":"media.audio","required":true,"validation":{"regular_file":true,"decodable":true,"minimum_duration_ms":40}}
      ],
      "parameters":[
        {"name":"seed","type":"integer","required":false,"default":5703,"validation":{"minimum":0,"maximum":"18446744073709551615"}}
      ],
      "outputs":[
        {"name":"output","type":"media.mp4","required":false,"default":"output.mp4","validation":{"parent_creatable_and_writable":true}}
      ]
    })JSON";
    return value;
}

const std::string& ttv_schema() {
    static const std::string value = R"JSON({
      "schema":"vrhino.product.input-schema.v1",
      "inputs":[{"name":"prompt","type":"text","required":true,"validation":{"min_length":1}}],
      "parameters":[{"name":"seed","type":"integer","required":false,"default":5703,"validation":{"minimum":0,"maximum":"18446744073709551615"}}],
      "outputs":[{"name":"output","type":"media.mp4","required":false,"default":"output.mp4","validation":{"parent_creatable_and_writable":true}}]
    })JSON";
    return value;
}

const std::string& ttv_frozen() {
    static const std::string value = R"JSON({
      "output":{"width":704,"height":480,"frames":121,"fps":{"numerator":25,"denominator":1},"duration":"fixed","audio":"none"},
      "sampling":{"steps":40,"guidance_scale":3.0}
    })JSON";
    return value;
}

}  // namespace

int main() {
    try {
        const product::ProductInputSchema all_types =
            product::parse_product_input_schema(vrhino::Json::parse(all_types_schema()));
        require_test(all_types.inputs.size() == 3 &&
                         all_types.parameters.size() == 1 &&
                         all_types.outputs.size() == 1,
                     "all five v1 types did not parse deterministically");
        const product::ProductInputSchema repeated =
            product::parse_product_input_schema(vrhino::Json::parse(all_types_schema()));
        require_test(repeated.identity == all_types.identity &&
                         repeated.inputs.size() == all_types.inputs.size() &&
                         repeated.find_parameter("seed") != nullptr &&
                         std::get<uint64_t>(repeated.find_parameter("seed")->default_value) ==
                             5703,
                     "repeated deterministic parse changed the schema");

        const product::ProductInputSchema ttv =
            product::parse_product_input_schema(vrhino::Json::parse(ttv_schema()));
        const product::ProductFrozenProfile frozen =
            product::parse_product_frozen_profile(vrhino::Json::parse(ttv_frozen()));
        product::validate_product_contract_for_family(
            "text_to_video", "", ttv, frozen);
        require_test(product::resolve_product_seed(&ttv, std::nullopt, 5703) == 5703,
                     "canonical seed default did not resolve");
        require_test(product::resolve_product_seed(&ttv, 99, 5703) == 99,
                     "user seed did not override Product default");
        require_test(product::resolve_product_output(&ttv, "") == "output.mp4" &&
                         product::resolve_product_output(&ttv, "chosen.mp4") ==
                             "chosen.mp4",
                     "output default/override precedence failed");
        require_test(product::resolve_product_seed(nullptr, std::nullopt, 17) == 17 &&
                         product::resolve_product_output(nullptr, "") == "output.mp4",
                     "legacy absence fallback failed");

        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(),
                             "\"schema\":\"vrhino.product.input-schema.v1\",", "")));
        }, "missing schema identity");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "input-schema.v1", "input-schema.v2")));
        }, "unknown schema version");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"text\"", "\"image\"")));
        }, "unknown type");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"name\":\"seed\"",
                              "\"name\":\"prompt\"")));
        }, "duplicate names across groups");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(all_types_schema(), "\"name\":\"video\"",
                              "\"name\":\"prompt\"")));
        }, "duplicate names within a group");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"required\":true,\"validation\"",
                              "\"required\":true,\"default\":\"x\",\"validation\"")));
        }, "required field with default");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"minimum\":0",
                              "\"minimum\":-1")));
        }, "negative integer bound");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(),
                             "\"maximum\":\"18446744073709551615\"",
                             "\"maximum\":\"18446744073709551616\"")));
        }, "overflowing decimal uint64 bound");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"minimum\":0",
                              "\"minimum\":6000")));
        }, "minimum greater than default");
        expect_invalid([&] {
            std::string invalid = replace_once(
                ttv_schema(), "\"minimum\":0", "\"minimum\":6000");
            invalid = replace_once(
                invalid, "\"maximum\":\"18446744073709551615\"",
                "\"maximum\":\"5000\"");
            (void)product::parse_product_input_schema(vrhino::Json::parse(invalid));
        }, "minimum greater than maximum");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(),
                             "\"maximum\":\"18446744073709551615\"",
                             "\"maximum\":\"01\"")));
        }, "malformed decimal uint64 bound");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(all_types_schema(), "\"denominator\":1",
                              "\"denominator\":0")));
        }, "invalid rational FPS");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"min_length\":1",
                              "\"script\":\"accept()\"")));
        }, "unsupported validation kind");
        expect_invalid([&] {
            (void)product::parse_product_input_schema(vrhino::Json::parse(
                replace_once(ttv_schema(), "\"default\":5703",
                              "\"default\":\"seed\"")));
        }, "invalid default type");
        expect_invalid([&] {
            product::ProductInputSchema invalid =
                product::parse_product_input_schema(vrhino::Json::parse(
                    replace_once(ttv_schema(), "\"output.mp4\"", "\"other.mp4\"")));
            product::validate_product_contract_for_family(
                "text_to_video", "", invalid, frozen);
        }, "invalid output default semantics");
        expect_invalid([&] {
            product::ProductInputSchema invalid =
                product::parse_product_input_schema(vrhino::Json::parse(ttv_schema()));
            (void)product::resolve_product_seed(&invalid, std::nullopt, 5704);
        }, "execution seed drift");

        std::cout << "ProductInputSchema parser/semantic tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ProductInputSchema parser/semantic tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
