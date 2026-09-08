#include "vrhino/product/input_schema.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>
#include <string_view>
#include <tuple>

#include "vrhino/product/model_package.h"

namespace vrhino::product {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw ModelPackageError(
        ModelPackageErrorCode::PackageInvalid,
        "ProductInputSchema v1: " + message);
}

void require_object(const Json& value, const std::string& context) {
    if (!value.is_object()) fail(context + " must be an object");
}

void require_exact_keys(const Json& value,
                        const std::set<std::string>& admitted,
                        const std::string& context) {
    require_object(value, context);
    for (const auto& [key, ignored] : value.object()) {
        (void)ignored;
        if (!admitted.contains(key))
            fail(context + " contains unsupported semantic field: " + key);
    }
}

const Json& required_field(const Json& object, const std::string& name,
                           const std::string& context) {
    const Json* value = object.find(name);
    if (value == nullptr) fail(context + " is missing required field: " + name);
    return *value;
}

std::string required_string(const Json& object, const std::string& name,
                            const std::string& context) {
    const Json& value = required_field(object, name, context);
    if (!value.is_string() || value.string().empty())
        fail(context + "." + name + " must be a non-empty string");
    return value.string();
}

bool required_bool(const Json& object, const std::string& name,
                   const std::string& context) {
    const Json& value = required_field(object, name, context);
    if (!value.is_bool()) fail(context + "." + name + " must be boolean");
    return value.boolean();
}

bool safe_name(const std::string& value) {
    if (value.empty()) return false;
    const unsigned char first = static_cast<unsigned char>(value.front());
    if (!std::islower(first) && !std::isdigit(first)) return false;
    return std::all_of(value.begin(), value.end(), [](const unsigned char byte) {
        return std::islower(byte) || std::isdigit(byte) || byte == '_' ||
               byte == '.' || byte == '-';
    });
}

uint64_t nonnegative_integer(const Json& value, const std::string& context) {
    if (!value.is_int() || value.integer() < 0)
        fail(context + " must be a non-negative integer");
    return static_cast<uint64_t>(value.integer());
}

uint64_t positive_integer(const Json& value, const std::string& context) {
    const uint64_t parsed = nonnegative_integer(value, context);
    if (parsed == 0) fail(context + " must be positive");
    return parsed;
}

uint64_t decimal_uint64(const Json& value, const std::string& context) {
    if (value.is_int()) return nonnegative_integer(value, context);
    if (!value.is_string() || value.string().empty())
        fail(context + " must be a non-negative integer or decimal uint64 string");
    const std::string& text = value.string();
    if ((text.size() > 1 && text.front() == '0') ||
        !std::all_of(text.begin(), text.end(), [](const unsigned char byte) {
            return std::isdigit(byte);
        }))
        fail(context + " is not a canonical decimal uint64 string");
    uint64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        fail(context + " overflows uint64");
    return result;
}

double finite_number(const Json& value, const std::string& context) {
    if (!value.is_number() || !std::isfinite(value.number()))
        fail(context + " must be a finite number");
    return value.number();
}

ProductValueType parse_type(const std::string& value) {
    if (value == "text") return ProductValueType::Text;
    if (value == "integer") return ProductValueType::Integer;
    if (value == "media.video") return ProductValueType::MediaVideo;
    if (value == "media.audio") return ProductValueType::MediaAudio;
    if (value == "media.mp4") return ProductValueType::MediaMp4;
    fail("unsupported input type: " + value);
}

ProductRational parse_rational(const Json& value, const std::string& context) {
    require_exact_keys(value, {"denominator", "numerator"}, context);
    ProductRational result;
    result.numerator = positive_integer(
        required_field(value, "numerator", context), context + ".numerator");
    result.denominator = positive_integer(
        required_field(value, "denominator", context), context + ".denominator");
    return result;
}

ProductValidation parse_validation(const Json& value,
                                   const ProductValueType type,
                                   const std::string& context) {
    static const std::set<std::string> kValidationKeys = {
        "decodable", "fps", "maximum", "min_length", "minimum",
        "minimum_duration_ms", "parent_creatable_and_writable",
        "regular_file",
    };
    require_exact_keys(value, kValidationKeys, context);
    ProductValidation result;
    for (const auto& [name, field] : value.object()) {
        if (name == "min_length")
            result.min_length = nonnegative_integer(field, context + "." + name);
        else if (name == "minimum")
            result.minimum = decimal_uint64(field, context + "." + name);
        else if (name == "maximum")
            result.maximum = decimal_uint64(field, context + "." + name);
        else if (name == "regular_file") {
            if (!field.is_bool() || !field.boolean())
                fail(context + ".regular_file must be true when declared");
            result.regular_file = true;
        } else if (name == "decodable") {
            if (!field.is_bool() || !field.boolean())
                fail(context + ".decodable must be true when declared");
            result.decodable = true;
        } else if (name == "fps")
            result.fps = parse_rational(field, context + ".fps");
        else if (name == "minimum_duration_ms")
            result.minimum_duration_ms = positive_integer(
                field, context + ".minimum_duration_ms");
        else if (name == "parent_creatable_and_writable") {
            if (!field.is_bool() || !field.boolean())
                fail(context +
                     ".parent_creatable_and_writable must be true when declared");
            result.parent_creatable_and_writable = true;
        }
    }
    if (result.minimum && result.maximum && *result.minimum > *result.maximum)
        fail(context + " minimum exceeds maximum");

    const auto reject = [&](const bool condition, const std::string& name) {
        if (condition) fail(context + "." + name + " is invalid for type " +
                            product_value_type_name(type));
    };
    reject(result.min_length.has_value() && type != ProductValueType::Text,
           "min_length");
    reject((result.minimum.has_value() || result.maximum.has_value()) &&
               type != ProductValueType::Integer,
           "integer bound");
    reject((result.regular_file.has_value() || result.decodable.has_value()) &&
               type != ProductValueType::MediaVideo &&
               type != ProductValueType::MediaAudio,
           "media file validation");
    reject(result.fps.has_value() && type != ProductValueType::MediaVideo,
           "fps");
    reject(result.minimum_duration_ms.has_value() &&
               type != ProductValueType::MediaAudio,
           "minimum_duration_ms");
    reject(result.parent_creatable_and_writable.has_value() &&
               type != ProductValueType::MediaMp4,
           "parent_creatable_and_writable");
    return result;
}

ProductDefaultValue parse_default(const Json& value,
                                  const ProductValueType type,
                                  const std::string& context) {
    if (type == ProductValueType::Integer)
        return decimal_uint64(value, context);
    if (type == ProductValueType::Text ||
        type == ProductValueType::MediaVideo ||
        type == ProductValueType::MediaAudio ||
        type == ProductValueType::MediaMp4) {
        if (!value.is_string() || value.string().empty() ||
            value.string().find('\0') != std::string::npos)
            fail(context + " must be a non-empty string without NUL");
        return value.string();
    }
    fail(context + " has unsupported default type");
}

ProductFieldDeclaration parse_declaration(const Json& value,
                                          const std::string& group,
                                          const size_t index) {
    const std::string context = group + "[" + std::to_string(index) + "]";
    require_exact_keys(value, {"default", "name", "required", "type", "validation"},
                       context);
    ProductFieldDeclaration result;
    result.name = required_string(value, "name", context);
    if (!safe_name(result.name)) fail(context + ".name is not a safe semantic name");
    result.type = parse_type(required_string(value, "type", context));
    result.required = required_bool(value, "required", context);
    if (const Json* validation = value.find("validation"); validation != nullptr)
        result.validation = parse_validation(
            *validation, result.type, context + ".validation");
    if (const Json* default_value = value.find("default"); default_value != nullptr) {
        if (result.required) fail(context + " required declarations cannot have defaults");
        result.default_value = parse_default(
            *default_value, result.type, context + ".default");
    }

    if (group == "inputs" &&
        result.type != ProductValueType::Text &&
        result.type != ProductValueType::MediaVideo &&
        result.type != ProductValueType::MediaAudio)
        fail(context + " uses a type not admitted for inputs");
    if (group == "parameters" && result.type != ProductValueType::Integer)
        fail(context + " uses a type not admitted for parameters");
    if (group == "outputs" && result.type != ProductValueType::MediaMp4)
        fail(context + " uses a type not admitted for outputs");

    if (result.type == ProductValueType::Integer && result.has_default()) {
        const uint64_t default_integer = std::get<uint64_t>(result.default_value);
        if ((result.validation.minimum &&
             default_integer < *result.validation.minimum) ||
            (result.validation.maximum &&
             default_integer > *result.validation.maximum))
            fail(context + ".default is outside its declared integer bounds");
    }
    if (result.type == ProductValueType::Text && result.has_default() &&
        result.validation.min_length &&
        std::get<std::string>(result.default_value).size() <
            *result.validation.min_length)
        fail(context + ".default is shorter than min_length");
    return result;
}

std::vector<ProductFieldDeclaration> parse_group(const Json& root,
                                                 const std::string& name,
                                                 std::set<std::string>& names) {
    const Json& value = required_field(root, name, "input_schema");
    if (!value.is_array()) fail("input_schema." + name + " must be an array");
    std::vector<ProductFieldDeclaration> result;
    for (size_t index = 0; index < value.array().size(); ++index) {
        ProductFieldDeclaration declaration =
            parse_declaration(value.array()[index], name, index);
        if (!names.insert(declaration.name).second)
            fail("field name is duplicated across Product schema groups: " +
                 declaration.name);
        result.push_back(std::move(declaration));
    }
    return result;
}

template <typename T>
const ProductFieldDeclaration* find_declaration(
    const T& values, const std::string& name) {
    const auto found = std::find_if(
        values.begin(), values.end(), [&](const ProductFieldDeclaration& value) {
            return value.name == name;
        });
    return found == values.end() ? nullptr : &*found;
}

void require_declaration(const ProductFieldDeclaration* declaration,
                         const ProductValueType type, const bool required,
                         const std::string& context) {
    if (declaration == nullptr || declaration->type != type ||
        declaration->required != required)
        fail(context + " declaration does not match the bounded Product contract");
}

void require_common_seed_and_output(const ProductInputSchema& schema) {
    if (schema.parameters.size() != 1 || schema.outputs.size() != 1)
        fail("current Product contracts require exactly seed and output declarations");
    const ProductFieldDeclaration* seed = schema.find_parameter("seed");
    require_declaration(seed, ProductValueType::Integer, false, "seed");
    if (!seed->has_default() || !seed->validation.minimum ||
        !seed->validation.maximum || *seed->validation.minimum != 0 ||
        *seed->validation.maximum != std::numeric_limits<uint64_t>::max())
        fail("seed must declare an optional uint64 default and complete bounds");
    const ProductFieldDeclaration* output = schema.find_output("output");
    require_declaration(output, ProductValueType::MediaMp4, false, "output");
    if (!output->has_default() ||
        std::get<std::string>(output->default_value) != "output.mp4" ||
        output->validation.parent_creatable_and_writable != true)
        fail("output must be optional with writable destination default output.mp4");
}

uint64_t json_uint(const Json& object, const std::string& name,
                   const std::string& context) {
    return nonnegative_integer(required_field(object, name, context),
                               context + "." + name);
}

double json_number(const Json& object, const std::string& name,
                   const std::string& context) {
    return finite_number(required_field(object, name, context),
                         context + "." + name);
}

void require_equal(const uint64_t left, const uint64_t right,
                   const std::string& context) {
    if (left != right) fail(context + " contradicts Product contract");
}

void require_equal(const double left, const double right,
                   const std::string& context) {
    if (left != right) fail(context + " contradicts Product contract");
}

}  // namespace

const char* product_value_type_name(const ProductValueType type) {
    switch (type) {
        case ProductValueType::Text: return "text";
        case ProductValueType::Integer: return "integer";
        case ProductValueType::MediaVideo: return "media.video";
        case ProductValueType::MediaAudio: return "media.audio";
        case ProductValueType::MediaMp4: return "media.mp4";
    }
    return "unknown";
}

const ProductFieldDeclaration* ProductInputSchema::find_input(
        const std::string& name) const {
    return find_declaration(inputs, name);
}

const ProductFieldDeclaration* ProductInputSchema::find_parameter(
        const std::string& name) const {
    return find_declaration(parameters, name);
}

const ProductFieldDeclaration* ProductInputSchema::find_output(
        const std::string& name) const {
    return find_declaration(outputs, name);
}

ProductInputSchema parse_product_input_schema(const Json& value) {
    require_exact_keys(value, {"inputs", "outputs", "parameters", "schema"},
                       "input_schema");
    ProductInputSchema result;
    result.identity = required_string(value, "schema", "input_schema");
    if (result.identity != kProductInputSchemaV1)
        fail("unsupported schema identity: " + result.identity);
    std::set<std::string> names;
    result.inputs = parse_group(value, "inputs", names);
    result.parameters = parse_group(value, "parameters", names);
    result.outputs = parse_group(value, "outputs", names);
    return result;
}

ProductFrozenProfile parse_product_frozen_profile(const Json& value) {
    require_exact_keys(value, {"output", "sampling", "temporal"},
                       "frozen_profile");
    ProductFrozenProfile result;
    const Json& output = required_field(value, "output", "frozen_profile");
    require_exact_keys(output,
        {"audio", "duration", "fps", "frames", "height", "width"},
        "frozen_profile.output");
    if (const Json* width = output.find("width"))
        result.output.width = positive_integer(*width, "frozen_profile.output.width");
    if (const Json* height = output.find("height"))
        result.output.height = positive_integer(*height, "frozen_profile.output.height");
    if (const Json* frames = output.find("frames"))
        result.output.frames = positive_integer(*frames, "frozen_profile.output.frames");
    result.output.fps = parse_rational(
        required_field(output, "fps", "frozen_profile.output"),
        "frozen_profile.output.fps");
    result.output.duration = required_string(
        output, "duration", "frozen_profile.output");
    if (result.output.duration != "fixed" &&
        result.output.duration != "audio_derived")
        fail("frozen_profile.output.duration is unsupported");
    result.output.audio = required_string(output, "audio", "frozen_profile.output");
    if (result.output.audio != "none" &&
        result.output.audio != "driving_audio")
        fail("frozen_profile.output.audio is unsupported");

    if (const Json* sampling = value.find("sampling")) {
        require_exact_keys(*sampling,
            {"eta", "guidance_scale", "method", "prediction", "steps"},
            "frozen_profile.sampling");
        ProductFrozenSampling parsed;
        if (const Json* method = sampling->find("method")) {
            if (!method->is_string() || method->string().empty())
                fail("frozen_profile.sampling.method must be a non-empty string");
            parsed.method = method->string();
        }
        if (const Json* prediction = sampling->find("prediction")) {
            if (!prediction->is_string() || prediction->string().empty())
                fail("frozen_profile.sampling.prediction must be a non-empty string");
            parsed.prediction = prediction->string();
        }
        if (const Json* steps = sampling->find("steps"))
            parsed.steps = positive_integer(*steps, "frozen_profile.sampling.steps");
        if (const Json* guidance = sampling->find("guidance_scale")) {
            parsed.guidance_scale = finite_number(
                *guidance, "frozen_profile.sampling.guidance_scale");
            if (*parsed.guidance_scale < 0)
                fail("frozen_profile.sampling.guidance_scale cannot be negative");
        }
        if (const Json* eta = sampling->find("eta")) {
            parsed.eta = finite_number(*eta, "frozen_profile.sampling.eta");
            if (*parsed.eta < 0)
                fail("frozen_profile.sampling.eta cannot be negative");
        }
        result.sampling = std::move(parsed);
    }
    if (const Json* temporal = value.find("temporal")) {
        require_exact_keys(*temporal, {"chunk_frames"},
                           "frozen_profile.temporal");
        ProductFrozenTemporal parsed;
        parsed.chunk_frames = positive_integer(
            required_field(*temporal, "chunk_frames", "frozen_profile.temporal"),
            "frozen_profile.temporal.chunk_frames");
        result.temporal = parsed;
    }
    return result;
}

void validate_product_contract_for_family(
        const std::string& family,
        const std::string& workflow_identity,
        const ProductInputSchema& schema,
        const ProductFrozenProfile& frozen_profile) {
    require_common_seed_and_output(schema);
    if (family == "text_to_video") {
        if (schema.inputs.size() != 1)
            fail("text_to_video requires exactly one Product input");
        const ProductFieldDeclaration* prompt = schema.find_input("prompt");
        require_declaration(prompt, ProductValueType::Text, true, "prompt");
        if (prompt->validation.min_length != 1)
            fail("prompt must declare min_length 1");
        if (!frozen_profile.output.width || !frozen_profile.output.height ||
            !frozen_profile.output.frames ||
            frozen_profile.output.fps.denominator != 1 ||
            frozen_profile.output.duration != "fixed" ||
            frozen_profile.output.audio != "none" ||
            !frozen_profile.sampling || !frozen_profile.sampling->steps ||
            !frozen_profile.sampling->guidance_scale ||
            frozen_profile.temporal)
            fail("text_to_video frozen profile is incomplete or over-broad");
        return;
    }
    if (family != "lip_sync") fail("unsupported Product family: " + family);
    if (schema.inputs.size() != 2)
        fail("lip_sync requires exactly video and audio Product inputs");
    const ProductFieldDeclaration* video = schema.find_input("video");
    const ProductFieldDeclaration* audio = schema.find_input("audio");
    require_declaration(video, ProductValueType::MediaVideo, true, "video");
    require_declaration(audio, ProductValueType::MediaAudio, true, "audio");
    if (video->validation.regular_file != true ||
        video->validation.decodable != true ||
        video->validation.fps != ProductRational{25, 1})
        fail("lip_sync video must declare regular decodable exact 25/1 FPS media");
    if (audio->validation.regular_file != true ||
        audio->validation.decodable != true ||
        audio->validation.minimum_duration_ms != 40)
        fail("lip_sync audio must declare regular decodable 40 ms minimum media");
    if (frozen_profile.output.width || frozen_profile.output.height ||
        frozen_profile.output.frames ||
        frozen_profile.output.fps != ProductRational{25, 1} ||
        frozen_profile.output.duration != "audio_derived" ||
        frozen_profile.output.audio != "driving_audio")
        fail("lip_sync frozen output profile is invalid");
    if (workflow_identity == "lip_sync_workflow_v1") {
        if (frozen_profile.sampling || frozen_profile.temporal)
            fail("lip_sync_workflow_v1 does not expose sampling or temporal profile");
    } else if (workflow_identity == "lip_sync_diffusion_workflow_v1") {
        if (!frozen_profile.sampling || !frozen_profile.sampling->method ||
            !frozen_profile.sampling->prediction ||
            !frozen_profile.sampling->steps ||
            !frozen_profile.sampling->guidance_scale ||
            !frozen_profile.sampling->eta ||
            !frozen_profile.temporal || !frozen_profile.temporal->chunk_frames)
            fail("lip_sync diffusion frozen sampling profile is incomplete");
    } else {
        fail("unsupported lip_sync workflow identity: " + workflow_identity);
    }
}

void validate_product_execution_consistency(
        const std::string& family,
        const std::string& workflow_identity,
        const ProductInputSchema& schema,
        const ProductFrozenProfile& frozen_profile,
        const Json& execution,
        const Json* workflow) {
    validate_product_contract_for_family(
        family, workflow_identity, schema, frozen_profile);
    require_object(execution, "execution");
    const ProductFieldDeclaration* seed = schema.find_parameter("seed");
    const uint64_t schema_seed = std::get<uint64_t>(seed->default_value);
    if (family == "text_to_video") {
        const Json& inputs = required_field(execution, "inputs", "execution");
        const Json& run = required_field(execution, "run", "execution");
        require_object(inputs, "execution.inputs");
        require_object(run, "execution.run");
        require_equal(json_uint(inputs, "width", "execution.inputs"),
                      *frozen_profile.output.width, "execution width");
        require_equal(json_uint(inputs, "height", "execution.inputs"),
                      *frozen_profile.output.height, "execution height");
        require_equal(json_uint(inputs, "frames", "execution.inputs"),
                      *frozen_profile.output.frames, "execution frames");
        require_equal(json_uint(inputs, "fps", "execution.inputs"),
                      frozen_profile.output.fps.numerator /
                          frozen_profile.output.fps.denominator,
                      "execution FPS");
        require_equal(json_uint(inputs, "steps", "execution.inputs"),
                      *frozen_profile.sampling->steps, "execution steps");
        require_equal(json_number(inputs, "cfg", "execution.inputs"),
                      *frozen_profile.sampling->guidance_scale,
                      "execution guidance");
        require_equal(json_uint(run, "default_seed", "execution.run"),
                      schema_seed, "execution seed default");
        return;
    }

    if (required_string(execution, "product_family", "execution") != family ||
        required_string(execution, "workflow_identity", "execution") !=
            workflow_identity)
        fail("lip_sync execution identity contradicts Product contract");
    require_equal(json_uint(execution, "default_seed", "execution"),
                  schema_seed, "execution seed default");
    const Json& execution_inputs = required_field(execution, "inputs", "execution");
    require_object(execution_inputs, "execution.inputs");
    for (const auto& [name, type, required] : std::array{
             std::tuple{"video", "media.video", true},
             std::tuple{"audio", "media.audio", true},
             std::tuple{"output", "media.mp4", false}}) {
        const Json& declaration = required_field(execution_inputs, name,
                                                 "execution.inputs");
        require_object(declaration, "execution.inputs." + std::string(name));
        if (required_string(declaration, "type", "execution.inputs." +
                            std::string(name)) != type ||
            required_bool(declaration, "required", "execution.inputs." +
                          std::string(name)) != required)
            fail("execution input contradicts Product declaration: " +
                 std::string(name));
    }
    if (workflow == nullptr) fail("lip_sync consistency requires workflow metadata");
    const Json& profile = required_field(*workflow, "profile", "workflow");
    require_equal(json_uint(profile, "fps", "workflow.profile"),
                  frozen_profile.output.fps.numerator /
                      frozen_profile.output.fps.denominator,
                  "workflow FPS");
    if (workflow_identity == "lip_sync_diffusion_workflow_v1") {
        const Json& sampling = required_field(profile, "sampling", "workflow.profile");
        if (*frozen_profile.sampling->method != "ddim" ||
            required_string(sampling, "scheduler", "workflow.profile.sampling") !=
                "ddim_scaled_linear_alpha_cumprod" ||
            required_string(sampling, "prediction", "workflow.profile.sampling") !=
                *frozen_profile.sampling->prediction)
            fail("workflow sampling identity contradicts frozen profile");
        require_equal(json_uint(sampling, "steps", "workflow.profile.sampling"),
                      *frozen_profile.sampling->steps, "workflow sampling steps");
        require_equal(json_number(sampling, "guidance", "workflow.profile.sampling"),
                      *frozen_profile.sampling->guidance_scale,
                      "workflow sampling guidance");
        require_equal(json_number(sampling, "eta", "workflow.profile.sampling"),
                      *frozen_profile.sampling->eta, "workflow sampling eta");
        require_equal(json_uint(profile, "temporal_chunk", "workflow.profile"),
                      *frozen_profile.temporal->chunk_frames,
                      "workflow temporal chunk");
    }
}

uint64_t resolve_product_seed(const ProductInputSchema* schema,
                              const std::optional<uint64_t> requested,
                              const uint64_t legacy_execution_default) {
    if (requested) return *requested;
    if (schema == nullptr) return legacy_execution_default;
    const ProductFieldDeclaration* seed = schema->find_parameter("seed");
    if (seed == nullptr || !seed->has_default() ||
        !std::holds_alternative<uint64_t>(seed->default_value))
        fail("seed default is missing from canonical Product contract");
    const uint64_t canonical = std::get<uint64_t>(seed->default_value);
    if (canonical != legacy_execution_default)
        fail("execution seed default contradicts canonical Product contract");
    return canonical;
}

std::string resolve_product_output(const ProductInputSchema* schema,
                                   const std::string& requested,
                                   const std::string& legacy_default) {
    if (!requested.empty()) return requested;
    if (schema == nullptr) return legacy_default;
    const ProductFieldDeclaration* output = schema->find_output("output");
    if (output == nullptr || !output->has_default() ||
        !std::holds_alternative<std::string>(output->default_value))
        fail("output default is missing from canonical Product contract");
    return std::get<std::string>(output->default_value);
}

}  // namespace vrhino::product
