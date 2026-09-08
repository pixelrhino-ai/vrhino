#include "vrhino/json.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string_view>

#include "vrhino/error.h"

namespace vrhino {

void require(bool condition, const std::string& message) {
    if (!condition) throw Error(message);
}

namespace {

bool valid_utf8(std::string_view value);

JsonParseLimits unbounded_parse_limits() {
    const size_t maximum = std::numeric_limits<size_t>::max();
    return JsonParseLimits{maximum, maximum, maximum, maximum, maximum};
}

class Parser {
public:
    Parser(const std::string& text, JsonParseLimits limits)
        : text_(text), limits_(limits) {}

    Json parse() {
        require(text_.size() <= limits_.maximum_document_bytes,
                "JSON document exceeds parser byte limit");
        require(valid_utf8(text_), "JSON document is not valid UTF-8");
        require(limits_.maximum_depth > 0 &&
                    limits_.maximum_container_entries > 0 &&
                    limits_.maximum_total_values > 0 &&
                    limits_.maximum_string_bytes > 0,
                "JSON parser limits must be positive");
        Json result = value(1);
        whitespace();
        require(position_ == text_.size(), "JSON has trailing data");
        return result;
    }

private:
    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }

    char peek() {
        whitespace();
        require(position_ < text_.size(), "Unexpected end of JSON");
        return text_[position_];
    }

    bool consume(char value) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == value) {
            ++position_;
            return true;
        }
        return false;
    }

    void literal(const char* value) {
        while (*value) {
            require(position_ < text_.size() && text_[position_] == *value,
                    "Invalid JSON literal");
            ++position_;
            ++value;
        }
    }

    Json value(const size_t depth) {
        require(depth <= limits_.maximum_depth,
                "JSON nesting exceeds parser depth limit");
        require(++total_values_ <= limits_.maximum_total_values,
                "JSON value count exceeds parser limit");
        const char current = peek();
        if (current == 'n') { literal("null"); return Json(); }
        if (current == 't') { literal("true"); return Json(Json::Value(true)); }
        if (current == 'f') { literal("false"); return Json(Json::Value(false)); }
        if (current == '"') return Json(Json::Value(string()));
        if (current == '[') return array(depth);
        if (current == '{') return object(depth);
        return number();
    }

    static uint32_t hex(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        throw Error("Invalid JSON unicode escape");
    }

    static void append_utf8(std::string& output, uint32_t codepoint) {
        if (codepoint <= 0x7f) output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    uint32_t unicode_code_unit() {
        require(position_ + 4 <= text_.size(),
                "Truncated JSON unicode escape");
        uint32_t result = 0;
        for (int index = 0; index < 4; ++index)
            result = (result << 4) | hex(text_[position_++]);
        return result;
    }

    void require_string_bound(const std::string& value) const {
        require(value.size() <= limits_.maximum_string_bytes,
                "JSON string exceeds parser byte limit");
    }

    std::string string() {
        require(consume('"'), "Expected JSON string");
        std::string output;
        while (position_ < text_.size()) {
            char current = text_[position_++];
            if (current == '"') return output;
            require(static_cast<unsigned char>(current) >= 0x20, "Control byte in JSON string");
            if (current != '\\') {
                output.push_back(current);
                require_string_bound(output);
                continue;
            }
            require(position_ < text_.size(), "Truncated JSON escape");
            current = text_[position_++];
            switch (current) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = unicode_code_unit();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        require(position_ + 2 <= text_.size() &&
                                    text_[position_] == '\\' &&
                                    text_[position_ + 1] == 'u',
                                "JSON high surrogate has no low surrogate");
                        position_ += 2;
                        const uint32_t low = unicode_code_unit();
                        require(low >= 0xdc00 && low <= 0xdfff,
                                "JSON high surrogate has invalid low surrogate");
                        codepoint = 0x10000 +
                            ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else {
                        require(codepoint < 0xdc00 || codepoint > 0xdfff,
                                "JSON contains an unpaired low surrogate");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: throw Error("Unknown JSON escape");
            }
            require_string_bound(output);
        }
        throw Error("Unterminated JSON string");
    }

    Json number() {
        whitespace();
        const size_t begin = position_;
        if (position_ < text_.size() && text_[position_] == '-') ++position_;
        require(position_ < text_.size(), "Truncated JSON number");
        if (text_[position_] == '0') ++position_;
        else {
            require(text_[position_] >= '1' && text_[position_] <= '9', "Invalid JSON number");
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        bool floating = false;
        if (position_ < text_.size() && text_[position_] == '.') {
            floating = true; ++position_;
            require(position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])), "Invalid JSON fraction");
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            floating = true; ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            require(position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])), "Invalid JSON exponent");
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        const std::string token = text_.substr(begin, position_ - begin);
        if (!floating) {
            int64_t result = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), result);
            require(parsed.ec == std::errc(), "JSON integer overflow");
            return Json(Json::Value(result));
        }
        char* end = nullptr;
        const double result = std::strtod(token.c_str(), &end);
        require(end == token.c_str() + token.size() && std::isfinite(result), "Invalid JSON float");
        return Json(Json::Value(result));
    }

    Json array(const size_t depth) {
        require(consume('['), "Expected JSON array");
        Json::Array output;
        if (consume(']')) return Json(Json::Value(std::move(output)));
        do {
            require(output.size() < limits_.maximum_container_entries,
                    "JSON array exceeds parser entry limit");
            output.push_back(value(depth + 1));
        } while (consume(','));
        require(consume(']'), "Unterminated JSON array");
        return Json(Json::Value(std::move(output)));
    }

    Json object(const size_t depth) {
        require(consume('{'), "Expected JSON object");
        Json::Object output;
        if (consume('}')) return Json(Json::Value(std::move(output)));
        do {
            require(output.size() < limits_.maximum_container_entries,
                    "JSON object exceeds parser entry limit");
            require(peek() == '"', "JSON object key is not a string");
            std::string key = string();
            require(consume(':'), "JSON object missing colon");
            require(output.emplace(std::move(key), value(depth + 1)).second,
                    "Duplicate JSON object key");
        } while (consume(','));
        require(consume('}'), "Unterminated JSON object");
        return Json(Json::Value(std::move(output)));
    }

    const std::string& text_;
    JsonParseLimits limits_;
    size_t position_ = 0;
    size_t total_values_ = 0;
};

bool valid_utf8(const std::string_view value) {
    for (size_t index = 0; index < value.size();) {
        const uint8_t first = static_cast<uint8_t>(value[index]);
        if (first <= 0x7f) {
            ++index;
            continue;
        }
        size_t continuation_count = 0;
        uint8_t second_minimum = 0x80;
        uint8_t second_maximum = 0xbf;
        if (first >= 0xc2 && first <= 0xdf) {
            continuation_count = 1;
        } else if (first == 0xe0) {
            continuation_count = 2;
            second_minimum = 0xa0;
        } else if (first >= 0xe1 && first <= 0xec) {
            continuation_count = 2;
        } else if (first == 0xed) {
            continuation_count = 2;
            second_maximum = 0x9f;
        } else if (first >= 0xee && first <= 0xef) {
            continuation_count = 2;
        } else if (first == 0xf0) {
            continuation_count = 3;
            second_minimum = 0x90;
        } else if (first >= 0xf1 && first <= 0xf3) {
            continuation_count = 3;
        } else if (first == 0xf4) {
            continuation_count = 3;
            second_maximum = 0x8f;
        } else {
            return false;
        }
        if (index + continuation_count >= value.size()) return false;
        const uint8_t second = static_cast<uint8_t>(value[index + 1]);
        if (second < second_minimum || second > second_maximum) return false;
        for (size_t offset = 2; offset <= continuation_count; ++offset) {
            const uint8_t byte = static_cast<uint8_t>(value[index + offset]);
            if (byte < 0x80 || byte > 0xbf) return false;
        }
        index += continuation_count + 1;
    }
    return true;
}

void append_json_string(std::string& output, const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    require(valid_utf8(value), "Cannot serialize invalid UTF-8 JSON string");
    output.push_back('"');
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character < 0x20) {
                    output += "\\u00";
                    output.push_back(digits[character >> 4]);
                    output.push_back(digits[character & 0x0f]);
                } else {
                    output.push_back(static_cast<char>(character));
                }
        }
    }
    output.push_back('"');
}

void append_serialized_json(std::string& output, const Json& value) {
    if (value.is_null()) { output += "null"; return; }
    if (value.is_bool()) { output += value.boolean() ? "true" : "false"; return; }
    if (value.is_int()) { output += std::to_string(value.integer()); return; }
    if (value.is_number()) {
        const double number = value.number();
        require(std::isfinite(number), "Cannot serialize non-finite JSON number");
        char buffer[64];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), number);
        require(converted.ec == std::errc(), "Cannot serialize JSON number");
        const size_t begin = output.size();
        output.append(buffer, converted.ptr);
        if (std::floor(number) == number &&
            output.find_first_of(".eE", begin) == std::string::npos)
            output += ".0";
        return;
    }
    if (value.is_string()) { append_json_string(output, value.string()); return; }
    if (value.is_array()) {
        output.push_back('[');
        bool first = true;
        for (const Json& item : value.array()) {
            if (!first) output.push_back(',');
            first = false;
            append_serialized_json(output, item);
        }
        output.push_back(']');
        return;
    }
    output.push_back('{');
    bool first = true;
    for (const auto& [key, item] : value.object()) {
        if (!first) output.push_back(',');
        first = false;
        append_json_string(output, key);
        output.push_back(':');
        append_serialized_json(output, item);
    }
    output.push_back('}');
}

}  // namespace

Json Json::parse(const std::string& text) {
    return Parser(text, unbounded_parse_limits()).parse();
}
Json Json::parse(const std::string& text, const JsonParseLimits& limits) {
    return Parser(text, limits).parse();
}
bool Json::is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool Json::is_bool() const { return std::holds_alternative<bool>(value_); }
bool Json::is_int() const { return std::holds_alternative<int64_t>(value_); }
bool Json::is_number() const { return is_int() || std::holds_alternative<double>(value_); }
bool Json::is_string() const { return std::holds_alternative<std::string>(value_); }
bool Json::is_array() const { return std::holds_alternative<Array>(value_); }
bool Json::is_object() const { return std::holds_alternative<Object>(value_); }
bool Json::boolean() const { require(is_bool(), "JSON value is not bool"); return std::get<bool>(value_); }
int64_t Json::integer() const { require(is_int(), "JSON value is not integer"); return std::get<int64_t>(value_); }
double Json::number() const { require(is_number(), "JSON value is not number"); return is_int() ? static_cast<double>(integer()) : std::get<double>(value_); }
const std::string& Json::string() const { require(is_string(), "JSON value is not string"); return std::get<std::string>(value_); }
const Json::Array& Json::array() const { require(is_array(), "JSON value is not array"); return std::get<Array>(value_); }
const Json::Object& Json::object() const { require(is_object(), "JSON value is not object"); return std::get<Object>(value_); }
const Json& Json::at(const std::string& key) const {
    const auto& values = object();
    const auto found = values.find(key);
    require(found != values.end(), "Missing JSON key: " + key);
    return found->second;
}
const Json* Json::find(const std::string& key) const {
    if (!is_object()) return nullptr;
    const auto found = object().find(key);
    return found == object().end() ? nullptr : &found->second;
}
std::string Json::serialize() const {
    std::string output;
    append_serialized_json(output, *this);
    return output;
}

}  // namespace vrhino
