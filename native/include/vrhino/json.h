#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace vrhino {

struct JsonParseLimits {
    size_t maximum_document_bytes = 64 * 1024;
    size_t maximum_depth = 32;
    size_t maximum_container_entries = 1024;
    size_t maximum_total_values = 4096;
    size_t maximum_string_bytes = 32 * 1024;
};

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    using Value = std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object>;

    Json() : value_(nullptr) {}
    explicit Json(Value value) : value_(std::move(value)) {}

    static Json parse(const std::string& text);
    static Json parse(const std::string& text, const JsonParseLimits& limits);
    bool is_null() const;
    bool is_bool() const;
    bool is_int() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;
    bool boolean() const;
    int64_t integer() const;
    double number() const;
    const std::string& string() const;
    const Array& array() const;
    const Object& object() const;
    const Json& at(const std::string& key) const;
    const Json* find(const std::string& key) const;
    std::string serialize() const;

private:
    Value value_;
};

}  // namespace vrhino
