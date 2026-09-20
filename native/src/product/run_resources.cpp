#include "vrhino/product/run_resources.h"

#include <charconv>
#include <fstream>
#include <limits>

#include "vrhino/product/model_package.h"

namespace vrhino::product {
namespace {
[[noreturn]] void invalid(const std::string& message) {
    throw ModelPackageError(ModelPackageErrorCode::InvalidInput, message);
}
}  // namespace

RunResources parse_run_resources(const Json& declaration) {
    if (!declaration.is_object()) invalid("run resources must be an object");
    RunResources result;
    for (const auto& [key, value] : declaration.object()) {
        if (key != "weight_cache_budget_bytes")
            invalid("unknown run resource: " + key);
        uint64_t bytes = 0;
        if (value.is_int() && value.integer() > 0) {
            bytes = static_cast<uint64_t>(value.integer());
        } else if (value.is_string()) {
            const auto& text = value.string();
            if (text.empty() || text.front() < '1' || text.front() > '9')
                invalid("weight_cache_budget_bytes must be a positive canonical integer");
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), bytes);
            if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
                invalid("invalid weight_cache_budget_bytes");
        } else {
            invalid("weight_cache_budget_bytes must be a positive integer");
        }
        if (bytes == 0 || bytes > std::numeric_limits<size_t>::max())
            invalid("weight_cache_budget_bytes is outside the host size domain");
        result.weight_cache_budget_bytes = static_cast<size_t>(bytes);
    }
    return result;
}

RunResources load_run_resources(const std::filesystem::path& path) {
    constexpr size_t limit = 64 * 1024;
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        invalid("run resources must reference a regular local JSON file");
    std::ifstream input(path, std::ios::binary);
    if (!input) invalid("cannot open run resources");
    std::string text(limit + 1, '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<size_t>(input.gcount()));
    if (input.bad() || text.size() > limit) invalid("run resources unreadable or too large");
    return parse_run_resources(Json::parse(text, JsonParseLimits{}));
}

MemoryBudget constrain_run_memory(const MemoryBudget& declared,
                                  const RunResources& resources) {
    declared.validate();
    MemoryBudget result = declared;
    if (resources.weight_cache_budget_bytes) {
        const size_t cap = *resources.weight_cache_budget_bytes;
        if (cap == 0 || cap > declared.device_weight_budget_bytes())
            invalid("run weight cache budget must tighten the declared weight budget");
        result.weight_cache_budget_bytes = cap;
    }
    result.validate();
    return result;
}
}  // namespace vrhino::product
