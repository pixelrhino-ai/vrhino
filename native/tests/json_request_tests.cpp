#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void rejects(Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": input unexpectedly succeeded");
    } catch (const vrhino::Error&) {
    }
}

}  // namespace

int main() {
    try {
        const vrhino::JsonParseLimits limits{
            64 * 1024, 32, 1024, 4096, 32 * 1024};
        const vrhino::Json escaped = vrhino::Json::parse(
            R"({"emoji":"\ud83d\ude80","escaped":"a\n\t\\\"\/b","utf8":"犀牛"})",
            limits);
        require_test(escaped.at("emoji").string() == "🚀" &&
                         escaped.at("escaped").string() == "a\n\t\\\"/b" &&
                         escaped.at("utf8").string() == "犀牛",
                     "JSON escape/Unicode decoding drift");

        rejects([&] { (void)vrhino::Json::parse(R"({"a":1,"a":2})", limits); },
                "duplicate key");
        rejects([&] { (void)vrhino::Json::parse("\"\\ud800\"", limits); },
                "unpaired high surrogate");
        rejects([&] { (void)vrhino::Json::parse("\"\\udc00\"", limits); },
                "unpaired low surrogate");
        rejects([&] { (void)vrhino::Json::parse("\"\\ud800\\u0041\"", limits); },
                "invalid surrogate pair");
        rejects([&] {
            std::string invalid = "\"";
            invalid.push_back(static_cast<char>(0xc0));
            invalid.push_back(static_cast<char>(0x80));
            invalid.push_back('"');
            (void)vrhino::Json::parse(invalid, limits);
        }, "invalid UTF-8");

        for (const std::string malformed : {
                 "01", "1.", "1e", "NaN", "Infinity", "1 trailing",
                 "9223372036854775808", "-9223372036854775809"}) {
            rejects([&] { (void)vrhino::Json::parse(malformed, limits); },
                    "malformed number " + malformed);
        }

        std::string nested = "0";
        for (size_t index = 0; index < 32; ++index)
            nested = "[" + nested + "]";
        rejects([&] { (void)vrhino::Json::parse(nested, limits); },
                "nesting depth");

        std::string array = "[";
        for (size_t index = 0; index < 1025; ++index) {
            if (index != 0) array.push_back(',');
            array.push_back('0');
        }
        array.push_back(']');
        rejects([&] { (void)vrhino::Json::parse(array, limits); },
                "container entries");

        vrhino::JsonParseLimits total_limits = limits;
        total_limits.maximum_container_entries = 4096;
        std::string values = "[";
        for (size_t index = 0; index < 4096; ++index) {
            if (index != 0) values.push_back(',');
            values.push_back('0');
        }
        values.push_back(']');
        rejects([&] { (void)vrhino::Json::parse(values, total_limits); },
                "total values");

        rejects([&] {
            (void)vrhino::Json::parse(
                "\"" + std::string(32 * 1024 + 1, 'x') + "\"", limits);
        }, "string bytes");
        rejects([&] {
            (void)vrhino::Json::parse(
                "\"" + std::string(64 * 1024, 'x') + "\"", limits);
        }, "document bytes");

        const vrhino::Json exact = vrhino::Json::parse(
            "9007199254740993", limits);
        require_test(exact.is_int() &&
                         exact.integer() == 9007199254740993LL,
                     "integer above 2^53 lost precision");

        require_test(vrhino::Json::parse("-9223372036854775808", limits)
                             .integer() == std::numeric_limits<int64_t>::min() &&
                         vrhino::Json::parse("9223372036854775807", limits)
                             .integer() == std::numeric_limits<int64_t>::max(),
                     "int64 boundary parsing drift");
        const std::string embedded_nul =
            vrhino::Json::parse(R"("a\u0000b")", limits).string();
        require_test(embedded_nul.size() == 3 && embedded_nul[1] == '\0',
                     "escaped JSON NUL was not represented safely");
        rejects([&] {
            (void)vrhino::Json::parse(
                R"({"outer":{"same":1,"same":2}})", limits);
        }, "nested duplicate key");

        // Deterministic bounded mutation corpus: a successful parse must
        // serialize and reparse byte-identically; every other mutation must
        // fail through the bounded parser rather than crash or escape limits.
        const std::vector<std::string> seeds = {
            R"({"model":"vrhino/example:1","inputs":{"prompt":"犀牛"}})",
            R"([null,true,false,-1,0,1,1.25,1e3,"\ud83d\ude80"])",
            R"({"nested":{"array":[1,2,3],"escaped":"a\n\\\"b"}})",
        };
        constexpr char mutation_bytes[] =
            "{}[],:\"\\0123456789eE+-%Gabcdefghijklmnopqrstuvwxyz";
        uint64_t state = 0x6a09e667f3bcc909ULL;
        for (size_t iteration = 0; iteration < 4096; ++iteration) {
            std::string candidate = seeds[iteration % seeds.size()];
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            const size_t mutations = 1 + static_cast<size_t>(state % 8);
            for (size_t mutation = 0; mutation < mutations; ++mutation) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                const char byte = mutation_bytes[
                    state % (sizeof(mutation_bytes) - 1)];
                const size_t position = candidate.empty()
                    ? 0 : static_cast<size_t>(state % (candidate.size() + 1));
                if ((state & 3U) == 0 && !candidate.empty()) {
                    candidate.erase(std::min(position, candidate.size() - 1), 1);
                } else if ((state & 3U) == 1 && candidate.size() < 4096) {
                    candidate.insert(candidate.begin() +
                                         static_cast<std::ptrdiff_t>(position),
                                     byte);
                } else if (!candidate.empty()) {
                    candidate[std::min(position, candidate.size() - 1)] = byte;
                }
            }
            try {
                const vrhino::Json parsed = vrhino::Json::parse(candidate, limits);
                const std::string serialized = parsed.serialize();
                require_test(vrhino::Json::parse(serialized, limits).serialize() ==
                                 serialized,
                             "JSON fuzz round-trip was not deterministic");
            } catch (const vrhino::Error&) {
            }
        }

        std::cout << "bounded HTTP JSON parser tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bounded HTTP JSON parser tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
