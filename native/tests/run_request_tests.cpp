#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/run_request.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void rejects(Operation&& operation, const product::RunRequestErrorCode code,
             const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": input unexpectedly succeeded");
    } catch (const product::RunRequestError& error) {
        require_test(error.code() == code, context + ": wrong error code");
    }
}

product::ResolvedRunnableModel model(const fs::path& manifest) {
    product::ResolvedRunnableModel result;
    result.manifest = product::load_model_package_manifest(manifest);
    return result;
}

product::ProductRunDocument request(const std::string& text) {
    const vrhino::JsonParseLimits limits{
        64 * 1024, 32, 1024, 4096, 32 * 1024};
    return product::parse_product_run_document(
        vrhino::Json::parse(text, limits));
}

}  // namespace

int main() {
    const fs::path temporary = fs::temp_directory_path() /
        ("vrhino-run-request-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(temporary);
        const fs::path video = temporary / "input.mp4";
        const fs::path audio = temporary / "input.wav";
        std::ofstream(video).put('v');
        std::ofstream(audio).put('a');
        const fs::path specs = VRHINO_TEST_SOURCE_ROOT;

        struct Fixture {
            const char* manifest;
            uint64_t seed;
            bool lip_sync;
        };
        const std::vector<Fixture> fixtures = {
            {"ltx_v0_9_1/successors/1.1.1/vrhino-model.json", 5703, false},
            {"wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json", 5701, false},
            {"mochi_1_preview/successors/1.0.1/vrhino-model.json", 11001, false},
            {"public_musetalk_v15/successors/1.0.1/vrhino-model.json", 11001, true},
            {"public_latentsync_16/successors/1.0.1/vrhino-model.json", 1247, true},
        };
        for (const Fixture& fixture : fixtures) {
            const product::ResolvedRunnableModel runnable =
                model(specs / fixture.manifest);
            const std::string body = fixture.lip_sync
                ? "{\"model\":\"" + runnable.manifest.identity.reference() +
                      "\",\"inputs\":{\"video\":\"" + video.string() +
                      "\",\"audio\":\"" + audio.string() + "\"}}"
                : "{\"model\":\"" + runnable.manifest.identity.reference() +
                      "\",\"inputs\":{\"prompt\":\"a rhino\"}}";
            const product::RunOptions options =
                product::map_product_run_options(runnable, request(body));
            require_test(options.seed == fixture.seed &&
                             options.output.empty() && !options.overwrite &&
                             (fixture.lip_sync
                                  ? (!options.video.empty() && !options.audio.empty() &&
                                     options.prompt.empty())
                                  : (options.prompt == "a rhino" &&
                                     options.video.empty() && options.audio.empty())),
                         "successor defaults/request mapping drift: " +
                             runnable.manifest.identity.reference());
        }

        const product::ResolvedRunnableModel ltx = model(
            specs / "ltx_v0_9_1/successors/1.1.1/vrhino-model.json");
        product::RunOptions exact = product::map_product_run_options(
            ltx, request("{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\","
                         "\"inputs\":{\"prompt\":\"x\"},"
                         "\"parameters\":{\"seed\":\"18446744073709551615\"}}"));
        require_test(exact.seed == std::numeric_limits<uint64_t>::max(),
                     "full uint64 seed was not preserved");
        exact = product::map_product_run_options(
            ltx, request("{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\","
                         "\"inputs\":{\"prompt\":\"x\"},"
                         "\"parameters\":{\"seed\":9007199254740993}}"));
        require_test(exact.seed == 9007199254740993ULL,
                     "integer above 2^53 was rounded");

        for (const char* body : {
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\"}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\",\"unknown\":1}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":1}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\"},\"parameters\":{\"seed\":-1}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\"},\"parameters\":{\"seed\":\"18446744073709551616\"}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\\u0000y\"}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\"},\"outputs\":{\"output\":\"relative.mp4\"}}",
                 "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\",\"inputs\":{\"prompt\":\"x\"},\"extra\":{}}"}) {
            rejects([&] {
                (void)product::map_product_run_options(ltx, request(body));
            }, product::RunRequestErrorCode::InvalidRequest,
            "invalid TTV request");
        }

        const product::ResolvedRunnableModel lip = model(
            specs /
            "public_musetalk_v15/successors/1.0.1/vrhino-model.json");
        for (const std::string& body : {
                 "{\"model\":\"vrhino/musetalk-v1.5:1.0.1\",\"inputs\":{\"audio\":\"" + audio.string() + "\"}}",
                 "{\"model\":\"vrhino/musetalk-v1.5:1.0.1\",\"inputs\":{\"video\":\"relative.mp4\",\"audio\":\"" + audio.string() + "\"}}",
                 "{\"model\":\"vrhino/musetalk-v1.5:1.0.1\",\"inputs\":{\"video\":\"https://example.invalid/input.mp4\",\"audio\":\"" + audio.string() + "\"}}",
                 "{\"model\":\"vrhino/musetalk-v1.5:1.0.1\",\"inputs\":{\"video\":\"data:video/mp4;base64,AAAA\",\"audio\":\"" + audio.string() + "\"}}",
                 "{\"model\":\"vrhino/musetalk-v1.5:1.0.1\",\"inputs\":{\"video\":\"" + (temporary / "missing.mp4").string() + "\",\"audio\":\"" + audio.string() + "\"}}"}) {
            rejects([&] {
                (void)product::map_product_run_options(lip, request(body));
            }, product::RunRequestErrorCode::InvalidRequest,
            "invalid lip-sync request");
        }

        const product::ResolvedRunnableModel legacy = model(
            specs / "wan2_1_t2v_1_3b/vrhino-model.json");
        rejects([&] {
            (void)product::map_product_run_options(
                legacy,
                request("{\"model\":\"vrhino/wan2.1-t2v-1.3b:1.0.0\","
                        "\"inputs\":{\"prompt\":\"x\"}}"));
        }, product::RunRequestErrorCode::ModelUnavailable,
        "legacy execution");

        // Bounded deterministic mutations jointly exercise JSON parsing,
        // envelope validation, and schema-driven Product mapping. Successful
        // mutations must retain the resolved Product identity and safe output
        // policy; rejected mutations must fail through a typed parser/request
        // exception rather than crash or hang.
        const std::string valid =
            "{\"model\":\"vrhino/ltx-video-v0.9.1:1.1.1\","
            "\"inputs\":{\"prompt\":\"a rhino\"},"
            "\"parameters\":{\"seed\":5703},\"outputs\":{}}";
        constexpr char mutation_bytes[] =
            "{}[],:\"\\0123456789-%Gabcdefghijklmnopqrstuvwxyz";
        uint64_t state = 0xbb67ae8584caa73bULL;
        for (size_t iteration = 0; iteration < 2048; ++iteration) {
            std::string candidate = valid;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            const size_t mutations = 1 + static_cast<size_t>(state % 6);
            for (size_t mutation = 0; mutation < mutations; ++mutation) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                const size_t position = static_cast<size_t>(
                    state % (candidate.size() + 1));
                const char byte = mutation_bytes[
                    state % (sizeof(mutation_bytes) - 1)];
                if ((state & 3U) == 0 && !candidate.empty()) {
                    candidate.erase(std::min(position, candidate.size() - 1), 1);
                } else if ((state & 3U) == 1 && candidate.size() < 2048) {
                    candidate.insert(candidate.begin() +
                                         static_cast<std::ptrdiff_t>(position),
                                     byte);
                } else if (!candidate.empty()) {
                    candidate[std::min(position, candidate.size() - 1)] = byte;
                }
            }
            try {
                const product::RunOptions options =
                    product::map_product_run_options(ltx, request(candidate));
                require_test(options.model_reference ==
                                 ltx.manifest.identity.reference() &&
                                 !options.overwrite,
                             "run-request fuzz escaped canonical mapping");
            } catch (const vrhino::Error&) {
            } catch (const product::RunRequestError&) {
            }
        }

        fs::remove_all(temporary);
        std::cout << "ProductInputSchema Native run request tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(temporary);
        std::cerr << "ProductInputSchema Native run request tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
