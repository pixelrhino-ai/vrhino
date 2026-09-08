#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vrhino/api/server.h"
#include "vrhino/json.h"
#include "vrhino/product/info.h"
#include "vrhino/product/model_list.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/version.h"

namespace fs = std::filesystem;
namespace api = vrhino::api;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    require_test(static_cast<bool>(output), "cannot write API fixture");
}

product::PackageIdentity publish_fixture(const fs::path& cache_root,
                                         const fs::path& manifest_path) {
    const product::ModelPackageManifest manifest =
        product::load_model_package_manifest(manifest_path);
    const fs::path installed_manifest = cache_root / "models" /
        manifest.identity.name_space / manifest.identity.name /
        manifest.identity.version / product::kModelManifestName;
    write_text(installed_manifest, manifest.raw_json);
    for (const product::ArtifactDeclaration& artifact : manifest.artifacts) {
        if (!artifact.required) continue;
        const fs::path blob = cache_root / "blobs/sha256" /
                              artifact.sha256.substr(0, 2) / artifact.sha256;
        if (fs::exists(blob)) {
            require_test(fs::file_size(blob) == artifact.size,
                         "shared sparse fixture size mismatch");
            continue;
        }
        fs::create_directories(blob.parent_path());
        std::ofstream output(blob, std::ios::binary | std::ios::trunc);
        require_test(static_cast<bool>(output), "cannot create sparse CAS fixture");
        output.close();
        fs::resize_file(blob, artifact.size);
    }
    return manifest.identity;
}

uint16_t available_port() {
    const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require_test(socket_fd >= 0, "cannot create port-selection socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    require_test(::bind(socket_fd, reinterpret_cast<sockaddr*>(&address),
                        sizeof(address)) == 0,
                 "cannot bind port-selection socket");
    socklen_t size = sizeof(address);
    require_test(::getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address),
                               &size) == 0,
                 "cannot read selected port");
    const uint16_t result = ntohs(address.sin_port);
    ::close(socket_fd);
    return result;
}

struct HttpResponse {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
    std::string raw;
};

void send_all(const int socket_fd, const std::string& request) {
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t count = ::send(socket_fd, request.data() + sent,
                                     request.size() - sent, 0);
        require_test(count > 0, "raw HTTP request send failed");
        sent += static_cast<size_t>(count);
    }
}

HttpResponse raw_request(const uint16_t port, const std::string& request) {
    int socket_fd = -1;
    for (size_t attempt = 0; attempt < 100; ++attempt) {
        socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        require_test(socket_fd >= 0, "cannot create raw HTTP socket");
        timeval timeout{3, 0};
        require_test(::setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO,
                                  &timeout, sizeof(timeout)) == 0,
                     "cannot set raw HTTP timeout");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::connect(socket_fd, reinterpret_cast<sockaddr*>(&address),
                      sizeof(address)) == 0) {
            break;
        }
        ::close(socket_fd);
        socket_fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require_test(socket_fd >= 0, "cannot connect to Native API loopback socket");
    send_all(socket_fd, request);
    ::shutdown(socket_fd, SHUT_WR);

    std::string raw;
    char buffer[4096];
    for (;;) {
        const ssize_t count = ::recv(socket_fd, buffer, sizeof(buffer), 0);
        if (count == 0) break;
        require_test(count > 0, "raw HTTP response receive failed");
        raw.append(buffer, static_cast<size_t>(count));
    }
    ::close(socket_fd);

    const size_t header_end = raw.find("\r\n\r\n");
    require_test(header_end != std::string::npos, "HTTP response has no header terminator");
    const size_t status_end = raw.find("\r\n");
    require_test(status_end != std::string::npos &&
                     raw.rfind("HTTP/1.1 ", 0) == 0,
                 "HTTP status line is invalid");
    HttpResponse result;
    result.raw = raw;
    result.status = std::stoi(raw.substr(9, 3));
    size_t cursor = status_end + 2;
    while (cursor < header_end) {
        const size_t line_end = raw.find("\r\n", cursor);
        const std::string line = raw.substr(cursor, line_end - cursor);
        const size_t colon = line.find(':');
        require_test(colon != std::string::npos, "HTTP response header is invalid");
        size_t value = colon + 1;
        while (value < line.size() && line[value] == ' ') ++value;
        result.headers.emplace(line.substr(0, colon), line.substr(value));
        cursor = line_end + 2;
    }
    result.body = raw.substr(header_end + 4);
    const auto length = result.headers.find("Content-Length");
    require_test(length != result.headers.end(), "HTTP response lacks Content-Length");
    require_test(std::stoull(length->second) == result.body.size(),
                 "HTTP Content-Length mismatch");
    return result;
}

std::string get_request(const std::string& target) {
    return "GET " + target + " HTTP/1.1\r\nHost: 127.0.0.1\r\n"
           "Connection: close\r\n\r\n";
}

std::string encoded_reference(const std::string& reference) {
    std::string result;
    for (const char byte : reference) {
        if (byte == '/') result += "%2F";
        else if (byte == ':') result += "%3A";
        else result += byte;
    }
    return result;
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(static_cast<bool>(input), "cannot open Native API contract fixture");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

const vrhino::Json& require_json(const HttpResponse& response,
                                const int expected_status) {
    require_test(response.status == expected_status,
                 "unexpected HTTP status " + std::to_string(response.status));
    const auto content_type = response.headers.find("Content-Type");
    require_test(content_type != response.headers.end() &&
                     content_type->second == "application/json; charset=utf-8",
                 "JSON Content-Type mismatch");
    static thread_local vrhino::Json parsed;
    parsed = vrhino::Json::parse(response.body);
    return parsed;
}

void require_private(const HttpResponse& response, const fs::path& cache_root) {
    for (const std::string& forbidden : {
             std::string("/root"), std::string("/tmp"), cache_root.string(),
             std::string("VRHINO_HOME"), std::string("manifest_path"),
             std::string("CAS path"), std::string("token=")}) {
        require_test(response.body.find(forbidden) == std::string::npos,
                     "HTTP response leaked private local state");
    }
    require_test(response.headers.find("Access-Control-Allow-Origin") ==
                     response.headers.end(),
                 "Phase 2 response enabled CORS");
}

void require_error(const HttpResponse& response, const int status,
                   const std::string& code, const fs::path& cache_root) {
    const vrhino::Json& parsed = require_json(response, status);
    require_test(parsed.at("error").at("code").string() == code &&
                     !parsed.at("error").at("message").string().empty(),
                 "Native API error envelope mismatch");
    require_private(response, cache_root);
}

}  // namespace

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-native-api-test-" + std::to_string(stamp));
    try {
        const fs::path source = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        const fs::path cache_root = root / "isolated-vrhino-home";

        const fs::path examples = fs::path(VRHINO_REPOSITORY_ROOT) /
            "docs/api/examples/native-api-v1";
        const vrhino::JsonParseLimits example_limits{
            64 * 1024, 32, 1024, 4096, 32 * 1024};
        const vrhino::Json version_example = vrhino::Json::parse(
            read_text(examples / "version-response.json"), example_limits);
        const vrhino::Json list_example = vrhino::Json::parse(
            read_text(examples / "model-list-response.json"), example_limits);
        const vrhino::Json detail_example = vrhino::Json::parse(
            read_text(examples / "model-detail-successor-response.json"),
            example_limits);
        const vrhino::Json request_example = vrhino::Json::parse(
            read_text(examples / "run-request.json"), example_limits);
        const vrhino::Json accepted_example = vrhino::Json::parse(
            read_text(examples / "run-accepted-response.json"), example_limits);
        const vrhino::Json statuses_example = vrhino::Json::parse(
            read_text(examples / "run-status-responses.json"), example_limits);
        const vrhino::Json truncation_example = vrhino::Json::parse(
            read_text(examples / "truncation-error.json"), example_limits);
        require_test(version_example.at("schema_version").integer() == 1 &&
                         list_example.at("schema_version").integer() == 1 &&
                         detail_example.at("product").at("input_schema")
                             .at("schema").string() ==
                             product::kProductInputSchemaV1 &&
                         request_example.at("model").is_string() &&
                         accepted_example.at("status").string() == "queued" &&
                         statuses_example.at("running").at("status").string() ==
                             "running" &&
                         statuses_example.at("succeeded").at("status").string() ==
                             "succeeded" &&
                         statuses_example.at("failed").at("status").string() ==
                             "failed" &&
                         statuses_example.at("cancelled").at("status").string() ==
                             "cancelled" &&
                         truncation_example.at("error").at("code").string() ==
                             "event_history_truncated",
                     "Native API v1 JSON contract examples drifted");
        std::istringstream event_examples(
            read_text(examples / "progress.ndjson"));
        std::string event_line;
        uint64_t prior_sequence = 0;
        size_t event_count = 0;
        while (std::getline(event_examples, event_line)) {
            const vrhino::Json event = vrhino::Json::parse(
                event_line, example_limits);
            const uint64_t current = static_cast<uint64_t>(
                event.at("sequence").integer());
            require_test(current > prior_sequence && event.at("kind").is_string(),
                         "Native API v1 NDJSON example ordering drifted");
            prior_sequence = current;
            ++event_count;
        }
        require_test(event_count == 4,
                     "Native API v1 NDJSON example count drifted");

        // Deliberately publish in non-sorted order.
        const product::PackageIdentity lip = publish_fixture(
            cache_root, source /
                "public_musetalk_v15/successors/1.0.1/vrhino-model.json");
        const product::PackageIdentity legacy = publish_fixture(
            cache_root, source / "wan2_1_t2v_1_3b/vrhino-model.json");
        const product::PackageIdentity ttv = publish_fixture(
            cache_root, source /
                "ltx_v0_9_1/successors/1.1.1/vrhino-model.json");

        product::LocalModelCache cache(cache_root);
        const std::vector<product::LocalModelSummary> summaries =
            product::list_local_models(cache);
        require_test(summaries.size() == 3, "local model summary count mismatch");
        require_test(std::is_sorted(
                         summaries.begin(), summaries.end(),
                         [](const auto& left, const auto& right) {
                             return left.reference < right.reference;
                         }),
                     "local model projection is not deterministically sorted");
        for (const product::LocalModelSummary& summary : summaries) {
            require_test(summary.installed && !summary.architecture.empty() &&
                             !summary.product_family.empty(),
                         "local model summary is incomplete");
        }

        const product::VersionInfo version = product::current_version_info();
        require_test(version_example.at("version").string() == version.version &&
                         product::build_version_info().at("version").string() ==
                         version.version &&
                         product::format_cli_version(version).find(version.git_head) !=
                             std::string::npos,
                     "canonical version projection drift");

        require_test(api::valid_server_port(1) &&
                         api::valid_server_port(65535) &&
                         !api::valid_server_port(0) &&
                         !api::valid_server_port(65536),
                     "server port validation drift");
        require_test(api::is_loopback_host("127.0.0.1") &&
                         api::is_loopback_host("::1") &&
                         !api::non_loopback_bind_warning("127.0.0.1") &&
                         api::non_loopback_bind_warning("0.0.0.0") &&
                         api::non_loopback_bind_warning("192.0.2.1"),
                     "non-loopback warning policy drift");

        api::ServerConfig config;
        config.host = "127.0.0.1";
        config.cache_root = cache_root;
        std::unique_ptr<api::Server> server;
        uint16_t port = 0;
        for (size_t attempt = 0; attempt < 20; ++attempt) {
            port = available_port();
            config.port = port;
            server = std::make_unique<api::Server>(config);
            if (server->bind() == port) break;
            server.reset();
        }
        require_test(server != nullptr, "Native API bind failed");

        api::Server conflict(config);
        require_test(conflict.bind() < 0, "occupied port was accepted");

        bool listen_result = false;
        std::thread listener([&] { listen_result = server->listen(); });

        const HttpResponse version_response = raw_request(
            port, get_request("/api/v1/version"));
        const vrhino::Json& version_json = require_json(version_response, 200);
        require_test(version_json.serialize() ==
                         product::build_version_info().serialize(),
                     "HTTP version differs from canonical projection");
        require_private(version_response, cache_root);

        const HttpResponse models_first = raw_request(
            port, get_request("/api/v1/models"));
        const HttpResponse models_second = raw_request(
            port, get_request("/api/v1/models"));
        const vrhino::Json& models_json = require_json(models_first, 200);
        require_test(models_first.body == models_second.body &&
                         models_first.body ==
                             product::build_local_model_list(cache).serialize() &&
                         models_json.at("models").array().size() == 3,
                     "HTTP model list is nondeterministic or noncanonical");
        std::vector<std::string> references;
        for (const vrhino::Json& model : models_json.at("models").array()) {
            references.push_back(model.at("reference").string());
            require_test(model.object().size() == 7 &&
                             model.at("installed").boolean(),
                         "model-list entry fields drifted");
        }
        require_test(std::is_sorted(references.begin(), references.end()),
                     "HTTP model list ordering drifted");
        require_private(models_first, cache_root);

        std::string mixed_case = encoded_reference(ttv.reference());
        const size_t slash_escape = mixed_case.find("%2F");
        const size_t colon_escape = mixed_case.find("%3A");
        require_test(slash_escape != std::string::npos &&
                         colon_escape != std::string::npos,
                     "encoded fixture reference lacks separators");
        mixed_case[slash_escape + 2] = 'f';
        mixed_case[colon_escape + 2] = 'a';
        require_test(require_json(raw_request(
                         port, get_request("/api/v1/models/" + mixed_case)), 200)
                         .at("model").at("reference").string() == ttv.reference(),
                     "mixed-case percent escapes changed decode semantics");

        for (const product::PackageIdentity& identity : {ttv, lip}) {
            const HttpResponse detail = raw_request(
                port, get_request("/api/v1/models/" +
                                  encoded_reference(identity.reference())));
            const vrhino::Json& parsed = require_json(detail, 200);
            const product::ResolvedRunnableModel resolved =
                cache.resolve(identity.reference(), false);
            require_test(parsed.serialize() ==
                             product::build_model_info(resolved).serialize(),
                         "successor HTTP detail differs from build_model_info");
            require_test(parsed.at("product").at("input_schema").is_object() &&
                             parsed.at("product").at("frozen_profile").is_object(),
                         "successor Product contract missing over HTTP");
            require_private(detail, cache_root);
        }

        const HttpResponse legacy_detail = raw_request(
            port, get_request("/api/v1/models/" +
                              encoded_reference(legacy.reference())));
        const vrhino::Json& legacy_json = require_json(legacy_detail, 200);
        require_test(legacy_json.at("product").at("input_schema").is_null() &&
                         legacy_json.at("product").at("frozen_profile").is_null(),
                     "legacy detail inferred a successor Product contract");
        require_private(legacy_detail, cache_root);

        require_error(raw_request(
                          port, get_request(
                              "/api/v1/models/vrhino%2Fmissing%3A1.0.0")),
                      404, "model_not_found", cache_root);

        const std::vector<std::string> malformed = {
            "/api/v1/models/vrhino%2Fbad%",
            "/api/v1/models/vrhino%2Fbad%2",
            "/api/v1/models/vrhino%2Fbad%GG%3A1",
            "/api/v1/models/vrhino%2Fbad%00name%3A1",
            "/api/v1/models/vrhino%252Fbad%3A1",
            "/api/v1/models/vrhino%2Fbad%253A1",
            "/api/v1/models/vrhino%2Fbad%5Cname%3A1",
            "/api/v1/models/vrhino%2Fbad%E7%8A%80%3A1",
            "/api/v1/models/vrhino/bad%3A1",
            "/api/v1/models/vrhino%2Fbad:1",
            "/api/v1/models/vrhino%2Fbad%3A1?unexpected=1",
            "/api/v1/models/vrhino%2Fbad%23fragment%3A1",
            "/api/v1/models/..%2Fbad%3A1",
            "/api/v1/models/vrhino%2F..%3A1",
            "/api/v1/models/vrhino%2F" + std::string(193, 'a') + "%3A1",
        };
        for (const std::string& target : malformed) {
            require_error(raw_request(port, get_request(target)), 400,
                          "invalid_request", cache_root);
        }

        // A deterministic raw-target corpus exercises the percent parser with
        // bounded hostile bytes. Every generated identity contains an invalid
        // escape and must fail as bounded JSON rather than crash or decode
        // differently on a second pass.
        constexpr char raw_bytes[] = "%0123456789ABCDEFabcdefGg";
        uint64_t fuzz_state = 0x3c6ef372fe94f82bULL;
        for (size_t iteration = 0; iteration < 512; ++iteration) {
            std::string raw = "vrhino%2Ffuzz";
            const size_t length = 1 + (iteration % 128);
            for (size_t index = 0; index < length; ++index) {
                fuzz_state ^= fuzz_state << 13;
                fuzz_state ^= fuzz_state >> 7;
                fuzz_state ^= fuzz_state << 17;
                raw.push_back(raw_bytes[fuzz_state % (sizeof(raw_bytes) - 1)]);
            }
            raw += "%GZ%3A1";
            require_error(raw_request(
                              port, get_request("/api/v1/models/" + raw)),
                          400, "invalid_request", cache_root);
        }

        require_error(raw_request(port, get_request("/api/v1/unknown")),
                      404, "not_found", cache_root);
        require_error(raw_request(port, get_request("/api/v1/runs")),
                      405, "method_not_allowed", cache_root);
        require_error(raw_request(
                          port, get_request("/api/v1/events")),
                      404, "not_found", cache_root);
        require_error(raw_request(
                          port,
                          "POST /api/v1/version HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Length: 0\r\nConnection: close\r\n\r\n"),
                      405, "method_not_allowed", cache_root);
        require_error(raw_request(
                          port,
                          "GET /api/v1/version HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                          "Content-Length: 65537\r\nConnection: close\r\n\r\n"),
                      413, "invalid_request", cache_root);
        require_error(raw_request(
                          port, get_request("/api/v1/" + std::string(2050, 'a'))),
                      414, "invalid_request", cache_root);

        server->stop();
        listener.join();
        require_test(listen_result, "Native API listener returned an error");

        api::Server released(config);
        require_test(released.bind() == port, "server shutdown did not release port");
        released.stop();

        fs::remove_all(root);
        std::cout << "bounded read-only Native API raw-socket tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << "bounded read-only Native API raw-socket tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
