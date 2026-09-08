#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vrhino/api/server.h"
#include "vrhino/json.h"
#include "vrhino/product/model_package.h"

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
    write_text(cache_root / "models" / manifest.identity.name_space /
                   manifest.identity.name / manifest.identity.version /
                   product::kModelManifestName,
               manifest.raw_json);
    for (const product::ArtifactDeclaration& artifact : manifest.artifacts) {
        if (!artifact.required) continue;
        const fs::path blob = cache_root / "blobs/sha256" /
                              artifact.sha256.substr(0, 2) / artifact.sha256;
        if (fs::exists(blob)) continue;
        fs::create_directories(blob.parent_path());
        std::ofstream(blob, std::ios::binary).close();
        fs::resize_file(blob, artifact.size);
    }
    return manifest.identity;
}

uint16_t available_port() {
    const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
    require_test(descriptor >= 0, "cannot create port-selection socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    require_test(::bind(descriptor, reinterpret_cast<sockaddr*>(&address),
                        sizeof(address)) == 0,
                 "cannot bind port-selection socket");
    socklen_t size = sizeof(address);
    require_test(::getsockname(descriptor,
                               reinterpret_cast<sockaddr*>(&address),
                               &size) == 0,
                 "cannot read selected port");
    const uint16_t result = ntohs(address.sin_port);
    ::close(descriptor);
    return result;
}

struct HttpResponse {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

HttpResponse request(const uint16_t port, const std::string& method,
                     const std::string& target,
                     const std::string& body = {},
                     const std::string& content_type =
                         "application/json; charset=utf-8") {
    int descriptor = -1;
    for (size_t attempt = 0; attempt < 100; ++attempt) {
        descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
        require_test(descriptor >= 0, "cannot create HTTP socket");
        timeval timeout{5, 0};
        require_test(::setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO,
                                  &timeout, sizeof(timeout)) == 0,
                     "cannot set HTTP timeout");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::connect(descriptor, reinterpret_cast<sockaddr*>(&address),
                      sizeof(address)) == 0)
            break;
        ::close(descriptor);
        descriptor = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require_test(descriptor >= 0, "cannot connect to Job API");

    std::string wire = method + " " + target +
        " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n";
    if (!body.empty() || method == "POST") {
        wire += "Content-Type: " + content_type + "\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n";
    }
    wire += "\r\n" + body;
    size_t sent = 0;
    while (sent < wire.size()) {
        const ssize_t count = ::send(descriptor, wire.data() + sent,
                                     wire.size() - sent, 0);
        require_test(count > 0, "HTTP send failed");
        sent += static_cast<size_t>(count);
    }
    ::shutdown(descriptor, SHUT_WR);

    std::string raw;
    char buffer[4096];
    for (;;) {
        const ssize_t count = ::recv(descriptor, buffer, sizeof(buffer), 0);
        if (count == 0) break;
        require_test(count > 0, "HTTP receive failed");
        raw.append(buffer, static_cast<size_t>(count));
    }
    ::close(descriptor);
    const size_t line_end = raw.find("\r\n");
    const size_t header_end = raw.find("\r\n\r\n");
    require_test(line_end != std::string::npos &&
                     header_end != std::string::npos &&
                     raw.rfind("HTTP/1.1 ", 0) == 0,
                 "malformed HTTP response");
    HttpResponse result;
    result.status = std::stoi(raw.substr(9, 3));
    size_t cursor = line_end + 2;
    while (cursor < header_end) {
        const size_t end = raw.find("\r\n", cursor);
        const std::string line = raw.substr(cursor, end - cursor);
        const size_t colon = line.find(':');
        require_test(colon != std::string::npos, "malformed HTTP header");
        size_t value = colon + 1;
        while (value < line.size() && line[value] == ' ') ++value;
        result.headers.emplace(line.substr(0, colon), line.substr(value));
        cursor = end + 2;
    }
    result.body = raw.substr(header_end + 4);
    require_test(result.headers.at("Content-Type") ==
                     "application/json; charset=utf-8" &&
                     std::stoull(result.headers.at("Content-Length")) ==
                         result.body.size(),
                 "Job API response framing drift");
    return result;
}

vrhino::Json json(const HttpResponse& response, const int status) {
    require_test(response.status == status,
                 "unexpected HTTP status: " + std::to_string(response.status) +
                     " body=" + response.body);
    return vrhino::Json::parse(response.body);
}

std::string run_path(const vrhino::Json& response) {
    return "/api/v1/runs/" + response.at("id").string();
}

vrhino::Json wait_status(const uint16_t port, const std::string& path,
                         const std::set<std::string>& desired) {
    for (size_t attempt = 0; attempt < 5000; ++attempt) {
        const vrhino::Json current = json(request(port, "GET", path), 200);
        if (desired.contains(current.at("status").string())) return current;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("job status wait timed out");
}

struct FakeExecutor {
    std::mutex mutex;
    bool block = false;
    size_t active = 0;
    size_t maximum_active = 0;
    std::vector<std::string> calls;
    std::map<std::string, uint64_t> seeds;

    product::RunExecutor callback() {
        return [this](const product::ResolvedRunnableModel& model,
                      const product::RunOptions& options,
                      product::RunEventSink events) {
            const std::string reference = model.manifest.identity.reference();
            {
                std::lock_guard lock(mutex);
                calls.push_back(reference + "|" + options.prompt);
                seeds[reference] = options.seed.value_or(0);
                ++active;
                maximum_active = std::max(maximum_active, active);
            }
            const auto finish = [this] {
                std::lock_guard lock(mutex);
                --active;
            };
            if (events) {
                events(product::RunEvent::stage_changed(
                    product::RunStage::Sampling, "fake sampling"));
                events(product::RunEvent::progress(
                    product::RunStage::Sampling, 1, 3,
                    product::RunProgressUnit::Step));
            }
            for (;;) {
                bool blocked = false;
                {
                    std::lock_guard lock(mutex);
                    blocked = block;
                }
                if (!blocked) break;
                if (options.cancellation_requested &&
                    options.cancellation_requested()) {
                    finish();
                    throw product::ModelPackageError(
                        product::ModelPackageErrorCode::Cancelled,
                        "fake cancellation");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (options.cancellation_requested &&
                options.cancellation_requested()) {
                finish();
                throw product::ModelPackageError(
                    product::ModelPackageErrorCode::Cancelled,
                    "fake cancellation");
            }
            if (options.prompt == "FAIL") {
                finish();
                throw std::runtime_error("private fake path /tmp/secret");
            }
            fs::create_directories(options.output.parent_path());
            std::ofstream(options.output, std::ios::binary) << "fake";
            product::RunResult result;
            result.identity = model.manifest.identity;
            result.seed = options.seed.value_or(0);
            result.output_bytes = 4;
            finish();
            return result;
        };
    }

    void set_block(const bool value) {
        std::lock_guard lock(mutex);
        block = value;
    }

    bool called_with(const std::string& needle) {
        std::lock_guard lock(mutex);
        return std::any_of(calls.begin(), calls.end(), [&](const auto& call) {
            return call.find(needle) != std::string::npos;
        });
    }
};

std::string ttv_body(const std::string& model, const std::string& prompt,
                     const fs::path& output = {}) {
    std::string result = "{\"model\":\"" + model +
        "\",\"inputs\":{\"prompt\":\"" + prompt + "\"}";
    if (!output.empty())
        result += ",\"outputs\":{\"output\":\"" + output.string() + "\"}";
    return result + "}";
}

std::string lip_body(const std::string& model, const fs::path& video,
                     const fs::path& audio) {
    return "{\"model\":\"" + model +
        "\",\"inputs\":{\"video\":\"" + video.string() +
        "\",\"audio\":\"" + audio.string() + "\"}}";
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-native-job-api-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::unique_ptr<api::Server> server;
    std::thread listener;
    try {
        const fs::path cache_root = root / "home";
        const fs::path specs = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        const std::vector<product::PackageIdentity> models = {
            publish_fixture(cache_root, specs /
                "ltx_v0_9_1/successors/1.1.1/vrhino-model.json"),
            publish_fixture(cache_root, specs /
                "wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json"),
            publish_fixture(cache_root, specs /
                "mochi_1_preview/successors/1.0.1/vrhino-model.json"),
            publish_fixture(cache_root, specs /
                "public_musetalk_v15/successors/1.0.1/vrhino-model.json"),
            publish_fixture(cache_root, specs /
                "public_latentsync_16/successors/1.0.1/vrhino-model.json"),
        };
        const product::PackageIdentity legacy = publish_fixture(
            cache_root, specs / "wan2_1_t2v_1_3b/vrhino-model.json");
        const fs::path video = root / "input.mp4";
        const fs::path audio = root / "input.wav";
        write_text(video, "video");
        write_text(audio, "audio");

        FakeExecutor fake;
        api::ServerConfig config;
        config.host = "127.0.0.1";
        config.port = available_port();
        config.cache_root = cache_root;
        config.run_executor = fake.callback();
        server = std::make_unique<api::Server>(config);
        require_test(server->bind() == config.port, "Job API bind failed");
        bool listened = false;
        listener = std::thread([&] { listened = server->listen(); });

        const std::map<std::string, uint64_t> defaults = {
            {models[0].reference(), 5703}, {models[1].reference(), 5701},
            {models[2].reference(), 11001}, {models[3].reference(), 11001},
            {models[4].reference(), 1247},
        };
        std::set<std::string> generated_outputs;
        for (size_t index = 0; index < models.size(); ++index) {
            const std::string body = index < 3
                ? ttv_body(models[index].reference(), "success-" +
                                                       std::to_string(index))
                : lip_body(models[index].reference(), video, audio);
            const HttpResponse accepted = request(
                config.port, "POST", "/api/v1/runs", body);
            const vrhino::Json accepted_json = json(accepted, 202);
            const std::string path = run_path(accepted_json);
            require_test(accepted.headers.at("Location") == path &&
                             accepted_json.at("status").string() == "queued",
                         "POST acceptance/Location contract drift");
            const vrhino::Json terminal = wait_status(
                config.port, path, {"succeeded"});
            const fs::path output = terminal.at("output").at("path").string();
            require_test(terminal.at("output").at("managed").boolean() &&
                             fs::file_size(output) == 4 &&
                             generated_outputs.insert(output.string()).second,
                         "managed output is not unique/successful");
        }
        {
            std::lock_guard lock(fake.mutex);
            require_test(fake.seeds == defaults,
                         "successor ProductInputSchema defaults drifted");
        }

        const vrhino::Json legacy_error = json(request(
            config.port, "POST", "/api/v1/runs",
            ttv_body(legacy.reference(), "legacy")), 400);
        require_test(legacy_error.at("error").at("code").string() ==
                         "model_unavailable",
                     "legacy execution contract was inferred");
        require_test(json(request(config.port, "POST", "/api/v1/runs",
                                  ttv_body("vrhino/missing:1", "x")), 404)
                         .at("error").at("code").string() == "model_not_found",
                     "unknown model status drift");

        const std::vector<std::string> invalid = {
            "{", "{\"model\":\"" + models[0].reference() +
                "\",\"model\":\"x\"}",
            "{\"model\":\"" + models[0].reference() + "\"}",
            "{\"model\":\"" + models[0].reference() +
                "\",\"inputs\":{\"prompt\":1}}",
            "{\"model\":\"" + models[0].reference() +
                "\",\"inputs\":{\"prompt\":\"x\"},\"parameters\":{\"seed\":-1}}",
            "{\"model\":\"" + models[0].reference() +
                "\",\"inputs\":{\"prompt\":\"x\"},\"parameters\":{\"seed\":\"18446744073709551616\"}}",
            "{\"model\":\"" + models[0].reference() +
                "\",\"inputs\":{\"prompt\":\"x\"},\"outputs\":{\"output\":\"relative.mp4\"}}",
            "{\"model\":\"" + models[3].reference() +
                "\",\"inputs\":{\"video\":\"relative.mp4\",\"audio\":\"" +
                audio.string() + "\"}}",
        };
        for (const std::string& body : invalid)
            require_test(json(request(config.port, "POST", "/api/v1/runs", body),
                              400).at("error").at("code").string() ==
                             "invalid_request",
                         "invalid request did not fail closed");

        const fs::path existing = root / "explicit" / "existing.mp4";
        write_text(existing, "preserve");
        require_test(json(request(config.port, "POST", "/api/v1/runs",
                                  ttv_body(models[0].reference(), "existing",
                                           existing)), 409)
                         .at("error").at("code").string() == "conflict",
                     "existing output was not rejected");

        fake.set_block(true);
        const fs::path reserved = root / "explicit" / "reserved.mp4";
        const vrhino::Json active = json(request(
            config.port, "POST", "/api/v1/runs",
            ttv_body(models[0].reference(), "blocked", reserved)), 202);
        const std::string active_path = run_path(active);
        const vrhino::Json running = wait_status(
            config.port, active_path, {"running"});
        require_test(running.at("stage").string() == "sampling" &&
                         running.at("progress").at("completed").integer() == 1,
                     "RunSession stage/progress projection drift");
        require_test(json(request(config.port, "POST", "/api/v1/runs",
                                  ttv_body(models[0].reference(), "collision",
                                           reserved)), 409)
                         .at("error").at("code").string() == "conflict",
                     "reserved output collision was accepted");

        std::vector<std::string> queued;
        for (size_t index = 0; index < 4; ++index) {
            const vrhino::Json accepted = json(request(
                config.port, "POST", "/api/v1/runs",
                ttv_body(models[0].reference(),
                         "queued-" + std::to_string(index))), 202);
            queued.push_back(run_path(accepted));
        }
        require_test(json(request(config.port, "POST", "/api/v1/runs",
                                  ttv_body(models[0].reference(), "overload")),
                                  429)
                         .at("error").at("code").string() == "overloaded",
                     "bounded FIFO overload did not return 429");
        require_test(json(request(config.port, "DELETE", queued[1]), 200)
                         .at("status").string() == "cancelled",
                     "queued DELETE did not cancel immediately");
        const std::string replacement = run_path(json(request(
            config.port, "POST", "/api/v1/runs",
            ttv_body(models[0].reference(), "replacement")), 202));

        std::atomic<bool> concurrent_failed{false};
        std::vector<std::thread> readers;
        for (size_t index = 0; index < 8; ++index) {
            readers.emplace_back([&] {
                try {
                    if (request(config.port, "GET", active_path).status != 200)
                        concurrent_failed.store(true);
                } catch (...) {
                    concurrent_failed.store(true);
                }
            });
        }
        readers.emplace_back([&] {
            try {
                const HttpResponse concurrent_post = request(
                    config.port, "POST", "/api/v1/runs",
                    ttv_body(models[0].reference(), "concurrent-post"));
                if (concurrent_post.status != 202 &&
                    concurrent_post.status != 429)
                    concurrent_failed.store(true);
            } catch (...) {
                concurrent_failed.store(true);
            }
        });
        const vrhino::Json cancel_pending = json(
            request(config.port, "DELETE", active_path), 202);
        require_test(cancel_pending.at("cancel_requested").boolean(),
                     "running DELETE did not expose pending cancellation");
        for (std::thread& reader : readers) reader.join();
        require_test(!concurrent_failed.load(),
                     "concurrent GET/DELETE failed");
        const vrhino::Json cancelled = wait_status(
            config.port, active_path, {"cancelled"});
        require_test(!fs::exists(cancelled.at("output").at("path").string()) &&
                         !fs::exists(cancelled.at("output").at("path").string() +
                                     ".partial"),
                     "cancelled output or partial remained");
        fake.set_block(false);
        for (const std::string& path : queued)
            (void)wait_status(config.port, path, {"succeeded", "cancelled"});
        (void)wait_status(config.port, replacement, {"succeeded"});
        require_test(!fake.called_with("queued-1"),
                     "queued cancelled job invoked Product executor");
        {
            std::lock_guard lock(fake.mutex);
            require_test(fake.maximum_active == 1,
                         "more than one Product execution overlapped");
        }

        const vrhino::Json failure = wait_status(
            config.port,
            run_path(json(request(config.port, "POST", "/api/v1/runs",
                                  ttv_body(models[0].reference(), "FAIL")), 202)),
            {"failed"});
        require_test(failure.at("error").at("code").string() ==
                         "execution_failed" &&
                         failure.at("error").at("message").string().find(
                             "/tmp") == std::string::npos,
                     "asynchronous failure was not privacy-safe");
        require_test(json(request(config.port, "DELETE",
                                  "/api/v1/runs/r0000000000000000-0000000000000000"),
                                  404).at("error").at("code").string() ==
                         "job_not_found",
                     "unknown job status drift");
        require_test(json(request(config.port, "GET",
                                  "/api/v1/events"), 404)
                         .at("error").at("code").string() == "not_found",
                     "global event route was accidentally implemented");

        fake.set_block(true);
        const std::string shutdown_active = run_path(json(request(
            config.port, "POST", "/api/v1/runs",
            ttv_body(models[0].reference(), "shutdown-active")), 202));
        (void)wait_status(config.port, shutdown_active, {"running"});
        (void)json(request(config.port, "POST", "/api/v1/runs",
                          ttv_body(models[0].reference(), "shutdown-queued")), 202);
        server->stop();
        listener.join();
        require_test(listened && !fake.called_with("shutdown-queued"),
                     "server shutdown did not cancel queued work");
        server.reset();

        fs::remove_all(root);
        std::cout << "bounded Native Job API loopback tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        if (server) server->stop();
        if (listener.joinable()) listener.join();
        fs::remove_all(root);
        std::cerr << "bounded Native Job API loopback tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
