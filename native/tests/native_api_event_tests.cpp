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
#include <memory>
#include <mutex>
#include <optional>
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
    require_test(static_cast<bool>(output), "cannot write event fixture");
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

int connect_socket(const uint16_t port) {
    for (size_t attempt = 0; attempt < 200; ++attempt) {
        const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
        require_test(descriptor >= 0, "cannot create HTTP socket");
        timeval timeout{5, 0};
        require_test(::setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO,
                                  &timeout, sizeof(timeout)) == 0,
                     "cannot set HTTP receive timeout");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::connect(descriptor, reinterpret_cast<sockaddr*>(&address),
                      sizeof(address)) == 0)
            return descriptor;
        ::close(descriptor);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("cannot connect to event API");
}

void send_all(const int descriptor, const std::string& value) {
    size_t sent = 0;
    while (sent < value.size()) {
        const ssize_t count = ::send(descriptor, value.data() + sent,
                                     value.size() - sent, 0);
        require_test(count > 0, "HTTP send failed");
        sent += static_cast<size_t>(count);
    }
}

struct HttpResponse {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

HttpResponse request(const uint16_t port, const std::string& method,
                     const std::string& target,
                     const std::string& body = {}) {
    const int descriptor = connect_socket(port);
    std::string wire = method + " " + target +
        " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n";
    if (method == "POST") {
        wire += "Content-Type: application/json; charset=utf-8\r\n"
                "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    wire += "\r\n" + body;
    send_all(descriptor, wire);
    ::shutdown(descriptor, SHUT_WR);
    std::string raw;
    char bytes[4096];
    for (;;) {
        const ssize_t count = ::recv(descriptor, bytes, sizeof(bytes), 0);
        if (count == 0) break;
        require_test(count > 0, "HTTP receive failed");
        raw.append(bytes, static_cast<size_t>(count));
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
        require_test(colon != std::string::npos, "malformed response header");
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
                 "bounded JSON response framing drift");
    return result;
}

vrhino::Json response_json(const HttpResponse& response, const int status) {
    require_test(response.status == status,
                 "unexpected HTTP status " + std::to_string(response.status) +
                     ": " + response.body);
    return vrhino::Json::parse(response.body);
}

class ChunkedStream {
public:
    ChunkedStream(const uint16_t port, const std::string& target)
        : descriptor_(connect_socket(port)) {
        const std::string wire = "GET " + target +
            " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
        send_all(descriptor_, wire);
        read_headers();
        require_test(status_ == 200, "event stream returned " + std::to_string(status_) + " for " + target);
        require_test(headers_.at("Content-Type") ==
                         "application/x-ndjson; charset=utf-8" &&
                     headers_.at("Transfer-Encoding") == "chunked" &&
                     headers_.at("Cache-Control") == "no-store" &&
                     !headers_.contains("Access-Control-Allow-Origin"),
                     "NDJSON response headers/framing drift");
    }

    ChunkedStream(const ChunkedStream&) = delete;
    ChunkedStream& operator=(const ChunkedStream&) = delete;
    ~ChunkedStream() { close(); }

    std::optional<std::string> next_line() {
        if (finished_) return std::nullopt;
        const std::string size_line = take_until("\r\n");
        require_test(!size_line.empty() &&
                         size_line.find_first_not_of("0123456789abcdefABCDEF") ==
                             std::string::npos,
                     "invalid HTTP chunk size");
        const size_t size = static_cast<size_t>(std::stoull(size_line, nullptr, 16));
        if (size == 0) {
            require_test(take_bytes(2) == "\r\n",
                         "invalid terminal chunk");
            finished_ = true;
            // Connection: close is requested. Observe server completion before
            // a later capacity assertion opens another subscriber batch.
            wait_for_close();
            close();
            return std::nullopt;
        }
        const std::string payload = take_bytes(size);
        require_test(take_bytes(2) == "\r\n" && !payload.empty() &&
                         payload.back() == '\n' &&
                         payload.size() <= api::kEventLineMaxBytes,
                     "invalid NDJSON chunk framing/bound");
        return payload;
    }

    std::optional<vrhino::Json> next_event() {
        const std::optional<std::string> line = next_line();
        if (!line) return std::nullopt;
        return vrhino::Json::parse(line->substr(0, line->size() - 1));
    }

    std::vector<vrhino::Json> collect(const size_t maximum = 4096) {
        std::vector<vrhino::Json> result;
        while (result.size() < maximum) {
            std::optional<vrhino::Json> event = next_event();
            if (!event) return result;
            result.push_back(std::move(*event));
        }
        throw std::runtime_error("event stream exceeded expected bound");
    }

    void close() noexcept {
        if (descriptor_ >= 0) {
            ::shutdown(descriptor_, SHUT_RDWR);
            ::close(descriptor_);
            descriptor_ = -1;
        }
    }

    void wait_for_close() {
        char byte;
        for (;;) {
            const ssize_t count = ::recv(descriptor_, &byte, 1, 0);
            if (count == 0) return;
            if (count < 0) return;
        }
    }

private:
    void read_more() {
        char bytes[4096];
        const ssize_t count = ::recv(descriptor_, bytes, sizeof(bytes), 0);
        require_test(count > 0, "event stream closed unexpectedly");
        buffer_.append(bytes, static_cast<size_t>(count));
    }

    std::string take_until(const std::string& delimiter) {
        for (;;) {
            const size_t found = buffer_.find(delimiter);
            if (found != std::string::npos) {
                std::string result = buffer_.substr(0, found);
                buffer_.erase(0, found + delimiter.size());
                return result;
            }
            read_more();
        }
    }

    std::string take_bytes(const size_t size) {
        while (buffer_.size() < size) read_more();
        std::string result = buffer_.substr(0, size);
        buffer_.erase(0, size);
        return result;
    }

    void read_headers() {
        const std::string header = take_until("\r\n\r\n");
        const size_t line_end = header.find("\r\n");
        require_test(line_end != std::string::npos &&
                         header.rfind("HTTP/1.1 ", 0) == 0,
                     "malformed stream status line");
        status_ = std::stoi(header.substr(9, 3));
        size_t cursor = line_end + 2;
        while (cursor < header.size()) {
            const size_t end = header.find("\r\n", cursor);
            const size_t line_stop = end == std::string::npos
                ? header.size() : end;
            const std::string line = header.substr(cursor, line_stop - cursor);
            const size_t colon = line.find(':');
            require_test(colon != std::string::npos,
                         "malformed stream header");
            size_t value = colon + 1;
            while (value < line.size() && line[value] == ' ') ++value;
            headers_.emplace(line.substr(0, colon), line.substr(value));
            if (end == std::string::npos) break;
            cursor = end + 2;
        }
    }

    int descriptor_ = -1;
    int status_ = 0;
    bool finished_ = false;
    std::map<std::string, std::string> headers_;
    std::string buffer_;
};

std::string run_path(const vrhino::Json& accepted) {
    return "/api/v1/runs/" + accepted.at("id").string();
}

vrhino::Json wait_status(const uint16_t port, const std::string& path,
                         const std::set<std::string>& desired) {
    for (size_t attempt = 0; attempt < 10000; ++attempt) {
        const vrhino::Json current = response_json(
            request(port, "GET", path), 200);
        if (desired.contains(current.at("status").string())) return current;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("job status wait timed out");
}

struct ControlledExecutor {
    struct Control {
        std::atomic<uint64_t> permits{0};
        std::atomic<uint64_t> burst{0};
        std::atomic<bool> finish{false};
        std::atomic<bool> started{false};
        std::atomic<bool> cancelled{false};
    };

    ControlledExecutor() {
        for (const char* prompt : {"cancellation", "reconnect", "queue-owner",
                                   "queued-stream", "overrun", "multi",
                                   "shutdown-stream"})
            controls.emplace(prompt, std::make_shared<Control>());
    }

    std::map<std::string, std::shared_ptr<Control>> controls;

    product::RunExecutor callback() {
        return [this](const product::ResolvedRunnableModel& model,
                      const product::RunOptions& options,
                      product::RunEventSink events) {
            const std::string prompt = options.prompt;
            const auto control_entry = controls.find(prompt);
            std::shared_ptr<Control> control = control_entry == controls.end()
                ? nullptr : control_entry->second;
            if (control)
                control->started.store(true, std::memory_order_release);
            if (events)
                events(product::RunEvent::stage_changed(
                    product::RunStage::Sampling, "Streaming fake sampling"));

            uint64_t completed = 0;
            if (prompt == "many") {
                for (uint64_t index = 0; index < 300; ++index)
                    events(product::RunEvent::progress(
                        product::RunStage::Sampling, index + 1, 300,
                        product::RunProgressUnit::Step));
            } else if (prompt == "privacy") {
                events(product::RunEvent::diagnostic(
                    "private /tmp/secret cache-token=credential"));
                events(product::RunEvent::stage_changed(
                    product::RunStage::Encoding, std::string(5000, 'x')));
            } else if (prompt == "hostile-message") {
                std::string hostile =
                    "quote=\" backslash=\\ newline=\n tab=\t control=";
                hostile.push_back(static_cast<char>(1));
                hostile += " 犀牛";
                events(product::RunEvent::stage_changed(
                    product::RunStage::Sampling, hostile));
                std::string invalid_utf8 = "invalid=";
                invalid_utf8.push_back(static_cast<char>(0xff));
                events(product::RunEvent::stage_changed(
                    product::RunStage::Encoding, invalid_utf8));
            } else if (prompt == "failure") {
                events(product::RunEvent::progress(
                    product::RunStage::Sampling, 1, 2,
                    product::RunProgressUnit::Step));
                throw std::runtime_error("private /tmp/failure");
            } else if (prompt.rfind("success-", 0) != 0) {
                if (!control)
                    throw std::runtime_error("missing controlled test state");
                for (;;) {
                    uint64_t emit = control->burst.exchange(
                        0, std::memory_order_acq_rel);
                    if (emit == 0) {
                        uint64_t permitted = control->permits.load(
                            std::memory_order_acquire);
                        while (permitted > 0 &&
                               !control->permits.compare_exchange_weak(
                                   permitted, permitted - 1,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire)) {}
                        if (permitted > 0) emit = 1;
                    }
                    const bool finish = control->finish.load(
                        std::memory_order_acquire) &&
                        control->permits.load(std::memory_order_acquire) == 0 &&
                        control->burst.load(std::memory_order_acquire) == 0;
                    if (options.cancellation_requested &&
                        options.cancellation_requested()) {
                        control->cancelled.store(true,
                                                 std::memory_order_release);
                        throw product::ModelPackageError(
                            product::ModelPackageErrorCode::Cancelled,
                            "controlled cancellation");
                    }
                    for (uint64_t index = 0; index < emit; ++index) {
                        ++completed;
                        if (events)
                            events(product::RunEvent::progress(
                                product::RunStage::Sampling, completed, 100000,
                                product::RunProgressUnit::Step));
                    }
                    if (finish) break;
                    if (emit == 0)
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1));
                }
            }

            fs::create_directories(options.output.parent_path());
            std::ofstream(options.output, std::ios::binary) << "fake";
            product::RunResult result;
            result.identity = model.manifest.identity;
            result.seed = options.seed.value_or(0);
            result.output_bytes = 4;
            return result;
        };
    }

    void wait_started(const std::string& prompt) {
        const std::shared_ptr<Control> control = controls.at(prompt);
        for (size_t attempt = 0; attempt < 5000; ++attempt) {
            if (control->started.load(std::memory_order_acquire)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("controlled executor did not start");
    }

    void allow(const std::string& prompt, const uint64_t count) {
        controls.at(prompt)->permits.fetch_add(count,
                                               std::memory_order_acq_rel);
    }

    void finish(const std::string& prompt) {
        controls.at(prompt)->finish.store(true, std::memory_order_release);
    }

    void burst_and_finish(const std::string& prompt, const uint64_t count) {
        const std::shared_ptr<Control> control = controls.at(prompt);
        control->burst.store(count, std::memory_order_release);
        control->finish.store(true, std::memory_order_release);
    }

    bool cancelled(const std::string& prompt) {
        return controls.at(prompt)->cancelled.load(std::memory_order_acquire);
    }
};

std::string submit(const uint16_t port, const std::string& model,
                   const std::string& prompt) {
    const std::string body = "{\"model\":\"" + model +
        "\",\"inputs\":{\"prompt\":\"" + prompt + "\"}}";
    return run_path(response_json(
        request(port, "POST", "/api/v1/runs", body), 202));
}

uint64_t sequence(const vrhino::Json& event) {
    return static_cast<uint64_t>(event.at("sequence").integer());
}

void require_ordered(const std::vector<vrhino::Json>& events) {
    for (size_t index = 1; index < events.size(); ++index)
        require_test(sequence(events[index - 1]) < sequence(events[index]),
                     "event sequence is not strictly increasing");
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-native-event-api-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::unique_ptr<api::Server> server;
    std::thread listener;
    try {
        const fs::path cache_root = root / "home";
        const product::PackageIdentity model = publish_fixture(
            cache_root, fs::path(VRHINO_TEST_SOURCE_ROOT) /
                "specs/ltx_v0_9_1/successors/1.1.1/vrhino-model.json");
        ControlledExecutor controlled;
        api::ServerConfig config;
        config.host = "127.0.0.1";
        config.port = available_port();
        config.cache_root = cache_root;
        config.run_executor = controlled.callback();
        server = std::make_unique<api::Server>(config);
        require_test(server->bind() == config.port, "event API bind failed");
        bool listened = false;
        listener = std::thread([&] { listened = server->listen(); });

        const std::string success_a = submit(
            config.port, model.reference(), "success-a");
        (void)wait_status(config.port, success_a, {"succeeded"});
        ChunkedStream success_stream(config.port, success_a + "/events");
        const std::vector<vrhino::Json> success_events =
            success_stream.collect();
        require_test(success_events.size() == 3 &&
                         success_events.back().at("state").string() ==
                             "succeeded",
                     "success stream terminal contract drift");
        require_ordered(success_events);

        const std::string success_b = submit(
            config.port, model.reference(), "success-b");
        (void)wait_status(config.port, success_b, {"succeeded"});
        ChunkedStream deterministic_stream(config.port,
                                            success_b + "/events");
        const std::vector<vrhino::Json> deterministic =
            deterministic_stream.collect();
        require_test(deterministic.size() == success_events.size(),
                     "deterministic event count drift");
        for (size_t index = 0; index < deterministic.size(); ++index)
            require_test(deterministic[index].serialize() ==
                             success_events[index].serialize(),
                         "event serialization is not deterministic");

        const std::string failure = submit(
            config.port, model.reference(), "failure");
        (void)wait_status(config.port, failure, {"failed"});
        ChunkedStream failure_stream(config.port, failure + "/events");
        const std::vector<vrhino::Json> failure_events =
            failure_stream.collect();
        require_test(failure_events.back().at("state").string() == "failed" &&
                         response_json(request(config.port, "GET", failure), 200)
                                 .at("status").string() == "failed",
                     "failure stream/status mismatch");

        const std::string cancellation = submit(
            config.port, model.reference(), "cancellation");
        controlled.wait_started("cancellation");
        ChunkedStream cancel_stream(config.port, cancellation + "/events");
        controlled.allow("cancellation", 1);
        bool saw_progress = false;
        while (!saw_progress) {
            const auto event = cancel_stream.next_event();
            require_test(event.has_value(), "cancel stream closed early");
            saw_progress = event->at("kind").string() == "progress";
        }
        require_test(response_json(
                         request(config.port, "DELETE", cancellation), 202)
                         .at("cancel_requested").boolean(),
                     "streaming job cancellation was not accepted");
        const std::vector<vrhino::Json> cancelled_events =
            cancel_stream.collect();
        require_test(!cancelled_events.empty() &&
                         cancelled_events.back().at("state").string() ==
                             "cancelled",
                     "cancel stream missed terminal event");

        const std::string reconnect = submit(
            config.port, model.reference(), "reconnect");
        controlled.wait_started("reconnect");
        ChunkedStream first(config.port, reconnect + "/events");
        require_test(first.next_event().has_value() &&
                         first.next_event().has_value(),
                     "initial replay missing running/stage events");
        controlled.allow("reconnect", 3);
        uint64_t last = 0;
        for (size_t index = 0; index < 3; ++index) {
            const auto event = first.next_event();
            require_test(event.has_value(), "live progress event missing");
            last = sequence(*event);
        }
        first.close();
        controlled.allow("reconnect", 3);
        for (size_t attempt = 0; attempt < 5000; ++attempt) {
            const vrhino::Json status = response_json(
                request(config.port, "GET", reconnect), 200);
            if (status.find("progress") &&
                status.at("progress").at("completed").integer() >= 6)
                break;
            if (attempt == 4999)
                throw std::runtime_error("disconnected job stopped progressing");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ChunkedStream resumed(config.port, reconnect +
            "/events?after=" + std::to_string(last));
        controlled.finish("reconnect");
        const std::vector<vrhino::Json> resumed_events = resumed.collect();
        require_test(!resumed_events.empty() &&
                         sequence(resumed_events.front()) > last &&
                         resumed_events.back().at("state").string() ==
                             "succeeded",
                     "sequence reconnect contract drift");
        require_ordered(resumed_events);

        const std::string queue_owner = submit(
            config.port, model.reference(), "queue-owner");
        controlled.wait_started("queue-owner");
        const std::string queued = submit(
            config.port, model.reference(), "queued-stream");
        ChunkedStream queued_stream(config.port, queued + "/events");
        require_test(response_json(request(config.port, "DELETE", queued), 200)
                             .at("status").string() == "cancelled" &&
                         queued_stream.collect().empty(),
                     "queued cancellation manufactured RunEvents");
        (void)response_json(request(config.port, "DELETE", queue_owner), 202);
        (void)wait_status(config.port, queue_owner, {"cancelled"});

        const std::string many = submit(config.port, model.reference(), "many");
        (void)wait_status(config.port, many, {"succeeded"});
        const vrhino::Json truncation = response_json(
            request(config.port, "GET", many + "/events?after=1"), 409);
        require_test(truncation.at("error").at("code").string() ==
                         "event_history_truncated",
                     "stale cursor did not return truncation error");
        const uint64_t first_available = static_cast<uint64_t>(
            truncation.at("error").at("details")
                .at("first_available_sequence").integer());
        const uint64_t latest = static_cast<uint64_t>(
            truncation.at("error").at("details")
                .at("latest_sequence").integer());
        require_test(first_available > 1 && latest >= first_available,
                     "truncation replay metadata is invalid");
        ChunkedStream retained(config.port, many + "/events?after=" +
            std::to_string(first_available - 1));
        const auto retained_events = retained.collect();
        require_test(retained_events.size() ==
                         product::RunSession::kEventHistoryCapacity &&
                         sequence(retained_events.front()) == first_available &&
                         sequence(retained_events.back()) == latest,
                     "valid retained replay drift");
        ChunkedStream omitted(config.port, many + "/events");
        const auto omitted_events = omitted.collect();
        require_test(omitted_events.size() == retained_events.size() &&
                         sequence(omitted_events.front()) == first_available,
                     "omitted cursor rejected truncated retained history");
        ChunkedStream empty_terminal(config.port, many + "/events?after=" +
            std::to_string(latest));
        require_test(empty_terminal.collect().empty(),
                     "terminal after-latest stream was not empty");
        require_test(response_json(request(config.port, "GET", many +
                         "/events?after=" + std::to_string(latest + 1)), 400)
                             .at("error").at("code").string() ==
                         "invalid_request",
                     "future cursor did not fail closed");

        const std::vector<std::string> invalid_after = {
            "", "-1", "+1", "%201", "%00", "0x1", "1.0", "1x",
            "1%20", "1%2B",
            "18446744073709551616", "1&after=2", "1&x=2"};
        for (const std::string& value : invalid_after)
            require_test(response_json(request(config.port, "GET", many +
                             "/events?after=" + value), 400)
                                 .at("error").at("code").string() ==
                             "invalid_request",
                         "invalid after cursor was accepted: " + value);
        require_test(response_json(request(config.port, "GET",
                         "/api/v1/runs/bad/events"), 400)
                             .at("error").at("code").string() ==
                         "invalid_request" &&
                         response_json(request(config.port, "GET",
                         "/api/v1/runs/r0000000000000000-0000000000000000/events"),
                         404).at("error").at("code").string() ==
                         "job_not_found" &&
                         response_json(request(config.port, "POST",
                         many + "/events"), 405)
                             .at("error").at("code").string() ==
                         "method_not_allowed",
                         "event route validation/status contract drift");

        uint64_t cursor_state = 0x510e527fade682d1ULL;
        constexpr char cursor_digits[] = "0123456789";
        for (size_t iteration = 0; iteration < 256; ++iteration) {
            std::string value;
            const size_t length = 1 + (iteration % 128);
            value.reserve(length + 1);
            for (size_t index = 0; index < length; ++index) {
                cursor_state ^= cursor_state << 13;
                cursor_state ^= cursor_state >> 7;
                cursor_state ^= cursor_state << 17;
                value.push_back(cursor_digits[cursor_state % 10]);
            }
            value.insert(value.begin() +
                             static_cast<std::ptrdiff_t>(iteration %
                                 (value.size() + 1)),
                         'x');
            require_test(response_json(request(config.port, "GET", many +
                             "/events?after=" + value), 400)
                                 .at("error").at("code").string() ==
                             "invalid_request",
                         "cursor mutation corpus escaped strict parsing");
        }

        const std::string privacy = submit(
            config.port, model.reference(), "privacy");
        (void)wait_status(config.port, privacy, {"succeeded"});
        ChunkedStream privacy_stream(config.port, privacy + "/events");
        const auto privacy_events = privacy_stream.collect();
        bool saw_diagnostic = false;
        bool saw_encoding_stage = false;
        for (const vrhino::Json& event : privacy_events) {
            const std::string encoded = event.serialize();
            require_test(encoded.find("/tmp") == std::string::npos &&
                             encoded.find("credential") == std::string::npos &&
                             encoded.size() < api::kEventLineMaxBytes,
                         "event projection leaked diagnostic/private data");
            if (event.at("kind").string() == "message") {
                saw_diagnostic = true;
                require_test(event.find("message") == nullptr,
                             "diagnostic message text was exposed");
            }
            if (event.find("stage") &&
                event.at("stage").string() == "encoding") {
                saw_encoding_stage = true;
                require_test(event.find("message") == nullptr,
                             "oversized optional message was exposed");
            }
        }
        require_test(saw_diagnostic && saw_encoding_stage,
                     "privacy omission fixtures were not streamed");

        const std::string hostile_path = submit(
            config.port, model.reference(), "hostile-message");
        (void)wait_status(config.port, hostile_path, {"succeeded"});
        ChunkedStream hostile_stream(config.port, hostile_path + "/events");
        bool saw_escaped_message = false;
        bool saw_invalid_omission = false;
        for (;;) {
            const std::optional<std::string> physical =
                hostile_stream.next_line();
            if (!physical) break;
            require_test(std::count(physical->begin(), physical->end(), '\n') ==
                             1 && physical->back() == '\n' &&
                             physical->size() <= api::kEventLineMaxBytes,
                         "hostile event injected a physical NDJSON line break");
            const vrhino::Json event = vrhino::Json::parse(
                physical->substr(0, physical->size() - 1));
            if (event.find("stage") &&
                event.at("stage").string() == "sampling" &&
                event.find("message")) {
                const std::string& message = event.at("message").string();
                saw_escaped_message =
                    message.find('"') != std::string::npos &&
                    message.find('\\') != std::string::npos &&
                    message.find('\n') != std::string::npos &&
                    message.find(static_cast<char>(1)) != std::string::npos &&
                    message.find("犀牛") != std::string::npos;
            }
            if (event.find("stage") &&
                event.at("stage").string() == "encoding") {
                saw_invalid_omission = event.find("message") == nullptr;
            }
        }
        require_test(saw_escaped_message && saw_invalid_omission,
                     "hostile NDJSON message escaping/omission drifted");

        const std::string overrun = submit(
            config.port, model.reference(), "overrun");
        controlled.wait_started("overrun");
        ChunkedStream slow(config.port, overrun + "/events");
        const auto slow_first = slow.next_event();
        const auto slow_second = slow.next_event();
        require_test(slow_first && slow_second,
                     "slow-client initial replay missing");
        const uint64_t slow_cursor = sequence(*slow_second);
        const auto burst_started = std::chrono::steady_clock::now();
        controlled.burst_and_finish("overrun", 5000);
        (void)wait_status(config.port, overrun, {"succeeded"});
        require_test(std::chrono::steady_clock::now() - burst_started <
                         std::chrono::seconds(5),
                     "slow client blocked Product execution");
        require_test(response_json(request(config.port, "GET", overrun +
                         "/events?after=" + std::to_string(slow_cursor)), 409)
                             .at("error").at("code").string() ==
                         "event_history_truncated",
                     "mid-stream overrun recovery contract drift");
        // A client-side close alone does not acknowledge release of the
        // server subscriber lease; wait for the overrun response to end.
        slow.wait_for_close();
        slow.close();

        const std::string multi = submit(
            config.port, model.reference(), "multi");
        controlled.wait_started("multi");
        std::vector<std::unique_ptr<ChunkedStream>> subscribers;
        for (size_t index = 0; index < api::kEventSubscriberCapacity; ++index)
            subscribers.push_back(std::make_unique<ChunkedStream>(
                config.port, multi + "/events"));
        require_test(response_json(request(config.port, "GET",
                         multi + "/events"), 429)
                             .at("error").at("code").string() == "overloaded",
                     "event subscriber bound did not return 429");
        require_test(response_json(request(config.port, "GET",
                         "/api/v1/version"), 200).at("version").is_string() &&
                         response_json(request(config.port, "GET", multi), 200)
                             .at("status").string() == "running",
                     "active streams starved status/version control plane");
        require_test(response_json(request(config.port, "DELETE", multi), 202)
                             .at("cancel_requested").boolean(),
                     "active streams starved cancellation control plane");
        for (auto& subscriber : subscribers) {
            const auto events = subscriber->collect();
            require_test(!events.empty() &&
                             events.back().at("state").string() == "cancelled",
                         "multiple subscriber missed cancellation terminal");
        }

        const std::string shutdown = submit(
            config.port, model.reference(), "shutdown-stream");
        controlled.wait_started("shutdown-stream");
        ChunkedStream shutdown_stream(config.port, shutdown + "/events");
        require_test(shutdown_stream.next_event().has_value(),
                     "shutdown stream received no event");
        server->stop();
        listener.join();
        shutdown_stream.wait_for_close();
        shutdown_stream.close();
        require_test(listened && controlled.cancelled("shutdown-stream"),
                     "shutdown did not cancel Product/wake stream");
        server.reset();

        const int port_probe = ::socket(AF_INET, SOCK_STREAM, 0);
        require_test(port_probe >= 0, "cannot create port release probe");
        int reuse = 1;
        require_test(::setsockopt(port_probe, SOL_SOCKET, SO_REUSEADDR,
                                  &reuse, sizeof(reuse)) == 0,
                     "cannot configure port release probe");
        sockaddr_in released{};
        released.sin_family = AF_INET;
        released.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        released.sin_port = htons(config.port);
        const bool port_released = ::bind(
            port_probe, reinterpret_cast<sockaddr*>(&released),
            sizeof(released)) == 0;
        ::close(port_probe);
        require_test(port_released, "server shutdown did not release port");

        fs::remove_all(root);
        std::cout << "bounded NDJSON Native run event tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        if (server) server->stop();
        if (listener.joinable()) listener.join();
        fs::remove_all(root);
        std::cerr << "bounded NDJSON Native run event tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
