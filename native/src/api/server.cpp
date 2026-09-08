#include "vrhino/api/server.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "httplib.h"
#include "vrhino/application/job_manager.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/product/info.h"
#include "vrhino/product/model_list.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/run_request.h"
#include "vrhino/product/version.h"

namespace vrhino::api {
namespace {

constexpr const char* kJsonContentType = "application/json; charset=utf-8";
constexpr const char* kNdjsonContentType =
    "application/x-ndjson; charset=utf-8";
constexpr const char* kApiPrefix = "/api/v1/";
constexpr const char* kModelPrefix = "/api/v1/models/";
constexpr const char* kRunsPath = "/api/v1/runs";
constexpr const char* kRunPrefix = "/api/v1/runs/";
constexpr const char* kEventsSuffix = "/events";
constexpr size_t kEncodedModelReferenceMaxBytes =
    kModelReferenceMaxBytes * 3;

Json text(const std::string& value) {
    return Json(Json::Value(value));
}

Json object(Json::Object value) {
    return Json(Json::Value(std::move(value)));
}

Json integer(const uint64_t value) {
    if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return Json(Json::Value(static_cast<int64_t>(value)));
    return text(std::to_string(value));
}

Json boolean(const bool value) {
    return Json(Json::Value(value));
}

void set_json(httplib::Response& response, const int status,
              const Json& document) {
    response.status = status;
    response.set_content(document.serialize(), kJsonContentType);
}

void set_error(httplib::Response& response, const int status,
               const std::string& code, const std::string& message) {
    set_json(response, status, object({
        {"error", object({
            {"code", text(code)},
            {"message", text(message)},
        })},
    }));
}

void set_error_with_details(httplib::Response& response, const int status,
                            const std::string& code,
                            const std::string& message,
                            Json::Object details) {
    set_json(response, status, object({
        {"error", object({
            {"code", text(code)},
            {"details", object(std::move(details))},
            {"message", text(message)},
        })},
    }));
}

bool is_hex(const unsigned char value) {
    return std::isdigit(value) || (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
}

unsigned char decoded_hex(const char high, const char low) {
    const auto digit = [](const unsigned char value) -> unsigned char {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return value - 'A' + 10;
    };
    return static_cast<unsigned char>(digit(high) * 16 + digit(low));
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

bool has_event_path_shape(const std::string& path) {
    if (!starts_with(path, kRunPrefix) ||
        path.size() <= std::char_traits<char>::length(kRunPrefix) +
                           std::char_traits<char>::length(kEventsSuffix))
        return false;
    const std::string suffix = path.substr(
        std::char_traits<char>::length(kRunPrefix));
    const size_t separator = suffix.find('/');
    return separator != std::string::npos && separator > 0 &&
           suffix.substr(separator) == kEventsSuffix;
}

std::optional<std::string> validated_event_job_id(
    const httplib::Request& request) {
    if (!has_event_path_shape(request.path)) return std::nullopt;
    const size_t begin = std::char_traits<char>::length(kRunPrefix);
    const size_t length = request.path.size() - begin -
                          std::char_traits<char>::length(kEventsSuffix);
    const std::string id = request.path.substr(begin, length);
    if (!application::JobManager::valid_job_id(id)) return std::nullopt;
    return id;
}

struct AfterCursor {
    bool supplied = false;
    uint64_t sequence = 0;
};

std::optional<AfterCursor> parse_after_cursor(
    const httplib::Request& request) {
    const size_t query = request.target.find('?');
    if (query == std::string::npos) {
        if (request.target != request.path) return std::nullopt;
        return AfterCursor{};
    }
    if (request.target.substr(0, query) != request.path) return std::nullopt;
    const std::string parameter = request.target.substr(query + 1);
    constexpr const char* prefix = "after=";
    if (!starts_with(parameter, prefix)) return std::nullopt;
    const std::string value = parameter.substr(
        std::char_traits<char>::length(prefix));
    if (value.empty()) return std::nullopt;
    uint64_t result = 0;
    for (const char byte : value) {
        if (byte < '0' || byte > '9') return std::nullopt;
        const uint64_t digit = static_cast<uint64_t>(byte - '0');
        if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            return std::nullopt;
        result = result * 10 + digit;
    }
    return AfterCursor{true, result};
}

bool is_known_path(const std::string& path) {
    if (path == "/api/v1/version" || path == "/api/v1/models" ||
        path == kRunsPath)
        return true;
    if (starts_with(path, kRunPrefix)) {
        if (has_event_path_shape(path)) return true;
        const std::string suffix = path.substr(
            std::char_traits<char>::length(kRunPrefix));
        return !suffix.empty() && suffix.find('/') == std::string::npos;
    }
    return
           (starts_with(path, kModelPrefix) && path.size() >
                                                std::char_traits<char>::length(kModelPrefix));
}

bool method_allowed(const httplib::Request& request) {
    if (request.path == kRunsPath) return request.method == "POST";
    if (starts_with(request.path, kRunPrefix)) {
        if (has_event_path_shape(request.path))
            return request.method == "GET";
        const std::string suffix = request.path.substr(
            std::char_traits<char>::length(kRunPrefix));
        if (!suffix.empty() && suffix.find('/') == std::string::npos)
            return request.method == "GET" || request.method == "DELETE";
    }
    return request.method == "GET";
}

std::optional<std::string> validated_job_id(const httplib::Request& request) {
    if (!starts_with(request.path, kRunPrefix)) return std::nullopt;
    const std::string id = request.path.substr(
        std::char_traits<char>::length(kRunPrefix));
    if (id.find('/') != std::string::npos ||
        !application::JobManager::valid_job_id(id))
        return std::nullopt;
    return id;
}

std::optional<std::string> validated_model_reference(
    const httplib::Request& request) {
    const size_t query = request.target.find('?');
    const std::string raw_path = request.target.substr(0, query);
    if (!starts_with(raw_path, kModelPrefix)) return std::nullopt;

    const std::string raw = raw_path.substr(
        std::char_traits<char>::length(kModelPrefix));
    if (raw.empty() || raw.size() > kEncodedModelReferenceMaxBytes ||
        query != std::string::npos) {
        return std::nullopt;
    }

    size_t encoded_slashes = 0;
    size_t encoded_colons = 0;
    for (size_t index = 0; index < raw.size(); ++index) {
        const unsigned char byte = static_cast<unsigned char>(raw[index]);
        if (byte == '/' || byte == ':' || byte == '?' || byte == '#' ||
            byte == '\\' || byte == 0 || byte >= 0x80) {
            return std::nullopt;
        }
        if (byte != '%') continue;
        if (index + 2 >= raw.size() ||
            !is_hex(static_cast<unsigned char>(raw[index + 1])) ||
            !is_hex(static_cast<unsigned char>(raw[index + 2]))) {
            return std::nullopt;
        }
        const unsigned char decoded = decoded_hex(raw[index + 1], raw[index + 2]);
        if (decoded == 0 || decoded == '%' || decoded == '?' || decoded == '#' ||
            decoded == '\\' || decoded >= 0x80) {
            return std::nullopt;
        }
        if (decoded == '/') ++encoded_slashes;
        if (decoded == ':') ++encoded_colons;
        index += 2;
    }
    if (encoded_slashes != 1 || encoded_colons != 1) return std::nullopt;

    if (!starts_with(request.path, kModelPrefix)) return std::nullopt;
    const std::string decoded = request.path.substr(
        std::char_traits<char>::length(kModelPrefix));
    if (decoded.empty() || decoded.size() > kModelReferenceMaxBytes ||
        decoded.find('\0') != std::string::npos ||
        decoded.find('%') != std::string::npos ||
        decoded.find('?') != std::string::npos ||
        decoded.find('#') != std::string::npos ||
        decoded.find('\\') != std::string::npos) {
        return std::nullopt;
    }

    try {
        const product::PackageIdentity identity =
            product::parse_package_reference(decoded);
        if (identity.reference() != decoded) return std::nullopt;
    } catch (const product::ModelPackageError&) {
        return std::nullopt;
    }
    return decoded;
}

bool has_json_content_type(const httplib::Response& response) {
    const auto iterator = response.headers.find("Content-Type");
    return iterator != response.headers.end() &&
           iterator->second == kJsonContentType;
}

Json job_snapshot_json(const application::JobSnapshot& snapshot) {
    Json::Object result = {
        {"cancel_requested", boolean(snapshot.cancel_requested)},
        {"id", text(snapshot.id)},
        {"model", text(snapshot.model_reference)},
        {"output", object({
            {"managed", boolean(snapshot.output_managed)},
            {"path", text(snapshot.output_path.string())},
        })},
        {"status", text(product::run_lifecycle_state_name(snapshot.state))},
    };
    if (snapshot.stage)
        result.emplace("stage", text(product::run_stage_name(*snapshot.stage)));
    if (snapshot.completed && snapshot.total && snapshot.unit) {
        result.emplace("progress", object({
            {"completed", integer(*snapshot.completed)},
            {"total", integer(*snapshot.total)},
            {"unit", text(product::run_progress_unit_name(*snapshot.unit))},
        }));
    }
    if (snapshot.failure) {
        result.emplace("error", object({
            {"code", text(snapshot.failure->code)},
            {"message", text(snapshot.failure->message)},
        }));
    }
    return object(std::move(result));
}

bool terminal(const product::RunLifecycleState state) noexcept {
    return state == product::RunLifecycleState::Succeeded ||
           state == product::RunLifecycleState::Failed ||
           state == product::RunLifecycleState::Cancelled;
}

bool valid_event_message(const product::RunEvent& event) {
    if (event.message.empty() ||
        event.message.size() > kEventMessageMaxBytes ||
        event.kind == product::RunEventKind::Message)
        return false;
    try {
        const std::string encoded = text(event.message).serialize();
        return Json::parse(encoded).string() == event.message;
    } catch (...) {
        return false;
    }
}

std::optional<std::string> event_line(const product::RunEvent& event) {
    Json::Object fields = {
        {"kind", text(product::run_event_kind_name(event.kind))},
        {"sequence", integer(event.sequence)},
    };
    if (event.state)
        fields.emplace("state", text(product::run_lifecycle_state_name(
                                    *event.state)));
    if (event.stage)
        fields.emplace("stage", text(product::run_stage_name(*event.stage)));
    if (event.completed)
        fields.emplace("completed", integer(*event.completed));
    if (event.total) fields.emplace("total", integer(*event.total));
    if (event.unit)
        fields.emplace("unit", text(product::run_progress_unit_name(*event.unit)));
    const bool include_message = valid_event_message(event);
    if (include_message) fields.emplace("message", text(event.message));

    std::string result = object(fields).serialize() + '\n';
    if (result.size() > kEventLineMaxBytes && include_message) {
        fields.erase("message");
        result = object(std::move(fields)).serialize() + '\n';
    }
    if (result.size() > kEventLineMaxBytes) return std::nullopt;
    return result;
}

struct EventSubscriberLease {
    explicit EventSubscriberLease(std::atomic<size_t>& value) : count(value) {}
    ~EventSubscriberLease() { count.fetch_sub(1, std::memory_order_acq_rel); }
    std::atomic<size_t>& count;
};

std::shared_ptr<EventSubscriberLease> acquire_event_subscriber(
    std::atomic<size_t>& count) {
    size_t current = count.load(std::memory_order_acquire);
    while (current < kEventSubscriberCapacity) {
        if (count.compare_exchange_weak(current, current + 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire))
            return std::make_shared<EventSubscriberLease>(count);
    }
    return {};
}

class EventStreamState {
public:
    EventStreamState(application::JobManager& manager, std::string job_id,
                     uint64_t after,
                     application::JobEventSnapshot initial)
        : jobs_(manager), id_(std::move(job_id)), cursor_(after),
          current_(std::move(initial)) {}

    bool provide(httplib::DataSink& sink) {
        if (!sink.is_writable()) return false;
        for (;;) {
            if (next_event_ < current_.events.size()) {
                const product::RunEvent& event = current_.events[next_event_];
                const std::optional<std::string> line = event_line(event);
                if (!line || !sink.is_writable() ||
                    !sink.write(line->data(), line->size()))
                    return false;
                cursor_ = event.sequence;
                ++next_event_;
                return true;
            }
            if (terminal(current_.state) || current_.manager_stopping) {
                sink.done();
                return true;
            }

            std::optional<application::JobEventSnapshot> next =
                jobs_.wait_for_events(
                    id_, cursor_,
                    std::chrono::milliseconds(kEventWaitMilliseconds));
            if (!next || next->history_truncated) return false;
            current_ = std::move(*next);
            next_event_ = 0;
            if (!sink.is_writable()) return false;
        }
    }

private:
    application::JobManager& jobs_;
    std::string id_;
    uint64_t cursor_ = 0;
    application::JobEventSnapshot current_;
    size_t next_event_ = 0;
};

void set_job_not_found(httplib::Response& response) {
    set_error(response, 404, "job_not_found", "job was not found");
}

}  // namespace

bool valid_server_port(const uint32_t port) noexcept {
    return port >= 1 && port <= std::numeric_limits<uint16_t>::max();
}

bool is_loopback_host(const std::string& host) noexcept {
    return host == "127.0.0.1" || host == "localhost" || host == "::1" ||
           host == "[::1]";
}

std::optional<std::string> non_loopback_bind_warning(const std::string& host) {
    if (is_loopback_host(host)) return std::nullopt;
    return "WARNING: VRhino API is binding to a non-loopback address.\n"
           "The VRhino local API has no authentication or TLS.";
}

class Server::Impl {
public:
    explicit Impl(ServerConfig value)
        : config(std::move(value)), cache(config.cache_root) {
        if (config.host.empty()) {
            throw std::invalid_argument("server host must not be empty");
        }
        if (!valid_server_port(config.port)) {
            throw std::invalid_argument("server port must be between 1 and 65535");
        }
        if (config.run_executor) {
            jobs = std::make_unique<application::JobManager>(
                cache.layout().root / "runs", config.run_executor);
        }

        http.new_task_queue = [] {
            return new httplib::ThreadPool(
                kWorkerThreads, kWorkerThreads, kWorkerQueueCapacity);
        };
        // cpp-httplib defaults to SO_REUSEPORT where available, which permits
        // multiple listeners on one port. A single local API instance must
        // instead fail closed when its configured port is occupied.
        http.set_socket_options([](socket_t socket) {
            httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, 1);
        });
        http.set_payload_max_length(kRequestBodyMaxBytes);
        http.set_read_timeout(kReadTimeoutSeconds);
        http.set_write_timeout(kWriteTimeoutSeconds);
        http.set_keep_alive_timeout(kKeepAliveTimeoutSeconds);
        http.set_keep_alive_max_count(kKeepAliveMaxRequests);
        http.set_idle_interval(0, 100000);

        http.set_pre_routing_handler([](const httplib::Request& request,
                                        httplib::Response& response) {
            if (!starts_with(request.path, kApiPrefix)) {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            const bool event_path = has_event_path_shape(request.path);
            if ((request.target.find('?') != std::string::npos && !event_path) ||
                (event_path && !parse_after_cursor(request))) {
                set_error(response, 400, "invalid_request",
                          "request target is invalid");
                return httplib::Server::HandlerResponse::Handled;
            }
            if (request.has_header("Content-Encoding")) {
                set_error(response, 400, "invalid_request",
                          "content encoding is not supported");
                return httplib::Server::HandlerResponse::Handled;
            }
            const uint64_t content_length =
                request.get_header_value_u64("Content-Length");
            if (content_length > kRequestBodyMaxBytes ||
                request.body.size() > kRequestBodyMaxBytes) {
                set_error(response, 413, "invalid_request",
                          "request body is too large");
                return httplib::Server::HandlerResponse::Handled;
            }
            const bool run_submission =
                request.path == kRunsPath && request.method == "POST";
            if (run_submission) {
                const std::string content_type =
                    request.get_header_value("Content-Type");
                if (request.has_header("Transfer-Encoding") ||
                    (content_type != "application/json" &&
                     content_type != "application/json; charset=utf-8")) {
                    set_error(response, 400, "invalid_request",
                              "run request must be bounded JSON");
                    return httplib::Server::HandlerResponse::Handled;
                }
            } else if (content_length > 0 || !request.body.empty() ||
                       request.has_header("Transfer-Encoding")) {
                    set_error(response, 400, "invalid_request",
                              "request body is not allowed");
                    return httplib::Server::HandlerResponse::Handled;
            }
            if (is_known_path(request.path) && !method_allowed(request)) {
                set_error(response, 405, "method_not_allowed",
                          "method is not allowed for this resource");
                return httplib::Server::HandlerResponse::Handled;
            }
            if (starts_with(request.path, kModelPrefix) &&
                request.path.size() >
                    std::char_traits<char>::length(kModelPrefix) &&
                !validated_model_reference(request)) {
                set_error(response, 400, "invalid_request",
                          "model reference is invalid");
                return httplib::Server::HandlerResponse::Handled;
            }
            if (starts_with(request.path, kRunPrefix)) {
                const bool invalid_event_id = event_path &&
                    !validated_event_job_id(request);
                const std::string suffix = request.path.substr(
                    std::char_traits<char>::length(kRunPrefix));
                const bool invalid_status_id = !event_path && !suffix.empty() &&
                    suffix.find('/') == std::string::npos &&
                    !validated_job_id(request);
                if (invalid_event_id || invalid_status_id) {
                    set_error(response, 400, "invalid_request",
                              "job identifier is invalid");
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        http.set_exception_handler([](const httplib::Request&,
                                      httplib::Response& response,
                                      std::exception_ptr) {
            set_error(response, 500, "internal", "internal server error");
        });

        http.set_error_handler([](const httplib::Request&,
                                  httplib::Response& response) {
            if (has_json_content_type(response)) {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            switch (response.status) {
                case 404:
                    set_error(response, 404, "not_found", "route not found");
                    break;
                case 405:
                    set_error(response, 405, "method_not_allowed",
                              "method is not allowed for this resource");
                    break;
                case 413:
                    set_error(response, 413, "invalid_request",
                              "request body is too large");
                    break;
                case 414:
                    set_error(response, 414, "invalid_request",
                              "request target is too long");
                    break;
                case 400:
                    set_error(response, 400, "invalid_request",
                              "request is invalid");
                    break;
                default:
                    set_error(response, 500, "internal",
                              "internal server error");
                    break;
            }
            return httplib::Server::HandlerResponse::Handled;
        });

        http.Get("/api/v1/version", [](const httplib::Request&,
                                       httplib::Response& response) {
            set_json(response, 200, product::build_version_info());
        });
        http.Get("/api/v1/models", [this](const httplib::Request&,
                                          httplib::Response& response) {
            try {
                set_json(response, 200, product::build_local_model_list(cache));
            } catch (...) {
                set_error(response, 500, "internal", "internal server error");
            }
        });
        http.Get(R"(/api/v1/models/(.+))",
                 [this](const httplib::Request& request,
                        httplib::Response& response) {
            const std::optional<std::string> reference =
                validated_model_reference(request);
            if (!reference) {
                set_error(response, 400, "invalid_request",
                          "model reference is invalid");
                return;
            }
            try {
                const product::ResolvedRunnableModel model =
                    cache.resolve(*reference, false);
                set_json(response, 200, product::build_model_info(model));
            } catch (const product::ModelPackageError& error) {
                if (error.code() == product::ModelPackageErrorCode::ModelNotFound) {
                    set_error(response, 404, "model_not_found",
                              "model is not installed");
                } else {
                    set_error(response, 500, "internal", "internal server error");
                }
            } catch (...) {
                set_error(response, 500, "internal", "internal server error");
            }
        });

        http.Post(kRunsPath,
                  [this](const httplib::Request& request,
                         httplib::Response& response) {
            product::ProductRunDocument document;
            try {
                const JsonParseLimits limits{
                    kRequestBodyMaxBytes,
                    kRunRequestJsonMaxDepth,
                    kRunRequestJsonMaxContainerEntries,
                    kRunRequestJsonMaxTotalValues,
                    kRunRequestJsonMaxStringBytes,
                };
                document = product::parse_product_run_document(
                    Json::parse(request.body, limits));
            } catch (const Error&) {
                set_error(response, 400, "invalid_request",
                          "request JSON is invalid");
                return;
            } catch (const product::RunRequestError&) {
                set_error(response, 400, "invalid_request",
                          "run request is invalid");
                return;
            }

            try {
                product::ResolvedRunnableModel model =
                    cache.resolve(document.model_reference, false);
                product::RunOptions options =
                    product::map_product_run_options(model, document);
                if (!jobs) {
                    set_error(response, 503, "model_unavailable",
                              "Product execution is unavailable in this build");
                    return;
                }
                const application::JobSnapshot accepted = jobs->submit(
                    std::move(model), std::move(options));
                response.set_header("Location", "/api/v1/runs/" + accepted.id);
                set_json(response, 202, job_snapshot_json(accepted));
            } catch (const product::ModelPackageError& error) {
                if (error.code() == product::ModelPackageErrorCode::ModelNotFound)
                    set_error(response, 404, "model_not_found",
                              "model is not installed");
                else if (error.code() == product::ModelPackageErrorCode::PackageInvalid)
                    set_error(response, 400, "invalid_request",
                              "model reference is invalid");
                else
                    set_error(response, 500, "internal", "internal server error");
            } catch (const product::RunRequestError& error) {
                if (error.code() == product::RunRequestErrorCode::ModelUnavailable)
                    set_error(response, 400, "model_unavailable",
                              "model has no Native API execution contract");
                else
                    set_error(response, 400, "invalid_request",
                              "run request is invalid");
            } catch (const application::JobAdmissionError& error) {
                if (error.code() == application::JobAdmissionErrorCode::Conflict)
                    set_error(response, 409, "conflict", error.what());
                else if (error.code() ==
                         application::JobAdmissionErrorCode::Overloaded)
                    set_error(response, 429, "overloaded",
                              "Product job queue is full");
                else
                    set_error(response, 503, "model_unavailable",
                              "Product execution is shutting down");
            } catch (...) {
                set_error(response, 500, "internal", "internal server error");
            }
        });

        http.Get(R"(/api/v1/runs/([^/]+)/events)",
                 [this](const httplib::Request& request,
                        httplib::Response& response) {
            const std::optional<std::string> id =
                validated_event_job_id(request);
            const std::optional<AfterCursor> after =
                parse_after_cursor(request);
            if (!id || !after) {
                set_error(response, 400, "invalid_request",
                          "event stream request is invalid");
                return;
            }
            if (!jobs) {
                set_job_not_found(response);
                return;
            }

            std::optional<application::JobEventSnapshot> initial =
                jobs->events_after(*id, after->sequence);
            if (!initial) {
                set_job_not_found(response);
                return;
            }
            if (after->supplied && after->sequence > initial->latest_sequence) {
                set_error(response, 400, "invalid_request",
                          "event cursor is newer than this job");
                return;
            }
            if (after->supplied && initial->history_truncated) {
                set_error_with_details(response, 409,
                    "event_history_truncated",
                    "requested event history is no longer retained",
                    {
                        {"first_available_sequence",
                         integer(initial->first_available_sequence)},
                        {"latest_sequence", integer(initial->latest_sequence)},
                    });
                return;
            }

            std::shared_ptr<EventSubscriberLease> lease =
                acquire_event_subscriber(event_subscribers);
            if (!lease) {
                set_error(response, 429, "overloaded",
                          "event subscriber capacity is full");
                return;
            }

            uint64_t stream_after = after->sequence;
            if (!after->supplied && initial->first_available_sequence > 0)
                stream_after = initial->first_available_sequence - 1;
            auto state = std::make_shared<EventStreamState>(
                *jobs, *id, stream_after, std::move(*initial));
            response.status = 200;
            response.set_header("Cache-Control", "no-store");
            response.set_chunked_content_provider(
                kNdjsonContentType,
                [state, lease](size_t, httplib::DataSink& sink) {
                    (void)lease;
                    return state->provide(sink);
                });
        });

        http.Get(R"(/api/v1/runs/([^/]+))",
                 [this](const httplib::Request& request,
                        httplib::Response& response) {
            const std::optional<std::string> id = validated_job_id(request);
            if (!id) {
                set_error(response, 400, "invalid_request",
                          "job identifier is invalid");
                return;
            }
            if (!jobs) {
                set_job_not_found(response);
                return;
            }
            const std::optional<application::JobSnapshot> current =
                jobs->snapshot(*id);
            if (!current) {
                set_job_not_found(response);
                return;
            }
            set_json(response, 200, job_snapshot_json(*current));
        });

        http.Delete(R"(/api/v1/runs/([^/]+))",
                    [this](const httplib::Request& request,
                           httplib::Response& response) {
            const std::optional<std::string> id = validated_job_id(request);
            if (!id) {
                set_error(response, 400, "invalid_request",
                          "job identifier is invalid");
                return;
            }
            if (!jobs) {
                set_job_not_found(response);
                return;
            }
            const std::optional<application::JobCancelResult> cancelled =
                jobs->cancel(*id);
            if (!cancelled) {
                set_job_not_found(response);
                return;
            }
            const int status = cancelled->disposition ==
                    application::JobCancelDisposition::
                        RunningCancellationRequested
                ? 202 : 200;
            set_json(response, status,
                     job_snapshot_json(cancelled->snapshot));
        });
    }

    ServerConfig config;
    product::LocalModelCache cache;
    std::unique_ptr<application::JobManager> jobs;
    std::atomic<size_t> event_subscribers{0};
    httplib::Server http;
    bool bound = false;
};

Server::Server(ServerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Server::~Server() = default;

int Server::bind() {
    if (impl_->bound) return -1;
    impl_->bound = impl_->http.bind_to_port(impl_->config.host,
                                            impl_->config.port);
    return impl_->bound ? impl_->config.port : -1;
}

bool Server::listen() {
    if (!impl_->bound) return false;
    const bool result = impl_->http.listen_after_bind();
    impl_->bound = false;
    return result;
}

void Server::stop() noexcept {
    impl_->http.stop();
    if (impl_->jobs) impl_->jobs->shutdown();
}

const ServerConfig& Server::config() const noexcept {
    return impl_->config;
}

}  // namespace vrhino::api
