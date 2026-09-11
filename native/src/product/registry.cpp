#include "registry_transport.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include "vrhino/product/windows_cache.h"
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <curl/curl.h>

#include "vrhino/error.h"
#include "vrhino/json.h"

namespace vrhino::product {
namespace fs = std::filesystem;

namespace registry_detail {

constexpr uint64_t kMaximumRegistryDocumentBytes = 8U * 1024U * 1024U;

[[noreturn]] void fail(const ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

bool development_http_url(const std::string& url) {
    return starts_with(url, "http://127.0.0.1:") ||
           starts_with(url, "http://localhost:") ||
           starts_with(url, "http://[::1]:");
}

void validate_remote_url(const std::string& url, const RegistryOptions& options) {
    if (starts_with(url, "https://")) return;
    if (options.allow_development_http && development_http_url(url)) return;
    fail(ModelPackageErrorCode::RegistryUnavailable,
         "registry and artifact URLs must use HTTPS; loopback HTTP requires --allow-http");
}

std::string trim_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}

const Json& object_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_object()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry field " + key + " must be an object");
    }
    return *result;
}

const Json& array_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_array()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry field " + key + " must be an array");
    }
    return *result;
}

std::string string_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_string() || result->string().empty()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry field " + key + " must be a non-empty string");
    }
    return result->string();
}

uint64_t unsigned_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_int() || result->integer() < 0) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry field " + key + " must be a non-negative integer");
    }
    return static_cast<uint64_t>(result->integer());
}

int64_t integer_field(const Json& value, const std::string& key) {
    const Json* result = value.find(key);
    if (result == nullptr || !result->is_int()) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry field " + key + " must be an integer");
    }
    return result->integer();
}



struct CurlHandle {
    CURL* value = curl_easy_init();
    ~CurlHandle() { if (value != nullptr) curl_easy_cleanup(value); }
};

struct CurlHeaders {
    curl_slist* value = nullptr;
    ~CurlHeaders() { curl_slist_free_all(value); }
};

class CurlGlobalState {
public:
    CurlGlobalState() {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            fail(ModelPackageErrorCode::NetworkError, "libcurl initialization failed");
        }
    }
    ~CurlGlobalState() { curl_global_cleanup(); }
};

CurlGlobalState& curl_global_state() {
    static CurlGlobalState state;
    return state;
}

void configure_common(CURL* curl,
                      const std::string& url,
                      const RegistryOptions& options,
                      char* error_buffer) {
    validate_remote_url(url, options);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "vrhino/0.1");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, options.maximum_redirects);
    // Never forward origin credentials across hosts. Cookies and netrc are not
    // part of Native acquisition and are explicitly disabled as well.
    curl_easy_setopt(curl, CURLOPT_UNRESTRICTED_AUTH, 0L);
    curl_easy_setopt(curl, CURLOPT_NETRC, CURL_NETRC_IGNORED);
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, static_cast<const char*>(nullptr));
    curl_easy_setopt(curl, CURLOPT_COOKIEJAR, static_cast<const char*>(nullptr));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, options.connect_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, options.low_speed_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    if (starts_with(url, "https://localhost:") ||
        starts_with(url, "https://127.0.0.1:") ||
        starts_with(url, "https://[::1]:") || development_http_url(url)) {
        curl_easy_setopt(curl, CURLOPT_NOPROXY, "localhost,127.0.0.1,::1");
    }
    // CURLOPT_*_PROTOCOLS_STR was introduced in libcurl 7.85. Keep the same
    // fail-closed HTTPS policy when building the Linux package against the
    // Ubuntu 22.04 baseline (libcurl 7.81).
#if LIBCURL_VERSION_NUM >= 0x075500
    const char* protocols = options.allow_development_http ? "https,http" : "https";
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, protocols);
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, protocols);
#else
    const long protocols = CURLPROTO_HTTPS |
        (options.allow_development_http ? CURLPROTO_HTTP : 0L);
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS, protocols);
    curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, protocols);
#endif
    if (!options.ca_file.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, options.ca_file.c_str());
    }
}

size_t append_string(char* data, const size_t size, const size_t count, void* user_data) {
    const size_t bytes = size * count;
    auto* output = static_cast<std::string*>(user_data);
    if (bytes > kMaximumRegistryDocumentBytes - output->size()) return 0;
    output->append(data, bytes);
    return bytes;
}

bool retryable_curl_error(const CURLcode code) {
    return code == CURLE_COULDNT_CONNECT || code == CURLE_OPERATION_TIMEDOUT ||
           code == CURLE_RECV_ERROR || code == CURLE_SEND_ERROR ||
           code == CURLE_PARTIAL_FILE || code == CURLE_GOT_NOTHING;
}

bool retryable_http_status(const long status) {
    return status == 429 || status == 500 || status == 502 || status == 503 ||
           status == 504;
}

NativeDownloadFailureClass classify_download_failure(const CURLcode code,
                                                      const long response) {
    if (code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_COULDNT_RESOLVE_PROXY)
        return NativeDownloadFailureClass::DnsResolution;
    if (code == CURLE_COULDNT_CONNECT)
        return NativeDownloadFailureClass::Connection;
    if (code == CURLE_OPERATION_TIMEDOUT)
        return NativeDownloadFailureClass::Timeout;
    if (code == CURLE_SSL_CONNECT_ERROR)
        return NativeDownloadFailureClass::TlsHandshake;
    if (code == CURLE_RECV_ERROR || code == CURLE_SEND_ERROR ||
        code == CURLE_PARTIAL_FILE || code == CURLE_GOT_NOTHING)
        return NativeDownloadFailureClass::InterruptedTransfer;
    if (response != 0 || code == CURLE_HTTP_RETURNED_ERROR)
        return NativeDownloadFailureClass::HttpStatus;
    return NativeDownloadFailureClass::Other;
}

int cancellation_progress(void* user_data, const curl_off_t, const curl_off_t,
                          const curl_off_t, const curl_off_t) {
    const auto* cancellation =
        static_cast<const std::function<bool()>*>(user_data);
    return cancellation != nullptr && *cancellation && (*cancellation)() ? 1 : 0;
}

std::string http_get_text(const std::string& url,
                          const RegistryOptions& options,
                          const ModelPackageErrorCode failure_code) {
    (void)curl_global_state();
    for (int attempt = 0; attempt <= options.retry_count; ++attempt) {
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled, "Pull interrupted.");
        CurlHandle handle;
        if (handle.value == nullptr) fail(ModelPackageErrorCode::NetworkError,
                                          "cannot allocate libcurl handle");
        std::string body;
        char error_buffer[CURL_ERROR_SIZE] = {};
        configure_common(handle.value, url, options, error_buffer);
        CurlHeaders headers;
        headers.value = curl_slist_append(headers.value, "Accept-Encoding: identity");
        curl_easy_setopt(handle.value, CURLOPT_HTTPHEADER, headers.value);
        curl_easy_setopt(handle.value, CURLOPT_WRITEFUNCTION, append_string);
        curl_easy_setopt(handle.value, CURLOPT_WRITEDATA, &body);
        if (options.cancellation_requested) {
            curl_easy_setopt(handle.value, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(handle.value, CURLOPT_XFERINFOFUNCTION,
                             cancellation_progress);
            curl_easy_setopt(handle.value, CURLOPT_XFERINFODATA,
                             &options.cancellation_requested);
        }
        const CURLcode code = curl_easy_perform(handle.value);
        long response = 0;
        curl_easy_getinfo(handle.value, CURLINFO_RESPONSE_CODE, &response);
        if (code == CURLE_OK && response >= 200 && response < 300) return body;
        if (code == CURLE_ABORTED_BY_CALLBACK && options.cancellation_requested &&
            options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled, "Pull interrupted.");
        if (!retryable_curl_error(code) || attempt == options.retry_count) {
            const std::string detail = error_buffer[0] != '\0' ? error_buffer
                                                                : curl_easy_strerror(code);
            fail(failure_code,
                 "GET " + url + " failed (HTTP " + std::to_string(response) + "): " + detail);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
    }
    fail(failure_code, "GET retry loop exhausted");
}

struct ProgressState {
    std::ostream* output = nullptr;
    const RegistryOptions* options = nullptr;
    std::string artifact;
    uint64_t initial_bytes = 0;
    uint64_t expected_bytes = 0;
    uint64_t aggregate_complete = 0;
    uint64_t aggregate_total = 0;
    int last_reported_percent = -1;
};

int transfer_progress(void* user_data,
                      const curl_off_t,
                      const curl_off_t downloaded,
                      const curl_off_t,
                      const curl_off_t) {
    auto* state = static_cast<ProgressState*>(user_data);
    if (state->options != nullptr && state->options->cancellation_requested &&
        state->options->cancellation_requested())
        return 1;
    if (downloaded < 0) return 0;
    const uint64_t current = std::min(
        state->expected_bytes,
        state->initial_bytes + static_cast<uint64_t>(downloaded));
    uint64_t aggregate = std::min(
        state->aggregate_total, state->aggregate_complete + current);
    if (aggregate != 0 && aggregate == state->aggregate_total)
        --aggregate;
    if (state->options != nullptr && state->options->progress)
        state->options->progress(aggregate, state->aggregate_total);
    if (state->output != nullptr &&
        (state->options == nullptr || !state->options->progress)) {
        const int percent = state->aggregate_total == 0
            ? 100
            : static_cast<int>(100 * aggregate / state->aggregate_total);
        if (aggregate == state->aggregate_total || state->last_reported_percent < 0 ||
            percent >= state->last_reported_percent + 10) {
            state->last_reported_percent = percent;
            *state->output << "Downloading " << aggregate << '/'
                           << state->aggregate_total << " bytes (" << percent << "%)\n";
            state->output->flush();
        }
    }
    return 0;
}

struct DownloadWriteState {
#ifdef _WIN32
    windows_cache::StagedFile* output = nullptr;
    std::exception_ptr write_error;
#else
    FILE* output = nullptr;
#endif
    CURL* handle = nullptr;
    uint64_t requested_offset = 0;
    uint64_t expected_size = 0;
    bool content_range_seen = false;
    bool content_range_valid = false;
    bool range_ignored = false;
    bool range_invalid = false;
};

size_t download_header(char* data, const size_t size, const size_t count,
                       void* user_data) {
    const size_t bytes = size * count;
    auto* state = static_cast<DownloadWriteState*>(user_data);
    const std::string line(data, bytes);
    if (line.starts_with("HTTP/")) {
        state->content_range_seen = false;
        state->content_range_valid = false;
        state->range_ignored = false;
        state->range_invalid = false;
        return bytes;
    }
    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (lower.starts_with("content-range:")) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        unsigned long long total = 0;
        state->content_range_seen = true;
        const char* value = line.c_str() + std::string("content-range:").size();
        if (std::sscanf(value, " bytes %llu-%llu/%llu", &start, &end, &total) == 3 &&
            start == state->requested_offset && end >= start &&
            total == state->expected_size) {
            state->content_range_valid = true;
        }
    }
    if (line == "\r\n" && state->requested_offset > 0) {
        long response = 0;
        curl_easy_getinfo(state->handle, CURLINFO_RESPONSE_CODE, &response);
        if (response == 200) state->range_ignored = true;
        else if (response == 206 &&
                 (!state->content_range_seen || !state->content_range_valid))
            state->range_invalid = true;
    }
    return bytes;
}

size_t write_download(char* data, const size_t size, const size_t count,
                      void* user_data) {
    auto* state = static_cast<DownloadWriteState*>(user_data);
    if (state->range_ignored || state->range_invalid) return 0;
    const size_t bytes = size * count;
#ifdef _WIN32
    try { state->output->write(data, bytes); return bytes; }
    catch (...) { state->write_error = std::current_exception(); return 0; }
#else
    return std::fwrite(data, 1, bytes, state->output);
#endif
}

NativeDownloadResult download_file(const std::string& url,
                             const fs::path& partial,
                             const uint64_t expected_size,
                             const std::string& label,
                             const uint64_t aggregate_complete,
                             const uint64_t aggregate_total,
                             const RegistryOptions& options,
                             std::ostream* progress_output,
                             const std::string& bearer_token) {
    (void)curl_global_state();
    std::error_code error;
#ifdef _WIN32
    windows_cache::ensure_directory(partial.parent_path());
    bool existing = fs::exists(partial);
    auto sink = std::make_shared<windows_cache::StagedFile>(partial, existing, true);
    uint64_t offset = sink->size();
    if (offset > expected_size) { sink->truncate(0); offset = 0; }
    const uint64_t original_offset = offset;
    NativeDownloadResult result;
    result.staging = sink;
    if (offset == expected_size) { result.resumed_bytes = offset; return result; }
#else
    fs::create_directories(partial.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create download directory: " + error.message());
    uint64_t offset = fs::exists(partial) ? fs::file_size(partial, error) : 0;
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot inspect partial download: " + error.message());
    if (offset > expected_size) {
        fs::remove(partial, error);
        if (error)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot discard oversized partial download: " + error.message());
        offset = 0;
    }
    const uint64_t original_offset = offset;
    if (offset == expected_size) return NativeDownloadResult{0, offset};

    NativeDownloadResult result;
#endif
    int failures = 0;
    bool restarted_for_no_range = false;
    while (failures <= options.retry_count) {
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Download interrupted. Partial download preserved for resume.");
#ifdef _WIN32
        const uint64_t attempt_offset = sink->size();
        auto* output = sink.get();
#else
        const uint64_t attempt_offset = fs::exists(partial) ? fs::file_size(partial, error) : 0;
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect partial download: " + error.message());
        FILE* output = std::fopen(partial.c_str(), attempt_offset == 0 ? "wb" : "ab");
        if (output == nullptr) fail(ModelPackageErrorCode::CacheError,
                                    "cannot open partial download: " + partial.string());
#endif
        CurlHandle handle;
        if (handle.value == nullptr) {
#ifndef _WIN32
            std::fclose(output);
#endif
            fail(ModelPackageErrorCode::NetworkError, "cannot allocate libcurl handle");
        }
        char error_buffer[CURL_ERROR_SIZE] = {};
        configure_common(handle.value, url, options, error_buffer);
        CurlHeaders headers;
        headers.value = curl_slist_append(headers.value, "Accept-Encoding: identity");
        if (!bearer_token.empty()) {
            // Use libcurl's origin-scoped bearer facility instead of a custom
            // header list, which also makes cross-host redirect isolation an
            // explicit transport contract.
            curl_easy_setopt(handle.value, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
            curl_easy_setopt(handle.value, CURLOPT_XOAUTH2_BEARER,
                             bearer_token.c_str());
        }
        curl_easy_setopt(handle.value, CURLOPT_HTTPHEADER, headers.value);
#ifdef _WIN32
        DownloadWriteState write_state;
        write_state.output = output; write_state.handle = handle.value;
        write_state.requested_offset = attempt_offset; write_state.expected_size = expected_size;
#else
        DownloadWriteState write_state{
            output, handle.value, attempt_offset, expected_size};
#endif
        curl_easy_setopt(handle.value, CURLOPT_HEADERFUNCTION, download_header);
        curl_easy_setopt(handle.value, CURLOPT_HEADERDATA, &write_state);
        curl_easy_setopt(handle.value, CURLOPT_WRITEFUNCTION, write_download);
        curl_easy_setopt(handle.value, CURLOPT_WRITEDATA, &write_state);
        curl_easy_setopt(handle.value, CURLOPT_NOPROGRESS, 0L);
        ProgressState progress{progress_output, &options, label, attempt_offset,
                               expected_size, aggregate_complete, aggregate_total, -1};
        curl_easy_setopt(handle.value, CURLOPT_XFERINFOFUNCTION, transfer_progress);
        curl_easy_setopt(handle.value, CURLOPT_XFERINFODATA, &progress);
        (void)transfer_progress(&progress, 0, 0, 0, 0);
        std::string range;
        if (attempt_offset > 0) {
            range = std::to_string(attempt_offset) + "-";
            curl_easy_setopt(handle.value, CURLOPT_RANGE, range.c_str());
        }
        const CURLcode code = curl_easy_perform(handle.value);
#ifdef _WIN32
        if (write_state.write_error) std::rethrow_exception(write_state.write_error);
        sink->flush();
#else
        std::fflush(output);
        std::fclose(output);
#endif
        curl_off_t transferred = 0;
        curl_easy_getinfo(handle.value, CURLINFO_SIZE_DOWNLOAD_T, &transferred);
        if (transferred > 0) result.network_bytes += static_cast<uint64_t>(transferred);
        long response = 0;
        curl_easy_getinfo(handle.value, CURLINFO_RESPONSE_CODE, &response);
        long proxy_response = 0;
        curl_easy_getinfo(handle.value, CURLINFO_HTTP_CONNECTCODE, &proxy_response);
        const long failure_response = response != 0 ? response : proxy_response;

        if (code == CURLE_ABORTED_BY_CALLBACK && options.cancellation_requested &&
            options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Download interrupted. Partial download preserved for resume.");

        if (attempt_offset > 0 &&
            (write_state.range_ignored || response == 200 || code == CURLE_RANGE_ERROR) &&
            !restarted_for_no_range) {
#ifdef _WIN32
            sink->truncate(0);
#else
            fs::resize_file(partial, 0, error);
#endif
            if (error) fail(ModelPackageErrorCode::CacheError,
                            "cannot restart non-range download: " + error.message());
            restarted_for_no_range = true;
            continue;
        }
        if (write_state.range_invalid)
            throw NativeDownloadError(
                ModelPackageErrorCode::DownloadResumeFailed,
                NativeDownloadFailureClass::ResumeProtocol, response,
                result.network_bytes,
                "server returned a mismatched Content-Range for " + label);

#ifdef _WIN32
        const uint64_t actual_size = sink->size();
#else
        const uint64_t actual_size = fs::file_size(partial, error);
#endif
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect completed download: " + error.message());
        if (code == CURLE_OK && response >= 200 && response < 300 &&
            actual_size == expected_size) {
            result.resumed_bytes = restarted_for_no_range ? 0 : original_offset;
            return result;
        }
        if (actual_size > expected_size) {
#ifdef _WIN32
            sink->discard();
#else
            fs::remove(partial, error);
#endif
            fail(ModelPackageErrorCode::DownloadFailed,
                 "download exceeds declared artifact size: " + label);
        }
        if ((!retryable_curl_error(code) &&
             !retryable_http_status(failure_response) && code != CURLE_OK) ||
            failures == options.retry_count) {
            const std::string detail = error_buffer[0] != '\0' ? error_buffer
                                                                : curl_easy_strerror(code);
            throw NativeDownloadError(
                attempt_offset > 0 ? ModelPackageErrorCode::DownloadResumeFailed
                                   : ModelPackageErrorCode::DownloadFailed,
                classify_download_failure(code, failure_response), failure_response,
                result.network_bytes,
                "artifact download failed for " + label + " (HTTP " +
                    std::to_string(failure_response) + "): " + detail);
        }
        ++failures;
        if (options.cancellation_requested && options.cancellation_requested())
            fail(ModelPackageErrorCode::Cancelled,
                 "Download interrupted. Partial download preserved for resume.");
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * failures));
    }
    throw NativeDownloadError(
        ModelPackageErrorCode::DownloadFailed,
        NativeDownloadFailureClass::Other, 0, result.network_bytes,
        "download retry loop exhausted: " + label);
}

void write_text_file(const fs::path& path, const std::string& text) {
#ifdef _WIN32
    windows_cache::write_text(path, text, false, true);
#else
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (error) fail(ModelPackageErrorCode::CacheError,
                    "cannot create manifest download directory: " + error.message());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) fail(ModelPackageErrorCode::CacheError,
                      "cannot write downloaded manifest: " + path.string());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
    if (!output) fail(ModelPackageErrorCode::CacheError,
                      "cannot write downloaded manifest: " + path.string());
#endif
}

struct RegistryDescriptor {
    std::string identity;
    std::string manifest_url;
    uint64_t manifest_size = 0;
    std::string manifest_sha256;
    std::map<std::string, std::string> artifact_urls;
};



ComponentRegistryDescriptor parse_component_registry_descriptor(
    const std::string& text, const RegistryOptions& options) {
    try {
        const Json root = Json::parse(text);
        if (!root.is_object()) fail(ModelPackageErrorCode::ComponentInvalid,
                                    "component registry descriptor root must be an object");
        if (integer_field(root, "registry_schema_version") != kRegistrySchemaVersion) {
            fail(ModelPackageErrorCode::ComponentVersionUnsupported,
                 "unsupported component registry descriptor schema");
        }
        ComponentRegistryDescriptor descriptor;
        descriptor.identity = string_field(root, "identity");
        const Json& package = object_field(root, "package");
        descriptor.package_url = string_field(package, "url");
        descriptor.package_size = unsigned_field(package, "size");
        descriptor.package_sha256 = string_field(package, "sha256");
        if (descriptor.package_sha256.size() != 64) {
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component package SHA256 must have 64 characters");
        }
        validate_remote_url(descriptor.package_url, options);
        return descriptor;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::ComponentInvalid, error.what());
    }
}

RegistryDescriptor parse_registry_descriptor(const std::string& text,
                                             const RegistryOptions& options) {
    try {
        const Json root = Json::parse(text);
        if (!root.is_object()) fail(ModelPackageErrorCode::PackageInvalid,
                                    "registry descriptor root must be an object");
        if (integer_field(root, "registry_schema_version") != kRegistrySchemaVersion) {
            fail(ModelPackageErrorCode::PackageVersionUnsupported,
                 "unsupported registry descriptor schema");
        }
        RegistryDescriptor descriptor;
        descriptor.identity = string_field(root, "identity");
        const Json& manifest = object_field(root, "manifest");
        descriptor.manifest_url = string_field(manifest, "url");
        descriptor.manifest_size = unsigned_field(manifest, "size");
        descriptor.manifest_sha256 = string_field(manifest, "sha256");
        if (descriptor.manifest_sha256.size() != 64) {
            fail(ModelPackageErrorCode::PackageInvalid,
                 "registry manifest SHA256 must have 64 characters");
        }
        validate_remote_url(descriptor.manifest_url, options);
        for (const Json& value : array_field(root, "artifacts").array()) {
            if (!value.is_object()) fail(ModelPackageErrorCode::PackageInvalid,
                                         "registry artifact must be an object");
            const std::string id = string_field(value, "id");
            const std::string url = string_field(value, "url");
            validate_remote_url(url, options);
            if (!descriptor.artifact_urls.emplace(id, url).second) {
                fail(ModelPackageErrorCode::PackageInvalid,
                     "duplicate registry artifact URL: " + id);
            }
        }
        return descriptor;
    } catch (const ModelPackageError&) {
        throw;
    } catch (const Error& error) {
        fail(ModelPackageErrorCode::PackageInvalid, error.what());
    }
}

std::string lock_safe_identity(const PackageIdentity& identity) {
    return identity.name_space + "-" + identity.name + "-" + identity.version;
}

}  // namespace registry_detail

using namespace registry_detail;

NativeDownloadResult download_native_artifact(
        const std::string& url,
        const fs::path& partial,
        const uint64_t expected_size,
        const std::string& label,
        const uint64_t aggregate_complete,
        const uint64_t aggregate_total,
        const RegistryOptions& options,
        std::ostream* progress_output,
        const std::string& bearer_token) {
    return download_file(url, partial, expected_size, label, aggregate_complete,
                         aggregate_total, options, progress_output, bearer_token);
}

RegistryClient::RegistryClient(LocalModelCache& cache, RegistryOptions options)
    : cache_(cache), options_(std::move(options)) {
    if (options_.base_url.empty()) {
        fail(ModelPackageErrorCode::RegistryUnavailable,
             "no registry configured; use --registry or VRHINO_REGISTRY");
    }
    options_.base_url = trim_trailing_slashes(options_.base_url);
    validate_remote_url(options_.base_url, options_);
    if (options_.retry_count < 0 || options_.maximum_redirects < 0 ||
        options_.connect_timeout_seconds <= 0 || options_.low_speed_timeout_seconds <= 0) {
        fail(ModelPackageErrorCode::PackageInvalid, "invalid registry network options");
    }
}

PullResult RegistryClient::pull(const std::string& exact_reference,
                                std::ostream* progress_output) {
    const PackageIdentity requested = parse_package_reference(exact_reference);
    try {
        const ResolvedRunnableModel installed = cache_.resolve(exact_reference, false);
        PullResult result;
        result.identity = installed.manifest.identity;
        result.manifest_path = installed.manifest_path;
        result.logical_bytes = installed.manifest.logical_size();
        result.reused_bytes = result.logical_bytes;
        result.artifacts_reused = installed.artifacts.size();
        result.already_installed = true;
        return result;
    } catch (const ModelPackageError& error) {
        if (error.code() != ModelPackageErrorCode::ModelNotFound) throw;
    }

    const fs::path lock_root = cache_.layout().temporary / "locks";
    FileLock package_lock(lock_root / ("package-" + lock_safe_identity(requested) + ".lock"));
    try {
        const ResolvedRunnableModel installed = cache_.resolve(exact_reference, false);
        PullResult result;
        result.identity = installed.manifest.identity;
        result.manifest_path = installed.manifest_path;
        result.logical_bytes = installed.manifest.logical_size();
        result.reused_bytes = result.logical_bytes;
        result.artifacts_reused = installed.artifacts.size();
        result.already_installed = true;
        return result;
    } catch (const ModelPackageError& error) {
        if (error.code() != ModelPackageErrorCode::ModelNotFound) throw;
    }

    const std::string descriptor_url = options_.base_url + "/v1/models/" +
                                       requested.name_space + "/" + requested.name + "/" +
                                       requested.version + "/index.json";
    if (progress_output != nullptr) {
        *progress_output << "Resolving " << exact_reference << '\n';
    }
    const RegistryDescriptor descriptor = parse_registry_descriptor(
        http_get_text(descriptor_url, options_, ModelPackageErrorCode::RegistryUnavailable),
        options_);
    if (descriptor.identity != exact_reference) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "registry descriptor identity does not match requested exact version");
    }
    const std::string manifest_text = http_get_text(
        descriptor.manifest_url, options_, ModelPackageErrorCode::DownloadFailed);
    if (manifest_text.size() != descriptor.manifest_size) {
        fail(ModelPackageErrorCode::DownloadFailed,
             "downloaded package manifest size does not match registry descriptor");
    }
    const fs::path manifest_download = cache_.layout().temporary / "manifests" /
                                       (descriptor.manifest_sha256 + ".json");
    write_text_file(manifest_download, manifest_text);
#ifdef _WIN32
    windows_cache::LocalSource manifest_source(manifest_download);
    const bool manifest_hash_matches = manifest_source.digest() == descriptor.manifest_sha256;
    if (!manifest_hash_matches) {
#else
    if (sha256_file(manifest_download) != descriptor.manifest_sha256) {
#endif
        std::error_code error;
        fs::remove(manifest_download, error);
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "downloaded package manifest SHA256 mismatch");
    }
    const ModelPackageManifest manifest = load_model_package_manifest(manifest_download);
#ifdef _WIN32
    manifest_source.verify();
    manifest_source.finish();
#endif
    if (manifest.identity.reference() != exact_reference) {
        fail(ModelPackageErrorCode::PackageInvalid,
             "package manifest identity does not match registry descriptor");
    }

    PullResult result;
    result.identity = manifest.identity;
    result.logical_bytes = manifest.logical_size();
    uint64_t bytes_to_acquire = 0;
    for (const ArtifactDeclaration& artifact : manifest.artifacts) {
        if (!cache_.contains_blob(artifact, false)) bytes_to_acquire += artifact.size;
    }
    uint64_t aggregate_complete = 0;
    for (const ArtifactDeclaration& artifact : manifest.artifacts) {
        FileLock artifact_lock(lock_root / ("artifact-" + artifact.sha256 + ".lock"));
        if (cache_.contains_blob(artifact, false)) {
            ++result.artifacts_reused;
            result.reused_bytes += artifact.size;
            continue;
        }
        const auto source = descriptor.artifact_urls.find(artifact.id);
        if (source == descriptor.artifact_urls.end()) {
            if (artifact.required) {
                fail(ModelPackageErrorCode::ArtifactMissing,
                     "registry has no URL for required artifact: " + artifact.id);
            }
            continue;
        }
        const fs::path partial = cache_.layout().temporary / "downloads" /
                                 (artifact.sha256 + ".partial");
        std::error_code error;
        const uint64_t partial_size = fs::exists(partial) ? fs::file_size(partial, error) : 0;
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect partial artifact: " + error.message());
        const uint64_t remaining = partial_size <= artifact.size
                                       ? artifact.size - partial_size
                                       : artifact.size;
        fs::create_directories(cache_.layout().root, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create cache root: " + error.message());
        const fs::space_info space = fs::space(cache_.layout().root, error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot inspect cache free space: " + error.message());
        if (space.available < remaining) {
            fail(ModelPackageErrorCode::InsufficientDiskSpace,
                 "artifact " + artifact.id + " needs " + std::to_string(remaining) +
                     " more bytes but cache filesystem has " +
                     std::to_string(space.available));
        }
        const NativeDownloadResult download = download_file(
            source->second, partial, artifact.size, artifact.id, aggregate_complete,
            bytes_to_acquire, options_, progress_output);
        result.downloaded_bytes += download.network_bytes;
        result.resumed_bytes += download.resumed_bytes;
#ifdef _WIN32
        const BlobAdmissionResult admission = cache_.admit_downloaded_blob(*download.staging, artifact);
#else
        const BlobAdmissionResult admission = cache_.admit_downloaded_blob(partial, artifact);
#endif
        if (admission.created) {
            ++result.artifacts_downloaded;
        } else {
            ++result.artifacts_reused;
            result.reused_bytes += artifact.size;
        }
        aggregate_complete += artifact.size;
        if (options_.progress)
            options_.progress(aggregate_complete, bytes_to_acquire);
    }
#ifdef _WIN32
    const InstallResult installed = cache_.publish_manifest(manifest_download, descriptor.manifest_sha256);
#else
    const InstallResult installed = cache_.publish_manifest(manifest_download);
#endif
    result.manifest_path = installed.manifest_path;
    if (progress_output != nullptr) {
        *progress_output << "Installed " << result.identity.reference() << '\n';
    }
    return result;
}

}  // namespace vrhino::product
