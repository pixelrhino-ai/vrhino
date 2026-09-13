#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#ifdef _WIN32
#include <memory>
#endif
#include <string>

#include "vrhino/product/model_package.h"
#include "vrhino/product/component_package.h"

namespace vrhino::product {

inline constexpr int64_t kRegistrySchemaVersion = 1;

struct RegistryOptions {
    std::string base_url;
    std::filesystem::path ca_file;
    bool allow_development_http = false;
    long connect_timeout_seconds = 30;
    long low_speed_timeout_seconds = 60;
    int retry_count = 3;
    long maximum_redirects = 5;
    std::function<bool()> cancellation_requested;
    std::function<void(uint64_t, uint64_t)> progress;
};

// Shared Native HTTPS transport used by registry, component and upstream
// source acquisition layers. Callers own integrity/CAS publication policy.
struct NativeDownloadResult {
    uint64_t network_bytes = 0;
    uint64_t resumed_bytes = 0;
#ifdef _WIN32
    std::shared_ptr<windows_cache::StagedFile> staging;
#endif
};

// Structured transport evidence lets provider-specific acquisition policy
// distinguish availability failures from immutable-source and resume errors.
// It deliberately contains no URL, request header, or credential material.
enum class NativeDownloadFailureClass {
    DnsResolution,
    Connection,
    Timeout,
    TlsHandshake,
    InterruptedTransfer,
    HttpStatus,
    ResumeProtocol,
    Other,
};

class NativeDownloadError final : public ModelPackageError {
public:
    NativeDownloadError(ModelPackageErrorCode code,
                        NativeDownloadFailureClass failure_class,
                        long http_status,
                        uint64_t network_bytes,
                        const std::string& message)
        : ModelPackageError(code, message),
          failure_class_(failure_class),
          http_status_(http_status),
          network_bytes_(network_bytes) {}

    NativeDownloadFailureClass failure_class() const noexcept {
        return failure_class_;
    }
    long http_status() const noexcept { return http_status_; }
    uint64_t network_bytes() const noexcept { return network_bytes_; }

private:
    NativeDownloadFailureClass failure_class_;
    long http_status_ = 0;
    uint64_t network_bytes_ = 0;
};

NativeDownloadResult download_native_artifact(
    const std::string& url,
    const std::filesystem::path& partial,
    uint64_t expected_size,
    const std::string& label,
    uint64_t aggregate_complete,
    uint64_t aggregate_total,
    const RegistryOptions& options,
    std::ostream* progress_output = nullptr,
    const std::string& bearer_token = {});

struct PullResult {
    PackageIdentity identity;
    std::filesystem::path manifest_path;
    uint64_t logical_bytes = 0;
    uint64_t downloaded_bytes = 0;
    uint64_t reused_bytes = 0;
    uint64_t resumed_bytes = 0;
    size_t artifacts_downloaded = 0;
    size_t artifacts_reused = 0;
    bool already_installed = false;
};

class RegistryClient {
public:
    RegistryClient(LocalModelCache& cache, RegistryOptions options);

    PullResult pull(const std::string& exact_reference,
                    std::ostream* progress_output = nullptr);

private:
    LocalModelCache& cache_;
    RegistryOptions options_;
};

struct ComponentPullResult {
    PackageIdentity identity;
    std::filesystem::path root;
    uint64_t downloaded_bytes = 0;
    uint64_t reused_bytes = 0;
    uint64_t resumed_bytes = 0;
    bool already_installed = false;
};

class ComponentRegistryClient {
public:
    ComponentRegistryClient(LocalModelCache& blob_cache,
                            LocalComponentCache& component_cache,
                            RegistryOptions options);

    ComponentPullResult pull(const std::string& exact_reference,
                             std::ostream* progress_output = nullptr);

private:
    LocalModelCache& blob_cache_;
    LocalComponentCache& component_cache_;
    RegistryOptions options_;
};

}  // namespace vrhino::product
