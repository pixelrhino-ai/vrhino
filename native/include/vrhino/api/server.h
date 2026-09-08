#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "vrhino/product/run_session.h"

namespace vrhino::api {

inline constexpr const char* kDefaultHost = "127.0.0.1";
inline constexpr uint16_t kDefaultPort = 11435;
inline constexpr size_t kRequestTargetMaxBytes = 2048;
inline constexpr size_t kModelReferenceMaxBytes = 192;
inline constexpr size_t kRequestBodyMaxBytes = 64 * 1024;
inline constexpr size_t kWorkerThreads = 8;
inline constexpr size_t kWorkerQueueCapacity = 16;
inline constexpr size_t kEventSubscriberCapacity = 4;
inline constexpr size_t kEventLineMaxBytes = 16 * 1024;
inline constexpr size_t kEventMessageMaxBytes = 4 * 1024;
inline constexpr int kEventWaitMilliseconds = 250;
inline constexpr int kReadTimeoutSeconds = 5;
inline constexpr int kWriteTimeoutSeconds = 5;
inline constexpr int kKeepAliveTimeoutSeconds = 2;
inline constexpr size_t kKeepAliveMaxRequests = 10;
inline constexpr size_t kRunRequestJsonMaxDepth = 32;
inline constexpr size_t kRunRequestJsonMaxContainerEntries = 1024;
inline constexpr size_t kRunRequestJsonMaxTotalValues = 4096;
inline constexpr size_t kRunRequestJsonMaxStringBytes = 32 * 1024;

struct ServerConfig {
    std::string host = kDefaultHost;
    uint16_t port = kDefaultPort;
    std::filesystem::path cache_root;
    product::RunExecutor run_executor;
};

bool valid_server_port(uint32_t port) noexcept;
bool is_loopback_host(const std::string& host) noexcept;
std::optional<std::string> non_loopback_bind_warning(
    const std::string& host);

class Server {
public:
    explicit Server(ServerConfig config = {});
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binding and listening are split so the foreground CLI can report a
    // successful bind without racing a test/control thread.
    int bind();
    bool listen();
    void stop() noexcept;
    const ServerConfig& config() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vrhino::api
