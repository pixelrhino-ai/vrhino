#pragma once
// Private transport/locking declarations shared only by the registry units.
#include "vrhino/product/registry.h"
#ifdef _WIN32
#include "vrhino/product/windows_cache.h"
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
namespace vrhino::product::registry_detail {
namespace fs = std::filesystem;
[[noreturn]] void fail(ModelPackageErrorCode code, const std::string& message);
std::string trim_trailing_slashes(std::string value);
void validate_remote_url(const std::string& url, const RegistryOptions& options);
std::string http_get_text(const std::string& url, const RegistryOptions& options, ModelPackageErrorCode failure_code);
NativeDownloadResult download_file(const std::string& url, const fs::path& partial,
    uint64_t expected_size, const std::string& label, uint64_t aggregate_complete,
    uint64_t aggregate_total, const RegistryOptions& options, std::ostream* progress_output,
    const std::string& bearer_token = {});
std::string lock_safe_identity(const PackageIdentity& identity);
#ifdef _WIN32
using FileLock = windows_cache::Lock;
#else
class FileLock {
public:
    explicit FileLock(const fs::path& path) {
        std::error_code error;
        fs::create_directories(path.parent_path(), error);
        if (error) fail(ModelPackageErrorCode::CacheError,
                        "cannot create lock directory: " + error.message());
        descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (descriptor_ < 0 || ::flock(descriptor_, LOCK_EX) != 0) {
            if (descriptor_ >= 0) ::close(descriptor_);
            fail(ModelPackageErrorCode::CacheError,
                 "cannot acquire cache lock: " + path.string());
        }
    }

    ~FileLock() {
        if (descriptor_ >= 0) {
            ::flock(descriptor_, LOCK_UN);
            ::close(descriptor_);
        }
    }

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
    int descriptor_ = -1;
};
#endif
struct ComponentRegistryDescriptor {
    std::string identity;
    std::string package_url;
    uint64_t package_size = 0;
    std::string package_sha256;
};
ComponentRegistryDescriptor parse_component_registry_descriptor(const std::string& text, const RegistryOptions& options);
} // namespace vrhino::product::registry_detail
