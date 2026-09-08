#include "vrhino/product/vrm_verification.h"

#include <chrono>
#include <cctype>
#include <exception>
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "vrhino/product/model_package.h"

namespace vrhino::product {
namespace {

using Clock = std::chrono::steady_clock;

class OwnedDescriptor {
public:
    explicit OwnedDescriptor(const int value = -1) : value_(value) {}
    ~OwnedDescriptor() { if (value_ >= 0) close(value_); }
    OwnedDescriptor(const OwnedDescriptor&) = delete;
    OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;
    int get() const noexcept { return value_; }
    int release() noexcept { const int value = value_; value_ = -1; return value; }
private:
    int value_;
};

struct FileIdentity {
    dev_t device{};
    ino_t inode{};
    off_t size{};
    timespec modified{};
    timespec changed{};
};

[[noreturn]] void fail(const ModelPackageErrorCode code,
                       const std::string& message) {
    throw ModelPackageError(code, message);
}

FileIdentity file_identity(const int descriptor, const std::string& context) {
    struct stat value{};
    if (fstat(descriptor, &value) != 0)
        fail(ModelPackageErrorCode::CacheError,
             "cannot inspect component file identity: " + context);
    if (!S_ISREG(value.st_mode))
        fail(ModelPackageErrorCode::ComponentInvalid,
             "component is not a regular file: " + context);
    return {value.st_dev, value.st_ino, value.st_size,
            value.st_mtim, value.st_ctim};
}

bool same_content_attributes(const FileIdentity& left, const FileIdentity& right) {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size &&
           left.modified.tv_sec == right.modified.tv_sec &&
           left.modified.tv_nsec == right.modified.tv_nsec;
}

bool valid_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const unsigned char byte : value)
        if (!std::isxdigit(byte) || (byte >= 'A' && byte <= 'F')) return false;
    return true;
}

double elapsed(const Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

std::unique_ptr<VrmModel> load_verified_vrm_component(
        const std::filesystem::path& path,
        const VrmComponentIntegrityContract& contract,
        VrmVerificationObservation* observation,
        const StableOpenVerificationHook& stable_open_hook) {
    if (!valid_sha256(contract.sha256))
        fail(ModelPackageErrorCode::ComponentInvalid,
             "invalid expected SHA-256 for component: " + contract.semantic_name);

    OwnedDescriptor authoritative(open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (authoritative.get() < 0)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "cannot open component: " + path.string());
    FileIdentity before = file_identity(authoritative.get(), path.string());
    if (before.size < 0 || static_cast<uint64_t>(before.size) != contract.bytes)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "component size mismatch: " + contract.semantic_name);

    if (stable_open_hook) stable_open_hook();

    VrmVerificationObservation totals{};
    for (unsigned attempt = 0; ; ++attempt) {
        OwnedDescriptor sha_descriptor(fcntl(authoritative.get(), F_DUPFD_CLOEXEC, 0));
        OwnedDescriptor vrm_descriptor(fcntl(authoritative.get(), F_DUPFD_CLOEXEC, 0));
        if (sha_descriptor.get() < 0 || vrm_descriptor.get() < 0)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot duplicate stable component descriptor: " + contract.semantic_name);

        const auto concurrent_started = Clock::now();
        std::string actual_sha256;
        double sha_seconds = 0.0;
        std::exception_ptr sha_error;
        std::thread sha_worker;
        try {
            sha_worker = std::thread([&] {
                const auto started = Clock::now();
                try {
                    actual_sha256 = sha256_file_descriptor(
                        sha_descriptor.get(), contract.bytes);
                } catch (...) {
                    sha_error = std::current_exception();
                }
                sha_seconds = elapsed(started);
            });
        } catch (...) {
            throw;
        }

        std::unique_ptr<VrmModel> model;
        std::exception_ptr vrm_error;
        try {
            model = std::make_unique<VrmModel>(vrm_descriptor.release(), true);
        } catch (...) {
            vrm_error = std::current_exception();
        }
        sha_worker.join();
        const double concurrent_seconds = elapsed(concurrent_started);
        const FileIdentity after = file_identity(authoritative.get(), path.string());

        if (observation != nullptr) {
            totals.sha256_bytes += contract.bytes;
            totals.blake2b_bytes += contract.bytes >= 128 ? contract.bytes - 128 : 0;
            totals.sha256_seconds += sha_seconds;
            totals.blake2b_seconds += model ? model->checksum_seconds() : 0.0;
            totals.concurrent_wall_seconds += concurrent_seconds;
            totals.sha256_completed = sha_error == nullptr;
            totals.blake2b_completed = vrm_error == nullptr;
            totals.maximum_digest_consumers = 2;
            *observation = totals;
        }

        if (!same_content_attributes(before, after))
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component changed during verification: " + contract.semantic_name);
        if (sha_error) std::rethrow_exception(sha_error);
        if (vrm_error) std::rethrow_exception(vrm_error);
        if (actual_sha256 != contract.sha256)
            fail(ModelPackageErrorCode::ChecksumMismatch,
                 "component SHA-256 mismatch: " + contract.semantic_name);
        if (model->architecture_id() != contract.architecture)
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component architecture mismatch: " + contract.semantic_name);
        if (before.changed.tv_sec == after.changed.tv_sec &&
            before.changed.tv_nsec == after.changed.tv_nsec)
            return model;

        // ctime may change on rename/unlink/chmod without a content change, but
        // also on writes with restored mtime. Never ignore it: discard this model
        // and repeat BOTH digests and parsing on the same open object once. The
        // accepted pass must have stable ctime too; further churn fails closed.
        if (attempt != 0)
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component remained unstable during verification: " + contract.semantic_name);
        before = after;
    }
}

}  // namespace vrhino::product
