#include "vrhino/product/vrm_verification.h"

#include <chrono>
#include <cctype>
#include <exception>
#include <fcntl.h>
#include <memory>
#ifdef _WIN32
#include <algorithm>
#include <iterator>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <thread>

#include "vrhino/product/model_package.h"

namespace vrhino::product {
#ifdef VRHINO_STABLE_VERIFICATION_TESTING
namespace stable_verification_testing {
std::function<void(Stage, unsigned)> hook;
void invoke(Stage stage, unsigned attempt) { if (hook) hook(stage, attempt); }
}
#endif
namespace {

using Clock = std::chrono::steady_clock;

class OwnedDescriptor {
public:
    explicit OwnedDescriptor(const int value = -1) : value_(value) {}
    ~OwnedDescriptor() {
#ifdef _WIN32
        if (value_ >= 0) _close(value_);
#else
        if (value_ >= 0) close(value_);
#endif
    }
    OwnedDescriptor(const OwnedDescriptor&) = delete;
    OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;
    int get() const noexcept { return value_; }
    int release() noexcept { const int value = value_; value_ = -1; return value; }
private:
    int value_;
};

struct FileIdentity {
#ifdef _WIN32
    FILE_ID_INFO identity{};
    int64_t size{};
    int64_t modified{};
    int64_t changed{};
#else
    dev_t device{};
    ino_t inode{};
    off_t size{};
    timespec modified{};
    timespec changed{};
#endif
};

[[noreturn]] void fail(const ModelPackageErrorCode code,
                       const std::string& message) {
    throw ModelPackageError(code, message);
}

FileIdentity file_identity(const int descriptor, const std::string& context) {
#ifdef _WIN32
    const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(descriptor));
    FILE_ID_INFO identity{};
    FILE_STANDARD_INFO standard{};
    FILE_BASIC_INFO basic{};
    if (!GetFileInformationByHandleEx(file, FileIdInfo, &identity, sizeof(identity)) ||
        !GetFileInformationByHandleEx(file, FileStandardInfo, &standard, sizeof(standard)) ||
        !GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic)))
        fail(ModelPackageErrorCode::CacheError, "cannot inspect component file identity: " + context);
    if (GetFileType(file) != FILE_TYPE_DISK || standard.Directory)
        fail(ModelPackageErrorCode::ComponentInvalid, "component is not a regular file: " + context);
    return {identity, standard.EndOfFile.QuadPart, basic.LastWriteTime.QuadPart, basic.ChangeTime.QuadPart};
#else
    struct stat value{};
    if (fstat(descriptor, &value) != 0)
        fail(ModelPackageErrorCode::CacheError,
             "cannot inspect component file identity: " + context);
    if (!S_ISREG(value.st_mode))
        fail(ModelPackageErrorCode::ComponentInvalid,
             "component is not a regular file: " + context);
    return {value.st_dev, value.st_ino, value.st_size,
            value.st_mtim, value.st_ctim};
#endif
}

bool same_content_attributes(const FileIdentity& left, const FileIdentity& right) {
#ifdef _WIN32
    return left.identity.VolumeSerialNumber == right.identity.VolumeSerialNumber &&
           std::equal(std::begin(left.identity.FileId.Identifier), std::end(left.identity.FileId.Identifier),
                      std::begin(right.identity.FileId.Identifier)) &&
           left.size == right.size && left.modified == right.modified;
#else
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size &&
           left.modified.tv_sec == right.modified.tv_sec &&
           left.modified.tv_nsec == right.modified.tv_nsec;
#endif
}

#ifdef _WIN32
// These helpers belong only to the stable-verification boundary.
int take_file_handle(HANDLE handle) noexcept {
    if (handle == INVALID_HANDLE_VALUE) return -1;
    const int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(handle),
        _O_RDONLY | _O_BINARY | _O_NOINHERIT);
    if (descriptor < 0) CloseHandle(handle);
    return descriptor;
}

int duplicate_file(int descriptor) noexcept {
    HANDLE copy = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(), reinterpret_cast<HANDLE>(_get_osfhandle(descriptor)),
                         GetCurrentProcess(), &copy, 0, FALSE, DUPLICATE_SAME_ACCESS)) return -1;
    return take_file_handle(copy);
}

std::string path_context(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

// A read oplock is an advisory change notification, not a writer exclusion.
// It also detects writes with delayed/restored timestamps and refuses existing
// writable mapped sections. Never accept a weaker timestamp-only fallback.
class ReadChangeWatch {
public:
    explicit ReadChangeWatch(int descriptor)
        : file_(reinterpret_cast<HANDLE>(_get_osfhandle(descriptor))) {
        overlapped_.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!overlapped_.hEvent) fail(ModelPackageErrorCode::CacheError, "cannot create verification change event");
        REQUEST_OPLOCK_INPUT_BUFFER input{};
        input.StructureVersion = REQUEST_OPLOCK_CURRENT_VERSION;
        input.StructureLength = sizeof(input);
        input.RequestedOplockLevel = OPLOCK_LEVEL_CACHE_READ;
        input.Flags = REQUEST_OPLOCK_INPUT_FLAG_REQUEST;
        const BOOL completed = DeviceIoControl(file_, FSCTL_REQUEST_OPLOCK, &input, sizeof(input),
            &output_, sizeof(output_), nullptr, &overlapped_);
        if (completed || GetLastError() != ERROR_IO_PENDING) {
            CloseHandle(overlapped_.hEvent);
            fail(ModelPackageErrorCode::ComponentInvalid, "cannot establish stable file change guard");
        }
    }
    ~ReadChangeWatch() {
        CancelIoEx(file_, &overlapped_);
        DWORD ignored = 0;
        // Drain completion before releasing the kernel's OVERLAPPED/output storage.
        GetOverlappedResult(file_, &overlapped_, &ignored, TRUE);
        CloseHandle(overlapped_.hEvent);
    }
    ReadChangeWatch(const ReadChangeWatch&) = delete;
    ReadChangeWatch& operator=(const ReadChangeWatch&) = delete;
    bool unchanged() const noexcept {
        return WaitForSingleObject(overlapped_.hEvent, 0) == WAIT_TIMEOUT;
    }
private:
    HANDLE file_;
    OVERLAPPED overlapped_{};
    REQUEST_OPLOCK_OUTPUT_BUFFER output_{};
};
#endif

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

#ifdef _WIN32
    const std::string context = path_context(path);
    if (path.native().find(L'\0') != std::wstring::npos)
        fail(ModelPackageErrorCode::ArtifactMissing, "invalid component path");
    OwnedDescriptor authoritative(take_file_handle(CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr)));
    if (authoritative.get() < 0)
        fail(ModelPackageErrorCode::ArtifactMissing, "cannot open component: " + context);
    ReadChangeWatch change_watch(authoritative.get());
    FileIdentity before = file_identity(authoritative.get(), context);
#else
    OwnedDescriptor authoritative(open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (authoritative.get() < 0)
        fail(ModelPackageErrorCode::ArtifactMissing,
             "cannot open component: " + path.string());
    FileIdentity before = file_identity(authoritative.get(), path.string());
#endif
    if (before.size < 0 || static_cast<uint64_t>(before.size) != contract.bytes)
        fail(ModelPackageErrorCode::ChecksumMismatch,
             "component size mismatch: " + contract.semantic_name);

    if (stable_open_hook) stable_open_hook();

    VrmVerificationObservation totals{};
    for (unsigned attempt = 0; ; ++attempt) {
#ifdef _WIN32
        OwnedDescriptor sha_descriptor(duplicate_file(authoritative.get()));
#ifdef VRHINO_STABLE_VERIFICATION_TESTING
        stable_verification_testing::invoke(stable_verification_testing::Stage::AfterShaDuplicate, attempt);
#endif
        OwnedDescriptor vrm_descriptor(duplicate_file(authoritative.get()));
#else
        OwnedDescriptor sha_descriptor(fcntl(authoritative.get(), F_DUPFD_CLOEXEC, 0));
        OwnedDescriptor vrm_descriptor(fcntl(authoritative.get(), F_DUPFD_CLOEXEC, 0));
#endif
        if (sha_descriptor.get() < 0 || vrm_descriptor.get() < 0)
            fail(ModelPackageErrorCode::CacheError,
                 "cannot duplicate stable component descriptor: " + contract.semantic_name);
#ifdef VRHINO_STABLE_VERIFICATION_TESTING
        stable_verification_testing::invoke(stable_verification_testing::Stage::AfterDuplicates, attempt);
#endif

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
                        sha_descriptor.get(), contract.bytes
#ifdef VRHINO_STABLE_VERIFICATION_TESTING
                        , [&](uint64_t) { stable_verification_testing::invoke(
                            stable_verification_testing::Stage::HashChunk, attempt); }
#endif
                    );
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
#ifdef _WIN32
            // new allocates before release() is evaluated; make_unique would
            // release first and leak the descriptor if object allocation fails.
            model.reset(new VrmModel(vrm_descriptor.release(), true));
#else
            model = std::make_unique<VrmModel>(vrm_descriptor.release(), true);
#endif
        } catch (...) {
            vrm_error = std::current_exception();
        }
        sha_worker.join();
#ifdef VRHINO_STABLE_VERIFICATION_TESTING
        stable_verification_testing::invoke(stable_verification_testing::Stage::AfterDigests, attempt);
#endif
        const double concurrent_seconds = elapsed(concurrent_started);
#ifdef _WIN32
        const FileIdentity after = file_identity(authoritative.get(), context);
#else
        const FileIdentity after = file_identity(authoritative.get(), path.string());
#endif

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

#ifdef _WIN32
        if (!change_watch.unchanged())
            fail(ModelPackageErrorCode::ComponentInvalid,
                 "component changed during verification (read oplock broken): " + contract.semantic_name);
#endif
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
#ifdef _WIN32
        if (before.changed == after.changed)
#else
        if (before.changed.tv_sec == after.changed.tv_sec &&
            before.changed.tv_nsec == after.changed.tv_nsec)
#endif
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
