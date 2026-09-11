#include "vrhino/product/windows_cache.h"
#ifdef _WIN32
#include <winioctl.h>
#include <io.h>
#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <utility>

namespace vrhino::product::windows_cache {
#ifdef VRHINO_WINDOWS_CACHE_TESTING
std::function<void(Point)> test_hook;
#define CHECKPOINT(point) do { const auto hook = test_hook; if (hook) hook(Point::point); } while (false)
#else
#define CHECKPOINT(point) ((void)0)
#endif
namespace {
constexpr DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
std::atomic<uint64_t> sequence{0};
[[noreturn]] void fail(const char* operation, DWORD error = GetLastError()) {
    throw ModelPackageError(ModelPackageErrorCode::CacheError,
        std::string(operation) + " (Windows error " + std::to_string(error) + ")");
}
void check(BOOL result, const char* operation) { if (!result) fail(operation); }
void supported_volume(std::wstring_view filesystem, UINT type, bool reparse) {
    if (filesystem != L"NTFS" || (type != DRIVE_FIXED && type != DRIVE_REMOVABLE) || reparse)
        fail("cache requires local NTFS without reparse redirection", ERROR_NOT_SUPPORTED);
}
void delete_handle(HANDLE file) {
    FILE_DISPOSITION_INFO_EX info{FILE_DISPOSITION_FLAG_DELETE |
        FILE_DISPOSITION_FLAG_POSIX_SEMANTICS | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE};
    check(SetFileInformationByHandle(file, FileDispositionInfoEx, &info, sizeof(info)),
          "cannot remove owned cache name");
}
void rename_handle(HANDLE file, const fs::path& destination, HANDLE parent, bool replace) {
    same_volume(file, parent);
    // Win32 FileRenameInfo on the supported Windows 10 host requires the
    // absolute form. Parents keeps every destination ancestor pinned.
    const auto name = native_path(destination).native();
    if (name.empty() || name.size() > (MAXDWORD - sizeof(FILE_RENAME_INFO)) / sizeof(wchar_t))
        fail("invalid publication name", ERROR_INVALID_NAME);
    const size_t bytes = sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t);
    std::vector<unsigned char> storage(bytes);
    auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    info->RootDirectory = nullptr;
    if (replace) info->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    else info->ReplaceIfExists = FALSE;
    info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, name.data(), info->FileNameLength);
    check(SetFileInformationByHandle(file, replace ? FileRenameInfoEx : FileRenameInfo,
          info, static_cast<DWORD>(bytes)), "cannot publish cache name");
}
DWORD read_at(HANDLE file, void* data, DWORD bytes, uint64_t offset) {
    Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) fail("cannot create cache read event");
    OVERLAPPED operation{};
    operation.hEvent = event.get();
    operation.Offset = static_cast<DWORD>(offset);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD count = 0;
    if (!ReadFile(file, data, bytes, &count, &operation)) {
        const DWORD error = GetLastError();
        if (error == ERROR_HANDLE_EOF) return 0;
        if (error != ERROR_IO_PENDING) fail("cannot read retained cache object", error);
        if (!GetOverlappedResult(file, &operation, &count, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) return 0;
            fail("cannot complete cache read");
        }
    }
    return count;
}
Handle open_metadata(const fs::path& path, DWORD access = FILE_READ_ATTRIBUTES) {
    Handle file(CreateFileW(native_path(path).c_str(), access, share, nullptr,
        OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS |
        FILE_FLAG_WRITE_THROUGH, nullptr));
    if (!file) fail("cannot open cache object");
    admit(file.get());
    return file;
}
}

Handle::~Handle() { if (*this) CloseHandle(value_); }
Handle::Handle(Handle&& other) noexcept : value_(other.release()) {}
Handle& Handle::operator=(Handle&& other) noexcept {
    if (this != &other) { if (*this) CloseHandle(value_); value_ = other.release(); }
    return *this;
}
HANDLE Handle::release() noexcept { return std::exchange(value_, INVALID_HANDLE_VALUE); }

fs::path native_path(const fs::path& path) {
    auto absolute = fs::absolute(path.empty() ? fs::current_path() : path).lexically_normal().native();
    if (absolute.find(L'\0') != std::wstring::npos || absolute.starts_with(L"\\\\")) {
        // Accept our own extended local paths, but never UNC/device namespaces.
        if (absolute.starts_with(L"\\\\?\\") && absolute.size() > 6 && absolute[5] == L':')
            absolute.erase(0, 4);
        else fail("cache requires an unambiguous local path", ERROR_NOT_SUPPORTED);
    }
    if (absolute.size() < 3 || absolute[1] != L':' || absolute[2] != L'\\' ||
        absolute.find(L':', 2) != std::wstring::npos)
        fail("invalid local cache path", ERROR_INVALID_NAME);
    return fs::path(L"\\\\?\\" + absolute);
}

fs::path unique_path(const fs::path& parent, const std::string& prefix) {
    return parent / (prefix + "-" + std::to_string(GetCurrentProcessId()) + "-" +
        std::to_string(GetTickCount64()) + "-" + std::to_string(sequence.fetch_add(1)));
}
uint64_t volume(HANDLE object) {
    FILE_ID_INFO identity{};
    check(GetFileInformationByHandleEx(object, FileIdInfo, &identity, sizeof(identity)), "cannot identify cache volume");
    return identity.VolumeSerialNumber;
}
void same_volume(HANDLE object, HANDLE destination) {
    if (volume(object) != volume(destination)) fail("cross-volume cache publication/hardlink is unsupported", ERROR_NOT_SAME_DEVICE);
}
void admit(HANDLE object) {
    CHECKPOINT(Admission);
    if (GetFileType(object) != FILE_TYPE_DISK) fail("cache object is not on a disk filesystem", ERROR_NOT_SUPPORTED);
    FILE_ATTRIBUTE_TAG_INFO info{};
    check(GetFileInformationByHandleEx(object, FileAttributeTagInfo, &info, sizeof(info)), "cannot inspect cache redirection");
    if (info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) fail("cache reparse points are unsupported", ERROR_NOT_SUPPORTED);
    wchar_t filesystem[64]{};
    check(GetVolumeInformationByHandleW(object, nullptr, 0, nullptr, nullptr, nullptr, filesystem, 64), "cannot inspect cache filesystem");
    if (std::wstring(filesystem) != L"NTFS") fail("cache requires local NTFS", ERROR_NOT_SUPPORTED);
    std::vector<wchar_t> resolved(32768);
    const DWORD length = GetFinalPathNameByHandleW(object, resolved.data(), static_cast<DWORD>(resolved.size()), VOLUME_NAME_GUID);
    if (!length || length >= resolved.size()) fail("cannot resolve local cache volume");
    std::wstring name(resolved.data(), length);
    const auto end = name.find(L"}\\");
    if (!name.starts_with(L"\\\\?\\Volume{") || end == std::wstring::npos)
        fail("remote cache volumes are unsupported", ERROR_NOT_SUPPORTED);
    name.resize(end + 2);
    const UINT type = GetDriveTypeW(name.c_str());
    supported_volume(filesystem, type, (info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);
}

Parents::Parents(const fs::path& directory) {
    const fs::path absolute = fs::path(native_path(directory).native().substr(4));
    fs::path current = absolute.root_path();
    const auto pin = [&] {
        Handle handle(CreateFileW(native_path(current).c_str(), FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
        if (!handle) fail("cannot pin cache parent directory");
        admit(handle.get());
        if (!handles_.empty()) same_volume(handles_.front().get(), handle.get());
        handles_.push_back(std::move(handle));
    };
    pin();
    for (const auto& component : absolute.relative_path()) { current /= component; pin(); }
}

void ensure_directory(const fs::path& path) {
    const fs::path absolute(native_path(path).native().substr(4));
    if (absolute == absolute.root_path()) { Parents root(absolute); return; }
    const DWORD attributes = GetFileAttributesW(native_path(absolute).c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES) {
        if (!(attributes & FILE_ATTRIBUTE_DIRECTORY)) fail("cache directory is not a directory", ERROR_DIRECTORY);
        Parents existing(absolute); return;
    }
    const DWORD missing = GetLastError();
    if (missing != ERROR_FILE_NOT_FOUND && missing != ERROR_PATH_NOT_FOUND) fail("cannot inspect cache directory", missing);
    ensure_directory(absolute.parent_path());
    Parents parent(absolute.parent_path());
    const auto staging = unique_path(absolute.parent_path(), ".vrhino-dir");
    check(CreateDirectoryW(native_path(staging).c_str(), nullptr), "cannot create directory staging");
    try {
        auto file = open_metadata(staging, GENERIC_READ | GENERIC_WRITE | DELETE);
        try { rename_handle(file.get(), absolute, parent.leaf(), false); }
        catch (const ModelPackageError&) {
            const DWORD current = GetFileAttributesW(native_path(absolute).c_str());
            if (current == INVALID_FILE_ATTRIBUTES || !(current & FILE_ATTRIBUTE_DIRECTORY)) throw;
            Parents winner(absolute);
            delete_handle(file.get());
        }
    } catch (...) { RemoveDirectoryW(native_path(staging).c_str()); throw; }
}

void require_space(const fs::path& directory, uint64_t bytes) {
    CHECKPOINT(Space);
    Parents parents(directory);
    ULARGE_INTEGER available{};
    check(GetDiskFreeSpaceExW(native_path(directory).c_str(), &available, nullptr, nullptr), "cannot inspect cache space");
    if (available.QuadPart < bytes)
        throw ModelPackageError(ModelPackageErrorCode::InsufficientDiskSpace,
            "Windows durability preparation requires an additional " + std::to_string(bytes) + " bytes");
}

Lock::Lock(const fs::path& path, LockMode mode) {
    ensure_directory(path.parent_path());
    parents_ = std::make_unique<Parents>(path.parent_path());
    file_ = Handle(CreateFileW(native_path(path).c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_OVERLAPPED, nullptr));
    if (!file_) fail("cannot open persistent cache lock");
    admit(file_.get());
    Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) fail("cannot create lock wait event");
    range_.hEvent = event.get();
    CHECKPOINT(Lock);
    if (!LockFileEx(file_.get(), mode == LockMode::Exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0,
                    0, 1, 0, &range_)) {
        if (GetLastError() != ERROR_IO_PENDING) fail("cannot acquire cache lock");
        DWORD ignored = 0;
        check(GetOverlappedResult(file_.get(), &range_, &ignored, TRUE), "cannot wait for cache lock");
    }
    locked_ = true;
    range_.hEvent = nullptr;
}
Lock::~Lock() { if (locked_) UnlockFileEx(file_.get(), 0, 1, 0, &range_); }

Snapshot snapshot(HANDLE file) {
    Snapshot state{};
    check(GetFileInformationByHandleEx(file, FileIdInfo, &state.identity, sizeof(state.identity)), "cannot identify cache file");
    check(GetFileInformationByHandleEx(file, FileBasicInfo, &state.basic, sizeof(state.basic)), "cannot inspect cache mutation metadata");
    check(GetFileInformationByHandleEx(file, FileStandardInfo, &state.standard, sizeof(state.standard)), "cannot inspect cache size");
    if (state.standard.Directory || state.standard.EndOfFile.QuadPart < 0) fail("cache payload is not a regular file", ERROR_INVALID_DATA);
    return state;
}
bool same_identity(const Snapshot& a, const Snapshot& b) {
    return a.identity.VolumeSerialNumber == b.identity.VolumeSerialNumber &&
        std::memcmp(a.identity.FileId.Identifier, b.identity.FileId.Identifier, 16) == 0;
}
bool unchanged(const Snapshot& a, const Snapshot& b) {
    return same_identity(a, b) && a.standard.EndOfFile.QuadPart == b.standard.EndOfFile.QuadPart &&
        a.basic.LastWriteTime.QuadPart == b.basic.LastWriteTime.QuadPart &&
        a.basic.ChangeTime.QuadPart == b.basic.ChangeTime.QuadPart &&
        a.basic.FileAttributes == b.basic.FileAttributes;
}
std::string hash(HANDLE file, uint64_t size, const WorkProgressCallback& progress) {
    HANDLE duplicate = INVALID_HANDLE_VALUE;
    check(DuplicateHandle(GetCurrentProcess(), file, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS), "cannot duplicate cache hash handle");
    Handle owner(duplicate);
    const int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(duplicate), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
    if (descriptor < 0) fail("cannot create cache hash descriptor");
    owner.release();
    struct Descriptor { int value; ~Descriptor() { _close(value); } } scoped{descriptor};
    return sha256_file_descriptor(descriptor, size, progress);
}

class ChangeWatch {
public:
    explicit ChangeWatch(HANDLE file) : file_(file), event_(CreateEventW(nullptr, TRUE, FALSE, nullptr)) {
        if (!event_) fail("cannot create acquisition change event");
        operation_.hEvent = event_.get();
        REQUEST_OPLOCK_INPUT_BUFFER input{};
        input.StructureVersion = REQUEST_OPLOCK_CURRENT_VERSION;
        input.StructureLength = sizeof(input);
        input.RequestedOplockLevel = OPLOCK_LEVEL_CACHE_READ;
        input.Flags = REQUEST_OPLOCK_INPUT_FLAG_REQUEST;
        if (DeviceIoControl(file_, FSCTL_REQUEST_OPLOCK, &input, sizeof(input), &output_, sizeof(output_), nullptr, &operation_) ||
            GetLastError() != ERROR_IO_PENDING) fail("cannot establish stable acquisition change guard");
    }
    ~ChangeWatch() {
        CancelIoEx(file_, &operation_);
        DWORD ignored = 0;
        GetOverlappedResult(file_, &operation_, &ignored, TRUE);
    }
    bool intact() const { return WaitForSingleObject(event_.get(), 0) == WAIT_TIMEOUT; }
private:
    HANDLE file_;
    Handle event_;
    OVERLAPPED operation_{};
    REQUEST_OPLOCK_OUTPUT_BUFFER output_{};
};

LocalSource::LocalSource(const fs::path& path, bool request_writable) : parents_(path.parent_path()) {
    const DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH;
    if (request_writable) {
        file_ = Handle(CreateFileW(native_path(path).c_str(), GENERIC_READ | GENERIC_WRITE,
            share, nullptr, OPEN_EXISTING, flags, nullptr));
        if (!file_ && GetLastError() != ERROR_ACCESS_DENIED) fail("cannot acquire writable local source");
        writable_ = static_cast<bool>(file_);
    }
    if (!file_) file_ = Handle(CreateFileW(native_path(path).c_str(), GENERIC_READ, share,
        nullptr, OPEN_EXISTING, flags, nullptr));
    if (!file_) fail("cannot acquire local source");
    admit(file_.get());
    watch_ = std::make_unique<ChangeWatch>(file_.get());
    before_ = snapshot(file_.get());
}
LocalSource::~LocalSource() = default;
std::string LocalSource::digest(const WorkProgressCallback& progress) {
    auto result = hash(file_.get(), static_cast<uint64_t>(before_.standard.EndOfFile.QuadPart), progress);
    verify(); return result;
}
void LocalSource::verify() const {
    if (!watch_ || !watch_->intact() || !unchanged(before_, snapshot(file_.get())))
        throw ModelPackageError(ModelPackageErrorCode::ChecksumMismatch, "local source changed during acquisition");
}
void LocalSource::finish() { verify(); watch_.reset(); }

StagedFile::StagedFile(const fs::path& path, bool existing, bool keep_partial)
    : path_(path), parents_(path.parent_path()), keep_(keep_partial) {
    file_ = Handle(CreateFileW(native_path(path).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
        FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, existing ? OPEN_EXISTING : CREATE_NEW,
        FILE_FLAG_WRITE_THROUGH | FILE_FLAG_OVERLAPPED | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    if (!file_) fail("cannot acquire write-through cache staging");
    try { admit(file_.get()); offset_ = size(); }
    catch (...) { if (!existing) { try { delete_handle(file_.get()); } catch (...) {} } throw; }
}
StagedFile::~StagedFile() {
    if (owned_ && !keep_ && file_) {
        try { delete_handle(file_.get()); } catch (...) { /* Explicit cleanup reports errors; unwinding must not throw. */ }
    }
}
uint64_t StagedFile::size() const { return static_cast<uint64_t>(snapshot(file_.get()).standard.EndOfFile.QuadPart); }
void StagedFile::write(const void* data, size_t bytes) {
    CHECKPOINT(Write);
    flushed_ = validated_ = false;
    auto* next = static_cast<const unsigned char*>(data);
    Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) fail("cannot create cache write event");
    while (bytes) {
        const DWORD wanted = static_cast<DWORD>(std::min<size_t>(bytes, 1U << 20));
        OVERLAPPED operation{}; operation.hEvent = event.get();
        operation.Offset = static_cast<DWORD>(offset_); operation.OffsetHigh = static_cast<DWORD>(offset_ >> 32);
        DWORD count = 0;
        if (!WriteFile(file_.get(), next, wanted, &count, &operation)) {
            if (GetLastError() != ERROR_IO_PENDING) fail("cannot write cache staging");
            check(GetOverlappedResult(file_.get(), &operation, &count, TRUE), "cannot complete staging write");
        }
        if (!count) fail("short staging write", ERROR_WRITE_FAULT);
        next += count; bytes -= count; offset_ += count;
    }
}
void StagedFile::truncate(uint64_t bytes) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail("staging size overflow", ERROR_FILE_TOO_LARGE);
    FILE_END_OF_FILE_INFO eof{}; eof.EndOfFile.QuadPart = static_cast<int64_t>(bytes);
    check(SetFileInformationByHandle(file_.get(), FileEndOfFileInfo, &eof, sizeof(eof)), "cannot resize retained staging");
    offset_ = bytes; flushed_ = validated_ = false;
}
void StagedFile::copy(LocalSource& source) {
    std::vector<unsigned char> buffer(1U << 20);
    const auto bytes = static_cast<uint64_t>(source.before().standard.EndOfFile.QuadPart);
    for (uint64_t offset = 0; offset < bytes;) {
        CHECKPOINT(SourceCopy);
        const DWORD count = read_at(source.get(), buffer.data(), static_cast<DWORD>(std::min<uint64_t>(buffer.size(), bytes - offset)), offset);
        if (!count) fail("local source ended during acquisition", ERROR_HANDLE_EOF);
        write(buffer.data(), count); offset += count;
    }
    source.verify();
}
void StagedFile::readonly() {
    auto info = snapshot(file_.get()).basic;
    info.FileAttributes |= FILE_ATTRIBUTE_READONLY;
    check(SetFileInformationByHandle(file_.get(), FileBasicInfo, &info, sizeof(info)), "cannot make cache object read-only");
    flushed_ = validated_ = false;
}
void StagedFile::flush() { CHECKPOINT(Flush); check(FlushFileBuffers(file_.get()), "cannot flush cache staging"); flushed_ = true; validated_ = false; }
void StagedFile::validate(uint64_t bytes, const std::string& digest, const WorkProgressCallback& progress) {
    if (!flushed_) fail("cache validation requires completed flush", ERROR_INVALID_STATE);
    ChangeWatch watch(file_.get());
    const auto before = snapshot(file_.get());
    CHECKPOINT(Validate);
    if (static_cast<uint64_t>(before.standard.EndOfFile.QuadPart) != bytes || hash(file_.get(), bytes, progress) != digest ||
        !watch.intact() || !unchanged(before, snapshot(file_.get())))
        throw ModelPackageError(ModelPackageErrorCode::ChecksumMismatch, "staging size/SHA256 or stability validation failed");
    validated_state_ = snapshot(file_.get());
    validated_ = true;
}
void StagedFile::publish(const fs::path& destination, bool replace) {
    if (!flushed_ || !validated_) fail("cache publication requires prepared staging", ERROR_INVALID_STATE);
    Parents destination_parent(destination.parent_path());
    same_volume(file_.get(), destination_parent.leaf());
    CHECKPOINT(BeforePublish);
    if (!unchanged(validated_state_, snapshot(file_.get())))
        fail("prepared cache object changed before publication", ERROR_INVALID_DATA);
    // A syscall error leaves ownership attached to the opened object. Never
    // delete an ambiguous final object as rollback: disable automatic cleanup
    // for the commit window, and restore it only if its old name still names us.
    keep_ = true;
    cleanup_safe_ = false;
    try { rename_handle(file_.get(), destination, destination_parent.leaf(), replace); }
    catch (...) {
        try { auto old = open_metadata(path_); if (same_identity(snapshot(old.get()), snapshot(file_.get()))) { keep_ = false; cleanup_safe_ = true; } } catch (...) {}
        throw;
    }
    owned_ = false;
    // Publication ends writable ownership. In particular, CRT readers need
    // not share DELETE access with a handle retained by a download result.
    try { CHECKPOINT(AfterPublish); }
    catch (...) { file_ = Handle(); throw; }
    file_ = Handle();
}
void StagedFile::discard() {
    if (!owned_ || !cleanup_safe_) fail("refusing to discard a committed or uncertain publication", ERROR_INVALID_STATE);
    CHECKPOINT(Cleanup); delete_handle(file_.get()); owned_ = false; file_ = Handle();
}

void write_text(const fs::path& destination, const std::string& text, bool immutable, bool replace) {
    ensure_directory(destination.parent_path());
    StagedFile file(unique_path(destination.parent_path(), ".vrhino-text"));
    file.write(text.data(), text.size());
    if (immutable) file.readonly();
    file.flush();
    std::vector<char> actual(text.size());
    if (text.size() > MAXDWORD || read_at(file.get(), actual.data(), static_cast<DWORD>(actual.size()), 0) != actual.size() ||
        !std::equal(actual.begin(), actual.end(), text.begin())) fail("cache text readback mismatch", ERROR_INVALID_DATA);
    file.validate(text.size(), hash(file.get(), text.size()));
    file.publish(destination, replace);
}
void publish_directory(const fs::path& staging, const fs::path& destination) {
    Parents source_parent(staging.parent_path());
    Parents target_parent(destination.parent_path());
    auto directory = open_metadata(staging, GENERIC_READ | GENERIC_WRITE | DELETE);
    FILE_STANDARD_INFO info{};
    check(GetFileInformationByHandleEx(directory.get(), FileStandardInfo, &info, sizeof(info)), "cannot inspect package staging");
    if (!info.Directory) fail("package staging is not a directory", ERROR_DIRECTORY);
    CHECKPOINT(BeforePublish);
    rename_handle(directory.get(), destination, target_parent.leaf(), false);
    CHECKPOINT(AfterPublish);
}
void migrate_partial(const fs::path& source, const fs::path& destination) {
    StagedFile file(source, true, true);
    file.flush();
    // Migration preserves incomplete bytes; remote integrity is checked later.
    file.validate(file.size(), hash(file.get(), file.size()));
    file.publish(destination);
}
void required_link(const fs::path& source, const fs::path& destination) {
    Parents source_parent(source.parent_path());
    ensure_directory(destination.parent_path());
    Parents target_parent(destination.parent_path());
    if (GetFileAttributesW(native_path(destination).c_str()) != INVALID_FILE_ATTRIBUTES)
        fail("required hardlink destination already exists", ERROR_ALREADY_EXISTS);
    auto original = open_metadata(source);
    const auto original_id = snapshot(original.get());
    same_volume(original.get(), target_parent.leaf());
    auto staging = unique_path(destination.parent_path(), ".vrhino-link");
    CHECKPOINT(Hardlink);
    check(CreateHardLinkW(native_path(staging).c_str(), native_path(source).c_str(), nullptr), "cannot create required CAS hardlink");
    try {
        auto linked = open_metadata(staging, GENERIC_READ | DELETE | FILE_WRITE_ATTRIBUTES);
        if (!same_identity(original_id, snapshot(linked.get()))) fail("CAS hardlink identity changed", ERROR_INVALID_DATA);
        // This links an already durable CAS payload. No bytes are written here;
        // the new link's name receives its own write-through rename barrier.
        rename_handle(linked.get(), destination, target_parent.leaf(), false);
    } catch (...) { try { windows_cache::remove(staging); } catch (...) {} throw; }
}
bool equivalent(const fs::path& first, const fs::path& second) {
    Parents a(first.parent_path()), b(second.parent_path());
    auto left = open_metadata(first), right = open_metadata(second);
    return same_identity(snapshot(left.get()), snapshot(right.get()));
}
uint64_t link_count(const fs::path& path) {
    Parents parent(path.parent_path()); auto file = open_metadata(path);
    return snapshot(file.get()).standard.NumberOfLinks;
}
void remove(const fs::path& path) {
    const DWORD attributes = GetFileAttributesW(native_path(path).c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES && (GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND)) return;
    Parents parent(path.parent_path());
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) && !fs::is_empty(native_path(path)))
        fail("refusing to retire a nonempty directory as a file", ERROR_DIR_NOT_EMPTY);
    auto file = open_metadata(path, DELETE | FILE_READ_ATTRIBUTES);
    const auto retired = unique_path(path.parent_path(), ".vrhino-remove");
    CHECKPOINT(Cleanup);
    rename_handle(file.get(), retired, parent.leaf(), false);
    delete_handle(file.get());
}
void remove_tree(const fs::path& path) {
    const DWORD attributes = GetFileAttributesW(native_path(path).c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES && (GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND)) return;
    if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) fail("refusing cache reparse cleanup", ERROR_NOT_SUPPORTED);
    if (attributes & FILE_ATTRIBUTE_DIRECTORY) {
        {
            Parents pinned(path);
            for (const auto& child : fs::directory_iterator(native_path(path))) remove_tree(child.path());
        }
    }
    windows_cache::remove(path);
}
void retire_tree(const fs::path& path, const fs::path& temporary) {
    ensure_directory(temporary);
    const auto retired = unique_path(temporary, "retired-package");
    publish_directory(path, retired);
    remove_tree(retired);
}
#ifdef VRHINO_WINDOWS_CACHE_TESTING
void test_volume_admission(std::wstring_view filesystem, UINT type, bool reparse) {
    supported_volume(filesystem, type, reparse);
}
#endif
} // namespace vrhino::product::windows_cache
#endif
