#pragma once
// Product converter file plumbing only. No tensor/conversion policy lives here.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <mutex>

namespace vrhino::product::windows_converter_io {
using ssize_t = int64_t;
using off_t = int64_t;
inline constexpr int O_CLOEXEC = _O_NOINHERIT;
inline std::mutex position_mutex;
inline int open(const wchar_t* path, int flags, int = 0) {
    const bool output = (flags & (_O_WRONLY | _O_RDWR)) != 0;
    const DWORD access = output ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
    HANDLE file = CreateFileW(path, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, flags & _O_CREAT ? CREATE_NEW : OPEN_EXISTING,
        output ? FILE_FLAG_WRITE_THROUGH : FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { errno = EIO; return -1; }
    const int result = _open_osfhandle(reinterpret_cast<intptr_t>(file),
        (flags & (_O_RDONLY | _O_WRONLY | _O_RDWR)) | _O_BINARY | _O_NOINHERIT);
    if (result < 0) CloseHandle(file);
    return result;
}
inline int close(int fd) { return _close(fd); }
inline int fstat(int fd, struct _stat64* state) { return _fstat64(fd, state); }
inline ssize_t write(int fd, const void* bytes, size_t count) {
    std::lock_guard lock(position_mutex);
    return _write(fd, bytes, static_cast<unsigned int>(std::min<size_t>(count, INT_MAX)));
}
inline ssize_t positional(int fd, void* bytes, size_t count, int64_t offset, bool writing) {
    if (offset < 0) { errno = EINVAL; return -1; }
    // Preserve pread/pwrite's independent cursor under concurrent use. All
    // converter cursor-changing access goes through this bounded adapter.
    std::lock_guard lock(position_mutex);
    const auto previous = _lseeki64(fd, 0, SEEK_CUR);
    if (previous < 0 || _lseeki64(fd, offset, SEEK_SET) < 0) return -1;
    const auto chunk = static_cast<unsigned int>(std::min<size_t>(count, INT_MAX));
    const int result = writing ? _write(fd, bytes, chunk) : _read(fd, bytes, chunk);
    const int saved = errno;
    if (_lseeki64(fd, previous, SEEK_SET) < 0) return -1;
    errno = saved;
    return result;
}
inline ssize_t pread(int fd, void* bytes, size_t count, int64_t offset) {
    return positional(fd, bytes, count, offset, false);
}
inline ssize_t pwrite(int fd, const void* bytes, size_t count, int64_t offset) {
    return positional(fd, const_cast<void*>(bytes), count, offset, true);
}
inline int fsync(int fd) {
    if (FlushFileBuffers(reinterpret_cast<HANDLE>(_get_osfhandle(fd)))) return 0;
    errno = EIO; return -1;
}
}
#endif
