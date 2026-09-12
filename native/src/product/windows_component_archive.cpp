#include "windows_component_archive.h"
#ifdef _WIN32
#define LIBARCHIVE_STATIC
#include <archive.h>
#include <archive_entry.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>

namespace vrhino::product::windows_component_archive {
namespace fs = std::filesystem;
namespace wc = windows_cache;
namespace {
[[noreturn]] void invalid(const std::string& detail) {
    throw ModelPackageError(ModelPackageErrorCode::ComponentInvalid, "component archive: " + detail);
}
void cancel(const std::function<bool()>& callback) {
    if (callback && callback())
        throw ModelPackageError(ModelPackageErrorCode::Cancelled, "component extraction cancelled");
}
uint64_t extent(int64_t offset, uint64_t count, uint64_t limit) {
    if (offset < 0 || static_cast<uint64_t>(offset) > limit ||
        count > limit - static_cast<uint64_t>(offset)) invalid("invalid or overflowing file extent");
    return static_cast<uint64_t>(offset) + count;
}
std::wstring wide(const std::string& value) {
    if (value.empty() || value.size() > INT_MAX || value.find('\0') != std::string::npos)
        invalid("empty, oversized, or NUL-containing path");
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                          static_cast<int>(value.size()), nullptr, 0);
    if (!length) invalid("path is not valid UTF-8");
    std::wstring result(static_cast<size_t>(length), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                           static_cast<int>(value.size()), result.data(), length) != length)
        invalid("cannot decode path");
    return result;
}
struct CaseLess {
    bool operator()(const std::wstring& a, const std::wstring& b) const {
        return CompareStringOrdinal(a.data(), static_cast<int>(a.size()), b.data(),
                                    static_cast<int>(b.size()), TRUE) == CSTR_LESS_THAN;
    }
};
void segment(const std::wstring& part) {
    if (part.empty() || part == L"." || part == L".." || part.back() == L'.' || part.back() == L' ')
        invalid("unsafe path segment");
    for (const wchar_t c : part)
        if (c < 32 || std::wstring_view(L"\\/:<>\"|?*").find(c) != std::wstring_view::npos)
            invalid("forbidden Windows path character");
    auto base = part.substr(0, part.find(L'.'));
    while (!base.empty() && base.back() == L' ') base.pop_back();
    const auto equal = [&](std::wstring_view other) {
        return CompareStringOrdinal(base.data(), static_cast<int>(base.size()), other.data(),
                                    static_cast<int>(other.size()), TRUE) == CSTR_EQUAL;
    };
    if (equal(L"CON") || equal(L"PRN") || equal(L"AUX") || equal(L"NUL") ||
        equal(L"CLOCK$") || equal(L"CONIN$") || equal(L"CONOUT$")) invalid("reserved Windows device name");
    if (base.size() == 4 && (std::wstring_view(L"123456789\u00b9\u00b2\u00b3").find(base[3]) != std::wstring_view::npos)) {
        base.resize(3);
        if (equal(L"COM") || equal(L"LPT")) invalid("reserved Windows device name");
    }
}
size_t read(wc::LocalSource& source, uint64_t& offset, void* output, DWORD capacity) {
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(offset);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
    wc::Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) invalid("cannot create read event");
    operation.hEvent = event.get();
    DWORD count = 0;
    if (!ReadFile(source.get(), output, capacity, &count, &operation)) {
        const DWORD error = GetLastError();
        if (error == ERROR_HANDLE_EOF) return 0;
        if (error != ERROR_IO_PENDING || !GetOverlappedResult(source.get(), &operation, &count, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) return 0;
            invalid("archive read failed");
        }
    }
    offset = extent(static_cast<int64_t>(offset), count, INT64_MAX);
    source.verify();
    return count;
}

// libarchive's gzip filter does not check CRC/ISIZE. zlib verifies every member
// and its trailer on this same retained source object before any extraction.
void verify_gzip(wc::LocalSource& source, const std::function<bool()>& cancelled) {
    z_stream stream{};
    if (inflateInit2(&stream, 15 + 16) != Z_OK) invalid("cannot initialize gzip validation");
    struct End { z_stream* stream; ~End() { inflateEnd(stream); } } end{&stream};
    std::array<unsigned char, 65536> input{}, output{};
    std::array<unsigned char, 1024> tail{};
    uint64_t position = 0, total = 0;
    bool complete = false, any = false;
    for (;;) {
        cancel(cancelled);
        if (stream.avail_in == 0) {
            const auto count = read(source, position, input.data(), static_cast<DWORD>(input.size()));
            if (!count) break;
            stream.next_in = input.data();
            stream.avail_in = static_cast<uInt>(count);
        }
        if (complete) {
            if (inflateReset2(&stream, 15 + 16) != Z_OK) invalid("cannot reset gzip validation");
            complete = false;
        }
        stream.next_out = output.data();
        stream.avail_out = static_cast<uInt>(output.size());
        const auto previous = stream.avail_in;
        const int result = inflate(&stream, Z_NO_FLUSH);
        const auto produced = output.size() - stream.avail_out;
        for (size_t i = 0; i < produced; ++i) tail[(total + i) % tail.size()] = output[i];
        total = extent(static_cast<int64_t>(total), produced, INT64_MAX);
        if (result == Z_STREAM_END) { complete = true; any = true; }
        else if (result != Z_OK || (produced == 0 && previous == stream.avail_in))
            invalid("corrupt gzip stream or checksum");
    }
    if (!any || !complete || total < 1024 || total % 512 != 0 ||
        std::any_of(tail.begin(), tail.end(), [](unsigned char c) { return c != 0; }))
        invalid("truncated gzip stream or tar terminator");
    source.verify();
}
struct Reader {
    wc::LocalSource& source;
    const std::function<bool()>& cancelled;
    uint64_t position = 0;
    std::array<unsigned char, 65536> buffer{};
    std::exception_ptr failure;
    static la_ssize_t callback(archive* object, void* context, const void** block) noexcept {
        auto& self = *static_cast<Reader*>(context);
        try {
            cancel(self.cancelled);
#ifdef VRHINO_WINDOWS_ARCHIVE_TESTING
            checkpoint(Point::Read);
#endif
            *block = self.buffer.data();
            return static_cast<la_ssize_t>(read(self.source, self.position, self.buffer.data(),
                                               static_cast<DWORD>(self.buffer.size())));
        } catch (...) {
            self.failure = std::current_exception();
            archive_set_error(object, EIO, "retained archive read failed");
            return -1;
        }
    }
    void check(int result, archive* object) {
        if (failure) std::rethrow_exception(failure);
        if (result != ARCHIVE_OK) {
            const char* message = archive_error_string(object);
            invalid(message ? message : "archive decoding failed");
        }
    }
};
fs::path member(const char* text, bool directory = false) {
    if (!text) invalid("missing UTF-8 member name");
    std::string name(text);
    if (directory && !name.empty() && name.back() == '/') name.pop_back();
    const auto path = relative_path(name);
    if (*path.begin() != L"vrhino-media") invalid("unexpected top-level directory");
    return path;
}
}

#ifdef VRHINO_WINDOWS_ARCHIVE_TESTING
std::function<void(Point)> test_hook;
void checkpoint(Point point) { if (test_hook) test_hook(point); }
uint64_t checked_extent(int64_t offset, uint64_t count, uint64_t limit) { return extent(offset, count, limit); }
#endif
fs::path relative_path(const std::string& utf8) {
    const auto value = wide(utf8);
    size_t start = 0;
    for (;;) {
        const auto end = value.find(L'/', start);
        segment(value.substr(start, end == std::wstring::npos ? end : end - start));
        if (end == std::wstring::npos) break;
        start = end + 1;
    }
    return fs::path(value);
}
void identity_segment(const std::string& value) { segment(wide(value)); }
std::string read_text(const fs::path& path) {
    wc::LocalSource source(path);
    std::string result;
    std::array<char, 65536> buffer{};
    uint64_t position = 0;
    for (;;) {
        const auto count = read(source, position, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (!count) break;
        if (count > result.max_size() - result.size()) invalid("manifest size overflow");
        result.append(buffer.data(), count);
    }
    source.verify();
    return result;
}
bool executable(const fs::path& path) {
    wc::Parents parents(path.parent_path());
    wc::Handle file(CreateFileW(wc::native_path(path).c_str(), FILE_READ_ATTRIBUTES | FILE_EXECUTE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    if (!file) return false;
    wc::admit(file.get());
    const auto state = wc::snapshot(file.get());
    return !state.standard.Directory && !(state.basic.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT);
}

void extract(wc::LocalSource& source, const fs::path& staging,
             const std::function<bool()>& cancelled) {
    verify_gzip(source, cancelled);
    std::unique_ptr<archive, decltype(&archive_read_free)> object(archive_read_new(), archive_read_free);
    if (!object) invalid("cannot allocate archive reader");
    Reader reader{source, cancelled};
    reader.check(archive_read_support_filter_gzip(object.get()), object.get());
    reader.check(archive_read_support_format_tar(object.get()), object.get());
    reader.check(archive_read_open(object.get(), &reader, nullptr, Reader::callback, nullptr), object.get());
    if (archive_filter_code(object.get(), 0) != ARCHIVE_FILTER_GZIP) invalid("archive must be gzip compressed tar");
    // Every prefix has a single spelling/type. Existing unexpected filesystem
    // names (including 8.3 aliases) fail closed instead of being reused.
    struct Name { std::wstring spelling; bool directory; bool explicit_entry; };
    std::map<std::wstring, Name, CaseLess> names;
    std::set<fs::path> regular;
    std::map<fs::path, fs::path> links;
    auto claim = [&](const fs::path& path, bool directory) {
        fs::path prefix;
        for (auto it = path.begin(); it != path.end(); ++it) {
            prefix /= *it;
            const bool leaf = std::next(it) == path.end();
            const bool is_directory = !leaf || directory;
            const auto spelling = prefix.native();
            const auto found = names.find(spelling);
            if (found != names.end()) {
                if (found->second.spelling != spelling || found->second.directory != is_directory ||
                    (leaf && found->second.explicit_entry)) invalid("duplicate, case alias, or conflicting archive path");
                if (leaf) found->second.explicit_entry = true;
            } else {
                if (fs::exists(wc::native_path(staging / prefix))) invalid("archive filesystem alias collision");
                names.emplace(spelling, Name{spelling, is_directory, leaf});
                if (is_directory) wc::ensure_directory(staging / prefix);
            }
        }
    };
    archive_entry* entry = nullptr;
    for (;;) {
        cancel(cancelled);
        const int next = archive_read_next_header(object.get(), &entry);
        if (next == ARCHIVE_EOF) break;
        reader.check(next, object.get());
        const auto type = archive_entry_filetype(entry);
        const bool directory = type == AE_IFDIR;
        const auto path = member(archive_entry_pathname_utf8(entry), directory);
        const char* target = archive_entry_hardlink_utf8(entry);
        if (archive_entry_symlink(entry) || (!directory && type != AE_IFREG && !target))
            invalid("symlink or special archive member");
        if (directory && target) invalid("directory hardlink");
        if ((directory || target) && archive_entry_size(entry) != 0) invalid("payload on non-regular archive entry");
        claim(path, directory);
        if (directory) continue;
        if (target) {
            links.emplace(path, member(target));
            continue;
        }
        if (!archive_entry_size_is_set(entry) || archive_entry_size(entry) < 0) invalid("missing or negative file size");
        const auto size = static_cast<uint64_t>(archive_entry_size(entry));
        wc::require_space(staging, size);
        wc::StagedFile file(wc::unique_path((staging / path).parent_path(), "archive-file-"));
        uint64_t written = 0;
        std::array<unsigned char, 65536> zeros{};
        for (;;) {
            const void* block = nullptr;
            size_t count = 0;
            la_int64_t offset = 0;
            const int result = archive_read_data_block(object.get(), &block, &count, &offset);
            if (result == ARCHIVE_EOF) break;
            reader.check(result, object.get());
            const uint64_t end = extent(offset, count, size);
            if (static_cast<uint64_t>(offset) < written) invalid("overlapping archive file extents");
            while (written < static_cast<uint64_t>(offset)) {
                cancel(cancelled);
                const auto gap = static_cast<size_t>(std::min<uint64_t>(zeros.size(), static_cast<uint64_t>(offset) - written));
                file.write(zeros.data(), gap);
                written += gap;
            }
            cancel(cancelled);
            file.write(block, count);
            written = end;
        }
        // A missing non-sparse payload must not be turned into valid zero data.
        if (written != size && archive_entry_sparse_count(entry) == 0) invalid("truncated archive member");
        while (written < size) {
            cancel(cancelled);
            const auto count = static_cast<size_t>(std::min<uint64_t>(zeros.size(), size - written));
            file.write(zeros.data(), count);
            written += count;
        }
        file.readonly();
        file.flush();
        const auto progress = [&](uint64_t) { cancel(cancelled); };
        file.validate(size, wc::hash(file.get(), size, progress), progress);
        file.publish(staging / path);
        regular.insert(path);
    }
    reader.check(archive_read_close(object.get()), object.get());
    while (!links.empty()) {
        bool progress = false;
        for (auto it = links.begin(); it != links.end();) {
            cancel(cancelled);
            if (!regular.contains(it->second)) { ++it; continue; }
            wc::required_link(staging / it->second, staging / it->first);
            regular.insert(it->first);
            it = links.erase(it);
            progress = true;
        }
        if (!progress) invalid("unresolved or cyclic archive hardlink");
    }
    if (!regular.contains(fs::path(L"vrhino-media/vrhino-component.json"))) invalid("missing component manifest");
    source.verify();
#ifdef VRHINO_WINDOWS_ARCHIVE_TESTING
    checkpoint(Point::Extracted);
#endif
}
}
#endif
