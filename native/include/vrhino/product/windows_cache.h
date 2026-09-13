#pragma once

// Windows Product cache primitives, intentionally separate from Runtime.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <string_view>
#include "vrhino/product/model_package.h"

namespace vrhino::product::windows_cache {
namespace fs = std::filesystem;
class Handle {
public:
    explicit Handle(HANDLE value = INVALID_HANDLE_VALUE) noexcept : value_(value) {}
    ~Handle();
    Handle(Handle&& other) noexcept;
    Handle& operator=(Handle&& other) noexcept;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    HANDLE get() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ != INVALID_HANDLE_VALUE && value_ != nullptr; }
    HANDLE release() noexcept;
private:
    HANDLE value_;
};

fs::path native_path(const fs::path& path);
fs::path unique_path(const fs::path& parent, const std::string& prefix);
void ensure_directory(const fs::path& path);
void require_space(const fs::path& directory, uint64_t bytes);
uint64_t volume(HANDLE object);
void same_volume(HANDLE object, HANDLE destination);
void admit(HANDLE object);

// Pins every ancestor against rename/reparse substitution while paths are used.
class Parents {
public:
    explicit Parents(const fs::path& directory);
    HANDLE leaf() const { return handles_.back().get(); }
private:
    std::vector<Handle> handles_;
};

enum class LockMode { Exclusive, Shared };
class Lock {
public:
    explicit Lock(const fs::path& path, LockMode mode = LockMode::Exclusive);
    ~Lock();
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;
private:
    std::unique_ptr<Parents> parents_;
    Handle file_;
    OVERLAPPED range_{};
    bool locked_ = false;
};

struct Snapshot {
    FILE_ID_INFO identity{};
    FILE_BASIC_INFO basic{};
    FILE_STANDARD_INFO standard{};
};
Snapshot snapshot(HANDLE file);
bool same_identity(const Snapshot& a, const Snapshot& b);
bool unchanged(const Snapshot& a, const Snapshot& b);
std::string hash(HANDLE file, uint64_t size, const WorkProgressCallback& progress = {});

class ChangeWatch;
class LocalSource {
public:
    explicit LocalSource(const fs::path& path, bool request_writable = false);
    ~LocalSource();
    HANDLE get() const { return file_.get(); }
    const Snapshot& before() const { return before_; }
    bool writable() const { return writable_; }
    std::string digest(const WorkProgressCallback& progress = {});
    void verify() const;
    void finish();
private:
    Parents parents_;
    Handle file_;
    std::unique_ptr<ChangeWatch> watch_;
    Snapshot before_{};
    bool writable_ = false;
};

class StagedFile {
public:
    // Existing resume/input files are acquired once, before any validation.
    explicit StagedFile(const fs::path& path, bool existing = false, bool keep_partial = false);
    ~StagedFile();
    StagedFile(const StagedFile&) = delete;
    StagedFile& operator=(const StagedFile&) = delete;
    HANDLE get() const { return file_.get(); }
    const fs::path& path() const { return path_; }
    bool published() const { return !owned_; }
    uint64_t size() const;
    void write(const void* bytes, size_t count);
    void truncate(uint64_t size);
    void copy(LocalSource& source);
    void readonly();
    void flush();
    void validate(uint64_t bytes, const std::string& digest, const WorkProgressCallback& progress = {});
    void publish(const fs::path& destination, bool replace = false);
    void discard();
    void keep_partial(bool keep) { keep_ = keep; }
private:
    fs::path path_;
    Parents parents_;
    Handle file_;
    uint64_t offset_ = 0;
    bool owned_ = true;
    bool keep_ = false;
    bool flushed_ = false;
    bool validated_ = false;
    bool cleanup_safe_ = true;
    Snapshot validated_state_{};
};

void write_text(const fs::path& destination, const std::string& text, bool readonly = false, bool replace = false);
void publish_directory(const fs::path& staging, const fs::path& destination);
void migrate_partial(const fs::path& source, const fs::path& destination);
void required_link(const fs::path& source, const fs::path& destination);
bool equivalent(const fs::path& first, const fs::path& second);
uint64_t link_count(const fs::path& path);
void remove(const fs::path& path);
void remove_tree(const fs::path& path);
void retire_tree(const fs::path& path, const fs::path& temporary);

#ifdef VRHINO_WINDOWS_CACHE_TESTING
enum class Point { Admission, Lock, Write, Flush, Validate, BeforePublish, AfterPublish, Hardlink, Cleanup, SourceCopy, Space };
extern std::function<void(Point)> test_hook;
void test_volume_admission(std::wstring_view filesystem, UINT drive_type, bool reparse);
#endif
} // namespace vrhino::product::windows_cache
#endif
