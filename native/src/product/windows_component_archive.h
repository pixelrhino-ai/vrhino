#pragma once
#ifdef _WIN32
#include "vrhino/product/windows_cache.h"

namespace vrhino::product::windows_component_archive {
std::filesystem::path relative_path(const std::string& utf8);
void identity_segment(const std::string& value);
bool executable(const std::filesystem::path& path);
std::string read_text(const std::filesystem::path& path);
void extract(windows_cache::LocalSource& source, const std::filesystem::path& staging,
             const std::function<bool()>& cancelled);
#ifdef VRHINO_WINDOWS_ARCHIVE_TESTING
enum class Point { Read, Extracted, BeforeCommit };
extern std::function<void(Point)> test_hook;
void checkpoint(Point point);
uint64_t checked_extent(int64_t offset, uint64_t count, uint64_t limit);
#endif
}
#endif
