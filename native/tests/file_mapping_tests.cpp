#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <fcntl.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "vrhino/error.h"
#include "vrhino/loader.h"

#ifdef _WIN32
namespace {
thread_local size_t allocations_before_failure = std::numeric_limits<size_t>::max();
}

// Test-only allocation failure injection. This also reaches allocating MSVC
// container constructors before the VrmModel constructor body is entered.
void* operator new(size_t size) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = std::numeric_limits<size_t>::max();
        throw std::bad_alloc();
    }
    if (allocations_before_failure != std::numeric_limits<size_t>::max()) --allocations_before_failure;
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, size_t) noexcept { std::free(pointer); }
#endif

namespace {
namespace fs = std::filesystem;
using vrhino::require;
using vrhino::VrmModel;
using Bytes = std::vector<uint8_t>;

struct FixtureDirectory {
    fs::path path = fs::current_path() / ("file-mapping-fixtures-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { require(fs::create_directory(path), "cannot create fixture directory"); }
    ~FixtureDirectory() {
        std::error_code error;
        // Only the unique test-owned child of the working directory is removed.
        if (path.parent_path() == fs::current_path()) fs::remove_all(path, error);
    }
};

void put64(Bytes& bytes, size_t offset, uint64_t value) {
    for (size_t index = 0; index < 8; ++index)
        bytes.at(offset + index) = static_cast<uint8_t>(value >> (8 * index));
}

Bytes prefix(uint64_t elements, uint64_t second_offset) {
    const std::string metadata = "{\"architecture\":\"fixture\"}";
    const std::string graph = "{\"schema_version\":1}";
    const std::string table =
        "{\"schema_version\":1,\"tensors\":[{\"name\":\"a\",\"dtype\":\"u8\",\"shape\":[" +
        std::to_string(elements) + "],\"offset\":0,\"byte_length\":" + std::to_string(elements) +
        ",\"alignment\":64,\"layout\":\"contiguous\",\"quantization\":{\"type\":\"none\"}},"
        "{\"name\":\"b\",\"dtype\":\"i32\",\"shape\":[1],\"offset\":" + std::to_string(second_offset) +
        ",\"byte_length\":4,\"alignment\":64,\"layout\":\"contiguous\",\"quantization\":{\"type\":\"none\"}}]}";
    const size_t data_offset = (128 + metadata.size() + table.size() + graph.size() + 63) / 64 * 64;
    Bytes bytes(data_offset);
    const std::array<uint8_t, 8> magic{'V','R','H','I','N','O',0,1};
    std::copy(magic.begin(), magic.end(), bytes.begin());
    bytes[10] = 1; bytes[12] = 1; bytes[13] = 6;
    const std::string profile = "component", architecture = "fixture";
    std::copy(profile.begin(), profile.end(), bytes.begin() + 16);
    std::copy(architecture.begin(), architecture.end(), bytes.begin() + 32);
    size_t cursor = 128;
    size_t field = 48;
    for (const auto& section : {metadata, table, graph}) {
        put64(bytes, field, cursor); put64(bytes, field + 8, section.size());
        std::copy(section.begin(), section.end(), bytes.begin() + static_cast<ptrdiff_t>(cursor));
        cursor += section.size(); field += 16;
    }
    put64(bytes, 96, data_offset);
    put64(bytes, 104, data_offset + second_offset + 4);
    return bytes;
}

Bytes small_fixture() {
    Bytes bytes = prefix(8, 64);
    const size_t data_offset = bytes.size();
    bytes.resize(data_offset + 68);
    bytes[data_offset] = 17; bytes[data_offset + 1] = 23; bytes[data_offset + 64] = 29;
    // Independent Python hashlib.blake2b(payload, digest_size=16) golden digest.
    constexpr std::array<uint8_t, 16> digest{
        0xb8, 0x92, 0x0e, 0xa2, 0x7d, 0x17, 0x6e, 0x63,
        0xa0, 0xd4, 0x35, 0x1c, 0xae, 0x44, 0x10, 0xa0};
    std::copy(digest.begin(), digest.end(), bytes.begin() + 112);
    return bytes;
}

void write_file(const fs::path& path, const Bytes& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    require(static_cast<bool>(file), "cannot write fixture");
}

std::string utf8_path(const fs::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

template<typename Operation>
void rejects(Operation operation, const std::string& message) {
    try { operation(); }
    catch (const vrhino::Error& error) {
        require(std::string(error.what()).find(message) != std::string::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw vrhino::Error("missing rejection: " + message);
}

#ifdef _WIN32
DWORD handle_count() {
    DWORD count = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &count) != 0, "handle count failed");
    return count;
}

std::pair<size_t, size_t> mapped_regions() {
    std::pair<size_t, size_t> result{};
    uintptr_t cursor = 0;
    MEMORY_BASIC_INFORMATION info{};
    while (VirtualQuery(reinterpret_cast<void*>(cursor), &info, sizeof(info)) == sizeof(info)) {
        if (info.Type == MEM_MAPPED) { ++result.first; result.second += info.RegionSize; }
        const auto base = reinterpret_cast<uintptr_t>(info.BaseAddress);
        if (info.RegionSize > std::numeric_limits<uintptr_t>::max() - base) break;
        const uintptr_t next = base + info.RegionSize;
        if (next <= cursor) break;
        cursor = next;
    }
    return result;
}

bool write_is_blocked(void* data) {
    // This leaf has no C++ objects requiring unwinding. Catch only the expected
    // protection fault, without crashing a child or leaving a crash dump.
    __try {
        *static_cast<volatile uint8_t*>(data) = 99;
        return false;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION
                    ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
}

void require_unmapped(void* address) {
    MEMORY_BASIC_INFORMATION info{};
    require(VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && info.State == MEM_FREE,
            "mapped view survived model destruction");
}

int owned_descriptor(const fs::path& path, HANDLE& native) {
    native = CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    require(native != INVALID_HANDLE_VALUE, "cannot open stable fixture");
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(native), _O_RDONLY | _O_BINARY);
    if (fd < 0) CloseHandle(native);
    require(fd >= 0, "cannot transfer fixture handle");
    return fd;
}

void require_closed(HANDLE handle) {
    DWORD flags = 0;
    require(GetHandleInformation(handle, &flags) == 0 && GetLastError() == ERROR_INVALID_HANDLE,
            "owned file handle was not closed");
}
#endif

void lifetime_contract(const fs::path& path) {
    void* address = nullptr;
    {
        VrmModel model(utf8_path(path));
        require(model.file_size() == fs::file_size(path), "small mapped size mismatch");
        require(model.architecture_id() == "fixture" && model.profile_id() == "component",
                "header parse changed");
        auto copy = model.tensor("a");
        auto view = copy.reshape({2, 4});
        auto second = model.tensor("b");
        address = copy.data();
        copy = {};
        require(!view.owns_storage() && !second.owns_storage(), "mapped tensors became owning");
        require(view.data_as<const uint8_t>()[0] == 17 && view.data_as<const uint8_t>()[1] == 23 &&
                    second.data_as<const int32_t>()[0] == 29, "borrowed view lifetime mismatch");
        { VrmModel other(utf8_path(path)); }
        require(view.data_as<const uint8_t>()[1] == 23, "another model unmapped our view");
        view = {}; second = {};
        require(model.tensor("a").data_as<const uint8_t>()[0] == 17, "view release unmapped model");
#ifdef _WIN32
        MEMORY_BASIC_INFORMATION info{};
        require(VirtualQuery(address, &info, sizeof(info)) == sizeof(info) &&
                    info.Type == MEM_MAPPED && info.Protect == PAGE_READONLY,
                "mapping is not read-only");
        require(write_is_blocked(address), "mapped bytes accepted a write");
        require(model.tensor("a").data_as<const uint8_t>()[0] == 17, "write probe changed mapped data");
#endif
    }
    // Borrowed tensors do not prolong the model's mapping, on either OS.
    // No tensor/view is dereferenced after its model is destroyed.
#ifdef _WIN32
    require_unmapped(address);
#else
    (void)address;
#endif
    std::cout << "small/checksum/borrowed-lifetime: PASS\n";
#ifdef _WIN32
    std::cout << "read-only/access-violation/unmap: PASS\n";
#endif
}

void rejection_contract(const fs::path& root) {
    const auto path = root / "invalid.vrm";
    rejects([&] { VrmModel model(utf8_path(root / "missing.vrm")); }, "Cannot open VRM");
    for (const size_t size : {size_t{0}, size_t{127}}) {
        write_file(path, Bytes(size));
        rejects([&] { VrmModel model(utf8_path(path)); }, "Truncated VRM file");
    }
    auto bytes = small_fixture();
    bytes[0] = 0; write_file(path, bytes);
    rejects([&] { VrmModel model(utf8_path(path)); }, "Bad VRM magic");
    bytes = small_fixture(); bytes.pop_back(); write_file(path, bytes);
    rejects([&] { VrmModel model(utf8_path(path)); }, "VRM file size/header mismatch");
    bytes = small_fixture(); bytes.back() ^= 1; write_file(path, bytes);
    rejects([&] { VrmModel model(utf8_path(path)); }, "VRM payload checksum mismatch");
    bytes = small_fixture(); put64(bytes, 80, std::numeric_limits<uint64_t>::max()); write_file(path, bytes);
    rejects([&] { VrmModel model(utf8_path(path), false); }, "Invalid VRM section layout");
    bytes = small_fixture(); bytes[128] = '!'; write_file(path, bytes);
    rejects([&] { VrmModel model(utf8_path(path), false); }, "JSON");
    rejects([&] { VrmModel model(-1); }, "Invalid VRM file descriptor");
#ifdef _WIN32
    rejects([&] { VrmModel model(std::string("\xff")); }, "Invalid VRM UTF-8 path");
    rejects([&] { VrmModel model(utf8_path(path) + std::string(1, '\0') + "suffix"); }, "Invalid VRM UTF-8 path");
    HANDLE native = INVALID_HANDLE_VALUE;
    const int fd = owned_descriptor(path, native);
    rejects([&] { VrmModel model(fd, false); }, "JSON");
    require_closed(native);
    // Force section creation failure using a write-only descriptor; the
    // constructor must still consume/close it without leaving a mapping.
    write_file(path, small_fixture());
    native = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(native != INVALID_HANDLE_VALUE, "cannot open write-only fixture");
    const int write_fd = _open_osfhandle(reinterpret_cast<intptr_t>(native), _O_WRONLY | _O_BINARY);
    if (write_fd < 0) CloseHandle(native);
    require(write_fd >= 0, "cannot own write-only fixture");
    rejects([&] { VrmModel model(write_fd); }, "VRM file mapping failed");
    require_closed(native);
#endif
    std::cout << "missing/empty/truncated/corrupt/overflow-layout/unwind: PASS\n";
}

void stable_open_contract(const fs::path& root) {
    const auto path = root / "stable.vrm", old_path = root / "old.vrm";
    write_file(path, small_fixture());
#ifdef _WIN32
    HANDLE native = INVALID_HANDLE_VALUE;
    const int fd = owned_descriptor(path, native);
#else
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "cannot open stable fixture");
#endif
    // Replace the name after opening; parsing must still use the old object.
    fs::rename(path, old_path);
    auto replacement = small_fixture(); replacement.back() = 1;
    write_file(path, replacement);
    {
        VrmModel model(fd);
        require(model.tensor("b").data_as<const int32_t>()[0] == 29, "stable descriptor reopened path");
        fs::remove(old_path);
        require(model.tensor("a").data_as<const uint8_t>()[0] == 17, "unlink invalidated mapped storage");
    }
#ifdef _WIN32
    require_closed(native);
#else
    require(fcntl(fd, F_GETFD) == -1, "owned descriptor leaked");
#endif
    std::cout << "stable-open/path-replacement/unlink/descriptor-close: PASS\n";
}

void unicode_contract(const fs::path& root) {
    const auto path = root / fs::path(u8"映射-测试-模型-🦏.vrm");
    write_file(path, small_fixture());
    VrmModel utf8(utf8_path(path));
    require(utf8.tensor("b").data_as<const int32_t>()[0] == 29, "UTF-8 path failed");
#ifdef _WIN32
    VrmModel native(path.native());
    require(native.tensor("b").data_as<const int32_t>()[0] == 29, "native UTF-16 path failed");
#endif
    std::cout << "Unicode path: PASS\n";
}

void large_file_contract(const fs::path& root) {
    static_assert(sizeof(size_t) >= 8, "large-file test requires a 64-bit host");
    constexpr uint64_t elements = (uint64_t{1} << 32) + 64;
    const auto path = root / "large-sparse.vrm";
    const Bytes header = prefix(elements, elements);
    write_file(path, header);
#ifdef _WIN32
    const HANDLE sparse_handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(sparse_handle != INVALID_HANDLE_VALUE, "cannot open sparse fixture");
    DWORD returned = 0;
    const BOOL sparse = DeviceIoControl(sparse_handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
    CloseHandle(sparse_handle);
    require(sparse != 0, "fixture filesystem must support sparse files");
#endif
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(header.size())); file.put(17);
        file.seekp(static_cast<std::streamoff>(header.size() + elements));
        const std::array<char, 4> tail{29, 0, 0, 0}; file.write(tail.data(), tail.size());
        file.close(); require(static_cast<bool>(file), "large sparse fixture write failed");
    }
    {
        VrmModel model(utf8_path(path), false); // Avoid hashing 4 GiB of sparse zeros.
        require(model.file_size() == header.size() + elements + 4 && model.file_size() > (uint64_t{1} << 32),
                "file size truncated to 32 bits");
        const auto& a = model.tensor("a"); const auto& b = model.tensor("b");
        require(a.bytes() == elements && a.numel() == static_cast<int64_t>(elements) &&
                    model.tensors().at("b").relative_offset == elements &&
                    a.data_as<const uint8_t>()[0] == 17 &&
                    a.data_as<const uint8_t>()[elements - 1] == 0 &&
                    b.data_as<const int32_t>()[0] == 29, "large-offset tensor read failed");
        std::cout << "large sparse mapping: PASS bytes=" << model.file_size()
                  << " tensor_offset=" << elements << '\n';
    }
#ifdef _WIN32
    DWORD high = 0;
    const DWORD low = GetCompressedFileSizeW(path.c_str(), &high);
    require(low != INVALID_FILE_SIZE || GetLastError() == NO_ERROR, "sparse allocation query failed");
    const uint64_t allocated = (static_cast<uint64_t>(high) << 32) | low;
    require(allocated < 16 * 1024 * 1024, "large fixture consumed excessive disk space");
    std::cout << "sparse disk allocation=" << allocated << " bytes\n";
#endif
}

void repeated_cleanup(const fs::path& root, const fs::path& good) {
    const auto bad = root / "bad-magic.vrm";
    const auto late = root / "late-failure.vrm";
    auto bytes = small_fixture(); bytes[0] = 0; write_file(bad, bytes);
    bytes = small_fixture(); bytes[32] = 'F'; write_file(late, bytes);
    { VrmModel warmup(utf8_path(good)); }
    rejects([&] { VrmModel warmup(utf8_path(bad)); }, "Bad VRM magic");
#ifdef _WIN32
    const DWORD before = handle_count();
    const auto regions_before = mapped_regions();
#endif
    for (int iteration = 0; iteration < 256; ++iteration) {
        void* address = nullptr;
        { VrmModel model(utf8_path(good)); address = model.tensor("a").data(); }
#ifdef _WIN32
        require_unmapped(address);
#else
        (void)address;
#endif
        if (iteration % 2 == 0)
            rejects([&] { VrmModel model(utf8_path(bad)); }, "Bad VRM magic");
        else
            rejects([&] { VrmModel model(utf8_path(late), false); }, "Metadata/header architecture mismatch");
    }
#ifdef _WIN32
    const DWORD after = handle_count();
    const auto regions_after = mapped_regions();
    require(after == before, "repeated open/close leaked native handles");
    require(regions_after == regions_before, "repeated failed opens leaked mapped views");
    std::cout << "256 successful + 256 failed opens: PASS handles=" << before << "->" << after << '\n';
    std::cout << "mapped regions=" << regions_before.first << "->" << regions_after.first
              << " mapped bytes=" << regions_before.second << "->" << regions_after.second << '\n';
#else
    std::cout << "256 successful + 256 failed opens: PASS\n";
#endif
}

#ifdef _WIN32
void allocation_failure_cleanup(const fs::path& path) {
    const DWORD handles_before = handle_count();
    const auto regions_before = mapped_regions();
    size_t failures = 0;
    for (size_t index = 0; index < 128; ++index) {
        HANDLE native = INVALID_HANDLE_VALUE;
        const int fd = owned_descriptor(path, native);
        allocations_before_failure = index;
        try {
            VrmModel model(fd);
            allocations_before_failure = std::numeric_limits<size_t>::max();
        } catch (const std::bad_alloc&) {
            allocations_before_failure = std::numeric_limits<size_t>::max();
            ++failures;
        } catch (...) {
            allocations_before_failure = std::numeric_limits<size_t>::max();
            throw;
        }
        require_closed(native);
    }
    require(failures > 0, "allocation failure injection did not run");
    require(handle_count() == handles_before && mapped_regions() == regions_before,
            "allocation failure leaked file/section/view resources");
    std::cout << "allocation-failure cleanup: PASS injected=" << failures
              << " handles/views unchanged\n";
}
#endif
} // namespace

int main() {
    try {
        FixtureDirectory root;
        const auto good = root.path / "small.vrm";
        write_file(good, small_fixture());
        lifetime_contract(good);
        rejection_contract(root.path);
        stable_open_contract(root.path);
        unicode_contract(root.path);
        large_file_contract(root.path);
        repeated_cleanup(root.path, good);
#ifdef _WIN32
        allocation_failure_cleanup(good);
#endif
        std::cout << "file mapping contract: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file mapping contract: FAIL: " << error.what() << '\n';
        return 1;
    }
}
