#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#include <io.h>
#include <fcntl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <new>
#include <thread>

#include "vrhino/error.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/vrm_verification.h"
#include "fixtures/stable_verification_fixture.h"

namespace { thread_local size_t fail_allocation_after = SIZE_MAX; }
void* operator new(size_t size) {
    if (fail_allocation_after == 0) { fail_allocation_after = SIZE_MAX; throw std::bad_alloc(); }
    if (fail_allocation_after != SIZE_MAX) --fail_allocation_after;
    if (void* value = std::malloc(size ? size : 1)) return value;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, size_t) noexcept { std::free(value); }

namespace {
namespace fs = std::filesystem;
namespace p = vrhino::product;
namespace testing = p::stable_verification_testing;
using Stage = testing::Stage;
using vrhino::require;

struct Handle {
    HANDLE value;
    explicit Handle(HANDLE handle) : value(handle) { require(value != INVALID_HANDLE_VALUE && value, "test handle creation failed"); }
    ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
struct Descriptor {
    int value;
    explicit Descriptor(HANDLE handle) : value(_open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY)) {
        if (value < 0) CloseHandle(handle);
        require(value >= 0, "test descriptor creation failed");
    }
    ~Descriptor() { _close(value); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
};
HANDLE open_file(const fs::path& path, DWORD access = GENERIC_READ,
                 DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED) {
    return CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       nullptr, OPEN_EXISTING, flags, nullptr);
}
struct Directory {
    fs::path path = fs::current_path() / ("stable-verification-fixtures-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Directory() { require(fs::create_directory(path), "cannot create fixture directory"); }
    ~Directory() { std::error_code ec; if (path.parent_path() == fs::current_path()) fs::remove_all(path, ec); }
};
struct HookReset { ~HookReset() { testing::hook = {}; } };

void write_fixture(const fs::path& path, bool different = false) {
    auto bytes = stable_fixture::bytes;
    if (different) bytes.back() ^= 1;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.close(); require(static_cast<bool>(output), "fixture write failed");
}
p::VrmComponentIntegrityContract contract() { return {"fixture", "fixture", stable_fixture::bytes.size(), stable_fixture::sha256}; }

template<class Operation>
void reject(Operation operation, const std::string& message) {
    try { operation(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(message) != std::string::npos,
                "unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("missing rejection: " + message);
}
void payload(const vrhino::VrmModel& model) {
    require(model.tensor("a").data_as<const uint8_t>()[0] == 17 &&
            model.tensor("b").data_as<const int32_t>()[0] == 29, "accepted wrong opened object");
}
FILE_BASIC_INFO basic_info(HANDLE file) {
    FILE_BASIC_INFO basic{};
    require(GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic)) != 0, "basic info failed");
    return basic;
}
FILE_ID_INFO identity(HANDLE file) {
    FILE_ID_INFO info{};
    require(GetFileInformationByHandleEx(file, FileIdInfo, &info, sizeof(info)) != 0, "identity query failed");
    return info;
}
bool same_id(const FILE_ID_INFO& a, const FILE_ID_INFO& b) {
    return a.VolumeSerialNumber == b.VolumeSerialNumber &&
        std::equal(std::begin(a.FileId.Identifier), std::end(a.FileId.Identifier), std::begin(b.FileId.Identifier));
}
void write_byte(HANDLE writer, uint64_t offset, uint8_t value) {
    LARGE_INTEGER position{}; position.QuadPart = static_cast<LONGLONG>(offset);
    require(SetFilePointerEx(writer, position, nullptr, FILE_BEGIN) != 0, "mutation seek failed");
    DWORD count = 0;
    require(WriteFile(writer, &value, 1, &count, nullptr) != 0 && count == 1, "mutation write failed");
    require(FlushFileBuffers(writer) != 0, "mutation flush failed");
}
std::pair<DWORD, size_t> resources() {
    DWORD handles = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &handles) != 0, "handle count failed");
    size_t mapped = 0;
    uintptr_t cursor = 0;
    MEMORY_BASIC_INFORMATION info{};
    while (VirtualQuery(reinterpret_cast<void*>(cursor), &info, sizeof(info)) == sizeof(info)) {
        if (info.Type == MEM_MAPPED) mapped += info.RegionSize;
        const auto base = reinterpret_cast<uintptr_t>(info.BaseAddress);
        if (info.RegionSize > UINTPTR_MAX - base) break;
        const auto next = base + info.RegionSize;
        if (next <= cursor) break;
        cursor = next;
    }
    return {handles, mapped};
}

void unchanged_and_unicode(const fs::path& root) {
    const auto path = root / fs::path(u8"验证-模型-🦏.vrm");
    write_fixture(path);
    const auto utf8 = path.u8string();
    const std::string boundary(reinterpret_cast<const char*>(utf8.data()), utf8.size());
    p::VrmVerificationObservation result;
    const std::u8string native_boundary(boundary.begin(), boundary.end());
    auto model = p::load_verified_vrm_component(fs::path(native_boundary), contract(), &result);
    payload(*model);
    require(result.sha256_completed && result.blake2b_completed &&
            result.sha256_bytes == contract().bytes && result.blake2b_bytes == contract().bytes - 128 &&
            result.maximum_digest_consumers == 2, "verification work accounting changed");
    require(p::sha256_file(path) == stable_fixture::sha256, "golden SHA-256 mismatch");
    Descriptor fd(open_file(path));
    require(p::sha256_file_descriptor(fd.value, contract().bytes) == stable_fixture::sha256, "descriptor SHA mismatch");
    auto other = std::async(std::launch::async, [&] { return p::sha256_file_descriptor(fd.value, contract().bytes); });
    require(p::sha256_file_descriptor(fd.value, contract().bytes) == other.get(), "concurrent positional reads interfered");
    Descriptor synchronous(open_file(path, GENERIC_READ, FILE_ATTRIBUTE_NORMAL));
    require(p::sha256_file_descriptor(synchronous.value, contract().bytes) == stable_fixture::sha256,
            "synchronous descriptor positioned read failed");
    reject([&] { (void)p::sha256_file_descriptor(fd.value, contract().bytes + 1); }, "premature EOF");
    std::cout << "unchanged/golden SHA/concurrent digests/UTF-8/UTF-16: PASS\n";
}

void replacement_tests(const fs::path& root) {
    for (const std::string operation : {"replace", "replace-large", "rename", "unlink", "recreate", "metadata"}) {
        const auto path = root / (operation + ".vrm"), other = root / (operation + "-other.vrm"), moved = root / (operation + "-old.vrm");
        write_fixture(path); write_fixture(other, true);
        if (operation == "replace-large") { std::ofstream out(other, std::ios::binary | std::ios::app); out << "extra"; }
        Handle observer(open_file(path));
        const auto before = identity(observer.value);
        p::VrmVerificationObservation observation;
        auto model = p::load_verified_vrm_component(path, contract(), &observation, [&] {
            if (operation == "replace" || operation == "replace-large") {
                Handle source(open_file(other, DELETE, FILE_ATTRIBUTE_NORMAL));
                const auto name_bytes = path.native().size() * sizeof(wchar_t);
                const auto size = sizeof(FILE_RENAME_INFO) + name_bytes;
                auto storage = std::make_unique<unsigned char[]>(size);
                auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(storage.get());
                rename->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
                rename->FileNameLength = static_cast<DWORD>(name_bytes);
                std::memcpy(rename->FileName, path.c_str(), name_bytes);
                if (!SetFileInformationByHandle(source.value, FileRenameInfoEx, rename, static_cast<DWORD>(size)))
                    throw std::runtime_error("replacement failed Win32=" + std::to_string(GetLastError()));
            }
            if (operation == "rename") fs::rename(path, moved);
            if (operation == "unlink" || operation == "recreate") fs::remove(path);
            if (operation == "recreate") fs::copy_file(other, path);
            if (operation == "metadata") {
                Handle metadata(open_file(path, FILE_WRITE_ATTRIBUTES, FILE_ATTRIBUTE_NORMAL));
                auto info = basic_info(metadata.value); info.ChangeTime.QuadPart += 10000000;
                info.LastAccessTime.QuadPart = 0;
                require(SetFileInformationByHandle(metadata.value, FileBasicInfo, &info, sizeof(info)) != 0, "metadata change failed");
            }
            require(same_id(before, identity(observer.value)), "opened identity changed on namespace operation");
            if (operation == "replace" || operation == "replace-large" || operation == "recreate") {
                Handle current(open_file(path));
                require(!same_id(before, identity(current.value)), "replacement did not change pathname identity");
            }
        });
        payload(*model);
        if (operation == "metadata") require(observation.sha256_bytes == 2 * contract().bytes, "ChangeTime-only retry missing");
        std::cout << "same opened object " << operation << ": PASS digest_passes="
                  << observation.sha256_bytes / contract().bytes << '\n';
    }
}

void mutation_tests(const fs::path& root) {
    const auto path = root / "mutate.vrm";
    for (const std::string action : {"content", "content-during-hash", "restore-all", "timestamp", "truncate", "grow"}) {
        write_fixture(path);
        Handle writer(open_file(path, GENERIC_READ | GENERIC_WRITE, FILE_ATTRIBUTE_NORMAL));
        const auto original = basic_info(writer.value);
        HookReset reset;
        bool executed = false;
        testing::hook = [&](Stage stage, unsigned) {
            const Stage desired = (action == "truncate" || action == "grow") ? Stage::AfterDuplicates :
                action == "content-during-hash" ? Stage::HashChunk : Stage::AfterDigests;
            if (stage != desired || executed) return;
            executed = true;
            // A different thread mutates the object within the verification interval.
            auto mutation = std::async(std::launch::async, [&] {
                if (action == "truncate" || action == "grow") {
                    FILE_END_OF_FILE_INFO size{}; size.EndOfFile.QuadPart = action == "truncate" ? 100 : 1024;
                    require(SetFileInformationByHandle(writer.value, FileEndOfFileInfo, &size, sizeof(size)) != 0, "size mutation failed");
                } else if (action == "timestamp") {
                    auto basic = original; basic.LastWriteTime.QuadPart += 10000000;
                    require(SetFileInformationByHandle(writer.value, FileBasicInfo, &basic, sizeof(basic)) != 0, "timestamp mutation failed");
                } else {
                    write_byte(writer.value, 512, 99);
                    if (action == "restore-all") {
                        write_byte(writer.value, 512, 17);
                        auto restore = original;
                        require(SetFileInformationByHandle(writer.value, FileBasicInfo, &restore, sizeof(restore)) != 0, "timestamp restore failed");
                        const auto now = basic_info(writer.value);
                        require(now.LastWriteTime.QuadPart == original.LastWriteTime.QuadPart &&
                                now.ChangeTime.QuadPart == original.ChangeTime.QuadPart, "timestamps were not restored");
                    }
                }
            });
            mutation.get();
        };
        reject([&] { (void)p::load_verified_vrm_component(path, contract()); }, "changed during verification");
        require(executed, "mutation hook did not execute");
        std::cout << "concurrent mutation " << action << ": PASS\n";
    }
    write_fixture(path);
    HookReset reset;
    testing::hook = [&](Stage stage, unsigned attempt) {
        if (stage != Stage::AfterDigests) return;
        Handle writer(open_file(path, FILE_WRITE_ATTRIBUTES, FILE_ATTRIBUTE_NORMAL));
        auto info = basic_info(writer.value); info.ChangeTime.QuadPart += 10000000 + attempt;
        require(SetFileInformationByHandle(writer.value, FileBasicInfo, &info, sizeof(info)) != 0, "metadata churn failed");
    };
    reject([&] { (void)p::load_verified_vrm_component(path, contract()); }, "remained unstable");
    std::cout << "repeated metadata-only churn: PASS\n";
}

void writable_mapping_tests(const fs::path& root) {
    const auto path = root / "writable-map.vrm";
    write_fixture(path);
    Handle writer(open_file(path, GENERIC_READ | GENERIC_WRITE, FILE_ATTRIBUTE_NORMAL));
    auto mapped_write = [&] {
        Handle section(CreateFileMappingW(writer.value, nullptr, PAGE_READWRITE, 0, 0, nullptr));
        struct View {
            void* value;
            ~View() { if (value) UnmapViewOfFile(value); }
        } view{MapViewOfFile(section.value, FILE_MAP_WRITE, 0, 0, 0)};
        require(view.value != nullptr, "writable view creation failed");
        auto* bytes = static_cast<volatile uint8_t*>(view.value);
        bytes[512] = 99;
        require(FlushViewOfFile(view.value, 0) != 0, "writable view flush failed");
        bytes[512] = 17;
        require(FlushViewOfFile(view.value, 0) != 0, "writable view restore failed");
    };
    {
        Handle existing(CreateFileMappingW(writer.value, nullptr, PAGE_READWRITE, 0, 0, nullptr));
        reject([&] { (void)p::load_verified_vrm_component(path, contract()); }, "cannot establish stable file change guard");
    }
    HookReset reset;
    testing::hook = [&](Stage stage, unsigned) { if (stage == Stage::AfterDigests) mapped_write(); };
    reject([&] { (void)p::load_verified_vrm_component(path, contract()); }, "changed during verification");
    std::cout << "existing/new writable mappings fail closed: PASS\n";
}

void large_file_test(const fs::path& root) {
    const auto path = root / "large-sparse.bin";
    { std::ofstream out(path, std::ios::binary); out << 'x'; }
    constexpr uint64_t offset = (uint64_t{1} << 32) + 64;
    {
        Handle writer(open_file(path, GENERIC_READ | GENERIC_WRITE, FILE_ATTRIBUTE_NORMAL));
        DWORD ignored = 0;
        require(DeviceIoControl(writer.value, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ignored, nullptr) != 0, "sparse setup failed");
        for (size_t index = 0; index < 4; ++index) write_byte(writer.value, offset + index, index == 0 ? 29 : 0);
    }
    Descriptor reader(open_file(path));
    require(testing::descriptor_range_digest(reader.value, offset, 4) == stable_fixture::tail_sha256, "64-bit SHA read offset truncated");
    auto expected = contract(); expected.bytes = offset + 4;
    bool reached = false;
    reject([&] { (void)p::load_verified_vrm_component(path, expected, nullptr, [&] {
        reached = true; throw std::runtime_error("large metadata accepted");
    }); }, "large metadata accepted");
    require(reached, "64-bit size metadata truncated");
    expected.bytes = 68;
    reject([&] { (void)p::load_verified_vrm_component(path, expected); }, "size mismatch");
    std::cout << ">4 GiB identity/size/actual positioned SHA read: PASS offset=" << offset << '\n';
}

void failure_and_resource_tests(const fs::path& root) {
    const auto path = root / "resource.vrm";
    write_fixture(path);
    { auto warmup = p::load_verified_vrm_component(path, contract()); }
    const auto before = resources();
    for (int index = 0; index < 64; ++index) { auto model = p::load_verified_vrm_component(path, contract()); payload(*model); }
    require(resources() == before, "successful verification leaked resources");
    for (Stage target : {Stage::AfterShaDuplicate, Stage::AfterDuplicates, Stage::HashChunk, Stage::AfterDigests}) {
        HookReset reset;
        testing::hook = [=](Stage stage, unsigned) { if (stage == target) throw std::runtime_error("injected stage failure"); };
        for (int index = 0; index < 16; ++index)
            reject([&] { (void)p::load_verified_vrm_component(path, contract()); }, "injected stage failure");
        require(resources() == before, "injected failure leaked duplicate/event/view handles");
    }
    size_t allocation_failures = 0;
    for (size_t index = 0; index < 128; ++index) {
        const auto expected = contract();
        fail_allocation_after = index;
        try { auto model = p::load_verified_vrm_component(path, expected); fail_allocation_after = SIZE_MAX; }
        catch (const std::bad_alloc&) { fail_allocation_after = SIZE_MAX; ++allocation_failures; }
        catch (...) { fail_allocation_after = SIZE_MAX; throw; }
    }
    require(resources() == before && allocation_failures > 0, "allocation failure leaked resources");
    auto expected = contract(); expected.sha256[0] = expected.sha256[0] == '0' ? '1' : '0';
    reject([&] { (void)p::load_verified_vrm_component(path, expected); }, "SHA-256 mismatch");
    expected = contract(); expected.architecture = "wrong";
    reject([&] { (void)p::load_verified_vrm_component(path, expected); }, "architecture mismatch");
    expected = contract(); expected.sha256 = "invalid";
    reject([&] { (void)p::load_verified_vrm_component(path, expected); }, "invalid expected SHA");
    reject([&] { (void)p::load_verified_vrm_component(root / "missing", contract()); }, "cannot open component");
    reject([&] { (void)p::sha256_file_descriptor(-1, 1); }, "invalid artifact descriptor");
    require(resources() == before, "validation rejection leaked resources");
    const auto after = resources();
    std::cout << "64 repeated verifications + 64 staged failures + " << allocation_failures << " allocation failures: PASS\n"
              << "handles=" << before.first << "->" << after.first
              << " mapped_bytes=" << before.second << "->" << after.second << '\n';
}
}

int main() {
    try {
        Directory root;
        unchanged_and_unicode(root.path);
        replacement_tests(root.path);
        mutation_tests(root.path);
        writable_mapping_tests(root.path);
        large_file_test(root.path);
        failure_and_resource_tests(root.path);
        std::cout << "stable file verification: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        testing::hook = {}; fail_allocation_after = SIZE_MAX;
        std::cerr << "stable file verification: FAIL: " << error.what() << '\n';
        return 1;
    }
}
