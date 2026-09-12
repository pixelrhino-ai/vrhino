#include "vrhino/product/component_package.h"
#include "vrhino/product/registry.h"
#include "windows_component_archive.h"
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;
using namespace vrhino::product;
namespace wc = windows_cache;
namespace wa = windows_component_archive;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
fs::path utf8_path(const std::string& value) {
    const auto length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    require(length > 0, "test UTF-8 decode");
    std::wstring wide(static_cast<size_t>(length), 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), wide.data(), length);
    return fs::path(wide);
}
void clean(const fs::path& root) { if (fs::exists(root)) wc::remove_tree(root); }
void empty_staging(const LocalComponentCache& cache) {
    if (!fs::exists(cache.layout().temporary)) return;
    for (const auto& entry : fs::directory_iterator(cache.layout().temporary))
        require(!entry.path().filename().native().starts_with(L"component-extraction-"), "orphan extraction staging");
}
template<class Function> void rejected(Function operation) {
    bool failed = false;
    try { operation(); } catch (const std::exception&) { failed = true; }
    require(failed, "unsafe operation unexpectedly succeeded");
}
DWORD handles() { DWORD count = 0; require(GetProcessHandleCount(GetCurrentProcess(), &count) != 0, "handle count"); return count; }
std::wstring quote(const fs::path& path) { return L"\"" + path.native() + L"\""; }
struct Child {
    wc::Handle process;
    ~Child() { if (process && WaitForSingleObject(process.get(), 0) == WAIT_TIMEOUT) { TerminateProcess(process.get(), 90); WaitForSingleObject(process.get(), 5000); } }
    void wait() { require(WaitForSingleObject(process.get(), 30000) == WAIT_OBJECT_0, "child timeout"); DWORD code = 1; GetExitCodeProcess(process.get(), &code); require(code == 0, "child failure"); }
};
void launch(Child& child, const fs::path& exe, const fs::path& archive, const fs::path& cache,
            const std::wstring& name, bool after) {
    std::wstring command = quote(exe) + L" --child " + quote(archive) + L" " + quote(cache) + L" " + name + (after ? L" after" : L" before");
    STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION info{};
    require(CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                           nullptr, nullptr, &startup, &info) != 0, "native child launch");
    CloseHandle(info.hThread); child.process = wc::Handle(info.hProcess);
}
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc == 6 && std::wstring(argv[1]) == L"--child") {
            const std::wstring name(argv[4]);
            wc::Handle ready(OpenEventW(EVENT_MODIFY_STATE, FALSE, (name + L"-ready").c_str()));
            wc::Handle proceed(OpenEventW(SYNCHRONIZE, FALSE, (name + L"-go").c_str()));
            require(ready && proceed, "child synchronization");
            wa::test_hook = [&](wa::Point point) {
                if (point != wa::Point::BeforeCommit) return;
                SetEvent(ready.get());
                require(WaitForSingleObject(proceed.get(), 30000) == WAIT_OBJECT_0, "publication synchronization timeout");
            };
            LocalComponentCache(argv[3]).install_archive(argv[2]);
            return 0;
        }
        require(argc == 3, "run with fixture directory and loopback URL (use fixture Python runner)");
        const fs::path fixtures = wc::native_path(argv[1]);
        const fs::path work = fixtures / "cache tests";
        clean(work); wc::ensure_directory(work);
        std::ifstream cases(fixtures / "cases.txt");
        std::string line;
        size_t passed = 0;
        while (std::getline(cases, line)) {
            const bool valid = line[0] == '1';
            const std::string name = line.substr(2);
            const fs::path root = work / std::to_string(passed);
            LocalComponentCache cache(root);
            const fs::path archive = fixtures / (utf8_path(name).native() + L".tar.gz");
            if (valid) {
                const auto result = cache.install_archive(archive);
                require(!result.already_installed, "first install reuse");
                require(cache.resolve(kMediaComponentReference).manifest.artifacts.size() == 1, "manifest resolution");
                require(cache.install_archive(archive).already_installed, "repeat install reuse");
                if (name == "hardlink") {
                    require(wc::equivalent(result.root / "alias", result.root / "bin/vrhino-ffmpeg.exe"), "archive hardlink identity");
                    require(wc::link_count(result.root / "alias") == 2, "archive hardlink count");
                    wc::remove(result.root / "alias");
                    require(GetFileAttributesW(wc::native_path(result.root / "bin/vrhino-ffmpeg.exe").c_str()) & FILE_ATTRIBUTE_READONLY,
                            "cleanup changed surviving alias attributes");
                }
                if (name == "Unicode 档案") require(fs::exists(wc::native_path(result.root / L"目录/媒体 文件.txt")), "Unicode member");
                if (name == "sparse") {
                    const auto bytes = wa::read_text(result.root / "sparse.bin");
                    require(bytes.size() == 1048576 && bytes.substr(0, 3) == "abc" && bytes.substr(bytes.size() - 3) == "xyz",
                            "sparse file extent reconstruction");
                    require(bytes.find_first_not_of('\0', 3) == bytes.size() - 3, "sparse hole bytes");
                }
            } else rejected([&] { cache.install_archive(archive); });
            empty_staging(cache);
            if (!valid) require(!fs::exists(cache.components_root() / "vrhino/media-windows-x86_64/1.0.0"), "partial final component");
            clean(root);
            ++passed;
            std::cout << "fixture " << name << " PASS\n";
        }
        require(passed >= 45, "fixture coverage missing");
        const fs::path good = fixtures / "valid.tar.gz";
        // Test the real checked arithmetic above 4 GiB without allocating a large payload.
        require(wa::checked_extent(INT64_C(0x100000000), 17, INT64_MAX) == UINT64_C(0x100000011), "64-bit extent truncated");
        rejected([] { wa::checked_extent(INT64_MAX, 1, INT64_MAX); });
        rejected([] { wa::checked_extent(-1, 1, INT64_MAX); });
        rejected([] { wa::checked_extent(7, UINT64_MAX, INT64_MAX); });
        std::cout << "64-bit accounting PASS\n";

        for (const auto point : {wc::Point::Write, wc::Point::Flush, wc::Point::Validate,
                                 wc::Point::BeforePublish, wc::Point::Space, wc::Point::Hardlink}) {
            const fs::path root = work / "failure";
            LocalComponentCache cache(root);
            bool injected = false;
            wc::test_hook = [&](wc::Point seen) {
                if (!injected && seen == point) { injected = true; throw std::runtime_error("injected filesystem failure"); }
            };
            rejected([&] { cache.install_archive(point == wc::Point::Hardlink ? fixtures / "hardlink.tar.gz" : good); });
            wc::test_hook = {};
            require(injected, "fault not reached");
            empty_staging(cache); clean(root);
        }
        std::cout << "write/flush/validation/rename/space/hardlink failure cleanup PASS\n";
        {
            LocalComponentCache cache(work / "cancel");
            bool cancelled = false;
            wa::test_hook = [&](wa::Point point) { if (point == wa::Point::Extracted) cancelled = true; };
            rejected([&] { cache.install_archive(good, [&] { return cancelled; }); });
            wa::test_hook = {}; empty_staging(cache); clean(cache.layout().root);
        }
        {
            LocalComponentCache cache(work / "cleanup");
            bool armed = false, injected = false;
            wa::test_hook = [&](wa::Point point) { if (point == wa::Point::BeforeCommit) armed = true; };
            wc::test_hook = [&](wc::Point point) { if (armed && !injected && point == wc::Point::Cleanup) { injected = true; throw std::runtime_error("cleanup fault"); } };
            rejected([&] { cache.install_archive(good); });
            wa::test_hook = {}; wc::test_hook = {};
            require(injected, "cleanup fault not reached");
            (void)cache.resolve(kMediaComponentReference); empty_staging(cache); clean(cache.layout().root);
        }
        std::cout << "cancellation and post-publication cleanup failure PASS\n";
        {
            const fs::path copy = fixtures / "readonly.tar.gz";
            clean(copy);
            fs::copy_file(good, copy, fs::copy_options::overwrite_existing);
            require(SetFileAttributesW(copy.c_str(), FILE_ATTRIBUTE_READONLY) != 0, "readonly setup");
            wc::LocalSource original(copy);
            const auto before = original.before();
            LocalComponentCache cache(work / "readonly");
            cache.install_archive(copy);
            require(wc::unchanged(before, wc::snapshot(original.get())), "archive source metadata changed");
            original.verify(); original.finish();
            clean(cache.layout().root);
        }
        std::cout << "retained read-only archive source identity PASS\n";

        for (const bool substitute : {false, true}) {
            const fs::path input = fixtures / "mutable.tar.gz";
            const fs::path displaced = fixtures / "displaced.tar.gz";
            clean(input); clean(displaced);
            fs::copy_file(good, input);
            LocalComponentCache cache(work / "mutation");
            bool changed = false;
            wa::test_hook = [&](wa::Point point) {
                if (changed || point != wa::Point::Extracted) return;
                changed = true;
                if (substitute) {
                    require(MoveFileExW(input.c_str(), displaced.c_str(), 0) != 0, "archive source name substitution");
                    fs::copy_file(good, input);
                } else {
                    wc::Handle writer(CreateFileW(input.c_str(), GENERIC_READ | GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr));
                    require(static_cast<bool>(writer), "archive mutation writer");
                    char original = 0; DWORD count = 0; LARGE_INTEGER zero{};
                    require(ReadFile(writer.get(), &original, 1, &count, nullptr) != 0, "mutation original byte");
                    FILETIME before_write{}; GetFileTime(writer.get(), nullptr, nullptr, &before_write);
                    SetFilePointerEx(writer.get(), zero, nullptr, FILE_BEGIN);
                    require(WriteFile(writer.get(), "X", 1, &count, nullptr) != 0, "mutation write");
                    SetFilePointerEx(writer.get(), zero, nullptr, FILE_BEGIN);
                    require(WriteFile(writer.get(), &original, 1, &count, nullptr) != 0, "mutation restore");
                    SetFileTime(writer.get(), nullptr, nullptr, &before_write);
                }
            };
            rejected([&] { cache.install_archive(input); });
            wa::test_hook = {};
            require(changed, "archive mutation checkpoint");
            empty_staging(cache); clean(cache.layout().root); clean(input); clean(displaced);
        }
        std::cout << "archive mutation and same-bytes source name substitution rejection PASS\n";
        {
            LocalComponentCache cache(work / "publication-identity");
            wc::Snapshot before{};
            bool recorded = false;
            wa::test_hook = [&](wa::Point point) {
                if (point != wa::Point::BeforeCommit) return;
                for (const auto& entry : fs::directory_iterator(cache.layout().temporary)) {
                    if (!entry.path().filename().native().starts_with(L"component-extraction-")) continue;
                    wc::LocalSource file(entry.path() / "vrhino-media/bin/vrhino-ffmpeg.exe");
                    before = file.before(); recorded = true;
                }
            };
            const auto installed = cache.install_archive(good);
            wa::test_hook = {};
            {
                wc::LocalSource after(installed.root / "bin/vrhino-ffmpeg.exe");
                require(recorded && wc::same_identity(before, after.before()), "directory publication changed child identity");
            }
            empty_staging(cache); clean(cache.layout().root);
        }
        std::cout << "same file identity across enclosing directory publication PASS\n";

        // Native child pauses immediately before the enclosing directory rename.
        wchar_t executable[32768]{}; GetModuleFileNameW(nullptr, executable, 32768);
        for (const bool kill : {false, true}) {
            const std::wstring name = L"Local\\vrhino-archive-" + std::to_wstring(GetCurrentProcessId()) + (kill ? L"-kill" : L"-atomic");
            wc::Handle ready(CreateEventW(nullptr, TRUE, FALSE, (name + L"-ready").c_str()));
            wc::Handle proceed(CreateEventW(nullptr, TRUE, FALSE, (name + L"-go").c_str()));
            const fs::path root = work / (kill ? "interrupted" : "atomic");
            LocalComponentCache cache(root);
            Child child;
            launch(child, executable, good, root, name, false);
            require(WaitForSingleObject(ready.get(), 30000) == WAIT_OBJECT_0, "child preparation timeout");
            const auto final = wc::native_path(cache.components_root() / "vrhino/media-windows-x86_64/1.0.0");
            require(!fs::exists(final), "partial directory exposed before publication");
            if (kill) {
                require(TerminateProcess(child.process.get(), 0) != 0, "terminate extraction child"); child.wait();
                cache.install_archive(good);
            } else {
                std::atomic<bool> done{false}, bad{false};
                std::thread reader([&] {
                    while (!done) {
                        if (fs::exists(final) && (!fs::exists(final / "vrhino-component.json") ||
                            !fs::exists(wc::native_path(final / "bin/vrhino-ffmpeg.exe")))) bad = true;
                        std::this_thread::yield();
                    }
                });
                SetEvent(proceed.get());
                try { child.wait(); } catch (...) { done = true; reader.join(); throw; }
                done = true; reader.join(); require(!bad, "reader saw incomplete component");
            }
            (void)cache.resolve(kMediaComponentReference);
            empty_staging(cache); clean(root);
        }
        std::cout << "native child atomic visibility and process-death retry recovery PASS\n";
        {
            std::wstring url(argv[2]);
            RegistryOptions options;
            for (const auto c : url) options.base_url.push_back(static_cast<char>(c));
            options.allow_development_http = true;
            LocalModelCache blobs(work / "registry");
            LocalComponentCache components(work / "registry");
            ComponentRegistryClient registry(blobs, components, options);
            const auto result = registry.pull(kMediaComponentReference);
            require(result.downloaded_bytes == fs::file_size(good), "registry download accounting");
            require(registry.pull(kMediaComponentReference).already_installed, "registry reuse");
            (void)components.resolve(kMediaComponentReference);
            empty_staging(components); clean(work / "registry");
        }
        std::cout << "integrated registry download/admission/extraction/reuse PASS\n";
        // Warm all library paths, then measure real repeated success/failure cycles.
        const DWORD before = handles();
        for (int i = 0; i < 24; ++i) {
            LocalComponentCache cache(work / "repeat");
            cache.install_archive(good);
            rejected([&] { cache.install_archive(fixtures / "bad-hash.tar.gz"); });
            (void)cache.resolve(kMediaComponentReference);
            empty_staging(cache); clean(cache.layout().root);
        }
        const DWORD after = handles();
        require(after == before, "handle growth across extraction cycles");
        std::cout << "24 success/failure cycles handles " << before << " -> " << after << " PASS\n";
        clean(work);
        std::cout << "WINDOWS_COMPONENT_ARCHIVE_TESTS=PASS fixtures=" << passed << '\n';
        return 0;
    } catch (const std::exception& error) {
        wc::test_hook = {}; wa::test_hook = {};
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
