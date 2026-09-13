#include "vrhino/product/windows_cache.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <array>
#include "vrhino/product/source_acquisition.h"
#include "fixtures/cache_package_fixture.h"

namespace wc = vrhino::product::windows_cache;
namespace p = vrhino::product;
namespace fs = std::filesystem;
const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void raw(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary); file << text; file.close(); require(static_cast<bool>(file), "fixture write");
}
template<class F> void rejected(F operation) {
    bool failed = false;
    try { operation(); } catch (const p::ModelPackageError&) { failed = true; }
    require(failed, "expected fail-closed error");
}
void stage_tests(const fs::path& root) {
    wc::ensure_directory(root);
    const auto destination = root / L"完整 数据.bin";
    {
        wc::StagedFile file(root / "first"); file.write("abc", 3); file.readonly(); file.flush(); file.validate(3, abc); file.publish(destination);
    }
    require(p::sha256_file(destination) == abc, "published bytes");
    rejected([&] {
        wc::StagedFile file(root / "readonly-replacement");
        file.write("abc", 3); file.flush(); file.validate(3, abc); file.publish(destination, true);
    });
    require(!fs::exists(root / "readonly-replacement") && p::sha256_file(destination) == abc,
            "read-only replacement failure must preserve old object and clean staging");
    for (const auto point : {wc::Point::Write, wc::Point::Flush, wc::Point::Validate, wc::Point::BeforePublish}) {
        rejected([&] {
            wc::StagedFile file(root / "failed");
            wc::test_hook = [=](wc::Point reached) { if (point == reached) throw p::ModelPackageError(p::ModelPackageErrorCode::CacheError, "injected"); };
            file.write("abc", 3); file.flush(); file.validate(3, abc); file.publish(destination, true);
        });
        wc::test_hook = {};
        require(!fs::exists(root / "failed") && p::sha256_file(destination) == abc, "preparation failure cleanup/old preservation");
    }
    wc::required_link(destination, root / "alias");
    require(wc::equivalent(destination, root / "alias") && wc::link_count(destination) == 2, "required hardlink identity");
    wc::remove(root / "alias");
    require(wc::link_count(destination) == 1 && (GetFileAttributesW(destination.c_str()) & FILE_ATTRIBUTE_READONLY), "readonly alias cleanup");
    wc::test_hook = [](wc::Point point) { if (point == wc::Point::Hardlink) throw p::ModelPackageError(p::ModelPackageErrorCode::CacheError, "injected hardlink failure"); };
    rejected([&] { wc::required_link(destination, root / "failed-link"); }); wc::test_hook = {};
    require(!fs::exists(root / "failed-link"), "hardlink failure must not copy");
    rejected([&] { wc::required_link(destination, destination); });
    require(p::sha256_file(destination) == abc && wc::link_count(destination) == 1, "hardlink collision cleanup");
    rejected([&] {
        wc::StagedFile file(root / "cleanup-failure");
        wc::test_hook = [](wc::Point point) { if (point == wc::Point::Cleanup) throw p::ModelPackageError(p::ModelPackageErrorCode::CacheError, "injected cleanup failure"); };
        file.discard();
    }); wc::test_hook = {};
    require(!fs::exists(root / "cleanup-failure"), "cleanup unwinding must release owned staging");
    rejected([&] {
        wc::StagedFile file(root / "after-commit"); file.write("abc", 3); file.flush(); file.validate(3, abc);
        wc::test_hook = [](wc::Point point) { if (point == wc::Point::AfterPublish) throw p::ModelPackageError(p::ModelPackageErrorCode::CacheError, "lost acknowledgment"); };
        file.publish(root / "committed");
    }); wc::test_hook = {};
    require(p::sha256_file(root / "committed") == abc, "lost acknowledgment must not delete a committed object");
    std::cout << "staging_publication_faults_hardlink=PASS\n";
}
void local_admission(const fs::path& root) {
    p::ArtifactDeclaration artifact{"fixture", "data", "data", 3, abc, true};
    const auto source = root / "readonly-input";
    raw(source, "abc");
    require(SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_READONLY) != 0, "source readonly");
    require(CreateHardLinkW((root / "source-alias").c_str(), source.c_str(), nullptr) != 0, "source alias");
    p::LocalModelCache cache(root / "cache");
    const auto admitted = cache.admit_local_blob(source, artifact);
    require(admitted.created && !wc::equivalent(source, admitted.path), "read-only exception copy identity");
    require(wc::equivalent(source, root / "source-alias") && wc::link_count(source) == 2, "source aliases unchanged");
    require((GetFileAttributesW(source.c_str()) & FILE_ATTRIBUTE_READONLY) != 0, "source attributes unchanged");
    require(!cache.admit_local_blob(source, artifact).created, "local CAS reuse");
    const auto writable = root / "writable-input"; raw(writable, "abc");
    p::LocalModelCache writable_cache(root / "writable-cache");
    const auto linked = writable_cache.admit_local_blob(writable, artifact);
    require(linked.created && wc::equivalent(writable, linked.path), "ordinary writable admission must hardlink");
    std::cout << "local_admission_readonly_copy_writable_link_reuse=PASS\n";
}
void package_and_source(const fs::path& root) {
    const auto package = cache_fixture::make_package(root, "v1");
    p::LocalModelCache cache(root / "packages");
    const auto installed = cache.install(package);
    require(installed.blobs_created == 3 && cache.resolve("vrhino/fixture:v1", true).artifacts.size() == 3, "package publication");
    rejected([&] { cache.install(package); });
    cache.remove("vrhino/fixture:v1");
    require(!fs::exists(installed.manifest_path), "durable package retirement");

    p::LocalSourceCache sources(root / "sources");
    p::SourceArtifactPlanDocument plan;
    plan.schema_version = 1; plan.model_reference = "vrhino/fixture:v1";
    plan.requested_source = {"huggingface", "example/fixture", std::string(40, 'a')};
    p::SourceArtifactPlan item;
    item.id = "input"; item.role = "weights"; item.repository = "example/fixture";
    item.revision = std::string(40, 'a'); item.upstream_path = "input.bin";
    item.local_path = "weights/input.bin"; item.size = 3; item.sha256 = abc;
    plan.artifacts.push_back(item); plan.raw_json = "{\"schema_version\":1}";
    const auto blob = sources.layout().blobs / abc.substr(0, 2) / abc;
    wc::write_text(blob, "abc");
    const auto acquired = sources.acquire(plan.requested_source, plan);
    require(acquired.reused_bytes == 3 && wc::equivalent(blob, acquired.materialized_directory / item.local_path), "source CAS materialization identity");
    p::LocalModelCache shared(root / "shared");
    p::ArtifactDeclaration artifact{"input", "weights", "input.bin", 3, abc, true};
    const auto admitted = shared.admit_local_blob(blob, artifact);
    require(wc::equivalent(blob, admitted.path), "source/model shared CAS identity");
    const auto cleaned = sources.reclaim_after_install(acquired);
    require(cleaned.reclaimed_bytes == plan.raw_json.size() && fs::exists(blob) && p::sha256_file(admitted.path) == abc, "source cleanup reuse accounting");
    const auto reacquired = sources.acquire(plan.requested_source, plan);
    require(reacquired.reused_bytes == 3 && wc::equivalent(blob, reacquired.materialized_directory / item.local_path), "read-only source CAS reuse");
    std::cout << "package_source_materialization_reuse_cleanup=PASS\n";
}
void source_mutation_and_space(const fs::path& root) {
    const auto source = root / "input"; raw(source, "abc");
    SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_READONLY);
    p::ArtifactDeclaration artifact{"fixture", "data", "input", 3, abc, true};
    p::LocalModelCache cache(root / "cache");
    bool mutated = false;
    wc::test_hook = [&](wc::Point point) {
        if (point != wc::Point::SourceCopy || mutated) return;
        mutated = true;
        SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_NORMAL);
        wc::Handle writer(CreateFileW(source.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, 0, nullptr));
        require(static_cast<bool>(writer), "mutation writer");
        FILETIME before{}; GetFileTime(writer.get(), nullptr, nullptr, &before);
        DWORD count = 0; WriteFile(writer.get(), "xyz", 3, &count, nullptr);
        LARGE_INTEGER zero{}; SetFilePointerEx(writer.get(), zero, nullptr, FILE_BEGIN);
        WriteFile(writer.get(), "abc", 3, &count, nullptr); SetFileTime(writer.get(), nullptr, nullptr, &before);
        SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_READONLY);
    };
    rejected([&] { cache.admit_local_blob(source, artifact); });
    wc::test_hook = {};
    require(mutated && !cache.contains_blob(artifact), "restored-byte source mutation must fail closed");
    bool substituted = false;
    wc::test_hook = [&](wc::Point point) {
        if (point != wc::Point::SourceCopy || substituted) return;
        substituted = true;
        // Fixture adversary: replace the pathname with identical bytes while
        // acquisition still owns the original source object.
        require(MoveFileExW(source.c_str(), (root / "displaced-input").c_str(), 0) != 0,
                "source-name substitution fixture");
        raw(source, "abc");
        require(SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_READONLY) != 0,
                "substituted source readonly");
    };
    rejected([&] { cache.admit_local_blob(source, artifact); }); wc::test_hook = {};
    require(substituted && !cache.contains_blob(artifact) &&
            p::sha256_file(root / "displaced-input") == abc,
            "source-name substitution must fail closed and preserve the opened object");
    bool copied = false;
    wc::test_hook = [&](wc::Point point) {
        if (point == wc::Point::SourceCopy) copied = true;
        if (point == wc::Point::Space) throw p::ModelPackageError(p::ModelPackageErrorCode::InsufficientDiskSpace, "injected space shortage");
    };
    rejected([&] { cache.admit_local_blob(source, artifact); }); wc::test_hook = {};
    require(!copied && !cache.contains_blob(artifact), "additional copy space must be checked first");
    SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_NORMAL);
    {
        wc::Handle writer(CreateFileW(source.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr));
        require(static_cast<bool>(writer), "mapping writer");
        wc::Handle mapping(CreateFileMappingW(writer.get(), nullptr, PAGE_READWRITE, 0, 0, nullptr));
        require(static_cast<bool>(mapping), "writable mapping fixture");
        void* view = MapViewOfFile(mapping.get(), FILE_MAP_WRITE, 0, 0, 0);
        require(view != nullptr, "writable view");
        SetFileAttributesW(source.c_str(), FILE_ATTRIBUTE_READONLY);
        bool failed = false;
        try { cache.admit_local_blob(source, artifact); } catch (const p::ModelPackageError&) { failed = true; }
        UnmapViewOfFile(view);
        require(failed && !cache.contains_blob(artifact), "pre-existing writable mappings must fail closed");
    }
    std::cout << "source_mutation_name_substitution_and_extra_space=PASS\n";
}

struct Event {
    std::wstring name;
    wc::Handle handle;
    explicit Event(const std::wstring& suffix) : name(L"Local\\VRhino-cache-" + std::to_wstring(GetCurrentProcessId()) + suffix),
        handle(CreateEventW(nullptr, TRUE, FALSE, name.c_str())) { require(static_cast<bool>(handle), "event create"); }
    void signal() { require(SetEvent(handle.get()) != 0, "event signal"); }
    bool wait(DWORD milliseconds = 10000) { return WaitForSingleObject(handle.get(), milliseconds) == WAIT_OBJECT_0; }
};
std::wstring executable() { std::vector<wchar_t> path(32768); auto count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size())); require(count > 0 && count < path.size(), "executable path"); return {path.data(), count}; }
struct Process {
    wc::Handle process, thread;
    explicit Process(const std::vector<std::wstring>& arguments) {
        const auto app = executable(); std::wstring command = L"\"" + app + L"\"";
        for (const auto& argument : arguments) command += L" \"" + argument + L"\"";
        STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION info{};
        require(CreateProcessW(app.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
            nullptr, nullptr, &startup, &info) != 0, "child launch");
        process = wc::Handle(info.hProcess); thread = wc::Handle(info.hThread);
    }
    ~Process() { if (WaitForSingleObject(process.get(), 0) == WAIT_TIMEOUT) { TerminateProcess(process.get(), 91); WaitForSingleObject(process.get(), 10000); } }
    bool done() { return WaitForSingleObject(process.get(), 0) == WAIT_OBJECT_0; }
    void join() { require(WaitForSingleObject(process.get(), 10000) == WAIT_OBJECT_0, "child timeout"); DWORD result = 1; GetExitCodeProcess(process.get(), &result); require(result == 0, "child failed"); }
    void kill() { require(TerminateProcess(process.get(), 90) != 0, "child termination"); require(WaitForSingleObject(process.get(), 10000) == WAIT_OBJECT_0, "child termination wait"); }
};
wc::Handle open_event(const wchar_t* name) { wc::Handle event(OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, name)); require(static_cast<bool>(event), "child event open"); return event; }
int child_lock(wchar_t** argv) {
    auto ready = open_event(argv[4]), acquired = open_event(argv[5]), release = open_event(argv[6]);
    wc::test_hook = [&](wc::Point point) { if (point == wc::Point::Lock) SetEvent(ready.get()); };
    wc::Lock lock(argv[2], std::wstring(argv[3]) == L"shared" ? wc::LockMode::Shared : wc::LockMode::Exclusive);
    SetEvent(acquired.get()); require(WaitForSingleObject(release.get(), 10000) == WAIT_OBJECT_0, "child release timeout"); return 0;
}
void lock_processes(const fs::path& root) {
    wc::ensure_directory(root);
    wc::test_hook = [](wc::Point point) { if (point == wc::Point::Lock) throw p::ModelPackageError(p::ModelPackageErrorCode::CacheError, "injected lock failure"); };
    rejected([&] { wc::Lock failure(root / "failure.lock"); }); wc::test_hook = {};
    { wc::Lock recovered(root / "failure.lock"); }
    unsigned index = 0;
    for (const auto modes : {std::pair{wc::LockMode::Exclusive, wc::LockMode::Exclusive},
                             std::pair{wc::LockMode::Shared, wc::LockMode::Shared},
                             std::pair{wc::LockMode::Shared, wc::LockMode::Exclusive},
                             std::pair{wc::LockMode::Exclusive, wc::LockMode::Shared}}) {
        const auto suffix = std::to_wstring(index++);
        Event ready(L"ready" + suffix), acquired(L"acquired" + suffix), release(L"release" + suffix);
        auto held = std::make_unique<wc::Lock>(root / "persistent.lock", modes.first);
        Process child({L"--lock", (root / "persistent.lock").native(), modes.second == wc::LockMode::Shared ? L"shared" : L"exclusive", ready.name, acquired.name, release.name});
        require(ready.wait(), "lock attempt not reached");
        const bool compatible = modes.first == wc::LockMode::Shared && modes.second == wc::LockMode::Shared;
        require(acquired.wait(compatible ? 10000 : 200) == compatible, "cross-process lock compatibility");
        held.reset(); require(acquired.wait(), "normal close must release OS lock"); release.signal(); child.join();
    }
    Event ar(L"death-ready-a"), aa(L"death-held-a"), ax(L"death-release-a");
    Event br(L"death-ready-b"), ba(L"death-held-b"), bx(L"death-release-b");
    Process first({L"--lock", (root / "persistent.lock").native(), L"exclusive", ar.name, aa.name, ax.name});
    require(aa.wait(), "first child did not lock");
    Process second({L"--lock", (root / "persistent.lock").native(), L"exclusive", br.name, ba.name, bx.name});
    require(br.wait() && !ba.wait(200), "second child must block");
    first.kill(); require(ba.wait(), "process death must release OS lock"); bx.signal(); second.join();
    std::cout << "cross_process_ex_ex_sh_sh_sh_ex_normal_death=PASS\n";
}

void admission_and_resources(const fs::path& root) {
    wc::test_volume_admission(L"NTFS", DRIVE_FIXED, false);
    rejected([&] { wc::test_volume_admission(L"exFAT", DRIVE_FIXED, false); });
    rejected([&] { wc::test_volume_admission(L"NTFS", DRIVE_REMOTE, false); });
    rejected([&] { wc::test_volume_admission(L"NTFS", DRIVE_FIXED, true); });
    wc::ensure_directory(root);
    auto long_directory = root / L"长路径 with spaces";
    while (long_directory.native().size() < 300) long_directory /= L"nested-directory";
    wc::write_text(long_directory / L"完整.txt", "abc", true);
    require(p::sha256_file(wc::native_path(long_directory / L"完整.txt")) == abc, "extended wide-path publication");
    std::cout << "unicode_spaces_long_path=PASS characters=" << long_directory.native().size() << '\n';
    const wchar_t* cross_root = _wgetenv(L"VRHINO_TEST_CROSS_VOLUME_ROOT");
    if (cross_root) {
        const auto other = fs::path(cross_root) / (L"vrhino-cross-" + std::to_wstring(GetCurrentProcessId()));
        wc::ensure_directory(other);
        wc::write_text(root / "link-source", "abc", true);
        rejected([&] { wc::required_link(root / "link-source", other / "link"); });
        {
            wc::StagedFile file(root / "cross-stage"); file.write("abc", 3); file.flush(); file.validate(3, abc);
            rejected([&] { file.publish(other / "published"); });
        }
        require(fs::is_empty(other), "cross-volume failure left a destination"); wc::remove_tree(other);
        std::cout << "real_cross_volume_link_publication=PASS\n";
    } else std::cout << "real_cross_volume_link_publication=NOT_RUN (set VRHINO_TEST_CROSS_VOLUME_ROOT)\n";
    DWORD before = 0, after = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &before) != 0, "initial handle count");
    for (unsigned n = 0; n < 32; ++n) {
        const auto path = root / ("cycle-" + std::to_string(n));
        { wc::Lock lock(root / "repeat.lock"); wc::write_text(path, "abc", true); }
        wc::remove(path);
        rejected([&] { wc::StagedFile file(path); file.write("bad", 3); file.flush(); file.validate(3, abc); });
    }
    require(GetProcessHandleCount(GetCurrentProcess(), &after) != 0 && after <= before, "HANDLE growth");
    std::cout << "handle_count_before=" << before << " after=" << after << "\nadmission_seam_and_repeated_lifetimes=PASS\n";
}
std::string complete_read(const fs::path& path) {
    wc::Handle file(CreateFileW(wc::native_path(path).c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr));
    require(static_cast<bool>(file), "reader observed missing final object");
    LARGE_INTEGER size{}; require(GetFileSizeEx(file.get(), &size) != 0 && size.QuadPart == 65536, "partial publication size");
    std::string bytes(65536, '\0'); DWORD count = 0;
    require(ReadFile(file.get(), bytes.data(), static_cast<DWORD>(bytes.size()), &count, nullptr) != 0 && count == bytes.size(), "partial read");
    require(bytes == std::string(65536, 'A') || bytes == std::string(65536, 'B'), "mixed/corrupt publication");
    return bytes;
}
int child_publish(wchar_t** argv) {
    const fs::path root = argv[2]; const auto value = static_cast<char>(argv[3][0]);
    const std::wstring wide_hash(argv[4]); std::string digest;
    for (const wchar_t value_byte : wide_hash) digest.push_back(static_cast<char>(value_byte));
    auto start = open_event(argv[5]); require(WaitForSingleObject(start.get(), 10000) == WAIT_OBJECT_0, "publisher start timeout");
    for (unsigned index = 0; index < 32; ++index) {
        wc::StagedFile file(wc::unique_path(root, "publisher"));
        const std::string payload(65536, value);
        file.write(payload.data(), payload.size()); file.flush(); file.validate(payload.size(), digest);
        file.publish(root / "final", true);
    }
    return 0;
}
void competing_publishers(const fs::path& root) {
    wc::ensure_directory(root);
    raw(root / "expected-a", std::string(65536, 'A')); raw(root / "expected-b", std::string(65536, 'B'));
    const auto a = p::sha256_file(root / "expected-a"), b = p::sha256_file(root / "expected-b");
    wc::write_text(root / "final", std::string(65536, 'A'));
    wc::Handle old(CreateFileW((root / "final").c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr));
    require(static_cast<bool>(old), "old reader"); const auto old_id = wc::snapshot(old.get());
    Event start(L"publish-start");
    Process first({L"--publish", root.native(), L"A", std::wstring(a.begin(), a.end()), start.name});
    Process second({L"--publish", root.native(), L"B", std::wstring(b.begin(), b.end()), start.name});
    start.signal(); unsigned reads = 0; const auto deadline = GetTickCount64() + 20000;
    do { complete_read(root / "final"); ++reads; require(GetTickCount64() < deadline, "publisher race deadline"); }
    while (!first.done() || !second.done());
    first.join(); second.join(); complete_read(root / "final");
    std::string original(65536, '\0'); DWORD count = 0;
    require(ReadFile(old.get(), original.data(), static_cast<DWORD>(original.size()), &count, nullptr) != 0 && count == original.size() &&
        original == std::string(65536, 'A') && wc::same_identity(old_id, wc::snapshot(old.get())), "old handle must retain old object");
    std::cout << "concurrent_publishers=2 replacements=64 complete_reader_opens=" << reads << " old_handle=PASS\n";
}
int child_interrupt(wchar_t** argv) {
    const fs::path root = argv[2]; const int phase = std::stoi(argv[3]);
    auto ready = open_event(argv[4]);
    wc::StagedFile stage(root / "partial", false, true);
    stage.write(phase == 0 ? "a" : "abc", phase == 0 ? 1 : 3);
    if (phase >= 1) stage.flush();
    if (phase >= 2) stage.validate(3, abc);
    if (phase >= 3) stage.publish(root / "final", true);
    SetEvent(ready.get()); Sleep(30000); return 0;
}
void interrupted_publication(const fs::path& root) {
    for (unsigned existing = 0; existing < 2; ++existing) for (int phase = 0; phase < 4; ++phase) {
        const auto directory = root / (std::to_string(existing) + "-" + std::to_string(phase));
        wc::ensure_directory(directory);
        if (existing) wc::write_text(directory / "final", "old");
        Event ready(L"interrupt-" + std::to_wstring(existing) + L"-" + std::to_wstring(phase));
        Process child({L"--interrupt", directory.native(), std::to_wstring(phase), ready.name});
        require(ready.wait(), "interruption checkpoint"); child.kill();
        if (phase == 3) require(p::sha256_file(directory / "final") == abc, "after-publication termination");
        else {
            require(fs::exists(directory / "final") == static_cast<bool>(existing), "before-publication visibility");
            if (existing) { std::ifstream old(directory / "final"); std::string bytes; old >> bytes; require(bytes == "old", "interruption changed old object"); }
            // Exercise initial acquisition of the surviving resumable object.
            wc::StagedFile resumed(directory / "partial", true);
            if (phase == 0) resumed.write("bc", 2);
            resumed.flush(); resumed.validate(3, abc); resumed.publish(directory / "recovered");
            require(p::sha256_file(directory / "recovered") == abc, "retained partial recovery");
        }
    }
    std::cout << "interruption_before_after_publication_and_partial_recovery=PASS cases=8\n";
}
void network_integration(const fs::path& root) {
    const char* endpoint = std::getenv("VRHINO_TEST_HTTP_URL");
    if (!endpoint) { std::cout << "registry_source_download_integration=NOT_RUN (use windows_cache_http_fixture.py)\n"; return; }
    const std::string base(endpoint);
    p::RegistryOptions options; options.base_url = base; options.allow_development_http = true; options.retry_count = 0;
    p::ArtifactDeclaration artifact{"fixture", "data", "input", 3, abc, true};
    for (const std::string mode : {"fresh", "resumed", "complete", "no-range"}) {
        const auto directory = root / mode; wc::ensure_directory(directory);
        const auto partial = directory / "input.partial";
        if (mode == "resumed" || mode == "no-range") raw(partial, "a");
        if (mode == "complete") raw(partial, "abc");
        auto transfer = p::download_native_artifact(base + (mode == "no-range" ? "/no-range" : "/payload"), partial, 3, "fixture", 0, 3, options);
        require(transfer.staging != nullptr && transfer.resumed_bytes == (mode == "resumed" ? 1U : mode == "complete" ? 3U : 0U), "resume accounting");
        p::LocalModelCache cache(directory / "cache");
        const auto result = cache.admit_downloaded_blob(*transfer.staging, artifact);
        require(result.created && p::sha256_file(result.path) == abc && !fs::exists(partial), "download retained-handle admission");
    }
    const auto cancelled = root / "cancelled.partial"; raw(cancelled, "a");
    options.cancellation_requested = [] { return true; };
    rejected([&] { p::download_native_artifact(base + "/payload", cancelled, 3, "fixture", 0, 3, options); });
    require(fs::file_size(cancelled) == 1, "cancelled partial preserved"); options.cancellation_requested = {};
    const auto wrong_range = root / "wrong-range.partial"; raw(wrong_range, "a");
    rejected([&] { p::download_native_artifact(base + "/wrong-range", wrong_range, 3, "fixture", 0, 3, options); });
    const auto corrupt = root / "corrupt.partial";
    auto transfer = p::download_native_artifact(base + "/corrupt", corrupt, 3, "fixture", 0, 3, options);
    p::LocalModelCache bad_cache(root / "bad-cache");
    rejected([&] { bad_cache.admit_downloaded_blob(*transfer.staging, artifact); });
    require(!fs::exists(corrupt) && !bad_cache.contains_blob(artifact), "corrupt transfer cleanup");

    p::LocalModelCache registry_cache(root / L"注册表 cache"); p::RegistryClient registry(registry_cache, options);
    const auto pulled = registry.pull("vrhino/fixture:v1");
    require(pulled.artifacts_downloaded == 3 && registry_cache.resolve("vrhino/fixture:v1", true).artifacts.size() == 3, "registry package fixture");
    require(registry.pull("vrhino/fixture:v1").already_installed, "registry reuse");
    p::LocalSourceCache sources(root / "source-download");
    p::SourceArtifactPlanDocument plan; plan.schema_version = 1; plan.model_reference = "vrhino/fixture:v1";
    plan.requested_source = {"huggingface", "example/fixture", std::string(40, 'a')};
    p::SourceArtifactPlan item; item.id = "input"; item.role = "weights"; item.repository = "example/fixture";
    item.revision = std::string(40, 'a'); item.upstream_path = "input.bin"; item.local_path = "input.bin"; item.size = 3; item.sha256 = abc;
    plan.artifacts.push_back(item); plan.raw_json = "{\"schema_version\":1}";
    p::AcquisitionOptions acquisition; acquisition.network = options; acquisition.automatic_huggingface_fallback = false;
    acquisition.huggingface_official.base_url = base;
    const auto acquired = sources.acquire(plan.requested_source, plan, acquisition);
    require(acquired.downloaded_bytes == 3 && wc::equivalent(acquired.artifacts.front().blob_path, acquired.materialized_directory / "input.bin"), "source download publication identity");
    require(sources.acquire(plan.requested_source, plan, acquisition).reused_bytes == 3, "source download reuse");
    std::cout << "registry_source_download_resume_cancellation_integrity=PASS\n";
}
int wmain(int argc, wchar_t** argv) {
    if (argc == 7 && std::wstring(argv[1]) == L"--lock") {
        try { return child_lock(argv); } catch (...) { return 2; }
    }
    if (argc == 6 && std::wstring(argv[1]) == L"--publish") {
        try { return child_publish(argv); } catch (...) { return 2; }
    }
    if (argc == 5 && std::wstring(argv[1]) == L"--interrupt") {
        try { return child_interrupt(argv); } catch (...) { return 2; }
    }
    const auto root = fs::current_path() / ("cache-fixture-" + std::to_string(GetCurrentProcessId()));
    try {
        stage_tests(root / "stage"); local_admission(root / "local");
        source_mutation_and_space(root / "mutation");
        package_and_source(root / "product");
        lock_processes(root / "locks");
        competing_publishers(root / "race");
        interrupted_publication(root / "interruption");
        network_integration(root / "network");
        admission_and_resources(root / "resources");
        wc::remove_tree(root);
        require(!fs::exists(root), "fixture cleanup");
        std::cout << "WINDOWS_CACHE_TESTS=PASS\n"; return 0;
    } catch (const std::exception& error) {
        wc::test_hook = {};
        std::cerr << error.what() << "\nfixture=" << root.string() << '\n'; return 1;
    }
}

