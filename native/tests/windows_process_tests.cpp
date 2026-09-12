#define NOMINMAX
#include <windows.h>
#include <aclapi.h>
#include "vrhino/product/windows_process.h"
#include "vrhino/lip_sync_workflow.h"
#include "run_media.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
namespace wp = vrhino::product::windows_process;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
struct Handle {
    HANDLE value = nullptr;
    explicit Handle(HANDLE h) : value(h) { require(h && h != INVALID_HANDLE_VALUE, "test handle"); }
    ~Handle() { CloseHandle(value); }
    Handle(const Handle&) = delete;
};
std::string text(const wp::Result& result) { return {result.standard_output.begin(), result.standard_output.end()}; }
template<class F> void failure(wp::Failure expected, F operation) {
    try { operation(); } catch (const wp::Error& error) { require(error.kind == expected, error.what()); return; }
    throw std::runtime_error("expected process failure");
}
bool stopped(DWORD pid) {
    const HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!process) return GetLastError() == ERROR_INVALID_PARAMETER;
    const bool result = WaitForSingleObject(process, 10000) == WAIT_OBJECT_0; CloseHandle(process); return result;
}
std::vector<std::wstring> adversarial() { return {L"", L"space value", L"\t", L"a\"b", L"a\\\"b", L"ends\\", L"ends \\", L"中文 媒体 🎬", L"&|<>^%PATH%$(echo)"}; }
void basic(const fs::path& fixture) {
    const DWORD prior_error_mode = GetErrorMode();
    auto args = adversarial(); std::string expected;
    for (const auto& arg : args) { const auto value = wp::utf8(arg); expected += std::to_string(value.size()) + ":" + value + "\n"; }
    args.insert(args.begin(), L"--args");
    const auto result = wp::run(fixture, args);
    require(result.exit_code == 37 && text(result) == expected && result.standard_error == "stderr captured\n", "argument/stdout/stderr/exit contract");
    SetEnvironmentVariableW(L"VRHINO_PROCESS_FIXTURE_ENV", L"继承 value");
    const auto environment = wp::run(fixture, {L"--environment"});
    SetEnvironmentVariableW(L"VRHINO_PROCESS_FIXTURE_ENV", nullptr);
    require(text(environment) == wp::utf8(L"继承 value") && environment.standard_error == wp::utf8(fs::current_path()), "inherited environment/cwd");
    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle sentinel(CreateEventW(&inherit, TRUE, FALSE, nullptr));
    require(wp::run(fixture, {L"--handle", std::to_wstring(reinterpret_cast<uintptr_t>(sentinel.value))}).exit_code == 0, "unrelated inheritable handle leaked");
    require(WaitForSingleObject(sentinel.value, 0) == WAIT_TIMEOUT, "unrelated inheritable event crossed the allowlist");
    wp::Options null; null.stdout_mode = null.stderr_mode = wp::Output::Null;
    require(wp::run(fixture, {L"-version"}, null).standard_output.empty(), "NUL output");
    require(text(wp::run(fixture, {L"--eof"})) == "0", "stdin EOF closure");
    require(GetErrorMode() == prior_error_mode, "parent error mode was not restored");
    std::cout << "arguments_unicode_environment_exit_stdio_null_inheritance=PASS\n";
}
void pipes(const fs::path& fixture) {
    wp::Options options; options.timeout = 10s; size_t offset = 0, out = 0, err = 0;
    options.input = [&] { if (offset++ == 32) return std::vector<uint8_t>{}; return std::vector<uint8_t>(65536, 'I'); };
    options.stdout_progress = [&](const uint8_t*, size_t count) { out += count; };
    options.stderr_progress = [&](const uint8_t*, size_t count) { err += count; };
    const auto result = wp::run(fixture, {L"--flood"}, options);
    require(result.exit_code == 0 && result.standard_output.size() == 1048576 + 7 &&
        result.standard_error.size() == 1048576 && out == result.standard_output.size() && err == result.standard_error.size(), "simultaneous pipe/progress drain");
    wp::Options early; early.require_full_input = true;
    early.input = [] { return std::vector<uint8_t>(65536, 'I'); };
    failure(wp::Failure::Pipe, [&] { wp::run(fixture, {L"-version"}, early); });
    wp::Options callback;
    callback.stdout_progress = [](const uint8_t*, size_t) { throw wp::Error(wp::Failure::Pipe, 0, "injected progress callback"); };
    failure(wp::Failure::Pipe, [&] { wp::run(fixture, {L"--flood"}, callback); });
    std::cout << "bidirectional_pipe_flood_progress_eof=PASS input_bytes=2097152\n";
}
void lifetimes(const fs::path& fixture) {
    for (const auto mode : {L"--sleep", L"--closed"}) {
        std::string received; wp::Options options; options.timeout = 150ms;
        options.stdout_progress = [&](const uint8_t* data, size_t size) { received.append(reinterpret_cast<const char*>(data), size); };
        failure(wp::Failure::Timeout, [&] { wp::run(fixture, {mode}, options); });
        require(!received.empty() && stopped(std::stoul(received)), "timeout child survived");
    }
    std::string received; wp::Options options;
    options.input = [] { return std::vector<uint8_t>(65536, 'x'); };
    options.stdout_progress = [&](const uint8_t* data, size_t size) { received.append(reinterpret_cast<const char*>(data), size); };
    options.cancelled = [&] { return !received.empty(); };
    failure(wp::Failure::Cancelled, [&] { wp::run(fixture, {L"--sleep"}, options); });
    require(stopped(std::stoul(received)), "cancelled blocked writer survived");
    const auto descendants = wp::run(fixture, {L"--descendant"});
    require(descendants.exit_code == 0 && stopped(std::stoul(text(descendants))), "descendant escaped normal parent exit");
    for (const auto point : {wp::Point::Created, wp::Point::BeforeResume, wp::Point::Pump}) {
        wp::test_hook = [=](wp::Point at) { if (at == point) throw wp::Error(wp::Failure::Launch, 0, "injected"); };
        failure(wp::Failure::Launch, [&] { wp::run(fixture, {L"--sleep"}); }); wp::test_hook = {};
    }
    std::cout << "cancel_timeout_closed_pipes_descendants_failure_cleanup=PASS\n";
}
void parent_death(const fs::path& fixture, const fs::path& root) {
    const auto name = L"Local\\VRhino-parent-death-" + std::to_wstring(GetCurrentProcessId());
    Handle ready(CreateEventW(nullptr, TRUE, FALSE, name.c_str()));
    const auto pidfile = root / "child.pid";
    std::wstring command = L"\"" + fixture.native() + L"\" --supervise \"" + pidfile.native() + L"\" \"" + name + L"\"";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION process{};
    require(CreateProcessW(fixture.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) != 0, "supervisor launch");
    Handle parent(process.hProcess), thread(process.hThread);
    const auto wait = WaitForSingleObject(ready.value, 10000);
    if (wait != WAIT_OBJECT_0) { TerminateProcess(parent.value, 90); throw std::runtime_error("supervisor readiness timeout"); }
    std::ifstream file(pidfile); DWORD pid = 0; file >> pid; require(pid != 0, "supervised child pid");
    Handle child(OpenProcess(SYNCHRONIZE, FALSE, pid));
    require(TerminateProcess(parent.value, 90) != 0, "terminate supervisor only");
    require(WaitForSingleObject(parent.value, 10000) == WAIT_OBJECT_0 &&
            WaitForSingleObject(child.value, 10000) == WAIT_OBJECT_0, "helper survived supervisor death");
    std::cout << "parent_death_atomic_job_ownership=PASS\n";
}
void failures(const fs::path& fixture, const fs::path& build, const fs::path& root) {
    failure(wp::Failure::Missing, [&] { wp::check_helper(root / "missing.exe"); });
    const auto wrong = root / "wrong-architecture.exe"; fs::copy_file(fixture, wrong);
    { std::fstream file(wrong, std::ios::in | std::ios::out | std::ios::binary); IMAGE_DOS_HEADER dos{}; file.read(reinterpret_cast<char*>(&dos), sizeof(dos)); file.seekp(dos.e_lfanew + sizeof(DWORD)); WORD machine = IMAGE_FILE_MACHINE_I386; file.write(reinterpret_cast<const char*>(&machine), sizeof(machine)); }
    failure(wp::Failure::Architecture, [&] { wp::check_helper(wrong); });
    const auto missing = root / "missing-dependency.exe";
    fs::copy_file(build / "vrhino-process-missing-dependency.exe", missing);
    // The fixture DLL exists only beside its original build executable. Neither
    // the copied image directory nor inherited cwd/PATH supplies it here.
    failure(wp::Failure::Dependency, [&] { wp::check_helper(missing); });
    const auto denied = root / "denied.exe"; fs::copy_file(fixture, denied);
    PACL original = nullptr; PSECURITY_DESCRIPTOR descriptor = nullptr;
    require(GetNamedSecurityInfoW(denied.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &original, nullptr, &descriptor) == ERROR_SUCCESS, "read fixture ACL");
    BYTE sid[SECURITY_MAX_SID_SIZE]{}; DWORD size = sizeof(sid); require(CreateWellKnownSid(WinWorldSid, nullptr, sid, &size) != 0, "fixture SID");
    EXPLICIT_ACCESSW access{}; access.grfAccessPermissions = FILE_EXECUTE; access.grfAccessMode = DENY_ACCESS;
    access.Trustee.TrusteeForm = TRUSTEE_IS_SID; access.Trustee.ptstrName = reinterpret_cast<LPWSTR>(sid);
    PACL acl = nullptr; require(SetEntriesInAclW(1, &access, original, &acl) == ERROR_SUCCESS, "fixture deny ACL");
    require(SetNamedSecurityInfoW(const_cast<LPWSTR>(denied.c_str()), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            nullptr, nullptr, acl, nullptr) == ERROR_SUCCESS, "apply fixture deny ACL");
    try { failure(wp::Failure::Permission, [&] { wp::check_helper(denied); }); }
    catch (...) { SetNamedSecurityInfoW(const_cast<LPWSTR>(denied.c_str()), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, original, nullptr); LocalFree(acl); LocalFree(descriptor); throw; }
    SetNamedSecurityInfoW(const_cast<LPWSTR>(denied.c_str()), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, original, nullptr);
    LocalFree(acl); LocalFree(descriptor);
    std::cout << "missing_wrong_architecture_missing_DLL_execute_permission=PASS\n";
}
void media(const fs::path& fixture, const fs::path& root) {
    const auto sibling = root / L"vrhino-ffmpeg.exe"; fs::copy_file(fixture, sibling);
    require(wp::run(fixture, {L"--discover"}).exit_code == 0, "packaged sibling discovery");
    wp::check_helper(sibling);
    const auto input = root / L"输入 媒体.dat", output = root / L"输出 视频.mp4";
    vrhino::product::run_media_detail::check_output_destination(output, false);
    { std::ofstream file(input); file << "synthetic input"; }
    const auto probe = vrhino::probe_media_bounded(sibling, input);
    require(probe.width == 2 && probe.height == 2 && probe.has_audio, "real workflow probe parsing");
    const auto video = vrhino::decode_video_rgb24(sibling, input, 1);
    const auto audio = vrhino::decode_audio_mono_f32_16khz(sibling, input, 2);
    require(video.frames.size() == 1 && audio.samples.size() == 2 && audio.samples[0] == 0.25f, "real workflow decode integration");
    const auto encoded = vrhino::encode_mux_mp4_atomic(sibling, video.frames, 25, input, output);
    require(encoded.bytes > 0 && fs::exists(output), "real workflow Unicode output");
    vrhino::encode_mux_mp4_atomic(sibling, video.frames, 25, input, output, {}, true);
    auto tensor = vrhino::Tensor::host({1,3,1,2,2}, vrhino::DType::F32);
    std::fill_n(tensor.data_as<float>(), 12, 0.5f);
    const auto product = vrhino::product::run_media_detail::encode_mp4(tensor, 25, sibling, output, true, {}, 0.0f, 1.0f);
    require(product.bytes > 0 && product.minimum == 0.5f && product.maximum == 0.5f, "real product RGB encoding boundary");
    bool cancelled = false;
    try { vrhino::encode_mux_mp4_atomic(sibling, video.frames, 25, input, output, [] { return true; }, true); }
    catch (const vrhino::LipSyncWorkflowError&) { cancelled = true; }
    require(cancelled && fs::exists(output) && !fs::exists(fs::path(output.native() + L".partial.mp4")), "cancelled media must preserve completed output and clean its partial");
    fs::remove(sibling);
    require(wp::run(fixture, {L"--discover"}).exit_code != 0, "missing packaged helper must fail without PATH fallback");
    std::cout << "packaged_helper_unicode_media_workflow_product_stream=PASS\n";
}
int wmain() {
    const auto build = wp::executable_path().parent_path();
    const auto original = fs::current_path();
    const auto root = build / (L"process fixture 中文 " + std::to_wstring(GetCurrentProcessId()));
    try {
        fs::create_directory(root); fs::current_path(root);
        const auto fixture = root / L"测试 helper.exe";
        fs::copy_file(build / "vrhino-process-fixture.exe", fixture);
        basic(fixture); pipes(fixture); lifetimes(fixture); parent_death(fixture, root); failures(fixture, build, root); media(fixture, root);
        DWORD before = 0, after = 0; GetProcessHandleCount(GetCurrentProcess(), &before);
        for (int i = 0; i < 32; ++i) {
            wp::run(fixture, {L"--eof"});
            wp::Options options; options.timeout = 20ms;
            failure(wp::Failure::Timeout, [&] { wp::run(fixture, {L"--sleep"}, options); });
            failure(wp::Failure::Missing, [&] { wp::run(root / "missing.exe", {}); });
            wp::test_hook = [i](wp::Point at) {
                if (at == (i % 2 ? wp::Point::Created : wp::Point::Pump))
                    throw wp::Error(wp::Failure::Launch, 0, "repeated ownership failure");
            };
            failure(wp::Failure::Launch, [&] { wp::run(fixture, {L"--sleep"}); });
            wp::test_hook = {};
        }
        GetProcessHandleCount(GetCurrentProcess(), &after);
        require(after == before, "process handle growth");
        fs::current_path(original); fs::remove_all(root);
        std::cout << "handle_count_before=" << before << " after=" << after << " cycles=32\nWINDOWS_PROCESS_TESTS=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        wp::test_hook = {}; fs::current_path(original);
        std::cerr << error.what() << "\nfixture=" << wp::utf8(root) << '\n'; return 1;
    }
}
