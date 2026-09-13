#define NOMINMAX
#include <windows.h>
#include "vrhino/product/windows_process.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
namespace wp = vrhino::product::windows_process;
void output(HANDLE handle, const void* bytes, size_t size) {
    const auto* data = static_cast<const char*>(bytes);
    while (size) { DWORD count = 0; if (!WriteFile(handle, data, static_cast<DWORD>(size), &count, nullptr) || !count) ExitProcess(81); data += count; size -= count; }
}
void text(HANDLE handle, const std::string& value) { output(handle, value.data(), value.size()); }
std::vector<char> input() {
    std::vector<char> result; std::array<char, 65536> buffer{};
    for (;;) { DWORD count = 0; if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr) || !count) break; result.insert(result.end(), buffer.begin(), buffer.begin() + count); }
    return result;
}
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE), err = GetStdHandle(STD_ERROR_HANDLE);
    if (argc < 2) return 80;
    const std::wstring mode = argv[1];
    if (mode == L"--args") {
        for (int i = 2; i < argc; ++i) { const auto value = wp::utf8(std::filesystem::path(argv[i])); text(out, std::to_string(value.size()) + ":" + value + "\n"); }
        text(err, "stderr captured\n"); return 37;
    }
    if (mode == L"--environment") {
        std::array<wchar_t, 1024> value{};
        GetEnvironmentVariableW(L"VRHINO_PROCESS_FIXTURE_ENV", value.data(), static_cast<DWORD>(value.size()));
        text(out, wp::utf8(value.data())); text(err, wp::utf8(std::filesystem::current_path())); return 0;
    }
    if (mode == L"--handle") {
        // Numeric values can be reused for unrelated child handles. Observe
        // the parent's actual event, not mere validity of the numeric value.
        const auto candidate = reinterpret_cast<HANDLE>(std::stoull(argv[2]));
        SetEvent(candidate); return 0;
    }
    if (mode == L"--flood") {
        std::vector<char> bytes(1024 * 1024, 'O');
        std::thread errors([&] { std::vector<char> data(bytes.size(), 'E'); output(err, data.data(), data.size()); });
        output(out, bytes.data(), bytes.size()); errors.join();
        const auto received = input();
        text(out, std::to_string(received.size())); return received.size() == 2 * 1024 * 1024 ? 0 : 82;
    }
    if (mode == L"--eof") { const auto received = input(); text(out, std::to_string(received.size())); return 0; }
    if (mode == L"--sleep" || mode == L"--closed") {
        text(out, std::to_string(GetCurrentProcessId()) + "\n");
        if (argc > 2) {
            const HANDLE ready = OpenEventW(EVENT_MODIFY_STATE, FALSE, argv[2]);
            if (!ready) return 88;
            SetEvent(ready); CloseHandle(ready);
        }
        if (mode == L"--closed") { CloseHandle(out); CloseHandle(err); CloseHandle(GetStdHandle(STD_INPUT_HANDLE)); }
        Sleep(60000); return 0;
    }
    if (mode == L"--supervise") {
        wp::Options options;
        options.stdout_progress = [&](const uint8_t* data, size_t count) {
            std::ofstream file(std::filesystem::path(argv[2]), std::ios::binary); file.write(reinterpret_cast<const char*>(data), count); file.close();
            HANDLE event = OpenEventW(EVENT_MODIFY_STATE, FALSE, argv[3]);
            if (event) { SetEvent(event); CloseHandle(event); }
        };
        return static_cast<int>(wp::run(wp::executable_path(), {L"--sleep"}, options).exit_code);
    }
    if (mode == L"--descendant") {
        const auto name = L"Local\\VRhino-descendant-" + std::to_wstring(GetCurrentProcessId());
        const HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, name.c_str());
        if (!ready) return 88;
        const auto app = wp::executable_path(); std::wstring command = L"\"" + app.native() + L"\" --sleep \"" + name + L"\"";
        STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION child{};
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE); startup.hStdOutput = out; startup.hStdError = err;
        if (!CreateProcessW(app.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child)) return 83;
        // Prove a live descendant owns the inherited pipes before its parent
        // exits; an already-failed descendant would not exercise job cleanup.
        const bool alive = WaitForSingleObject(ready, 10000) == WAIT_OBJECT_0 &&
            WaitForSingleObject(child.hProcess, 0) == WAIT_TIMEOUT;
        CloseHandle(ready);
        if (!alive) { TerminateProcess(child.hProcess, 89); CloseHandle(child.hThread); CloseHandle(child.hProcess); return 89; }
        text(out, std::to_string(child.dwProcessId) + "\n"); CloseHandle(child.hThread); CloseHandle(child.hProcess); return 0;
    }
    if (mode == L"--discover") { try { wp::check_helper(wp::discover_helper()); return 0; } catch (...) { return 84; } }
    if (mode == L"-version") { text(out, "VRhino synthetic media protocol fixture\n"); return 0; }
    std::vector<std::wstring> arguments(argv + 1, argv + argc);
    const auto has = [&](const wchar_t* value) { return std::find(arguments.begin(), arguments.end(), value) != arguments.end(); };
    // Small deterministic protocol responses, not an FFmpeg implementation.
    for (size_t i = 0; i + 1 < arguments.size(); ++i) {
        if (arguments[i] == L"-i" && arguments[i + 1] != L"pipe:0") {
            std::ifstream file(std::filesystem::path(arguments[i + 1]), std::ios::binary);
            if (!file) return 85;
        }
    }
    if (has(L"null")) {
        text(err, "Video: h264, yuv420p, 2x2, 25 fps\nAudio: aac, 48000 Hz, stereo\n"); return 0;
    }
    if (has(L"pipe:1")) {
        if (has(L"f32le")) { const float values[]{0.25f, -0.25f}; output(out, values, sizeof(values)); }
        else { const std::array<char, 12> rgb{}; output(out, rgb.data(), rgb.size()); }
        return 0;
    }
    if (has(L"mp4")) {
        const auto received = input();
        std::ofstream file(std::filesystem::path(arguments.back()), std::ios::binary); file << "fixture-mp4:" << received.size();
        return file ? 0 : 86;
    }
    return 87;
}
