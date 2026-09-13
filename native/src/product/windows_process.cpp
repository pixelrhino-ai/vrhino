#include "vrhino/product/windows_process.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace vrhino::product::windows_process {
namespace {
const char* failure_name(Failure kind) {
    switch (kind) {
    case Failure::Missing: return "missing executable";
    case Failure::Architecture: return "wrong or invalid executable architecture";
    case Failure::Permission: return "executable permission denied";
    case Failure::Dependency: return "missing or invalid executable dependency";
    case Failure::Launch: return "process launch failed";
    case Failure::Pipe: return "process pipe failed";
    case Failure::Cancelled: return "process cancelled";
    case Failure::Timeout: return "process timed out";
    }
    return "process failed";
}
}
Error::Error(Failure category, uint32_t code, const std::string& message)
    : std::runtime_error(std::string(failure_name(category)) + ": " + message + " (Windows code " + std::to_string(code) + ")"),
      kind(category), native_code(code) {}
#ifdef VRHINO_WINDOWS_PROCESS_TESTING
std::function<void(Point)> test_hook;
#define CHECKPOINT(p) do { if (test_hook) test_hook(Point::p); } while (false)
#else
#define CHECKPOINT(p) ((void)0)
#endif
namespace {
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    Handle() = default;
    explicit Handle(HANDLE h) : value(h) {}
    ~Handle() { if (valid()) CloseHandle(value); }
    Handle(Handle&& h) noexcept : value(std::exchange(h.value, INVALID_HANDLE_VALUE)) {}
    Handle& operator=(Handle&& h) noexcept {
        if (this != &h) { if (valid()) CloseHandle(value); value = std::exchange(h.value, INVALID_HANDLE_VALUE); }
        return *this;
    }
    Handle(const Handle&) = delete;
    bool valid() const { return value && value != INVALID_HANDLE_VALUE; }
};
[[noreturn]] void fail(Failure kind, const char* message, DWORD code = GetLastError()) {
    throw Error(kind, code, message);
}
void check(BOOL value, Failure kind, const char* message) { if (!value) fail(kind, message); }
Failure launch_kind(DWORD code) {
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) return Failure::Missing;
    if (code == ERROR_ACCESS_DENIED || code == ERROR_ELEVATION_REQUIRED) return Failure::Permission;
    if (code == ERROR_BAD_EXE_FORMAT || code == ERROR_EXE_MACHINE_TYPE_MISMATCH) return Failure::Architecture;
    if (code == ERROR_MOD_NOT_FOUND || code == ERROR_DLL_INIT_FAILED) return Failure::Dependency;
    return Failure::Launch;
}
std::wstring quote(const std::wstring& value) {
    if (value.find(L'\0') != std::wstring::npos) fail(Failure::Launch, "embedded NUL in process argument", ERROR_INVALID_PARAMETER);
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0; result += c;
    }
    result.append(2 * slashes, L'\\'); result += L'"'; return result;
}
Handle image(const std::filesystem::path& executable) {
    Handle file(CreateFileW(executable.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!file.valid()) { const DWORD code = GetLastError(); fail(launch_kind(code), "media executable cannot be opened", code); }
    IMAGE_DOS_HEADER dos{}; DWORD count = 0;
    if (!ReadFile(file.value, &dos, sizeof(dos), &count, nullptr) || count != sizeof(dos) ||
        dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < static_cast<LONG>(sizeof(dos)))
        fail(Failure::Architecture, "media executable is not a native x64 PE image", ERROR_BAD_EXE_FORMAT);
    LARGE_INTEGER offset{}; offset.QuadPart = dos.e_lfanew;
    check(SetFilePointerEx(file.value, offset, nullptr, FILE_BEGIN), Failure::Architecture, "invalid PE header offset");
    DWORD signature = 0; IMAGE_FILE_HEADER header{};
    if (!ReadFile(file.value, &signature, sizeof(signature), &count, nullptr) || count != sizeof(signature) ||
        signature != IMAGE_NT_SIGNATURE || !ReadFile(file.value, &header, sizeof(header), &count, nullptr) ||
        count != sizeof(header) || header.Machine != IMAGE_FILE_MACHINE_AMD64 ||
        !(header.Characteristics & IMAGE_FILE_EXECUTABLE_IMAGE) || (header.Characteristics & IMAGE_FILE_DLL))
        fail(Failure::Architecture, "media executable architecture is not native x64", ERROR_EXE_MACHINE_TYPE_MISMATCH);
    return file;
}
struct Pipe {
    Handle parent, child, event;
    OVERLAPPED operation{};
    bool pending = false, ended = false;
    std::vector<uint8_t> buffer;
    explicit Pipe(bool input) : event(CreateEventW(nullptr, TRUE, FALSE, nullptr)), buffer(65536) {
        if (!event.valid()) fail(Failure::Pipe, "cannot create pipe event");
        operation.hEvent = event.value;
        static std::atomic<uint64_t> sequence{0};
        const auto name = L"\\\\.\\pipe\\vrhino-process-" + std::to_wstring(GetCurrentProcessId()) +
            L"-" + std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(sequence++);
        parent = Handle(CreateNamedPipeW(name.c_str(), (input ? PIPE_ACCESS_OUTBOUND : PIPE_ACCESS_INBOUND) |
            FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE, PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1, 65536, 65536, 0, nullptr));
        if (!parent.valid()) fail(Failure::Pipe, "cannot create media pipe");
        SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        child = Handle(CreateFileW(name.c_str(), input ? GENERIC_READ : GENERIC_WRITE, 0,
                                  &inherit, OPEN_EXISTING, 0, nullptr));
        if (!child.valid()) fail(Failure::Pipe, "cannot open child media pipe");
        if (!ConnectNamedPipe(parent.value, &operation)) {
            const auto code = GetLastError();
            if (code != ERROR_PIPE_CONNECTED) fail(Failure::Pipe, "cannot connect media pipe", code);
        }
        ResetEvent(event.value);
    }
    ~Pipe() {
        if (pending) { CancelIoEx(parent.value, &operation); DWORD count = 0; GetOverlappedResult(parent.value, &operation, &count, TRUE); }
    }
    bool complete(DWORD& count) {
        if (!GetOverlappedResult(parent.value, &operation, &count, FALSE)) {
            const auto code = GetLastError();
            if (code == ERROR_IO_INCOMPLETE) return false;
            pending = false;
            if (code == ERROR_BROKEN_PIPE || code == ERROR_NO_DATA) { ended = true; return false; }
            fail(Failure::Pipe, "media pipe completion failed", code);
        }
        pending = false; return true;
    }
    void start(bool input, DWORD bytes) {
        ResetEvent(event.value);
        DWORD count = 0;
        const BOOL ok = input ? WriteFile(parent.value, buffer.data(), bytes, &count, &operation) :
                               ReadFile(parent.value, buffer.data(), bytes, &count, &operation);
        if (!ok) {
            const auto code = GetLastError();
            if (code == ERROR_BROKEN_PIPE || code == ERROR_NO_DATA) { ended = true; return; }
            if (code != ERROR_IO_PENDING) fail(Failure::Pipe, "media pipe I/O failed", code);
        }
        pending = true;
    }
};
Handle standard(Output mode, DWORD stream) {
    if (mode == Output::Inherit) {
        const HANDLE original = GetStdHandle(stream);
        if (original && original != INVALID_HANDLE_VALUE) {
            HANDLE copy = nullptr;
            check(DuplicateHandle(GetCurrentProcess(), original, GetCurrentProcess(), &copy, 0, TRUE,
                                  DUPLICATE_SAME_ACCESS), Failure::Pipe, "cannot inherit standard output");
            return Handle(copy);
        }
    }
    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle file(CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr));
    if (!file.valid()) fail(Failure::Pipe, "cannot open NUL output");
    return file;
}
struct Attributes {
    std::vector<uint8_t> buffer;
    LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
    Attributes() {
        SIZE_T size = 0; InitializeProcThreadAttributeList(nullptr, 2, 0, &size);
        buffer.resize(size); list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(buffer.data());
        check(InitializeProcThreadAttributeList(list, 2, 0, &size), Failure::Launch, "cannot initialize process attributes");
    }
    ~Attributes() { DeleteProcThreadAttributeList(list); }
};
struct Child {
    Handle job, process, thread;
    Child() : job(CreateJobObjectW(nullptr, nullptr)) {
        if (!job.valid()) fail(Failure::Launch, "cannot create helper job");
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        check(SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)),
              Failure::Launch, "cannot set helper lifetime limit");
    }
    ~Child() {
        if (process.valid()) { TerminateJobObject(job.value, ERROR_CANCELLED); WaitForSingleObject(process.value, INFINITE); }
    }
};
}

std::wstring wide(const std::string& text) {
    if (text.empty()) return {};
    if (text.size() > INT_MAX) fail(Failure::Launch, "UTF-8 process argument too long", ERROR_INVALID_PARAMETER);
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (!count) fail(Failure::Launch, "invalid UTF-8 process argument");
    std::wstring result(count, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), result.data(), count);
    return result;
}
std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string(); return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::filesystem::path executable_path() {
    std::vector<wchar_t> buffer(32768);
    const DWORD count = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!count || count >= buffer.size()) fail(Failure::Launch, "cannot locate product executable");
    return std::wstring(buffer.data(), count);
}
std::filesystem::path discover_helper() {
    const auto path = executable_path().parent_path() / L"vrhino-ffmpeg.exe";
    auto held = image(path); return path;
}
void check_helper(const std::filesystem::path& executable) {
    Options options; options.stdout_mode = options.stderr_mode = Output::Null;
    options.timeout = std::chrono::seconds(10);
    const auto result = run(executable, {L"-version"}, options);
    if (result.exit_code) fail(Failure::Launch, "media helper readiness check failed", result.exit_code);
}

Result run(const std::filesystem::path& requested, const std::vector<std::wstring>& arguments, const Options& options) {
    if (options.cancelled && options.cancelled()) fail(Failure::Cancelled, "media execution cancelled", ERROR_CANCELLED);
    if (requested.empty() || requested.native().find(L'\0') != std::wstring::npos)
        fail(Failure::Missing, "media executable path is empty or invalid", ERROR_INVALID_NAME);
    const auto executable = std::filesystem::absolute(requested);
    auto executable_owner = image(executable);
    std::wstring command = quote(executable.native());
    for (const auto& argument : arguments) command += L" " + quote(argument);
    if (command.size() >= 32767) fail(Failure::Launch, "process command line exceeds Windows limit", ERROR_INVALID_PARAMETER);
    Pipe input(true);
    auto output = options.stdout_mode == Output::Capture ? std::make_unique<Pipe>(false) : nullptr;
    auto error = options.stderr_mode == Output::Capture ? std::make_unique<Pipe>(false) : nullptr;
    Handle stdout_handle, stderr_handle;
    if (!output) stdout_handle = standard(options.stdout_mode, STD_OUTPUT_HANDLE);
    if (!error) stderr_handle = standard(options.stderr_mode, STD_ERROR_HANDLE);
    Child child; // destroyed before pipe OVERLAPPED storage on every exit path
    Attributes attributes;
    HANDLE inherited[] = {input.child.value, output ? output->child.value : stdout_handle.value,
                          error ? error->child.value : stderr_handle.value};
    check(UpdateProcThreadAttribute(attributes.list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          inherited, sizeof(inherited), nullptr, nullptr), Failure::Launch, "cannot restrict inherited handles");
    // Atomic job assignment removes the create-then-assign parent-death window.
    check(UpdateProcThreadAttribute(attributes.list, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST,
          &child.job.value, sizeof(HANDLE), nullptr, nullptr), Failure::Launch, "cannot assign helper job at creation");
    STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inherited[0]; startup.StartupInfo.hStdOutput = inherited[1]; startup.StartupInfo.hStdError = inherited[2];
    startup.lpAttributeList = attributes.list;
    PROCESS_INFORMATION process{};
    // Inherit environment and cwd exactly. No shell or PATH executable lookup.
    BOOL created = FALSE;
    DWORD launch_error = 0;
    {
        // Child loader failures must produce an exit status, not an unattended
        // modal dialog. Windows inherits the process error mode at creation.
        // Serialize our creation calls and restore the parent's mode immediately.
        static std::mutex creation_mutex;
        const std::lock_guard lock(creation_mutex);
        const DWORD prior_mode = GetErrorMode();
        SetErrorMode(prior_mode | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
        created = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
            nullptr, nullptr, &startup.StartupInfo, &process);
        launch_error = GetLastError();
        SetErrorMode(prior_mode);
    }
    if (!created) fail(launch_kind(launch_error), "cannot launch native media executable", launch_error);
    child.process = Handle(process.hProcess); child.thread = Handle(process.hThread);
    CHECKPOINT(Created);
    input.child = Handle(); if (output) output->child = Handle(); if (error) error->child = Handle();
    stdout_handle = Handle(); stderr_handle = Handle();
    CHECKPOINT(BeforeResume);
    if (ResumeThread(child.thread.value) == MAXDWORD) fail(Failure::Launch, "cannot resume helper");
    const auto started = std::chrono::steady_clock::now();
    Result result;
    bool exited = false, input_complete = false; size_t input_offset = 0;
    auto receive = [&](Pipe* pipe, auto& destination, const auto& progress) {
        if (!pipe || pipe->ended) return;
        if (pipe->pending) {
            DWORD count = 0;
            if (pipe->complete(count)) {
                if (!count) pipe->ended = true;
                else {
                    destination.insert(destination.end(), pipe->buffer.begin(), pipe->buffer.begin() + count);
                    if (progress) progress(pipe->buffer.data(), count);
                }
            }
        }
        if (!pipe->ended && !pipe->pending) pipe->start(false, static_cast<DWORD>(pipe->buffer.size()));
    };
    for (;;) {
        CHECKPOINT(Pump);
        if (options.cancelled && options.cancelled()) fail(Failure::Cancelled, "media execution cancelled", ERROR_CANCELLED);
        if (options.timeout.count() > 0 && std::chrono::steady_clock::now() - started >= options.timeout)
            fail(Failure::Timeout, "media execution timed out", WAIT_TIMEOUT);
        receive(output.get(), result.standard_output, options.stdout_progress);
        receive(error.get(), result.standard_error, options.stderr_progress);
        if (!exited && WaitForSingleObject(child.process.value, 0) == WAIT_OBJECT_0) {
            DWORD code = 0;
            check(GetExitCodeProcess(child.process.value, &code), Failure::Launch, "cannot read helper exit code");
            result.exit_code = code;
            exited = true;
            check(TerminateJobObject(child.job.value, ERROR_CANCELLED), Failure::Launch, "cannot retire helper descendants");
        }
        if (!input.ended) {
            if (input.pending) {
                DWORD count = 0;
                if (input.complete(count)) input_offset += count;
            }
            if (!input.pending && !input.ended) {
                if (exited) input.ended = true;
                else {
                    if (input_offset == input.buffer.size()) input.buffer.clear();
                    if (input.buffer.empty() || input_offset == 0) {
                        input.buffer = options.input ? options.input() : std::vector<uint8_t>{}; input_offset = 0;
                    }
                    if (input.buffer.empty()) { input_complete = true; input.ended = true; input.parent = Handle(); }
                    else {
                        // Keep only the unwritten bytes as stable OVERLAPPED storage.
                        if (input_offset) { input.buffer.erase(input.buffer.begin(), input.buffer.begin() + input_offset); input_offset = 0; }
                        input.start(true, static_cast<DWORD>(std::min<size_t>(input.buffer.size(), MAXDWORD)));
                    }
                }
            }
        }
        if (exited && (!output || output->ended) && (!error || error->ended)) break;
        HANDLE events[4]{}; DWORD count = 0;
        if (!exited) events[count++] = child.process.value;
        for (auto* pipe : {&input, output.get(), error.get()}) if (pipe && pipe->pending) events[count++] = pipe->event.value;
        if (count) WaitForMultipleObjects(count, events, FALSE, 10); else Sleep(1);
    }
    if (result.exit_code == 0xC0000135 || result.exit_code == 0xC0000142 || result.exit_code == 0xC0000139)
        fail(Failure::Dependency, "media helper DLL is missing or failed initialization", result.exit_code);
    if (result.exit_code == 0xC000007B) fail(Failure::Architecture, "media helper dependency has invalid architecture", result.exit_code);
    if (options.require_full_input && !input_complete && result.exit_code == 0)
        fail(Failure::Pipe, "media helper exited before consuming its complete input", ERROR_BROKEN_PIPE);
    return result;
}

void publish_output(const std::filesystem::path& partial, const std::filesystem::path& destination, bool replace) {
    Handle file(CreateFileW(partial.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, 0, nullptr));
    if (!file.valid()) fail(Failure::Launch, "cannot open completed media output for publication");
    const auto name = std::filesystem::absolute(destination).native();
    const size_t bytes = sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t);
    if (bytes > MAXDWORD) fail(Failure::Launch, "media output path is too long", ERROR_INVALID_NAME);
    std::vector<uint8_t> storage(bytes);
    auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    info->Flags = replace ? FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS : 0;
    info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, name.data(), info->FileNameLength);
    check(SetFileInformationByHandle(file.value, FileRenameInfoEx, info, static_cast<DWORD>(bytes)),
          Failure::Launch, "cannot atomically publish completed media output");
}
}
#endif
