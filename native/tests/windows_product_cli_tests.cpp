#include "vrhino/product/windows_process.h"
#include "vrhino/product/windows_cache.h"
#include "vrhino/product/component_package.h"
#include "windows_converter_io.h"
#include "fixtures/cache_package_fixture.h"
#include <aclapi.h>
#include <winioctl.h>
#include <zlib.h>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>

namespace fs = std::filesystem;
namespace p = vrhino::product;
namespace wc = p::windows_cache;
namespace wp = p::windows_process;
namespace io = p::windows_converter_io;
namespace {
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void clean(const fs::path& root) { if (fs::exists(root)) wc::remove_tree(root); }
std::string bytes(const fs::path& path) { std::ifstream file(path, std::ios::binary); require(static_cast<bool>(file), "fixture read"); return {std::istreambuf_iterator<char>(file), {}}; }
void raw(const fs::path& path, const std::string& data) { fs::create_directories(path.parent_path()); std::ofstream file(path, std::ios::binary); file.write(data.data(), static_cast<std::streamsize>(data.size())); require(static_cast<bool>(file), "fixture write"); }
struct Environment {
    std::wstring name, previous;
    bool present = false;
    Environment(const wchar_t* key, const std::wstring& value) : name(key) {
        DWORD size = GetEnvironmentVariableW(key, nullptr, 0);
        present = size != 0;
        if (present) { previous.resize(size); GetEnvironmentVariableW(key, previous.data(), size); previous.resize(size - 1); }
        require(SetEnvironmentVariableW(key, value.empty() ? nullptr : value.c_str()) != 0, "set child test environment");
    }
    ~Environment() { SetEnvironmentVariableW(name.c_str(), present ? previous.c_str() : nullptr); }
};
DWORD handles() { DWORD count = 0; require(GetProcessHandleCount(GetCurrentProcess(), &count) != 0, "handle measurement"); return count; }
size_t invocations = 0;
std::ofstream transcript;
std::string cli(const fs::path& exe, const std::vector<std::wstring>& args, bool success = true, const std::string& expected = {}) {
    wp::Options options; options.timeout = std::chrono::seconds(20);
    const auto result = wp::run(exe, args, options);
    ++invocations;
    const std::string output(result.standard_output.begin(), result.standard_output.end());
    const auto text = output + result.standard_error;
    transcript << "Invocation " << invocations << " exit=" << result.exit_code << '\n';
    for (const auto& argument : args) transcript << wp::utf8(fs::path(argument)) << " | ";
    transcript << '\n' << text << "\n\n"; transcript.flush();
    require((result.exit_code == 0) == success, "unexpected CLI exit: " + text);
    require(expected.empty() || text.find(expected) != std::string::npos, "missing diagnostic " + expected + ": " + text);
    return text;
}
void member(std::string& tar, const std::string& name, const std::string& contents) {
    std::array<char, 512> header{};
    require(name.size() < 100, "fixture name bound");
    std::memcpy(header.data(), name.data(), name.size());
    std::snprintf(header.data() + 100, 8, "%07o", 0755);
    std::snprintf(header.data() + 124, 12, "%011llo", static_cast<unsigned long long>(contents.size()));
    std::memset(header.data() + 148, ' ', 8); header[156] = '0';
    std::memcpy(header.data() + 257, "ustar", 5); header[263] = '0'; header[264] = '0';
    unsigned sum = 0; for (const unsigned char c : header) sum += c;
    std::snprintf(header.data() + 148, 7, "%06o", sum); header[155] = ' ';
    tar.append(header.data(), header.size()); tar += contents;
    tar.append((512 - contents.size() % 512) % 512, '\0');
}
void component_archive(const fs::path& target, const fs::path& helper) {
    const auto payload = bytes(helper);
    const std::string manifest = "{\"schema_version\":1,\"identity\":{\"namespace\":\"vrhino\",\"name\":\"media-windows-x86_64\",\"version\":\"1.0.0\",\"publisher\":\"fixture\"},"
        "\"contract\":{\"name\":\"vrhino.media.rgb24-h264-mp4.cli\",\"major\":1,\"minor\":0},"
        "\"platform\":{\"os\":\"windows\",\"architecture\":\"x86_64\"},\"entrypoint\":\"bin/vrhino-ffmpeg.exe\","
        "\"artifacts\":[{\"path\":\"bin/vrhino-ffmpeg.exe\",\"size\":" + std::to_string(payload.size()) +
        ",\"sha256\":\"" + p::sha256_file(helper) + "\",\"executable\":true}]}";
    std::string tar; member(tar, "vrhino-media/vrhino-component.json", manifest);
    member(tar, "vrhino-media/bin/vrhino-ffmpeg.exe", payload); tar.append(1024, '\0');
    z_stream stream{};
    require(deflateInit2(&stream, 6, Z_DEFLATED, 31, 8, Z_DEFAULT_STRATEGY) == Z_OK, "fixture gzip init");
    std::string compressed(compressBound(static_cast<uLong>(tar.size())) + 64, '\0');
    stream.next_in = reinterpret_cast<Bytef*>(tar.data()); stream.avail_in = static_cast<uInt>(tar.size());
    stream.next_out = reinterpret_cast<Bytef*>(compressed.data()); stream.avail_out = static_cast<uInt>(compressed.size());
    const int result = deflate(&stream, Z_FINISH); compressed.resize(stream.total_out); deflateEnd(&stream);
    require(result == Z_STREAM_END, "fixture gzip write"); raw(target, compressed);
}
std::set<fs::path> inventory(const fs::path& root) {
    std::set<fs::path> result;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        result.insert(entry.path().lexically_relative(root));
        const auto name = entry.path().filename().native();
        require(!name.starts_with(L"component-extraction-") || name.ends_with(L".lock"), "stale extraction staging");
        require(name.find(L".partial") == std::wstring::npos, "leaked partial artifact");
    }
    return result;
}
void permission_test(const fs::path& helper, const fs::path& exe, const fs::path& root) {
    PACL original = nullptr, denied = nullptr; PSECURITY_DESCRIPTOR descriptor = nullptr;
    require(GetNamedSecurityInfoW(helper.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
        nullptr, nullptr, &original, nullptr, &descriptor) == ERROR_SUCCESS, "read ACL");
    std::array<char, SECURITY_MAX_SID_SIZE> storage{}; DWORD size = static_cast<DWORD>(storage.size());
    require(CreateWellKnownSid(WinWorldSid, nullptr, storage.data(), &size) != 0, "everyone SID");
    EXPLICIT_ACCESSW access{}; access.grfAccessPermissions = FILE_READ_DATA | FILE_EXECUTE;
    access.grfAccessMode = DENY_ACCESS; access.grfInheritance = NO_INHERITANCE;
    access.Trustee.TrusteeForm = TRUSTEE_IS_SID; access.Trustee.ptstrName = reinterpret_cast<wchar_t*>(storage.data());
    require(SetEntriesInAclW(1, &access, original, &denied) == ERROR_SUCCESS, "deny ACL");
    require(SetNamedSecurityInfoW(const_cast<wchar_t*>(helper.c_str()), SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr, denied, nullptr) == ERROR_SUCCESS, "apply deny ACL");
    try { cli(exe, {L"--cache-root", root.native(), L"component", L"check"}, false, "permission"); }
    catch (...) { SetNamedSecurityInfoW(const_cast<wchar_t*>(helper.c_str()), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, original, nullptr); LocalFree(denied); LocalFree(descriptor); throw; }
    require(SetNamedSecurityInfoW(const_cast<wchar_t*>(helper.c_str()), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
        nullptr, nullptr, original, nullptr) == ERROR_SUCCESS, "restore ACL"); LocalFree(denied); LocalFree(descriptor);
}
}
int wmain(int argc, wchar_t** argv) {
    try {
        require(argc == 5, "expected package, helper fixture, missing-dependency fixture, work directory");
        const auto package = fs::absolute(argv[1]); const auto helper = fs::absolute(argv[2]);
        const auto missing = fs::absolute(argv[3]); const auto work = fs::absolute(argv[4]);
        clean(work); wc::ensure_directory(work);
        require(package.filename() == "vrhino-windows-test", "expected dedicated smoke package directory");
        for (const char* name : {"models", "blobs", "tmp", "locks", "components", "vrhino-ffmpeg.exe"}) clean(package / name);
        wc::ensure_directory(package / "components");
        transcript.open(work / "cli-transcript.txt", std::ios::binary);
        require(static_cast<bool>(transcript), "open CLI evidence transcript");
        std::array<wchar_t, 32768> system{}; require(GetSystemDirectoryW(system.data(), static_cast<UINT>(system.size())) != 0, "system directory");
        Environment path(L"PATH", system.data()), home(L"VRHINO_HOME", L""), unix_home(L"HOME", L""), spec(L"VRHINO_CONVERTER_SPEC_ROOT", L"");
        const auto original_cwd = fs::current_path(); fs::current_path(work);
        const auto exe = package / "vrhino.exe";
        cli(exe, {L"--help"}, true, "Windows inspection commands");
        cli(exe, {L"--version"}, true, "VRhino v");
        const auto default_root = p::cache_layout({}).root;
        cli(exe, {L"cache", L"info"}, true, wp::utf8(default_root));
        {
            Environment local(L"LOCALAPPDATA", (work / "local app data").native());
            cli(exe, {L"cache", L"info"}, true, wp::utf8(work / "local app data" / "VRhino"));
        }
        {
            Environment selected(L"VRHINO_HOME", (work / L"环境 缓存").native());
            cli(exe, {L"cache", L"info"}, true, wp::utf8(work / L"环境 缓存"));
            cli(exe, {L"--cache-root", L"relative cache", L"cache", L"info"}, true, wp::utf8(work / "relative cache"));
        }
        cli(exe, {L"--cache-root", package.native(), L"list"}, true, "NAME");
        cli(exe, {L"--cache-root", package.native(), L"info", L"invalid"}, false, "PACKAGE_INVALID");
        cli(exe, {L"--cache-root", package.native(), L"info", L"vrhino/missing:v1"}, false, "MODEL_NOT_FOUND");
        cli(exe, {L"--cache-root", package.native(), L"component", L"info"}, false, "COMPONENT_NOT_FOUND");
        std::cout << "help/version/default/VRHINO_HOME/cache-root/Unicode/relative/missing-errors PASS\n";

        p::LocalModelCache cache(package);
        cache.install(cache_fixture::make_package(work / "synthetic", "v1"));
        cli(exe, {L"--cache-root", package.native(), L"list"}, true, "vrhino/fixture");
        cli(exe, {L"--cache-root", package.native(), L"info", L"vrhino/fixture:v1"}, true, "test_arch");
        cli(exe, {L"--cache-root", package.native(), L"info", L"vrhino/fixture:v1", L"--json"}, true, "vrhino/fixture");
        const auto archive = work / L"组件 档案.tar.gz"; component_archive(archive, helper);
        cli(exe, {L"--cache-root", package.native(), L"component", L"install", archive.native()}, true, "Already installed: no");
        cli(exe, {L"--cache-root", package.native(), L"component", L"info"}, true, "media-windows-x86_64");
        cli(exe, {L"--cache-root", package.native(), L"component", L"check"}, true, "Media helper ready");
        raw(work / "invalid.tar.gz", "not gzip");
        cli(exe, {L"--cache-root", package.native(), L"component", L"install", (work / "invalid.tar.gz").native()}, false, "COMPONENT_INVALID");
        cli(exe, {L"--cache-root", L"\\\\unsupported-server\\share\\cache", L"cache", L"info"}, false, "local path");
        std::cout << "synthetic model list/info, component archive/cache helper, invalid archive/remote rejection PASS\n";

        const auto sibling = package / "vrhino-ffmpeg.exe";
        fs::copy_file(missing, sibling, fs::copy_options::overwrite_existing);
        cli(exe, {L"--cache-root", package.native(), L"component", L"check"}, false, "dependency");
        fs::copy_file(helper, sibling, fs::copy_options::overwrite_existing);
        permission_test(sibling, exe, package);
        cli(exe, {L"--cache-root", package.native(), L"component", L"check"}, true, "Media helper ready");
        std::cout << "missing helper DLL and actual execute/read permission diagnostics PASS\n";

        const auto relocated = work / L"搬迁 空间" / "vrhino-windows-test";
        fs::create_directories(relocated.parent_path());
        fs::copy(package, relocated, fs::copy_options::recursive);
        const auto moved_exe = relocated / "vrhino.exe";
        cli(moved_exe, {L"--help"}); cli(moved_exe, {L"--version"});
        cli(moved_exe, {L"--cache-root", relocated.native(), L"cache", L"info"}, true, wp::utf8(relocated / "share" / "vrhino" / "converters"));
        cli(moved_exe, {L"--cache-root", relocated.native(), L"info", L"vrhino/fixture:v1"}, true, "test_arch");
        cli(moved_exe, {L"--cache-root", relocated.native(), L"component", L"check"}, true, "Media helper ready");
        cli(moved_exe, {L"--cache-root", relocated.native(), L"component", L"install", archive.native()}, true, "Already installed: yes");
        std::cout << "relocated package Unicode/spaces, converter specs, cache and helper without developer PATH PASS\n";

        // Exercise the converter adapter at a sparse 64-bit offset; no model conversion.
        const auto io_path = work / L"文件 io.bin";
        const int fd = io::open(io_path.c_str(), _O_CREAT | _O_EXCL | _O_RDWR | io::O_CLOEXEC);
        require(fd >= 0, "converter wide create");
        try {
            DWORD ignored = 0;
            require(DeviceIoControl(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), FSCTL_SET_SPARSE,
                nullptr, 0, nullptr, 0, &ignored, nullptr) != 0, "sparse offset fixture");
            require(io::write(fd, "abc", 3) == 3, "converter write");
            require(io::pwrite(fd, "xyz", 3, INT64_C(0x100000000)) == 3, "converter 64-bit pwrite");
            char data[3]{}; require(io::pread(fd, data, 3, INT64_C(0x100000000)) == 3 && std::string(data, 3) == "xyz", "converter pread");
            require(io::write(fd, "d", 1) == 1, "positional IO preserved sequential cursor");
            require(io::pread(fd, data, 3, 1) == 3 && std::string(data, 3) == "bcd", "converter cursor isolation");
            struct _stat64 state{}; require(io::fstat(fd, &state) == 0 && state.st_size == INT64_C(0x100000003), "converter 64-bit stat");
            require(io::fsync(fd) == 0, "converter checked flush");
            require(io::pread(fd, data, 3, -1) == -1, "negative offset rejection");
        } catch (...) { io::close(fd); throw; }
        require(io::close(fd) == 0, "converter close"); clean(io_path);
        std::cout << "converter Windows IO wide/binary/cursor/64-bit/flush PASS\n";

        cli(exe, {L"--cache-root", package.native(), L"component", L"install", (work / "invalid.tar.gz").native()}, false);
        const auto before_files = inventory(package); const DWORD before = handles();
        for (int i = 0; i < 24; ++i) {
            cli(exe, {L"--cache-root", package.native(), L"list"});
            cli(exe, {L"--cache-root", package.native(), L"component", L"check"});
            cli(exe, {L"--cache-root", package.native(), L"component", L"install", (work / "invalid.tar.gz").native()}, false);
        }
        const DWORD after = handles(); require(after == before, "CLI invocation handle growth");
        require(inventory(package) == before_files, "CLI temporary artifact growth");
        std::cout << "24 invocation cycles handles " << before << " -> " << after << " stable cache inventory PASS\n";
        fs::current_path(original_cwd);
        // Preserve both package layouts as reviewable artifacts; private work fixtures are evidence.
        std::cout << "WINDOWS_PRODUCT_CLI_TESTS=PASS invocations=" << invocations << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
