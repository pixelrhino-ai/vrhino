#pragma once
#ifdef _WIN32
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vrhino::product::windows_process {
enum class Failure { Missing, Architecture, Permission, Dependency, Launch, Pipe, Cancelled, Timeout };
class Error : public std::runtime_error {
public:
    Error(Failure kind, uint32_t native_code, const std::string& message);
    Failure kind;
    uint32_t native_code;
};
enum class Output { Capture, Null, Inherit };
struct Options {
    // Called on the supervising thread. Empty input signals EOF; callbacks
    // must return promptly so cancellation and timeout can be observed.
    std::function<std::vector<uint8_t>()> input;
    bool require_full_input = false;
    std::function<bool()> cancelled;
    std::chrono::milliseconds timeout{0}; // zero preserves unbounded execution
    Output stdout_mode = Output::Capture;
    Output stderr_mode = Output::Capture;
    std::function<void(const uint8_t*, size_t)> stdout_progress;
    std::function<void(const uint8_t*, size_t)> stderr_progress;
};
struct Result {
    uint32_t exit_code = 0;
    std::vector<uint8_t> standard_output;
    std::string standard_error;
};
std::wstring wide(const std::string& utf8);
std::string utf8(const std::filesystem::path& path);
std::filesystem::path executable_path();
std::filesystem::path discover_helper(); // sibling vrhino-ffmpeg.exe; no PATH search
void check_helper(const std::filesystem::path& executable);
Result run(const std::filesystem::path& executable,
           const std::vector<std::wstring>& arguments, const Options& options = {});
void publish_output(const std::filesystem::path& partial,
                    const std::filesystem::path& destination, bool replace);
#ifdef VRHINO_WINDOWS_PROCESS_TESTING
enum class Point { Created, BeforeResume, Pump };
extern std::function<void(Point)> test_hook;
#endif
}
#endif
