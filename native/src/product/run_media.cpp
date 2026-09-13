#include "run_media.h"
#include "vrhino/product/model_package.h"
#include "vrhino/precision.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstring>
#ifdef _WIN32
#include "vrhino/product/windows_process.h"
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace vrhino::product::run_media_detail {
using Clock = std::chrono::steady_clock;
[[noreturn]] void product_fail(ModelPackageErrorCode code, const std::string& message) {
    throw ModelPackageError(code, message);
}
std::atomic<uint64_t> output_probe_counter = 0;
std::string display_path(const std::filesystem::path& path) {
#ifdef _WIN32
    return windows_process::utf8(path);
#else
    return path.string();
#endif
}
std::filesystem::path absolute_output_path(const std::filesystem::path& output) {
#ifdef _WIN32
    if (output.empty() || output.native().find(L'\0') != std::wstring::npos)
#else
    if (output.empty() || output.string().find('\0') != std::string::npos)
#endif
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output path is empty or contains an invalid character");
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(output, error);
    if (error)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot resolve output path " + display_path(output) + ": " +
                         error.message());
    return absolute.lexically_normal();
}

std::string output_probe_name(const char* suffix) {
    return ".vrhino-output-preflight-" + std::to_string(
#ifdef _WIN32
        GetCurrentProcessId()
#else
        getpid()
#endif
        ) + "-" +
           std::to_string(output_probe_counter.fetch_add(1)) + suffix;
}

void remove_probe(const std::filesystem::path& path) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

void check_output_destination(const std::filesystem::path& requested,
                              const bool overwrite) {
    const std::filesystem::path output = absolute_output_path(requested);
    std::error_code error;
    const std::filesystem::file_status output_status =
        std::filesystem::symlink_status(output, error);
    if (error && error != std::errc::no_such_file_or_directory)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot inspect output path " + display_path(output) + ": " +
                         error.message());
    if (!error && std::filesystem::exists(output_status)) {
        if (!std::filesystem::is_regular_file(output_status) &&
            !std::filesystem::is_symlink(output_status))
            product_fail(ModelPackageErrorCode::OutputInvalid,
                         "output path is not a regular file: " + display_path(output));
        if (!overwrite)
            product_fail(ModelPackageErrorCode::OutputExists,
                         "output already exists; pass --overwrite: " + display_path(output));
    }

    const std::filesystem::path parent = output.parent_path();
    error.clear();
    std::filesystem::create_directories(parent, error);
    if (error)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot create output parent " + display_path(parent) +
                         " for " + display_path(output) + ": " + error.message());
    const std::filesystem::file_status parent_status =
        std::filesystem::status(parent, error);
    if (error || !std::filesystem::is_directory(parent_status))
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output parent is not a directory: " + display_path(parent));

    std::filesystem::path partial = output;
    partial += ".partial";
    const std::filesystem::file_status partial_status =
        std::filesystem::symlink_status(partial, error);
    if (error && error != std::errc::no_such_file_or_directory)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot inspect temporary output path " + display_path(partial) +
                         ": " + error.message());
    if (!error && std::filesystem::exists(partial_status) &&
        !std::filesystem::is_regular_file(partial_status))
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "temporary output path is not a regular file: " +
                         display_path(partial));

    const std::filesystem::path probe = parent / output_probe_name(".partial");
#ifdef _WIN32
    const HANDLE descriptor = CreateFileW(probe.c_str(), GENERIC_WRITE | DELETE, 0, nullptr,
        CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (descriptor == INVALID_HANDLE_VALUE)
        product_fail(ModelPackageErrorCode::OutputInvalid, "output parent is not writable (Windows error " +
            std::to_string(GetLastError()) + ")");
    if (!CloseHandle(descriptor))
        product_fail(ModelPackageErrorCode::OutputInvalid, "cannot close output preflight probe");
#else
    const int descriptor = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                  S_IRUSR | S_IWUSR);
    if (descriptor < 0)
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "output parent is not writable for " + display_path(output) +
                         " (parent " + display_path(parent) + "): " +
                         std::strerror(errno));
    if (::close(descriptor) != 0) {
        const std::string reason = std::strerror(errno);
        remove_probe(probe);
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot close output preflight probe in " + display_path(parent) +
                         ": " + reason);
    }
    if (::unlink(probe.c_str()) != 0) {
        const std::string reason = std::strerror(errno);
        remove_probe(probe);
        product_fail(ModelPackageErrorCode::OutputInvalid,
                     "cannot remove output preflight probe from " + display_path(parent) +
                         ": " + reason);
    }
#endif
}


float load_float(const Tensor& tensor, int64_t index) {
    if (tensor.dtype() == DType::F32) return tensor.data_as<float>()[index];
    if (tensor.dtype() == DType::BF16) {
        const uint32_t bits = static_cast<uint32_t>(tensor.data_as<uint16_t>()[index]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    product_fail(ModelPackageErrorCode::RuntimeError,
                 "finite check requires FP32 or BF16 tensor");
}

void check_media_encoder(const std::filesystem::path& requested) {
#ifdef _WIN32
    try { windows_process::check_helper(requested); }
    catch (const windows_process::Error& error) {
        product_fail(ModelPackageErrorCode::VideoEncodingFailed, error.what());
    }
#else
    if (requested.empty())
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder path is empty; installation is "
                     "incomplete or damaged; inference did not start");
    std::error_code error;
    const std::filesystem::path encoder = std::filesystem::absolute(requested, error);
    if (error || !std::filesystem::is_regular_file(encoder, error) || error)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder is missing: " + requested.string() +
                         "; installation is incomplete or damaged; inference did not start");
    if (::access(encoder.c_str(), X_OK) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder is not executable: " + encoder.string() +
                         "; installation is incomplete or damaged; inference did not start");

    const pid_t child = ::fork();
    if (child < 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot launch bundled VRhino media encoder preflight: " +
                         encoder.string() + "; inference did not start");
    if (child == 0) {
        const int null_fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_fd >= 0) {
            ::dup2(null_fd, STDOUT_FILENO);
            ::dup2(null_fd, STDERR_FILENO);
            ::close(null_fd);
        }
        ::execl(encoder.c_str(), encoder.c_str(), "-version",
                static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled VRhino media encoder failed its readiness check: " +
                         encoder.string() +
                         "; installation is incomplete or damaged; inference did not start");
#endif
}
std::filesystem::path product_encoder_path() {
#ifdef _WIN32
    try { return windows_process::discover_helper(); }
    catch (const windows_process::Error& error) {
        product_fail(ModelPackageErrorCode::VideoEncodingFailed, error.what());
    }
#else
    std::array<char, 4096> executable {};
    const ssize_t length = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length <= 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot locate product executable");
    executable[static_cast<size_t>(length)] = '\0';
    return std::filesystem::path(executable.data()).parent_path() / "vrhino-ffmpeg";
#endif
}

uint8_t video_u8(const Tensor& video, int64_t index, float& minimum, float& maximum,
                 int64_t& nan_count, int64_t& inf_count,
                 const float declared_minimum, const float declared_maximum) {
    const float value = load_float(video, index);
    if (std::isnan(value)) ++nan_count;
    if (std::isinf(value)) ++inf_count;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    if (!std::isfinite(value)) return 0;
    const float normalized = (value - declared_minimum) /
                             (declared_maximum - declared_minimum);
    return static_cast<uint8_t>(std::lround(std::clamp(normalized, 0.0f, 1.0f) * 255.0f));
}

#ifndef _WIN32
void write_all(int fd, const uint8_t* data, size_t bytes) {
    while (bytes > 0) {
        const ssize_t written = write(fd, data, bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0)
            product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                         "video encoder input pipe failed");
        data += written;
        bytes -= static_cast<size_t>(written);
    }
}

#endif

EncodeResult encode_mp4(const Tensor& video, int64_t fps,
                        const std::filesystem::path& encoder,
                        const std::filesystem::path& output, bool overwrite,
                        const std::function<bool()>& cancelled,
                        const float declared_minimum, const float declared_maximum) {
    if (!video.device().is_host() || (video.dtype() != DType::F32 && video.dtype() != DType::BF16) ||
        video.shape().size() != 5 || video.dim(0) != 1 || video.dim(1) != 3)
        product_fail(ModelPackageErrorCode::RuntimeError,
                     "video output must be host FP32/BF16 BCTHW with batch=1 and RGB channels");
    if (fps <= 0) product_fail(ModelPackageErrorCode::PackageInvalid, "fps must be positive");
    if (!std::isfinite(declared_minimum) || !std::isfinite(declared_maximum) ||
        declared_minimum >= declared_maximum)
        product_fail(ModelPackageErrorCode::PackageInvalid,
                     "video output range must contain two increasing finite values");
    if (std::filesystem::exists(output) && !overwrite)
        product_fail(ModelPackageErrorCode::OutputExists,
                     "output already exists; pass --overwrite: " + output.string());
    if (!std::filesystem::exists(encoder))
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled encoder is missing: " + encoder.string());
    std::filesystem::path partial = output;
    partial += ".partial";
    std::error_code remove_error;
    std::filesystem::remove(partial, remove_error);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());

#ifdef _WIN32
    const auto started = Clock::now();
    EncodeResult result;
    bool interrupted = false;
    int status = 0;
    const int64_t frames = video.dim(2), height = video.dim(3), width = video.dim(4);
    int64_t t = 0;
    windows_process::Options process_options;
    process_options.require_full_input = true;
    process_options.cancelled = cancelled;
    process_options.stdout_mode = process_options.stderr_mode = windows_process::Output::Inherit;
    process_options.input = [&]() {
        if (t == frames) return std::vector<uint8_t>{};
        std::vector<uint8_t> frame(static_cast<size_t>(height * width * 3));
        size_t destination = 0;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t source = c * frames * height * width + t * height * width + y * width + x;
                    frame[destination++] = video_u8(video, source, result.minimum, result.maximum,
                        result.nan_count, result.inf_count, declared_minimum, declared_maximum);
                }
            }
        }
        ++t;
        return frame;
    };
    try {
        const auto encoded = windows_process::run(encoder, {L"-hide_banner", L"-loglevel", L"error",
            L"-f", L"rawvideo", L"-pix_fmt", L"rgb24", L"-s:v", std::to_wstring(width) + L"x" + std::to_wstring(height),
            L"-r", std::to_wstring(fps), L"-i", L"pipe:0", L"-an", L"-c:v", L"libx264",
            L"-pix_fmt", L"yuv420p", L"-movflags", L"+faststart", L"-f", L"mp4", partial.native()}, process_options);
        status = encoded.exit_code == 0 ? 0 : 1;
    } catch (const windows_process::Error& error) {
        std::filesystem::remove(partial, remove_error);
        product_fail(error.kind == windows_process::Failure::Cancelled ? ModelPackageErrorCode::Cancelled :
            ModelPackageErrorCode::VideoEncodingFailed, error.what());
    } catch (...) { std::filesystem::remove(partial, remove_error); throw; }
#else
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0)
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot create video encoder pipe");
    const std::string size = std::to_string(video.dim(4)) + "x" +
                             std::to_string(video.dim(3));
    const std::string rate = std::to_string(fps);
    const pid_t child = fork();
    if (child < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "cannot start bundled video encoder");
    }
    if (child == 0) {
        dup2(pipe_fds[0], STDIN_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        execl(encoder.c_str(), encoder.c_str(), "-hide_banner", "-loglevel", "error",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", size.c_str(),
              "-r", rate.c_str(), "-i", "pipe:0", "-an", "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
              partial.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(pipe_fds[0]);
    const auto started = Clock::now();
    EncodeResult result;
    const int64_t frames = video.dim(2), height = video.dim(3), width = video.dim(4);
    std::vector<uint8_t> frame(static_cast<size_t>(height * width * 3));
    bool interrupted = false;
    signal(SIGPIPE, SIG_IGN);
    for (int64_t t = 0; t < frames; ++t) {
        if (cancelled && cancelled()) { interrupted = true; break; }
        size_t destination = 0;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t source = c * frames * height * width +
                                           t * height * width + y * width + x;
                    frame[destination++] = video_u8(video, source, result.minimum,
                        result.maximum, result.nan_count, result.inf_count,
                        declared_minimum, declared_maximum);
                }
            }
        }
        if (!interrupted) write_all(pipe_fds[1], frame.data(), frame.size());
    }
    close(pipe_fds[1]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
#endif
    if (interrupted) {
        std::filesystem::remove(partial, remove_error);
        product_fail(ModelPackageErrorCode::Cancelled, "run cancelled during video encoding");
    }
#ifdef _WIN32
    if (status != 0 ||
#else
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
#endif
        result.nan_count != 0 || result.inf_count != 0 ||
        result.minimum < declared_minimum - 1.0e-6f ||
        result.maximum > declared_maximum + 1.0e-6f) {
        std::filesystem::remove(partial, remove_error);
        if (result.nan_count != 0 || result.inf_count != 0)
            product_fail(ModelPackageErrorCode::RuntimeError,
                         "video output contains NaN or Inf");
        if (result.minimum < declared_minimum - 1.0e-6f ||
            result.maximum > declared_maximum + 1.0e-6f)
            product_fail(ModelPackageErrorCode::RuntimeError,
                         "video output violates its declared range");
        product_fail(ModelPackageErrorCode::VideoEncodingFailed,
                     "bundled FFmpeg failed to encode MP4");
    }
    // POSIX rename replaces an existing regular file atomically. Do not unlink
    // first: --overwrite must never create a window in which the old completed
    // output is absent before the new completed output is published.
#ifdef _WIN32
    try { windows_process::publish_output(partial, output, overwrite); }
    catch (...) { std::filesystem::remove(partial, remove_error); throw; }
#else
    std::filesystem::rename(partial, output);
#endif
    result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    result.bytes = std::filesystem::file_size(output);
    return result;
}

}
