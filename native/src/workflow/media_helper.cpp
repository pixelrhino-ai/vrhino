#include "vrhino/lip_sync_workflow.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <regex>
#include <thread>
#ifdef _WIN32
#include "vrhino/product/windows_process.h"
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace vrhino {
namespace {
void fail(LipSyncStage stage, const std::string& message) { throw LipSyncWorkflowError(stage, message); }
void require_file(const std::filesystem::path& path, LipSyncStage stage) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error))
        fail(stage, "required file is unavailable: " + path.string());
}

struct ChildResult {
    std::vector<uint8_t> standard_output;
    std::string standard_error;
    int status = -1;
};

#ifndef _WIN32
void read_fd(int fd, std::vector<uint8_t>* binary, std::string* text) {
    std::array<uint8_t, 65536> buffer{};
    for (;;) {
        const ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count > 0) {
            if (binary) binary->insert(binary->end(), buffer.begin(), buffer.begin() + count);
            if (text) text->append(reinterpret_cast<const char*>(buffer.data()), count);
        } else if (count == 0) break;
        else if (errno != EINTR) break;
    }
    close(fd);
}
#endif

ChildResult run_helper(const std::filesystem::path& helper,
                       const std::vector<std::string>& arguments,
                       const uint8_t* input, size_t input_bytes,
                       LipSyncStage stage,
                       const std::function<bool()>& cancelled) {
#ifdef _WIN32
    product::windows_process::Options options;
    options.require_full_input = input_bytes != 0;
    options.cancelled = cancelled;
    size_t offset = 0;
    options.input = [&]() {
        const size_t count = std::min<size_t>(65536, input_bytes - offset);
        std::vector<uint8_t> bytes;
        if (count) bytes.assign(input + offset, input + offset + count);
        offset += count; return bytes;
    };
    std::vector<std::wstring> values;
    for (const auto& argument : arguments) values.push_back(product::windows_process::wide(argument));
    try {
        auto result = product::windows_process::run(helper, values, options);
        return {std::move(result.standard_output), std::move(result.standard_error), result.exit_code == 0 ? 0 : 1};
    } catch (const product::windows_process::Error& error) { fail(stage, error.what()); }
    throw std::logic_error("unreachable media launch result");
#else
    require_file(helper, stage);
    int in_pipe[2]{}, out_pipe[2]{}, err_pipe[2]{};
    if (pipe(in_pipe) || pipe(out_pipe) || pipe(err_pipe))
        fail(stage, "cannot create media-helper pipes");
    const pid_t child = fork();
    if (child < 0) fail(stage, "cannot fork media helper");
    if (child == 0) {
        dup2(in_pipe[0], STDIN_FILENO); dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        std::vector<char*> values;
        values.push_back(const_cast<char*>(helper.c_str()));
        for (const auto& argument : arguments)
            values.push_back(const_cast<char*>(argument.c_str()));
        values.push_back(nullptr);
        execv(helper.c_str(), values.data());
        _exit(127);
    }
    close(in_pipe[0]); close(out_pipe[1]); close(err_pipe[1]);
    ChildResult result;
    std::thread output_reader(read_fd, out_pipe[0], &result.standard_output, nullptr);
    std::thread error_reader(read_fd, err_pipe[0], nullptr, &result.standard_error);
    size_t written = 0;
    while (written < input_bytes) {
        if (cancelled && cancelled()) { kill(child, SIGINT); break; }
        const ssize_t count = write(in_pipe[1], input + written, input_bytes - written);
        if (count > 0) written += static_cast<size_t>(count);
        else if (count < 0 && errno == EINTR) continue;
        else break;
    }
    close(in_pipe[1]);
    for (;;) {
        const pid_t waited = waitpid(child, &result.status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) break;
        if (cancelled && cancelled()) kill(child, SIGINT);
        usleep(1000);
    }
    output_reader.join(); error_reader.join();
    if (cancelled && cancelled()) fail(stage, "workflow cancelled");
    return result;
#endif
}

ChildResult run_helper(const std::filesystem::path& helper,
                       const std::vector<std::string>& arguments,
                       LipSyncStage stage,
                       const std::function<bool()>& cancelled = {}) {
    return run_helper(helper, arguments, nullptr, 0, stage, cancelled);
}

bool succeeded(int status) {
#ifdef _WIN32
    return status == 0;
#else
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

std::string media_path(const std::filesystem::path& path) {
#ifdef _WIN32
    return product::windows_process::utf8(path);
#else
    return path.string();
#endif
}

}
LipSyncWorkflowError::LipSyncWorkflowError(LipSyncStage stage,
                                           const std::string& message)
    : std::runtime_error(message), stage_(stage) {}
MediaProbe probe_media_bounded(const std::filesystem::path& helper,
                               const std::filesystem::path& input
#ifdef _WIN32
                               , const std::function<bool()>& cancelled
#endif
                               ) {
    require_file(input, LipSyncStage::MediaInput);
    const auto result = run_helper(helper, {"-hide_banner", "-i", media_path(input),
        "-map", "0", "-c", "copy", "-f", "null", "-"},
        LipSyncStage::MediaInput
#ifdef _WIN32
        , cancelled
#endif
        );
    if (!succeeded(result.status)) fail(LipSyncStage::MediaInput, "media probe failed");
    MediaProbe probe;
    std::smatch match;
    // Pixel-format descriptors may contain color-range/space metadata with
    // commas (for example `bgr0(pc, gbr/unknown/unknown, progressive)`).
    // Bound codec and the leading pixel-format token, then locate dimensions
    // and frame rate on the same stream line.
    const std::regex video(
        R"(Video: ([^,]+), ([^,( ]+)[^\n]* ([0-9]+)x([0-9]+)[^\n]*, ([0-9]+(?:\.[0-9]+)?) fps)");
    if (std::regex_search(result.standard_error, match, video)) {
        probe.video_codec=match[1]; probe.pixel_format=match[2];
        probe.width=std::stoi(match[3]); probe.height=std::stoi(match[4]);
        const double fps=std::stod(match[5]); probe.fps_numerator=static_cast<int32_t>(std::lround(fps*1000)); probe.fps_denominator=1000;
        const int gcd=std::gcd(probe.fps_numerator,probe.fps_denominator); probe.fps_numerator/=gcd;probe.fps_denominator/=gcd;
    }
    const std::regex audio(R"(Audio: ([^,]+), ([0-9]+) Hz, ([^,]+))");
    if (std::regex_search(result.standard_error, match, audio)) {
        probe.has_audio=true;probe.audio_codec=match[1];probe.audio_sample_rate=std::stoi(match[2]);
        const std::string layout=match[3];probe.audio_channels=layout.find("stereo")!=std::string::npos?2:1;
    }
    return probe;
}

VideoInput decode_video_rgb24(const std::filesystem::path& helper,
                              const std::filesystem::path& input,
                              int64_t max_frames,
                              const std::function<bool()>& cancelled) {
    const auto probe=probe_media_bounded(helper,input
#ifdef _WIN32
        ,cancelled
#endif
        );
    if(probe.width<=0||probe.height<=0||probe.fps_numerator<=0)
        fail(LipSyncStage::MediaInput,"video metadata is invalid");
    std::vector<std::string> args={"-hide_banner","-loglevel","error","-i",media_path(input)};
    if(max_frames>0){args.push_back("-frames:v");args.push_back(std::to_string(max_frames));}
    args.insert(args.end(),{"-f","rawvideo","-pix_fmt","rgb24","pipe:1"});
    auto result=run_helper(helper,args,LipSyncStage::MediaInput,cancelled);
    if(!succeeded(result.status))fail(LipSyncStage::MediaInput,"video decode failed: "+result.standard_error);
    const size_t frame_bytes=static_cast<size_t>(probe.width)*probe.height*3;
    if(result.standard_output.empty()||result.standard_output.size()%frame_bytes)
        fail(LipSyncStage::MediaInput,"video frame stream is truncated");
    VideoInput output{probe.width,probe.height,probe.fps_numerator,probe.fps_denominator,{}};
    for(size_t offset=0;offset<result.standard_output.size();offset+=frame_bytes)
        output.frames.push_back({probe.width,probe.height,{result.standard_output.begin()+offset,result.standard_output.begin()+offset+frame_bytes}});
    return output;
}

AudioInput decode_audio_mono_f32_16khz(const std::filesystem::path& helper,
        const std::filesystem::path& input,int64_t max_samples,
        const std::function<bool()>& cancelled){
    require_file(input,LipSyncStage::MediaInput);
    const auto probe=probe_media_bounded(helper,input
#ifdef _WIN32
        ,cancelled
#endif
        );
    if(!probe.has_audio||probe.audio_sample_rate<=0||probe.audio_channels<=0)
        fail(LipSyncStage::MediaInput,"audio metadata is invalid");
    std::vector<std::string> args={"-hide_banner","-loglevel","error"};
    if(max_samples>0){args.push_back("-t");args.push_back(std::to_string(static_cast<double>(max_samples)/16000.0));}
    args.insert(args.end(),{"-i",media_path(input),"-vn"});
    const std::string resample="aresample=16000:resampler=soxr:precision=20";
    // librosa's frozen reference converts decoded PCM to float32, averages the
    // channels, and then invokes SoXR HQ.  Pin the intermediate sample format:
    // letting the helper retain s16 until after pan changes the waveform.
    if(probe.audio_channels==2)
        args.insert(args.end(),{"-af","aformat=sample_fmts=flt,pan=mono|c0=0.5*c0+0.5*c1,"+resample});
    else
        args.insert(args.end(),{"-af",resample});
    args.insert(args.end(),{"-ac","1","-ar","16000"});
    args.insert(args.end(),{"-c:a","pcm_f32le","-f","f32le","pipe:1"});
    auto result=run_helper(helper,args,LipSyncStage::MediaInput,cancelled);
    if(!succeeded(result.status)||result.standard_output.empty()||result.standard_output.size()%4)
        fail(LipSyncStage::MediaInput,"audio decode/resample failed: "+result.standard_error);
    AudioInput output;output.samples.resize(result.standard_output.size()/4);
    std::memcpy(output.samples.data(),result.standard_output.data(),result.standard_output.size());
    if(max_samples>0&&static_cast<int64_t>(output.samples.size())>max_samples)output.samples.resize(max_samples);
    return output;
}

MediaEncodeResult encode_mux_mp4_atomic(const std::filesystem::path& helper,
        const std::vector<RgbFrame>& frames,int32_t fps,
        const std::filesystem::path& audio,const std::filesystem::path& output,
        const std::function<bool()>& cancelled,const bool overwrite,
        const MediaEncodeContract& contract){
    if(frames.empty()||fps<=0)fail(LipSyncStage::MediaOutput,"no output frames");
    if(contract.audio_sample_rate<=0||(contract.audio_channels!=1&&
            contract.audio_channels!=2)||contract.h264_crf>51)
        fail(LipSyncStage::MediaOutput,"invalid bounded media encode contract");
    require_file(audio,LipSyncStage::MediaOutput);
    const int width=frames.front().width,height=frames.front().height;
    std::vector<uint8_t> bytes;bytes.reserve(static_cast<size_t>(width)*height*3*frames.size());
    for(const auto& frame:frames){if(frame.width!=width||frame.height!=height||frame.pixels.size()!=static_cast<size_t>(width)*height*3)fail(LipSyncStage::MediaOutput,"inconsistent output frame");bytes.insert(bytes.end(),frame.pixels.begin(),frame.pixels.end());}
    std::error_code ec;
    if(output.has_parent_path())std::filesystem::create_directories(output.parent_path(),ec);
    if(ec)fail(LipSyncStage::MediaOutput,"cannot create media output directory");
    if(std::filesystem::exists(output,ec)&&!overwrite)fail(LipSyncStage::MediaOutput,"refusing to overwrite published media output");
    std::filesystem::path partial=output;partial+=".partial.mp4";std::filesystem::remove(partial,ec);
    const double duration=static_cast<double>(frames.size())/fps;
    std::vector<std::string> args={"-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s:v",std::to_string(width)+"x"+std::to_string(height),"-r",std::to_string(fps),"-i","pipe:0","-i",media_path(audio),"-t",std::to_string(duration),"-c:v","libx264","-profile:v","high","-pix_fmt","yuv420p"};
    if(contract.h264_crf>=0){args.push_back("-crf");args.push_back(std::to_string(contract.h264_crf));}
    args.insert(args.end(),{"-c:a","aac","-ar",std::to_string(contract.audio_sample_rate),"-ac",std::to_string(contract.audio_channels),"-movflags","+faststart","-f","mp4",media_path(partial)});
#ifdef _WIN32
    ChildResult result;
    try { result=run_helper(helper,args,bytes.data(),bytes.size(),LipSyncStage::MediaOutput,cancelled); }
    catch (...) { std::filesystem::remove(partial,ec); throw; }
#else
    auto result=run_helper(helper,args,bytes.data(),bytes.size(),LipSyncStage::MediaOutput,cancelled);
#endif
    if(!succeeded(result.status)){std::filesystem::remove(partial,ec);fail(LipSyncStage::MediaOutput,"media encode/mux failed: "+result.standard_error);}
#ifdef _WIN32
    try { product::windows_process::publish_output(partial,output,overwrite); }
    catch (...) { std::filesystem::remove(partial,ec); fail(LipSyncStage::MediaOutput,"atomic media publication failed"); }
#else
    std::filesystem::rename(partial,output,ec);if(ec){std::filesystem::remove(partial,ec);fail(LipSyncStage::MediaOutput,"atomic media publication failed");}
#endif
    return {std::filesystem::file_size(output),output};
}

}
