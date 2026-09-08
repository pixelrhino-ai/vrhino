#include "vrhino/lip_sync_workflow.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "vrhino/loader.h"

namespace vrhino {
namespace {

void fail(LipSyncStage stage, const std::string& message) {
    throw LipSyncWorkflowError(stage, message);
}

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

ChildResult run_helper(const std::filesystem::path& helper,
                       const std::vector<std::string>& arguments,
                       const uint8_t* input, size_t input_bytes,
                       LipSyncStage stage,
                       const std::function<bool()>& cancelled) {
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
}

ChildResult run_helper(const std::filesystem::path& helper,
                       const std::vector<std::string>& arguments,
                       LipSyncStage stage,
                       const std::function<bool()>& cancelled = {}) {
    return run_helper(helper, arguments, nullptr, 0, stage, cancelled);
}

bool succeeded(int status) {
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int reflect(int value, int size) {
    while (value < 0 || value >= size) {
        if (value < 0) value = -value;
        else value = size * 2 - value - 2;
    }
    return value;
}

double lanczos(double value) {
    value = std::abs(value);
    if (value < 1e-12) return 1.0;
    if (value >= 4.0) return 0.0;
    const double pi = std::acos(-1.0);
    return std::sin(pi * value) * std::sin(pi * value / 4.0) /
           (pi * pi * value * value / 4.0);
}

constexpr int kLanczos4TapCount = 8;

struct Lanczos4CoordinatePlan {
    std::array<int, kLanczos4TapCount> reflected_indices{};
    std::array<double, kLanczos4TapCount> weights{};
};

struct Lanczos4ResizePlan {
    std::vector<Lanczos4CoordinatePlan> x;
    std::vector<Lanczos4CoordinatePlan> y;
};

std::vector<Lanczos4CoordinatePlan> make_lanczos4_coordinate_plan(
        int input_size, int output_size) {
    std::vector<Lanczos4CoordinatePlan> plan(
        static_cast<size_t>(output_size));
    const double scale = static_cast<double>(input_size) / output_size;
    for (int destination = 0; destination < output_size; ++destination) {
        const double source = (destination + 0.5) * scale - 0.5;
        const int base = static_cast<int>(std::floor(source));
        for (int tap = 0; tap < kLanczos4TapCount; ++tap) {
            const int offset = tap - 3;
            plan[static_cast<size_t>(destination)].reflected_indices[tap] =
                reflect(base + offset, input_size);
            plan[static_cast<size_t>(destination)].weights[tap] =
                lanczos(source - (base + offset));
        }
    }
    return plan;
}

Lanczos4ResizePlan make_lanczos4_resize_plan(
        int input_width, int input_height,
        int output_width, int output_height) {
    return {
        make_lanczos4_coordinate_plan(input_width, output_width),
        make_lanczos4_coordinate_plan(input_height, output_height),
    };
}

RgbFrame resize_lanczos(const RgbFrame& input, int output_width, int output_height) {
    RgbFrame output{output_width, output_height,
        std::vector<uint8_t>(static_cast<size_t>(output_width) * output_height * 3)};
    const Lanczos4ResizePlan plan = make_lanczos4_resize_plan(
        input.width, input.height, output_width, output_height);
    for (int y = 0; y < output_height; ++y) {
        const auto& y_plan = plan.y[static_cast<size_t>(y)];
        for (int x = 0; x < output_width; ++x) {
            const auto& x_plan = plan.x[static_cast<size_t>(x)];
            for (int channel = 0; channel < 3; ++channel) {
                double sum = 0.0, total = 0.0;
                for (int ky = 0; ky < kLanczos4TapCount; ++ky) {
                    const double wy = y_plan.weights[ky];
                    for (int kx = 0; kx < kLanczos4TapCount; ++kx) {
                        const double weight = wy * x_plan.weights[kx];
                        const int py = y_plan.reflected_indices[ky];
                        const int px = x_plan.reflected_indices[kx];
                        sum += input.pixels[(static_cast<size_t>(py) * input.width + px) * 3 + channel] * weight;
                        total += weight;
                    }
                }
                output.pixels[(static_cast<size_t>(y) * output_width + x) * 3 + channel] =
                    static_cast<uint8_t>(std::clamp<long>(std::lround(sum / total), 0, 255));
            }
        }
    }
    return output;
}

}  // namespace

LipSyncWorkflowError::LipSyncWorkflowError(LipSyncStage stage,
                                           const std::string& message)
    : std::runtime_error(message), stage_(stage) {}

int64_t lip_sync_output_frame_count(int64_t samples, int32_t rate, int32_t fps) {
    if (samples < 0 || rate <= 0 || fps <= 0)
        fail(LipSyncStage::MediaInput, "invalid audio/frame-rate contract");
    return static_cast<int64_t>((static_cast<__int128>(samples) * fps) / rate);
}

std::vector<int64_t> ping_pong_frame_cycle(int64_t source, int64_t output) {
    if (source <= 0 || output < 0) fail(LipSyncStage::SourcePreparation, "invalid frame cycle");
    if (source > std::numeric_limits<int64_t>::max() / 2)
        fail(LipSyncStage::SourcePreparation, "frame cycle range overflow");
    std::vector<int64_t> result; result.reserve(static_cast<size_t>(output));
    const int64_t cycle = source * 2;
    for (int64_t index = 0; index < output; ++index) {
        const int64_t position = index % cycle;
        result.push_back(position < source ? position : cycle - position - 1);
    }
    return result;
}

SourceFrameDemandPlan plan_source_frame_prefix_demand(
        int64_t source_frame_count,
        const std::vector<int64_t>& referenced_source_indices) {
    if (source_frame_count < 0 ||
        referenced_source_indices.size() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        fail(LipSyncStage::SourcePreparation,
             "invalid source-frame demand range");

    SourceFrameDemandPlan result;
    result.source_frame_count = source_frame_count;
    result.output_frame_count =
        static_cast<int64_t>(referenced_source_indices.size());
    for (const int64_t source_index : referenced_source_indices) {
        if (source_index < 0 || source_index >= source_frame_count)
            fail(LipSyncStage::SourcePreparation,
                 "source-frame demand index is out of range");
        result.maximum_referenced_source_index =
            std::max(result.maximum_referenced_source_index, source_index);
    }
    if (result.maximum_referenced_source_index >= 0)
        result.required_prefix_frame_count =
            result.maximum_referenced_source_index + 1;
    if (result.required_prefix_frame_count > source_frame_count)
        fail(LipSyncStage::SourcePreparation,
             "source-frame demand prefix exceeds source range");
    return result;
}

RngState lip_sync_component_rng(uint64_t seed, int64_t source_index,
                                uint64_t branch) {
    if (source_index < 0) fail(LipSyncStage::SourcePreparation, "negative source index");
    uint64_t value = seed ^ (static_cast<uint64_t>(source_index) + 0x9e3779b97f4a7c15ULL);
    value ^= branch + 0xbf58476d1ce4e5b9ULL + (value << 6) + (value >> 2);
    value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27; value *= 0x94d049bb133111ebULL; value ^= value >> 31;
    return {value, 0, "pytorch_compat.v1"};
}

MediaProbe probe_media_bounded(const std::filesystem::path& helper,
                               const std::filesystem::path& input) {
    require_file(input, LipSyncStage::MediaInput);
    const auto result = run_helper(helper, {"-hide_banner", "-i", input.string(),
        "-map", "0", "-c", "copy", "-f", "null", "-"},
        LipSyncStage::MediaInput);
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
    const auto probe=probe_media_bounded(helper,input);
    if(probe.width<=0||probe.height<=0||probe.fps_numerator<=0)
        fail(LipSyncStage::MediaInput,"video metadata is invalid");
    std::vector<std::string> args={"-hide_banner","-loglevel","error","-i",input.string()};
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
    const auto probe=probe_media_bounded(helper,input);
    if(!probe.has_audio||probe.audio_sample_rate<=0||probe.audio_channels<=0)
        fail(LipSyncStage::MediaInput,"audio metadata is invalid");
    std::vector<std::string> args={"-hide_banner","-loglevel","error"};
    if(max_samples>0){args.push_back("-t");args.push_back(std::to_string(static_cast<double>(max_samples)/16000.0));}
    args.insert(args.end(),{"-i",input.string(),"-vn"});
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
    std::vector<std::string> args={"-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s:v",std::to_string(width)+"x"+std::to_string(height),"-r",std::to_string(fps),"-i","pipe:0","-i",audio.string(),"-t",std::to_string(duration),"-c:v","libx264","-profile:v","high","-pix_fmt","yuv420p"};
    if(contract.h264_crf>=0){args.push_back("-crf");args.push_back(std::to_string(contract.h264_crf));}
    args.insert(args.end(),{"-c:a","aac","-ar",std::to_string(contract.audio_sample_rate),"-ac",std::to_string(contract.audio_channels),"-movflags","+faststart","-f","mp4",partial.string()});
    auto result=run_helper(helper,args,bytes.data(),bytes.size(),LipSyncStage::MediaOutput,cancelled);
    if(!succeeded(result.status)){std::filesystem::remove(partial,ec);fail(LipSyncStage::MediaOutput,"media encode/mux failed: "+result.standard_error);}
    std::filesystem::rename(partial,output,ec);if(ec){std::filesystem::remove(partial,ec);fail(LipSyncStage::MediaOutput,"atomic media publication failed");}
    return {std::filesystem::file_size(output),output};
}

RgbFrame crop_resize_lanczos4(const RgbFrame& source,const std::array<int32_t,4>& box,int32_t ow,int32_t oh){
    if(box[0]<0||box[1]<0||box[2]>source.width||box[3]>source.height||box[2]<=box[0]||box[3]<=box[1])fail(LipSyncStage::SourcePreparation,"crop bbox is invalid");
    RgbFrame crop{box[2]-box[0],box[3]-box[1],std::vector<uint8_t>(static_cast<size_t>(box[2]-box[0])*(box[3]-box[1])*3)};
    for(int y=0;y<crop.height;++y)std::copy_n(source.pixels.begin()+(static_cast<size_t>(box[1]+y)*source.width+box[0])*3,static_cast<size_t>(crop.width)*3,crop.pixels.begin()+static_cast<size_t>(y)*crop.width*3);
    return resize_lanczos(crop,ow,oh);
}

RgbFrame resize_bilinear_rgb(const RgbFrame& source,int32_t ow,int32_t oh){
    if(source.width<=0||source.height<=0||ow<=0||oh<=0||
            source.pixels.size()!=static_cast<size_t>(source.width)*source.height*3)
        fail(LipSyncStage::Composite,"invalid bilinear resize contract");
    constexpr int coefficient_scale=1<<11;
    struct Coefficient{int first=0,low=0,high=0;};
    const auto table=[](int input,int output,bool vertical){
        std::vector<Coefficient> values(static_cast<size_t>(output));
        const double inverse_scale=static_cast<double>(output)/input;
        const double scale=1.0/inverse_scale;
        for(int index=0;index<output;++index){
            float fraction=static_cast<float>((index+.5)*scale-.5);
            int first=static_cast<int>(std::floor(fraction));fraction-=first;
            // OpenCV's horizontal path explicitly collapses out-of-range taps;
            // its vertical path instead clips row pointers after retaining the
            // fractional coefficients.  The distinction is observable at the
            // first and last rows of an upsample.
            if(!vertical&&first<0){first=0;fraction=0;}
            if(!vertical&&first>=input-1){first=input-1;fraction=0;}
            values[static_cast<size_t>(index)]={first,
                static_cast<int>(std::lrint((1.0f-fraction)*coefficient_scale)),
                static_cast<int>(std::lrint(fraction*coefficient_scale))};
        }
        return values;
    };
    const auto horizontal=table(source.width,ow,false),vertical=table(source.height,oh,true);
    std::vector<int32_t> rows(static_cast<size_t>(source.height)*ow*3);
    for(int y=0;y<source.height;++y)for(int x=0;x<ow;++x){const auto& a=horizontal[x];const int x0=std::clamp(a.first,0,source.width-1),x1=std::clamp(a.first+1,0,source.width-1);for(int c=0;c<3;++c)rows[(static_cast<size_t>(y)*ow+x)*3+c]=source.pixels[(static_cast<size_t>(y)*source.width+x0)*3+c]*a.low+source.pixels[(static_cast<size_t>(y)*source.width+x1)*3+c]*a.high;}
    RgbFrame out{ow,oh,std::vector<uint8_t>(static_cast<size_t>(ow)*oh*3)};
    for(int y=0;y<oh;++y){const auto& b=vertical[y];const int y0=std::clamp(b.first,0,source.height-1),y1=std::clamp(b.first+1,0,source.height-1);for(int x=0;x<ow;++x)for(int c=0;c<3;++c){const int32_t s0=rows[(static_cast<size_t>(y0)*ow+x)*3+c],s1=rows[(static_cast<size_t>(y1)*ow+x)*3+c];const int value=((((s0>>4)*b.low)>>16)+(((s1>>4)*b.high)>>16)+2)>>2;out.pixels[(static_cast<size_t>(y)*ow+x)*3+c]=static_cast<uint8_t>(std::clamp(value,0,255));}}
    return out;
}

Tensor normalize_vae_rgb(const RgbFrame& source,bool mask){Tensor out=Tensor::host({1,3,source.height,source.width},DType::F32);for(int c=0;c<3;++c)for(int y=0;y<source.height;++y)for(int x=0;x<source.width;++x){uint8_t v=(mask&&y>=source.height/2)?0:source.pixels[(static_cast<size_t>(y)*source.width+x)*3+c];out.data_as<float>()[((c*source.height)+y)*source.width+x]=v/127.5f-1.0f;}return out;}

RgbFrame vae_rgb_tensor_to_frame(Backend& backend,const Tensor& rgb){auto host=backend.copy_to_host(rgb);if(host.dtype()!=DType::F32||host.ndim()!=4||host.dim(0)!=1||host.dim(1)!=3)fail(LipSyncStage::ComponentExecution,"decoded RGB tensor contract mismatch");RgbFrame out{static_cast<int32_t>(host.dim(3)),static_cast<int32_t>(host.dim(2)),std::vector<uint8_t>(static_cast<size_t>(host.dim(2)*host.dim(3)*3))};for(int y=0;y<out.height;++y)for(int x=0;x<out.width;++x)for(int c=0;c<3;++c){const float value=host.data_as<float>()[((c*out.height)+y)*out.width+x];if(!std::isfinite(value))fail(LipSyncStage::ComponentExecution,"decoded RGB contains non-finite values");out.pixels[(static_cast<size_t>(y)*out.width+x)*3+c]=static_cast<uint8_t>(std::clamp<double>(std::nearbyint(std::clamp(value,0.0f,1.0f)*255.0f),0,255));}return out;}

Tensor concatenate_latent_branches(Backend& backend,const Tensor& masked,const Tensor& full){if(masked.ndim()!=4||full.shape()!=masked.shape()||masked.dim(1)!=4)fail(LipSyncStage::SourcePreparation,"latent branch contract mismatch");return backend.concat({masked,full},1);}

RgbFrame place_generated_crop(const RgbFrame& source,const RgbFrame& generated,const std::array<int32_t,4>& box){RgbFrame output=source;auto resized=resize_bilinear_rgb(generated,box[2]-box[0],box[3]-box[1]);for(int y=0;y<resized.height;++y)for(int x=0;x<resized.width;++x)for(int c=0;c<3;++c)output.pixels[(static_cast<size_t>(box[1]+y)*output.width+box[0]+x)*3+c]=resized.pixels[(static_cast<size_t>(y)*resized.width+x)*3+c];return output;}

RgbFrame composite_rgb_alpha(const RgbFrame& source,const RgbFrame& generated,const AlphaMask& alpha,const std::array<int32_t,4>& box){if(generated.width!=source.width||generated.height!=source.height||alpha.width!=box[2]-box[0]||alpha.height!=box[3]-box[1])fail(LipSyncStage::Composite,"composite coordinate contract mismatch");RgbFrame out=source;for(int y=0;y<alpha.height;++y)for(int x=0;x<alpha.width;++x){const int a=alpha.values[static_cast<size_t>(y)*alpha.width+x];for(int c=0;c<3;++c){const size_t p=(static_cast<size_t>(box[1]+y)*source.width+box[0]+x)*3+c;out.pixels[p]=static_cast<uint8_t>((source.pixels[p]*(255-a)+generated.pixels[p]*a+127)/255);}}return out;}

}  // namespace vrhino
