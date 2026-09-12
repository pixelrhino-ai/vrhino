#include "vrhino/lip_sync_workflow.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "vrhino/loader.h"

namespace vrhino {
namespace {

void fail(LipSyncStage stage, const std::string& message) {
    throw LipSyncWorkflowError(stage, message);
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
