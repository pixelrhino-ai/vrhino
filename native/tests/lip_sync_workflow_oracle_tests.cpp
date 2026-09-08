#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/audio_conditioning.h"
#include "vrhino/audio_encoder.h"
#include "vrhino/autoencoder_kl.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/conditional_unet_2d.h"
#include "vrhino/face_mask.h"
#include "vrhino/face_roi.h"
#include "vrhino/lip_sync_workflow.h"
#include "vrhino/product/vrm_verification.h"
#include "vrhino/loader.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/pose_geometry.h"
#include "vrhino/semantic_segmenter.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;
namespace {

void check(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
std::string frame_name(int frame){std::ostringstream s;s<<"frame_"<<std::setw(2)<<std::setfill('0')<<frame;return s.str();}
std::string read_text(const fs::path& path){std::ifstream input(path,std::ios::binary);check(static_cast<bool>(input),"cannot read "+path.string());return {std::istreambuf_iterator<char>(input),{}};}

template<class T> struct Npy { std::vector<int64_t> shape; std::vector<T> values; };
template<class T>Npy<T> read_npy(const fs::path& path,const std::string& descriptor){
    std::ifstream input(path,std::ios::binary);check(static_cast<bool>(input),"cannot open "+path.string());
    uint8_t prefix[10]{};input.read(reinterpret_cast<char*>(prefix),10);check(input&&std::memcmp(prefix,"\x93NUMPY",6)==0&&prefix[6]==1,"unsupported NPY");
    const uint16_t n=prefix[8]|(static_cast<uint16_t>(prefix[9])<<8);std::string header(n,'\0');input.read(header.data(),n);check(header.find("'descr': '"+descriptor+"'")!=std::string::npos&&header.find("'fortran_order': False")!=std::string::npos,"NPY contract mismatch");
    const auto begin=header.find('(',header.find("'shape':")),end=header.find(')',begin);check(begin!=std::string::npos&&end!=std::string::npos,"NPY shape missing");
    Npy<T> result;std::istringstream fields(header.substr(begin+1,end-begin-1));std::string field;int64_t count=1;
    while(std::getline(fields,field,',')){field.erase(std::remove_if(field.begin(),field.end(),[](unsigned char c){return std::isspace(c);}),field.end());if(!field.empty()){result.shape.push_back(std::stoll(field));count*=result.shape.back();}}
    result.values.resize(static_cast<size_t>(count));input.read(reinterpret_cast<char*>(result.values.data()),static_cast<std::streamsize>(result.values.size()*sizeof(T)));check(input&&input.peek()==std::char_traits<char>::eof(),"NPY payload mismatch");return result;
}

struct Difference { double max_abs=0,mean_abs=0,cosine=0;uint64_t nan=0,inf=0; };
Difference compare(const vrhino::Tensor& actual,const vrhino::Tensor& expected){
    check(actual.device().is_host()&&expected.device().is_host()&&actual.dtype()==vrhino::DType::F32&&expected.dtype()==vrhino::DType::F32&&actual.shape()==expected.shape(),"tensor comparison contract mismatch");
    Difference d;long double sum=0,dot=0,aa=0,bb=0;for(int64_t i=0;i<actual.numel();++i){double a=actual.data_as<float>()[i],b=expected.data_as<float>()[i];d.nan+=std::isnan(a);d.inf+=std::isinf(a);double e=std::abs(a-b);d.max_abs=std::max(d.max_abs,e);sum+=e;dot+=a*b;aa+=a*a;bb+=b*b;}d.mean_abs=static_cast<double>(sum/actual.numel());d.cosine=static_cast<double>(dot/std::sqrt(aa*bb));return d;
}
Difference qualify(const std::string& name,vrhino::CudaBackend& backend,const vrhino::Tensor& actual,const fs::path& expected,double max_abs,double cosine,bool emit=true){auto host=actual.device().is_host()?actual:backend.copy_to_host(actual);auto d=compare(host,vrhino::test::read_npy_f32(expected.string()));if(emit)std::cout<<std::setprecision(10)<<name<<" max_abs="<<d.max_abs<<" mean_abs="<<d.mean_abs<<" cosine="<<d.cosine<<" nan="<<d.nan<<" inf="<<d.inf<<'\n';check(d.nan==0&&d.inf==0&&d.max_abs<=max_abs&&d.cosine>=cosine,name+" FP32 gate failed");return d;}

std::vector<float> tensor_values(const vrhino::Tensor& value){return {value.data_as<float>(),value.data_as<float>()+value.numel()};}
std::vector<vrhino::DetectorScaleHost> host_scales(vrhino::CudaBackend& backend,const vrhino::VisionDetectorResult& result){std::vector<vrhino::DetectorScaleHost> scales;for(const auto& scale:result.scales){auto c=backend.copy_to_host(scale.confidence_logits),l=backend.copy_to_host(scale.localization);scales.push_back({c.dim(0),c.dim(2),c.dim(3),tensor_values(c),tensor_values(l)});}return scales;}
const std::vector<int64_t>& flip_indices(){static const std::vector<int64_t> v={0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};return v;}

std::vector<uint8_t> rgb_to_bgr(const vrhino::RgbFrame& frame){auto out=frame.pixels;for(size_t i=0;i<out.size();i+=3)std::swap(out[i],out[i+2]);return out;}
vrhino::Tensor detector_input(const vrhino::RgbFrame& frame){auto values=vrhino::preprocess_detector_rgb_u8(frame.pixels.data(),frame.height,frame.width);vrhino::Tensor output=vrhino::Tensor::host({1,3,frame.height,frame.width},vrhino::DType::F32);std::copy(values.begin(),values.end(),output.data_as<float>());return output;}
vrhino::Tensor horizontal_flip(const vrhino::Tensor& input){vrhino::Tensor out=vrhino::Tensor::host(input.shape(),vrhino::DType::F32);for(int c=0;c<3;++c)for(int y=0;y<input.dim(2);++y)for(int x=0;x<input.dim(3);++x)out.data_as<float>()[(c*input.dim(2)+y)*input.dim(3)+x]=input.data_as<float>()[(c*input.dim(2)+y)*input.dim(3)+(input.dim(3)-1-x)];return out;}
vrhino::Tensor branch(const vrhino::Tensor& eight,int channel){vrhino::Tensor out=vrhino::Tensor::host({1,4,32,32},vrhino::DType::F32);std::copy_n(eight.data_as<float>()+channel*32*32,4*32*32,out.data_as<float>());return out;}
vrhino::Tensor effective_epsilon(vrhino::CudaBackend& backend,const vrhino::Tensor& mean,const vrhino::Tensor& logvar,const vrhino::Tensor& expected_scaled){auto m=backend.copy_to_host(mean),l=backend.copy_to_host(logvar);vrhino::Tensor out=vrhino::Tensor::host(m.shape(),vrhino::DType::F32);for(int64_t i=0;i<m.numel();++i)out.data_as<float>()[i]=(expected_scaled.data_as<float>()[i]/.18215f-m.data_as<float>()[i])/std::exp(.5f*l.data_as<float>()[i]);return out;}

struct PixelDifference { int maximum=0;double mean=0;uint64_t pixels=0,channels=0,outside_alpha=0,outside_alpha_neighborhood=0; };
PixelDifference compare_frame(const vrhino::RgbFrame& actual,const Npy<uint8_t>& expected,const std::vector<uint8_t>* alpha,const std::array<int32_t,4>& alpha_box){check(expected.shape==std::vector<int64_t>({actual.height,actual.width,3})&&expected.values.size()==actual.pixels.size(),"RGB reference contract mismatch");PixelDifference d;long double sum=0;for(int y=0;y<actual.height;++y)for(int x=0;x<actual.width;++x){bool pixel=false;for(int c=0;c<3;++c){size_t p=(static_cast<size_t>(y)*actual.width+x)*3+c;int e=std::abs(int(actual.pixels[p])-int(expected.values[p]));d.maximum=std::max(d.maximum,e);sum+=e;d.channels+=e!=0;pixel|=e!=0;}if(pixel){++d.pixels;bool exact=false,near=false;if(alpha)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){const int ax=x-alpha_box[0]+dx,ay=y-alpha_box[1]+dy;if(ax>=0&&ax<alpha_box[2]-alpha_box[0]&&ay>=0&&ay<alpha_box[3]-alpha_box[1]&&(*alpha)[static_cast<size_t>(ay)*(alpha_box[2]-alpha_box[0])+ax]!=0){near=true;if(dx==0&&dy==0)exact=true;}}if(!exact)++d.outside_alpha;if(!near)++d.outside_alpha_neighborhood;}}d.mean=static_cast<double>(sum/actual.pixels.size());return d;}

vrhino::RgbFrame frame_from_npy(const Npy<uint8_t>& value) {
    check(value.shape.size()==3&&value.shape[2]==3,"RGB NPY contract mismatch");
    return {static_cast<int32_t>(value.shape[1]),static_cast<int32_t>(value.shape[0]),value.values};
}

vrhino::RgbFrame quantize_rgb_0_1(const vrhino::Tensor& value) {
    check(value.device().is_host()&&value.dtype()==vrhino::DType::F32&&
          value.shape()==std::vector<int64_t>({1,3,256,256}),
          "decoded RGB float contract mismatch");
    vrhino::RgbFrame result{256,256,std::vector<uint8_t>(256*256*3)};
    for(int y=0;y<256;++y)for(int x=0;x<256;++x)for(int c=0;c<3;++c){
        const float sample=value.data_as<float>()[((c*256)+y)*256+x];
        check(std::isfinite(sample),"decoded RGB contains non-finite values");
        result.pixels[(static_cast<size_t>(y)*256+x)*3+c]=static_cast<uint8_t>(
            std::clamp<double>(std::nearbyint(std::clamp(sample,0.0f,1.0f)*255.0f),0,255));
    }
    return result;
}

struct FinalRgbAttribution {
    uint64_t differing_pixels=0;
    uint64_t differing_channels=0;
    uint64_t alpha_edge=0;
    uint64_t upstream_quantization=0;
    uint64_t unexplained=0;
    int maximum=0;
    int alpha_maximum=0;
    int upstream_maximum=0;
    double mean=0;
};

FinalRgbAttribution attribute_final_rgb(
        const vrhino::RgbFrame& authoritative,
        const vrhino::RgbFrame& reference_replay,
        const vrhino::RgbFrame& upstream_replay,
        const vrhino::RgbFrame& native,
        int frame_index) {
    check(authoritative.width==reference_replay.width&&
          authoritative.height==reference_replay.height&&
          authoritative.pixels==reference_replay.pixels,
          "frozen resize/composite replay differs from authoritative frame");
    check(authoritative.width==upstream_replay.width&&
          authoritative.height==upstream_replay.height&&
          authoritative.width==native.width&&
          authoritative.height==native.height,
          "attribution frame shape mismatch");
    FinalRgbAttribution result;
    long double sum=0;
    for(int y=0;y<native.height;++y)for(int x=0;x<native.width;++x){
        bool differs=false,alpha_cause=false,upstream_cause=false;
        for(int c=0;c<3;++c){
            const size_t p=(static_cast<size_t>(y)*native.width+x)*3+c;
            const int delta=std::abs(int(native.pixels[p])-int(authoritative.pixels[p]));
            result.maximum=std::max(result.maximum,delta);sum+=delta;
            if(delta){++result.differing_channels;differs=true;}
            alpha_cause|=native.pixels[p]!=upstream_replay.pixels[p];
            upstream_cause|=upstream_replay.pixels[p]!=reference_replay.pixels[p];
        }
        if(!differs)continue;
        ++result.differing_pixels;
        if(alpha_cause){++result.alpha_edge;for(int c=0;c<3;++c){const size_t p=(static_cast<size_t>(y)*native.width+x)*3+c;result.alpha_maximum=std::max(result.alpha_maximum,std::abs(int(native.pixels[p])-int(authoritative.pixels[p])));}}
        else if(upstream_cause){++result.upstream_quantization;for(int c=0;c<3;++c){const size_t p=(static_cast<size_t>(y)*native.width+x)*3+c;result.upstream_maximum=std::max(result.upstream_maximum,std::abs(int(native.pixels[p])-int(authoritative.pixels[p])));}}
        else ++result.unexplained;
        if(frame_index==0&&x==308&&(y==463||y==464))
            std::cout<<"trace.pixel x="<<x<<" y="<<y
                     <<" reference="<<int(authoritative.pixels[(static_cast<size_t>(y)*native.width+x)*3])
                     <<","<<int(authoritative.pixels[(static_cast<size_t>(y)*native.width+x)*3+1])
                     <<","<<int(authoritative.pixels[(static_cast<size_t>(y)*native.width+x)*3+2])
                     <<" upstream_replay="<<int(upstream_replay.pixels[(static_cast<size_t>(y)*native.width+x)*3])
                     <<","<<int(upstream_replay.pixels[(static_cast<size_t>(y)*native.width+x)*3+1])
                     <<","<<int(upstream_replay.pixels[(static_cast<size_t>(y)*native.width+x)*3+2])
                     <<" native="<<int(native.pixels[(static_cast<size_t>(y)*native.width+x)*3])
                     <<","<<int(native.pixels[(static_cast<size_t>(y)*native.width+x)*3+1])
                     <<","<<int(native.pixels[(static_cast<size_t>(y)*native.width+x)*3+2])<<'\n';
    }
    result.mean=static_cast<double>(sum/native.pixels.size());
    return result;
}

void trace_generated_support(const vrhino::Tensor& native_float,
        const vrhino::Tensor& reference_float,
        const vrhino::RgbFrame& native_u8,
        const vrhino::RgbFrame& reference_u8,
        const std::array<int32_t,4>& box,int output_x,int output_y) {
    const int relative_x=output_x-box[0],relative_y=output_y-box[1];
    const int output_width=box[2]-box[0],output_height=box[3]-box[1];
    const double source_x=(relative_x+.5)*native_u8.width/output_width-.5;
    const double source_y=(relative_y+.5)*native_u8.height/output_height-.5;
    const int x0=static_cast<int>(std::floor(source_x));
    const int y0=static_cast<int>(std::floor(source_y));
    std::cout<<std::setprecision(17)<<"trace.support output="<<output_x<<','<<output_y
             <<" relative="<<relative_x<<','<<relative_y
             <<" source="<<source_x<<','<<source_y<<'\n';
    for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
        const int x=std::clamp(x0+dx,0,native_u8.width-1),y=std::clamp(y0+dy,0,native_u8.height-1);
        std::cout<<"  input="<<x<<','<<y<<" reference_float=";
        for(int c=0;c<3;++c)std::cout<<(c?",":"")<<reference_float.data_as<float>()[((c*reference_u8.height)+y)*reference_u8.width+x];
        std::cout<<" native_float=";
        for(int c=0;c<3;++c)std::cout<<(c?",":"")<<native_float.data_as<float>()[((c*native_u8.height)+y)*native_u8.width+x];
        std::cout<<" reference_x255=";
        for(int c=0;c<3;++c)std::cout<<(c?",":"")<<reference_float.data_as<float>()[((c*reference_u8.height)+y)*reference_u8.width+x]*255.0f;
        std::cout<<" native_x255=";
        for(int c=0;c<3;++c)std::cout<<(c?",":"")<<native_float.data_as<float>()[((c*native_u8.height)+y)*native_u8.width+x]*255.0f;
        const size_t p=(static_cast<size_t>(y)*native_u8.width+x)*3;
        std::cout<<" reference_u8="<<int(reference_u8.pixels[p])<<','<<int(reference_u8.pixels[p+1])<<','<<int(reference_u8.pixels[p+2])
                 <<" native_u8="<<int(native_u8.pixels[p])<<','<<int(native_u8.pixels[p+1])<<','<<int(native_u8.pixels[p+2])<<'\n';
    }
}

}  // namespace

int main(int argc,char** argv){
    if(argc!=17){std::cerr<<"usage: lip_sync_workflow_oracle_tests WHISPER VAE UNET S3FD DWPOSE PARSER PREPROCESSOR HELPER VIDEO AUDIO PHASE1 PHASE2D PHASE2E PHASE2F PHASE3 OUTPUT_MP4\n";return 2;}
    try{
        const fs::path phase1=argv[11],phase2d=argv[12],phase2e=argv[13],phase2f=argv[14],phase3=argv[15];
        const fs::path official=phase1/"oracle/official-2/tensors",inputs=phase1/"oracle/input-oracles/tensors";
        const std::array<vrhino::ComponentIdentityContract,6> contracts={{{"audio_encoder_transformer","audio-encoder",32866496,"4527260f6727d202f0a964ff062d68bd03a00b6647e0809b18ba318afb76777d"},{"autoencoder_kl","autoencoder-kl",334727552,"49d6d814b8546535f36580543987ff0477ebd89040cc053d2a8f836b68db9b1e"},{"conditional_unet_2d","cond-unet-2d",3400098560ULL,"4fc8954eea557b636abe22450076b38ddea8a629fa38f9ab545014b1c1d22c47"},{"vision_detector_multiscale","vision-detector",89861568,"0d8086e0293f17cbf2b2db59f22e3bec8cb6760d97fe986394b2f6f89ade70e6"},{"pose_estimator_2d","pose-estimator",134774400,"3f457d8ff6eb49098d9626e232785b7158d8838e8054eb1f85d0684decc6a7c5"},{"semantic_segmenter_2d","segmenter-2d",52709376,"25ea9ae79a4c81b718ce2b47d42d72e0da242553c7f64c4b2d92942b885d0a3e"}}};
        const auto load_component=[&](const int argument,const int contract_index){const auto& value=contracts[contract_index];return vrhino::product::load_verified_vrm_component(argv[argument],{value.semantic_name,value.architecture,value.bytes,value.sha256});};
        auto whisper_storage=load_component(1,0),vae_storage=load_component(2,1),unet_storage=load_component(3,2),s3fd_storage=load_component(4,3),dwpose_storage=load_component(5,4),parser_storage=load_component(6,5);
        vrhino::VrmModel& whisper=*whisper_storage;vrhino::VrmModel& vae=*vae_storage;vrhino::VrmModel& unet=*unet_storage;vrhino::VrmModel& s3fd=*s3fd_storage;vrhino::VrmModel& dwpose=*dwpose_storage;vrhino::VrmModel& parser=*parser_storage;
        auto media_started=std::chrono::steady_clock::now();auto video=vrhino::decode_video_rgb24(argv[8],argv[9],8);auto audio=vrhino::decode_audio_mono_f32_16khz(argv[8],argv[10],5120);double media_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-media_started).count();
        check(video.frames.size()==8&&video.width==704&&video.height==1216&&video.fps_numerator==25&&video.fps_denominator==1,"real video contract mismatch");check(audio.samples.size()==5120,"real audio contract mismatch");
        double waveform_max=0,waveform_mean=0;auto waveform_ref=vrhino::test::read_npy_f32((inputs/"audio/waveform_mono_16khz.npy").string());vrhino::Tensor waveform=vrhino::Tensor::host({5120},vrhino::DType::F32);for(int i=0;i<5120;++i){waveform.data_as<float>()[i]=audio.samples[i];double e=std::abs(audio.samples[i]-waveform_ref.data_as<float>()[i]);waveform_max=std::max(waveform_max,e);waveform_mean+=e;}waveform_mean/=5120;check(waveform_max<=2e-5,"bundled audio boundary diverges");
        for(int frame=0;frame<8;++frame){auto ref=read_npy<uint8_t>(inputs/"media"/(frame_name(frame)+".bgr_u8.npy"),"|u1");auto bgr=rgb_to_bgr(video.frames[frame]);check(ref.values==bgr,"bundled video boundary diverges");}

        vrhino::CudaBackend backend;backend.set_execution_dtype(vrhino::DType::F32);backend.enable_weight_cache(true);
        vrhino::AudioEncoderComponentExecutor audio_encoder(backend,vrhino::WeightMap(whisper.bindings(whisper.graph())));vrhino::AutoencoderKLComponentExecutor autoencoder(backend,vrhino::WeightMap(vae.bindings(vae.graph())));vrhino::ConditionalUNet2DComponentExecutor conditional_unet(backend,vrhino::WeightMap(unet.bindings(unet.graph())));vrhino::VisionDetectorComponentExecutor detector(backend,vrhino::WeightMap(s3fd.bindings(s3fd.graph())));vrhino::PoseEstimator2DComponentExecutor pose(backend,vrhino::WeightMap(dwpose.bindings(dwpose.graph())));vrhino::SemanticSegmenter2DComponentExecutor segmenter(backend,vrhino::WeightMap(parser.bindings(parser.graph())));
        auto audio_started=std::chrono::steady_clock::now();auto mel=vrhino::whisper_log_mel_80(waveform,vrhino::Json::parse(read_text(argv[7])));auto mel_d=qualify("real_media.log_mel",backend,mel,official/"audio/whisper_log_mel.npy",3e-2,.99999999);check(mel_d.mean_abs<=2e-6,"real-media log-mel aggregate gate failed");auto encoded_audio=audio_encoder.execute(whisper.graph(),mel);std::vector<vrhino::Tensor> states;for(const char* name:{"frontend_hidden","encoder_block_1","encoder_block_2","encoder_block_3","encoder_block_4"})states.push_back(encoded_audio.outputs.at(name));auto stacked=vrhino::stack_audio_encoder_states(backend,states);auto windowed=vrhino::frame_audio_feature_windows(backend,stacked,5120,8);auto conditioning=vrhino::add_sinusoidal_position_encoding(backend,windowed);backend.synchronize();double audio_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-audio_started).count();
        double face_ms=0,source_ms=0,unet_ms=0,decode_ms=0,parser_ms=0,composite_ms=0;std::vector<vrhino::RgbFrame> final_frames;FinalRgbAttribution worst_rgb;int worst_frame=0,worst_alpha_rgb=0,worst_upstream_rgb=0;uint64_t total_pixels=0,total_channels=0,total_alpha_edge=0,total_upstream_quantization=0,total_unexplained=0,total_reference_alpha_repairs=0;double worst_unet=0,worst_latent=0,worst_conditioning=0,worst_decoded_float=0;uint64_t total_label_mismatch=0;int worst_alpha=0;double worst_alpha_mean=0;
        vrhino::Tensor timestep=vrhino::Tensor::host({1},vrhino::DType::I64);timestep.data_as<int64_t>()[0]=0;
        for(int frame=0;frame<8;++frame){const std::string name=frame_name(frame);auto source=video.frames[frame];auto bgr=rgb_to_bgr(source);
            auto started=std::chrono::steady_clock::now();auto detector_result=detector.execute(s3fd.graph(),detector_input(source));auto detection=vrhino::decode_multiscale_face_detector(host_scales(backend,detector_result));check(detection.selected.has_value(),name+" detector failed");auto prep=vrhino::preprocess_topdown_pose_bgr_u8(bgr.data(),source.height,source.width);auto original=pose.execute(dwpose.graph(),prep.normalized_nchw);auto mirrored=pose.execute(dwpose.graph(),horizontal_flip(prep.normalized_nchw));auto ox=backend.copy_to_host(original.simcc_x),oy=backend.copy_to_host(original.simcc_y),fx=backend.copy_to_host(mirrored.simcc_x),fy=backend.copy_to_host(mirrored.simcc_y);auto tta=vrhino::combine_simcc_flip_tta(ox,oy,fx,fy,flip_indices());auto keypoints=vrhino::decode_simcc_keypoints(tta.first,tta.second,prep.transform);auto geometry=vrhino::select_musetalk_face_geometry(keypoints,detection.selected,source.height,0,10);check(geometry.valid&&!geometry.used_fallback,name+" face geometry mismatch");face_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            auto face_ref=read_npy<int32_t>(official/"face"/(name+".face_points_68_int.npy"),"<i4");for(int i=0;i<68;++i)check(geometry.facial_landmarks[i][0]==face_ref.values[i*2]&&geometry.facial_landmarks[i][1]==face_ref.values[i*2+1],name+" face points mismatch");
            started=std::chrono::steady_clock::now();auto crop=vrhino::crop_resize_lanczos4(source,geometry.crop_bbox,256,256);if(frame==0){auto crop_ref=read_npy<uint8_t>(phase3/"frame_00_crop_256.rgb_u8.npy","|u1");auto crop_diff=compare_frame(crop,crop_ref,nullptr,{});std::cout<<"frame_00.crop max="<<crop_diff.maximum<<" mean="<<crop_diff.mean<<" pixels="<<crop_diff.pixels<<'\n';check(crop_diff.maximum<=2&&crop_diff.mean<=.05,"Lanczos crop boundary mismatch");}
            auto masked=autoencoder.encode(vae.graph(),vrhino::normalize_vae_rgb(crop,true));auto full=autoencoder.encode(vae.graph(),vrhino::normalize_vae_rgb(crop,false));auto expected_input=vrhino::test::read_npy_f32((official/"vae"/(name+".unet_input_8ch.npy")).string());auto expected_masked=branch(expected_input,0),expected_full=branch(expected_input,4);auto eps_masked=effective_epsilon(backend,masked.posterior_mean,masked.posterior_logvar,expected_masked),eps_full=effective_epsilon(backend,full.posterior_mean,full.posterior_logvar,expected_full);auto rng0=vrhino::lip_sync_component_rng(11001,frame,0),rng1=vrhino::lip_sync_component_rng(11001,frame,1);auto sampled_masked=autoencoder.sample(vae.graph(),masked.posterior_mean,masked.posterior_logvar,rng0,&eps_masked);auto sampled_full=autoencoder.sample(vae.graph(),full.posterior_mean,full.posterior_logvar,rng1,&eps_full);auto latent=vrhino::concatenate_latent_branches(backend,sampled_masked.scaled_latent,sampled_full.scaled_latent);auto latent_d=compare(backend.copy_to_host(latent),expected_input);worst_latent=std::max(worst_latent,latent_d.max_abs);check(latent_d.max_abs<=5e-5&&latent_d.cosine>=.999999,name+" source latent assembly mismatch");source_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            auto one_conditioning=backend.slice(conditioning,0,frame,frame+1);auto conditioning_d=compare(backend.copy_to_host(one_conditioning),vrhino::test::read_npy_f32((official/"audio"/(name+".position_encoded_conditioning.npy")).string()));worst_conditioning=std::max(worst_conditioning,conditioning_d.max_abs);std::cout<<name<<" conditioning_max="<<conditioning_d.max_abs<<" mean="<<conditioning_d.mean_abs<<" cosine="<<conditioning_d.cosine<<'\n';check(conditioning_d.max_abs<=2e-2&&conditioning_d.mean_abs<=2e-4&&conditioning_d.cosine>=.999999,name+" conditioning mismatch");
            backend.synchronize();started=std::chrono::steady_clock::now();auto predicted=conditional_unet.execute(unet.graph(),latent,timestep,one_conditioning);backend.synchronize();unet_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();auto unet_d=qualify(name+".unet",backend,predicted.predicted_latent,official/"unet"/(name+".output.npy"),7e-5,.999999,true);worst_unet=std::max(worst_unet,unet_d.max_abs);
            started=std::chrono::steady_clock::now();auto decoded=autoencoder.decode(vae.graph(),predicted.predicted_latent);backend.synchronize();decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();auto decoded_host=backend.copy_to_host(decoded.rgb_0_1);auto reference_float=vrhino::test::read_npy_f32((official/"vae_decode"/(name+".rgb_0_1.npy")).string());auto decoded_d=compare(decoded_host,reference_float);worst_decoded_float=std::max(worst_decoded_float,decoded_d.max_abs);check(decoded_d.max_abs<=3e-5&&decoded_d.cosine>=.999999,name+" decoded FP32 boundary mismatch");auto generated=quantize_rgb_0_1(decoded_host);auto reference_generated=quantize_rgb_0_1(reference_float);auto captured_reference_generated=read_npy<uint8_t>(phase3/"phase3a/reference_generated"/(name+".rgb_u8.npy"),"|u1");check(reference_generated.pixels==captured_reference_generated.values,"reference float quantization replay mismatch");if(frame==0){auto generated_ref=read_npy<uint8_t>(phase3/"generated_frame_00_crop.rgb_u8.npy","|u1");auto generated_diff=compare_frame(generated,generated_ref,nullptr,{});check(reference_generated.pixels==generated_ref.values,"reference generated fixture mismatch");std::cout<<"frame_00.generated max="<<generated_diff.maximum<<" mean="<<generated_diff.mean<<" pixels="<<generated_diff.pixels<<" decoded_float_max="<<decoded_d.max_abs<<" decoded_float_mean="<<decoded_d.mean_abs<<'\n';check(generated_diff.maximum<=1&&generated_diff.mean<=.01,"generated crop boundary mismatch");trace_generated_support(decoded_host,reference_float,generated,reference_generated,geometry.crop_bbox,308,463);trace_generated_support(decoded_host,reference_float,generated,reference_generated,geometry.crop_bbox,308,464);}
            started=std::chrono::steady_clock::now();auto parser_prep=vrhino::preprocess_face_parser_bgr_u8(bgr.data(),source.height,source.width,geometry.crop_bbox);auto prep_d=compare(parser_prep.normalized_nchw,vrhino::test::read_npy_f32((phase2f/name/"input_normalized.npy").string()));if(frame==0)std::cout<<name<<" parser_preprocess_max="<<prep_d.max_abs<<" mean="<<prep_d.mean_abs<<" cosine="<<prep_d.cosine<<'\n';auto logits=segmenter.execute(parser.graph(),parser_prep.normalized_nchw);auto labels=vrhino::semantic_argmax_first(backend.copy_to_host(logits.logits));auto ref_labels=read_npy<uint8_t>(phase2f/name/"labels.npy","|u1");uint64_t label_mismatch=0;for(size_t i=0;i<labels.labels.size();++i)label_mismatch+=labels.labels[i]!=ref_labels.values[i];total_label_mismatch+=label_mismatch;std::cout<<name<<" label_mismatch="<<label_mismatch<<'\n';check(label_mismatch<=128,name+" parser labels mismatch");auto alpha=vrhino::build_musetalk_jaw_alpha_mask(labels,geometry.crop_bbox,parser_prep.crop_box,90,90,.5);auto ref_alpha=read_npy<uint8_t>(phase2f/name/"alpha.npy","|u1");int alpha_max=0;long double alpha_sum=0;std::vector<uint8_t> alpha_difference(alpha.values.size());for(size_t i=0;i<alpha.values.size();++i){int e=std::abs(int(alpha.values[i])-int(ref_alpha.values[i]));alpha_max=std::max(alpha_max,e);alpha_sum+=e;alpha_difference[i]=static_cast<uint8_t>(e);}worst_alpha=std::max(worst_alpha,alpha_max);worst_alpha_mean=std::max(worst_alpha_mean,static_cast<double>(alpha_sum/alpha.values.size()));std::cout<<name<<" alpha_max="<<alpha_max<<" mean="<<static_cast<double>(alpha_sum/alpha.values.size())<<'\n';check(alpha_max<=3&&alpha_sum/alpha.values.size()<=.012,name+" alpha mismatch");parser_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            started=std::chrono::steady_clock::now();auto placed=vrhino::place_generated_crop(source,generated,geometry.crop_bbox);auto reference_placed=vrhino::place_generated_crop(source,reference_generated,geometry.crop_bbox);auto opencv_resized=read_npy<uint8_t>(phase3/"phase3a/reference_resized"/(name+".rgb_u8.npy"),"|u1");auto native_reference_resized=vrhino::resize_bilinear_rgb(reference_generated,geometry.crop_bbox[2]-geometry.crop_bbox[0],geometry.crop_bbox[3]-geometry.crop_bbox[1]);check(native_reference_resized.pixels==opencv_resized.values,"Native bilinear resize differs from frozen OpenCV output");auto effective_alpha=read_npy<uint8_t>(phase3/"phase3a/effective_alpha"/(name+".npy"),"|u1");auto reference_alpha=alpha;reference_alpha.values=effective_alpha.values;uint64_t alpha_repairs=0;int effective_alpha_delta=0;for(size_t i=0;i<reference_alpha.values.size();++i){const int delta=std::abs(int(reference_alpha.values[i])-int(ref_alpha.values[i]));alpha_repairs+=delta!=0;effective_alpha_delta=std::max(effective_alpha_delta,delta);}check(effective_alpha_delta<=3,"effective authoritative alpha exceeds Phase-2F gate");total_reference_alpha_repairs+=alpha_repairs;auto composite=vrhino::composite_rgb_alpha(source,placed,alpha,parser_prep.crop_box);auto upstream_replay=vrhino::composite_rgb_alpha(source,placed,reference_alpha,parser_prep.crop_box);auto reference_replay=vrhino::composite_rgb_alpha(source,reference_placed,reference_alpha,parser_prep.crop_box);auto reference=read_npy<uint8_t>(phase3/"reference_rgb"/(name+".rgb_u8.npy"),"|u1");auto authoritative=frame_from_npy(reference);auto rgb=attribute_final_rgb(authoritative,reference_replay,upstream_replay,composite,frame);if(rgb.mean>worst_rgb.mean){worst_rgb=rgb;worst_frame=frame;}worst_alpha_rgb=std::max(worst_alpha_rgb,rgb.alpha_maximum);worst_upstream_rgb=std::max(worst_upstream_rgb,rgb.upstream_maximum);total_pixels+=rgb.differing_pixels;total_channels+=rgb.differing_channels;total_alpha_edge+=rgb.alpha_edge;total_upstream_quantization+=rgb.upstream_quantization;total_unexplained+=rgb.unexplained;final_frames.push_back(std::move(composite));composite_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();std::cout<<name<<" rgb_max="<<rgb.maximum<<" mean="<<rgb.mean<<" pixels="<<rgb.differing_pixels<<" channels="<<rgb.differing_channels<<" alpha_edge="<<rgb.alpha_edge<<" upstream_quantization="<<rgb.upstream_quantization<<" unexplained="<<rgb.unexplained<<" reference_alpha_repairs="<<alpha_repairs<<'\n';check(rgb.upstream_maximum<=1&&rgb.unexplained==0,name+" final RGB contains unexplained parity differences");
        }
        backend.synchronize();const auto encode_started=std::chrono::steady_clock::now();auto encoded=vrhino::encode_mux_mp4_atomic(argv[8],final_frames,25,argv[10],argv[16]);double encode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-encode_started).count();auto probe=vrhino::probe_media_bounded(argv[8],argv[16]);check(probe.width==704&&probe.height==1216&&probe.fps_numerator==25&&probe.fps_denominator==1&&probe.has_audio&&probe.audio_codec.find("aac")!=std::string::npos&&probe.audio_sample_rate==48000&&probe.audio_channels==2,"Native MP4 stream contract mismatch");auto backcheck=vrhino::decode_video_rgb24(argv[8],argv[16],8);check(backcheck.frames.size()==8,"Native MP4 decode backcheck failed");bool overwrite_refused=false;try{(void)vrhino::encode_mux_mp4_atomic(argv[8],final_frames,25,argv[10],argv[16]);}catch(const vrhino::LipSyncWorkflowError& error){overwrite_refused=error.stage()==vrhino::LipSyncStage::MediaOutput;}check(overwrite_refused&&std::filesystem::file_size(argv[16])==encoded.bytes,"atomic media output overwrite guard failed");
        check(total_unexplained==0,"RGB drift contains unexplained pixels");std::cout<<"Bounded Native lip-sync workflow FP32 qualification: PASS\nwaveform_max_abs="<<waveform_max<<"\nwaveform_mean_abs="<<waveform_mean<<"\nworst_conditioning_max_abs="<<worst_conditioning<<"\nworst_latent_max_abs="<<worst_latent<<"\nworst_unet_max_abs="<<worst_unet<<"\nworst_decoded_float_max_abs="<<worst_decoded_float<<"\ntotal_label_mismatch="<<total_label_mismatch<<"\nworst_alpha_max_abs="<<worst_alpha<<"\nworst_alpha_mean_abs="<<worst_alpha_mean<<"\nreference_alpha_repair_pixels="<<total_reference_alpha_repairs<<"\nworst_rgb_frame="<<worst_frame<<"\nworst_rgb_max_abs="<<worst_rgb.maximum<<"\nworst_rgb_mean_abs="<<worst_rgb.mean<<"\nworst_alpha_rgb_max_abs="<<worst_alpha_rgb<<"\nworst_upstream_rgb_max_abs="<<worst_upstream_rgb<<"\ntotal_rgb_differing_pixels="<<total_pixels<<"\ntotal_rgb_differing_channels="<<total_channels<<"\nrgb_alpha_edge_pixels="<<total_alpha_edge<<"\nrgb_upstream_quantization_pixels="<<total_upstream_quantization<<"\nrgb_unexplained_pixels="<<total_unexplained<<"\nmedia_decode_ms="<<media_ms<<"\naudio_conditioning_ms="<<audio_ms<<"\nface_analysis_ms="<<face_ms<<"\nsource_vae_precompute_ms="<<source_ms<<"\nunet_ms="<<unet_ms<<"\nvae_decode_ms="<<decode_ms<<"\nparser_mask_ms="<<parser_ms<<"\ncomposite_ms="<<composite_ms<<"\nmedia_encode_ms="<<encode_ms<<"\noutput_bytes="<<encoded.bytes<<"\npeak_device_bytes="<<backend.peak_device_bytes()<<'\n';return 0;
    }catch(const std::exception& error){std::cerr<<"Bounded Native lip-sync workflow FP32 qualification: FAIL: "<<error.what()<<'\n';return 1;}
}
