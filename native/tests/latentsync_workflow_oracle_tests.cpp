#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/audio_encoder.h"
#include "vrhino/autoencoder_kl.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/json.h"
#include "vrhino/lip_sync_diffusion_workflow.h"
#include "vrhino/loader.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/product/model_package.h"
#include "vrhino/sampling.h"
#include "vrhino/temporal_conditional_unet_2d.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;

namespace {

void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "cannot read fixture: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

struct Difference {
    double maximum = 0.0;
    double mean = 0.0;
    double cosine = 0.0;
    uint64_t nan = 0;
    uint64_t inf = 0;
};

Difference compare(const vrhino::Tensor& actual,
                   const vrhino::Tensor& reference) {
    check(actual.device().is_host() && reference.device().is_host() &&
              actual.dtype() == vrhino::DType::F32 &&
              reference.dtype() == vrhino::DType::F32 &&
              actual.shape() == reference.shape(),
          "oracle tensor comparison contract mismatch");
    Difference result;
    long double sum=0.0, dot=0.0, left_norm=0.0, right_norm=0.0;
    for (int64_t index=0; index<actual.numel(); ++index) {
        const double left=actual.data_as<float>()[index];
        const double right=reference.data_as<float>()[index];
        result.nan += std::isnan(left);
        result.inf += std::isinf(left);
        const double delta=std::abs(left-right);
        result.maximum=std::max(result.maximum,delta);
        sum+=delta;dot+=left*right;left_norm+=left*left;right_norm+=right*right;
    }
    result.mean=static_cast<double>(sum/actual.numel());
    result.cosine=static_cast<double>(dot/std::sqrt(std::max<long double>(
        left_norm*right_norm,1.0e-30L)));
    return result;
}

Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual,
                   const vrhino::Tensor& reference, double maximum,
                   double cosine) {
    const auto host=actual.device().is_host()?actual:backend.copy_to_host(actual);
    const auto difference=compare(host,reference);
    std::cout<<std::setprecision(10)<<name<<" max_abs="<<difference.maximum
             <<" mean_abs="<<difference.mean<<" cosine="<<difference.cosine
             <<" nan="<<difference.nan<<" inf="<<difference.inf<<'\n';
    check(difference.nan==0&&difference.inf==0&&difference.maximum<=maximum&&
              difference.cosine>=cosine,name+" numerical gate failed");
    return difference;
}

vrhino::Tensor slice_host(const vrhino::Tensor& source, int64_t begin,
                          int64_t end) {
    check(source.device().is_host() && source.dtype()==vrhino::DType::F32 &&
              source.ndim()>=1 && begin>=0 && end<=source.dim(0) && begin<end,
          "host leading-dimension slice contract mismatch");
    auto shape=source.shape();shape[0]=end-begin;
    vrhino::Tensor result=vrhino::Tensor::host(shape,vrhino::DType::F32);
    const size_t row=source.bytes()/static_cast<size_t>(source.dim(0));
    std::memcpy(result.data(),static_cast<const uint8_t*>(source.data())+
        static_cast<size_t>(begin)*row,result.bytes());
    return result;
}

std::vector<float> tensor_values(vrhino::CudaBackend& backend,
                                 const vrhino::Tensor& tensor) {
    const auto host = tensor.device().is_host() ? tensor : backend.copy_to_host(tensor);
    return {host.data_as<float>(), host.data_as<float>() + host.numel()};
}

const std::vector<int64_t>& dwpose_flip_indices() {
    static const std::vector<int64_t> value={
0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};
    return value;
}

vrhino::Tensor retained_reference(const vrhino::Tensor& padded) {
    check(padded.shape()==std::vector<int64_t>({1,5,1500,384}),
          "Whisper padded-state oracle shape mismatch");
    vrhino::Tensor result=vrhino::Tensor::host({1,34,5,384},vrhino::DType::F32);
    for(int position=0;position<34;++position)for(int state=0;state<5;++state)
        std::copy_n(padded.data_as<float>()+(state*1500+position)*384,384,
            result.data_as<float>()+(position*5+state)*384);
    return result;
}

vrhino::Tensor one_state_reference(const vrhino::Tensor& padded,int state) {
    check(padded.shape()==std::vector<int64_t>({1,5,1500,384})&&state>=0&&state<5,
          "Whisper state oracle shape mismatch");
    vrhino::Tensor result=vrhino::Tensor::host({1,34,384},vrhino::DType::F32);
    for(int position=0;position<34;++position)
        std::copy_n(padded.data_as<float>()+(state*1500+position)*384,384,
                    result.data_as<float>()+position*384);
    return result;
}

vrhino::Tensor first_log_mel(const vrhino::Tensor& padded,int64_t frames) {
    check(padded.shape()==std::vector<int64_t>({1,80,3000}),
          "Native log-mel padded shape mismatch");
    vrhino::Tensor result=vrhino::Tensor::host({80,frames},vrhino::DType::F32);
    for(int channel=0;channel<80;++channel)
        std::copy_n(padded.data_as<float>()+channel*3000,frames,
                    result.data_as<float>()+channel*frames);
    return result;
}

vrhino::Tensor zeros_like(const vrhino::Tensor& source) {
    vrhino::Tensor result=vrhino::Tensor::host(source.shape(),vrhino::DType::F32);
    std::fill(result.data_as<float>(),result.data_as<float>()+result.numel(),0.0f);
    return result;
}

class WorkflowDenoiser final : public vrhino::Denoiser {
public:
    WorkflowDenoiser(vrhino::CudaBackend& backend,
        vrhino::TemporalConditionalUNet2DComponentExecutor& executor,
        const vrhino::Json& graph,const vrhino::Tensor& assembled,
        const vrhino::Tensor& audio)
        :backend_(backend),executor_(executor),graph_(graph){
        context_=backend_.slice(assembled,1,4,13);
        vrhino::Tensor batch=backend_.reshape(audio,
            {1,audio.dim(0),audio.dim(1),audio.dim(2)});
        vrhino::Tensor uncond=backend_.copy_to_device(
            zeros_like(batch),vrhino::DType::F32);
        audio_=backend_.concat({uncond,batch},0);
    }
    std::vector<vrhino::Tensor> evaluate(const vrhino::Tensor& latent,
                                         const vrhino::Tensor& timestep) override {
        auto input=backend_.concat({latent,context_},1);
        auto pair=backend_.concat({input,input},0);
        auto result=executor_.execute(graph_,pair,timestep,audio_);
        return backend_.split(result.epsilon,{1,1},0);
    }
private:
    vrhino::CudaBackend& backend_;
    vrhino::TemporalConditionalUNet2DComponentExecutor& executor_;
    const vrhino::Json& graph_;
    vrhino::Tensor context_,audio_;
};

vrhino::SamplingProgram program(int64_t frames) {
    vrhino::SamplingProgram result;
    result.latent_shape={1,4,frames,64,64};result.seed=1247;result.steps=20;
    result.guidance_mode=vrhino::GuidanceMode::CFG;
    result.guidance_coefficients={0.0f,1.5f};
    result.contract.emplace(
        vrhino::PredictionContract{vrhino::PredictionSemantic::Epsilon},
        vrhino::SolverContract{vrhino::SolverSemantic::AffineFirstOrder,1},
        vrhino::make_scaled_linear_ddim_schedule(
            1000,0.00085f,0.012f,20,1,false));
    return result;
}

struct ChunkResult {
    vrhino::Tensor final_latent;
    vrhino::Tensor reconstructed;
    double sampling_seconds=0.0;
    double decode_seconds=0.0;
};

vrhino::SimilarityAffine affine_from_npy(const fs::path& path) {
    const auto matrix=vrhino::test::read_npy_f32(path.string());
    check(matrix.shape()==std::vector<int64_t>({2,3}),
          "affine oracle shape mismatch");
    const float* value=matrix.data_as<float>();
    return {value[0],value[3],value[2],value[5]};
}

std::string frame_name(int frame) {
    std::ostringstream stream;stream<<"frame_"<<std::setw(2)<<std::setfill('0')<<frame;
    return stream.str();
}

vrhino::Tensor generated_audio_reference(const fs::path& capture,
                                          int begin, int end) {
    std::vector<vrhino::Tensor> frames;
    frames.reserve(static_cast<size_t>(end - begin));
    for (int frame = begin; frame < end; ++frame) {
        std::ostringstream name;
        name << "frame_" << std::setw(2) << std::setfill('0') << frame << ".npy";
        frames.push_back(vrhino::test::read_npy_f16_as_f32(
            (capture / "audio/windows" / name.str()).string()));
    }
    vrhino::Tensor result = vrhino::Tensor::host(
        {end - begin, 50, 384}, vrhino::DType::F32);
    for (int frame = begin; frame < end; ++frame) {
        std::memcpy(result.data_as<float>() +
                        static_cast<int64_t>(frame - begin) * 50 * 384,
                    frames[static_cast<size_t>(frame - begin)].data_as<float>(),
                    50 * 384 * sizeof(float));
    }
    return result;
}

}  // namespace

int main(int argc,char**argv){
    if(argc!=13){std::cerr<<"usage: latentsync_workflow_oracle_tests WHISPER_VRM VAE_VRM TEMPORAL_VRM BLAZE_VRM DWPOSE_VRM MEDIA_HELPER VIDEO AUDIO MASK PREPROCESSOR_CONFIG ORACLE_ROOT OUTPUT_MP4\n";return 2;}
    try{
        const fs::path root=argv[11];
        const bool generated_rng=fs::exists(root/"reference/capture/run-manifest.json");
        const fs::path capture=generated_rng?root/"reference/capture":root/"profile17/capture";
        if (!generated_rng) {
            check(vrhino::product::sha256_file(root/"qualification-report.json")==
                "1507b0e27b37f27e21a2ec2b7cecd3442fa276adfad64fb9bccd413accbf8bd4",
                "Phase-1 qualification report identity drift");
        } else {
            check(fs::exists(root/"rng/reference-manifest.json"),
                  "generated Product RNG manifest is missing");
        }
        vrhino::VrmModel whisper(argv[1],true),vae(argv[2],true),temporal(argv[3],true),
            blaze_model(argv[4],true),pose_model(argv[5],true);
        check(whisper.architecture_id()=="audio-encoder"&&
              vae.architecture_id()=="autoencoder-kl"&&
              temporal.architecture_id()=="temporal-unet","component identity mismatch");
        vrhino::CudaBackend backend;backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::AudioEncoderComponentExecutor audio_encoder(backend,
            vrhino::WeightMap(whisper.bindings(whisper.graph())));
        vrhino::AutoencoderKLComponentExecutor autoencoder(backend,
            vrhino::WeightMap(vae.bindings(vae.graph())));
        vrhino::TemporalConditionalUNet2DComponentExecutor unet(backend,
            vrhino::WeightMap(temporal.bindings(temporal.graph())));
        vrhino::VisionDetectorComponentExecutor blaze(backend,
            vrhino::WeightMap(blaze_model.bindings(blaze_model.graph())));
        vrhino::PoseEstimator2DComponentExecutor pose(backend,
            vrhino::WeightMap(pose_model.bindings(pose_model.graph())));

        const auto media_started=std::chrono::steady_clock::now();
        const auto video=vrhino::decode_video_rgb24(argv[6],argv[7],17);
        const auto audio=vrhino::decode_audio_mono_f32_16khz(argv[6],argv[8],10880);
        check(video.frames.size()==17&&video.width==1024&&video.height==1024&&
              audio.samples.size()==10880,"bounded media input mismatch");
        vrhino::Tensor waveform=vrhino::Tensor::host({10880},vrhino::DType::F32);
        std::copy(audio.samples.begin(),audio.samples.end(),waveform.data_as<float>());
        const auto waveform_difference=qualify("waveform",backend,waveform,
            vrhino::test::read_npy_f32((capture/"audio/waveform_f32.npy").string()),0.0,1.0);
        (void)waveform_difference;
        const double media_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-media_started).count();

        const auto face_started=std::chrono::steady_clock::now();
        std::vector<vrhino::RgbFrame> crops; crops.reserve(17);
        std::vector<vrhino::SimilarityAffine> transforms; transforms.reserve(17);
        vrhino::LipSyncDiffusionAlignmentState alignment_state;
        for(int frame=0;frame<17;++frame){
            const auto& rgb=video.frames[static_cast<size_t>(frame)];
            const auto dense_input=vrhino::preprocess_dense_face_detector_rgb_u8(
                rgb.pixels.data(),rgb.height,rgb.width);
            auto dense_tensor=vrhino::Tensor::host({1,3,128,128},vrhino::DType::F32);
            std::copy(dense_input.normalized_nchw.begin(),
                dense_input.normalized_nchw.end(),dense_tensor.data_as<float>());
            const auto dense_output=blaze.execute_dense(blaze_model.graph(),
                backend.copy_to_device(dense_tensor,vrhino::DType::F32));
            const auto stages=vrhino::decode_dense_face_detector({
                tensor_values(backend,dense_output.regressors),
                tensor_values(backend,dense_output.classification_logits)},dense_input);
            check(stages.selected.has_value(),"Native BlazeFace selected no face");
            std::vector<uint8_t> bgr(rgb.pixels.size());
            for(size_t pixel=0;pixel<rgb.pixels.size()/3;++pixel){
                bgr[pixel*3]=rgb.pixels[pixel*3+2];
                bgr[pixel*3+1]=rgb.pixels[pixel*3+1];
                bgr[pixel*3+2]=rgb.pixels[pixel*3];
            }
            const auto preparation=vrhino::preprocess_topdown_pose_bgr_u8(
                bgr.data(),rgb.height,rgb.width);
            auto original=pose.execute(pose_model.graph(),preparation.normalized_nchw);
            auto flipped=vrhino::Tensor::host({1,3,384,288},vrhino::DType::F32);
            for(int channel=0;channel<3;++channel) for(int y=0;y<384;++y)
                for(int x=0;x<288;++x)
                    flipped.data_as<float>()[(channel*384+y)*288+x]=
                        preparation.normalized_nchw.data_as<float>()[
                            (channel*384+y)*288+(287-x)];
            auto mirrored=pose.execute(pose_model.graph(),flipped);
            const auto combined=vrhino::combine_simcc_flip_tta(
                backend.copy_to_host(original.simcc_x),
                backend.copy_to_host(original.simcc_y),
                backend.copy_to_host(mirrored.simcc_x),
                backend.copy_to_host(mirrored.simcc_y),dwpose_flip_indices());
            const auto keypoints=vrhino::decode_simcc_keypoints(
                combined.first,combined.second,preparation.transform);
            const auto aligned=vrhino::align_lip_sync_diffusion_face(
                keypoints,*stages.selected,alignment_state);
            check(aligned.has_value(),"Native DWPose alignment rejected frame");
            transforms.push_back(aligned->smoothed);
            crops.push_back(vrhino::warp_lip_sync_diffusion_face(
                rgb,aligned->smoothed));
        }
        const double face_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-face_started).count();

        const auto audio_started=std::chrono::steady_clock::now();
        const vrhino::Json preprocessor=vrhino::Json::parse(read_text(
            argv[10]));
        const auto log_mel=vrhino::whisper_log_mel_80_variable_audio(
            waveform,preprocessor);
        qualify("log_mel",backend,first_log_mel(log_mel,68),
            vrhino::test::read_npy_f32((capture/"audio/log_mel_80.npy").string()),5e-5,.999999999);
        auto encoded=audio_encoder.execute(whisper.graph(),log_mel);
        const auto padded_states=vrhino::test::read_npy_f16_as_f32(
            (capture/"audio/whisper_five_states_padded.npy").string());
        std::vector<vrhino::Tensor> states;
        int state_index=0;
        for(const char* name:{"frontend_hidden","encoder_block_1","encoder_block_2","encoder_block_3","encoder_block_4"}) {
            states.push_back(encoded.outputs.at(name));
            qualify(std::string("whisper.")+name,backend,
                backend.slice(encoded.outputs.at(name),1,0,34),
                one_state_reference(padded_states,state_index++),.3,.9999);
        }
        auto stacked=vrhino::stack_audio_encoder_states(backend,states);
        auto retained=backend.slice(stacked,1,0,34);
        qualify("whisper_retained",backend,retained,retained_reference(
            padded_states),.3,.9999);
        auto windows=vrhino::assemble_lip_sync_diffusion_audio_windows(
            backend,stacked,10880,17);
        auto windows_host=backend.copy_to_host(windows);
        qualify("audio_chunk0",backend,slice_host(windows_host,0,16),
            generated_rng?generated_audio_reference(capture,0,16):
                vrhino::test::read_npy_f16_as_f32((capture/"audio/chunk0_conditioning.npy").string()),.2,.99999);
        qualify("audio_chunk1",backend,slice_host(windows_host,16,17),
            generated_rng?generated_audio_reference(capture,16,17):
                vrhino::test::read_npy_f16_as_f32((capture/"audio/chunk1_conditioning.npy").string()),.2,.99999);
        const double audio_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-audio_started).count();

        const auto source_started=std::chrono::steady_clock::now();
        const auto raw_mask=vrhino::decode_video_rgb24(argv[6],argv[9],1).frames[0];
        const auto resized_mask=vrhino::prepare_lip_sync_diffusion_fixed_mask(raw_mask);
        auto prepared=vrhino::prepare_lip_sync_diffusion_vae_input(crops,resized_mask);
        qualify("reference_pixels",backend,prepared.reference,
            vrhino::test::read_npy_f32((capture/"vae/reference_pixels_nchw.npy").string()),.25,.999);
        qualify("masked_pixels",backend,prepared.masked,
            vrhino::test::read_npy_f32((capture/"vae/masked_pixels_nchw.npy").string()),.25,.999);
        auto reference_mask=vrhino::test::read_npy_f64_as_f32(
            (capture/"vae/fixed_mask_rgb_f64.npy").string());
        vrhino::Tensor first_mask=vrhino::Tensor::host({1,512,512},vrhino::DType::F32);
        std::copy_n(prepared.mask.data_as<float>(),512*512,first_mask.data_as<float>());
        qualify("fixed_mask",backend,first_mask,slice_host(reference_mask,0,1),.02,.99999);

        std::vector<ChunkResult> chunk_results;
        const auto chunks=vrhino::plan_temporal_frame_chunks(17);
        double source_seconds=0.0;
        for(size_t chunk_index=0;chunk_index<chunks.size();++chunk_index){
            const auto chunk=chunks[chunk_index];
            const std::string prefix="chunk"+std::to_string(chunk_index);
            const auto chunk_start=std::chrono::steady_clock::now();
            auto masked_input=slice_host(prepared.masked,chunk.start,chunk.start+chunk.frames);
            auto reference_input=slice_host(prepared.reference,chunk.start,chunk.start+chunk.frames);
            auto mask_input=slice_host(prepared.mask,chunk.start,chunk.start+chunk.frames);
            auto masked_encoded=autoencoder.encode(vae.graph(),masked_input);
            auto reference_encoded=autoencoder.encode(vae.graph(),reference_input);
            qualify(prefix+".masked_mean",backend,masked_encoded.posterior_mean,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_masked_posterior_mean.npy")).string()),10.0,.995);
            qualify(prefix+".masked_logvar",backend,masked_encoded.posterior_logvar,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_masked_posterior_logvar.npy")).string()),10.0,.995);
            qualify(prefix+".reference_mean",backend,reference_encoded.posterior_mean,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_reference_posterior_mean.npy")).string()),10.0,.995);
            qualify(prefix+".reference_logvar",backend,reference_encoded.posterior_logvar,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_reference_posterior_logvar.npy")).string()),10.0,.995);
            vrhino::Tensor masked_sample_scaled;
            vrhino::Tensor reference_sample_scaled;
            if (generated_rng) {
                std::vector<vrhino::Tensor> masked_latents;
                std::vector<vrhino::Tensor> reference_latents;
                for (int64_t local=0; local<chunk.frames; ++local) {
                    const int64_t source_frame=chunk.start+local;
                    auto masked_rng=vrhino::lip_sync_diffusion_rng(1247,source_frame,
                        vrhino::LipSyncDiffusionRngBranch::MaskedSource);
                    auto reference_rng=vrhino::lip_sync_diffusion_rng(1247,source_frame,
                        vrhino::LipSyncDiffusionRngBranch::ReferenceSource);
                    auto masked_epsilon=backend.rng_normal(
                        masked_rng,{1,4,64,64},vrhino::DType::F32);
                    auto reference_epsilon=backend.rng_normal(
                        reference_rng,{1,4,64,64},vrhino::DType::F32);
                    std::ostringstream frame_name_stream;
                    frame_name_stream<<"frame_"<<std::setw(2)<<std::setfill('0')
                                     <<source_frame<<".npy";
                    qualify(prefix+".masked_epsilon."+std::to_string(local),backend,
                        masked_epsilon,vrhino::test::read_npy_f32(
                            (root/"rng/masked_source"/frame_name_stream.str()).string()),4e-6,.999999999);
                    qualify(prefix+".reference_epsilon."+std::to_string(local),backend,
                        reference_epsilon,vrhino::test::read_npy_f32(
                            (root/"rng/reference_source"/frame_name_stream.str()).string()),4e-6,.999999999);
                    auto masked=autoencoder.sample(vae.graph(),
                        backend.slice(masked_encoded.posterior_mean,0,local,local+1),
                        backend.slice(masked_encoded.posterior_logvar,0,local,local+1),
                        masked_rng,&masked_epsilon);
                    auto reference=autoencoder.sample(vae.graph(),
                        backend.slice(reference_encoded.posterior_mean,0,local,local+1),
                        backend.slice(reference_encoded.posterior_logvar,0,local,local+1),
                        reference_rng,&reference_epsilon);
                    masked_latents.push_back(masked.scaled_latent);
                    reference_latents.push_back(reference.scaled_latent);
                }
                masked_sample_scaled=masked_latents.size()==1?masked_latents[0]:
                    backend.concat(masked_latents,0);
                reference_sample_scaled=reference_latents.size()==1?reference_latents[0]:
                    backend.concat(reference_latents,0);
            } else {
                auto masked_epsilon=vrhino::test::read_npy_f16_as_f32(
                    (capture/"vae"/(prefix+"_masked_posterior_epsilon.npy")).string());
                auto reference_epsilon=vrhino::test::read_npy_f16_as_f32(
                    (capture/"vae"/(prefix+"_reference_posterior_epsilon.npy")).string());
                auto masked_rng=vrhino::lip_sync_diffusion_rng(1247,chunk.start,
                    vrhino::LipSyncDiffusionRngBranch::MaskedSource);
                auto reference_rng=vrhino::lip_sync_diffusion_rng(1247,chunk.start,
                    vrhino::LipSyncDiffusionRngBranch::ReferenceSource);
                masked_sample_scaled=autoencoder.sample(vae.graph(),
                    masked_encoded.posterior_mean,masked_encoded.posterior_logvar,
                    masked_rng,&masked_epsilon).scaled_latent;
                reference_sample_scaled=autoencoder.sample(vae.graph(),
                    reference_encoded.posterior_mean,reference_encoded.posterior_logvar,
                    reference_rng,&reference_epsilon).scaled_latent;
            }
            qualify(prefix+".masked_scaled",backend,masked_sample_scaled,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_masked_sample_scaled.npy")).string()),2.0,.995);
            qualify(prefix+".reference_scaled",backend,reference_sample_scaled,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_reference_sample_scaled.npy")).string()),2.0,.995);
            source_seconds+=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-chunk_start).count();
            const auto expected_input=vrhino::test::read_npy_f16_as_f32(
                (capture/"unet"/(prefix+"_13ch_input.npy")).string());
            vrhino::Tensor initial;
            if (generated_rng) {
                auto noise_rng=vrhino::lip_sync_diffusion_rng(1247,chunk.start,
                    vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
                auto base_noise=backend.rng_normal(
                    noise_rng,{1,4,1,64,64},vrhino::DType::F32);
                qualify(prefix+".base_noise",backend,base_noise,
                    vrhino::test::read_npy_f32((root/"rng/initial_noise"/
                        ("chunk_"+(chunk.start==0?std::string("00"):std::string("16"))+"_base.npy")).string()),4e-6,.999999999);
                std::vector<vrhino::Tensor> repeated_noise(
                    static_cast<size_t>(chunk.frames),base_noise);
                initial=chunk.frames==1?base_noise:backend.concat(repeated_noise,2);
            } else {
                initial=backend.slice(expected_input,1,0,4);
            }
            auto assembled=vrhino::assemble_lip_sync_diffusion_unet_input(
                backend,initial,mask_input,masked_sample_scaled,
                reference_sample_scaled);
            qualify(prefix+".13ch",backend,assembled,expected_input,2.0,.995);
            auto chunk_audio=slice_host(windows_host,chunk.start,chunk.start+chunk.frames);
            WorkflowDenoiser denoiser(backend,unet,temporal.graph(),assembled,chunk_audio);
            vrhino::SamplingRuntime runtime(backend,vrhino::PrecisionPolicy::fp32());
            const auto sampling_started=std::chrono::steady_clock::now();
            auto sampled=runtime.run_with_external_initial_state_for_test(
                denoiser,program(chunk.frames),initial);
            backend.synchronize();
            const double sampling_seconds=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-sampling_started).count();
            qualify(prefix+".final_latent",backend,sampled.final_latent,
                vrhino::test::read_npy_f16_as_f32((capture/"scheduler"/
                    (prefix+(generated_rng?"_final_latent.npy":"_final_latent_scaled.npy"))).string()),2.0,.99);
            const auto decode_started=std::chrono::steady_clock::now();
            auto decoded=autoencoder.decode(vae.graph(),
                vrhino::temporal_latents_to_frame_batch(
                    backend, sampled.final_latent));
            backend.synchronize();
            const double decode_seconds=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-decode_started).count();
            qualify(prefix+".decoder_raw",backend,decoded.decoded,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_decoder_raw.npy")).string()),2.0,.99);
            auto reconstructed=vrhino::reconstruct_lip_sync_diffusion_crop(
                backend,decoded.decoded,reference_input,mask_input);
            qualify(prefix+".reconstructed",backend,reconstructed,
                vrhino::test::read_npy_f16_as_f32((capture/"vae"/(prefix+"_reconstructed_crop.npy")).string()),2.0,.99);
            chunk_results.push_back({backend.copy_to_host(sampled.final_latent),
                backend.copy_to_host(reconstructed),sampling_seconds,decode_seconds});
        }
        const double source_total=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-source_started).count();

        const auto composite_started=std::chrono::steady_clock::now();
        std::vector<vrhino::RgbFrame> final_frames;final_frames.reserve(17);
        uint64_t differing_pixels=0,differing_channels=0,unexplained=0;
        uint64_t upstream_quantization=0,affine_interpolation=0;
        uint64_t composite_rounding=0;
        int maximum_channel_delta=0;long double mean_sum=0.0;
        for(int frame=0;frame<17;++frame){
            const int chunk=frame<16?0:1,index=frame<16?frame:0;
            auto face=slice_host(chunk_results[chunk].reconstructed,index,index+1);
            face=face.reshape({3,512,512});
            const std::string name=frame_name(frame);
            const auto transform=transforms[static_cast<size_t>(frame)];
            vrhino::LipSyncDiffusionCompositeObservation observation;
            auto result=vrhino::composite_lip_sync_diffusion_face(
                video.frames[frame],face,transform,&observation);
            const auto [shape,reference]=vrhino::test::read_npy_u8(
                (capture/"frames/raw"/(name+
                    (generated_rng?"_rgb_u8.npy":".rgb_u8.npy"))).string());
            check(shape==std::vector<int64_t>({1024,1024,3}),
                  "final RGB oracle shape mismatch");
            uint64_t frame_pixels=0,frame_channels=0,frame_unexplained=0;
            uint64_t frame_upstream=0,frame_affine=0,frame_composite=0;
            int frame_maximum=0;long double frame_sum=0.0;
            for(size_t pixel=0;pixel<1024ULL*1024;++pixel){
                bool changed=false;int pixel_maximum=0;
                for(int channel=0;channel<3;++channel){
                    const size_t offset=pixel*3+channel;
                    const int delta=std::abs(static_cast<int>(result.pixels[offset])-
                                             reference[offset]);
                    changed|=delta!=0;differing_channels+=delta!=0;
                    frame_channels+=delta!=0;
                    maximum_channel_delta=std::max(maximum_channel_delta,delta);
                    frame_maximum=std::max(frame_maximum,delta);
                    pixel_maximum=std::max(pixel_maximum,delta);
                    mean_sum+=delta;frame_sum+=delta;
                }
                differing_pixels+=changed;frame_pixels+=changed;
                if(changed&&observation.soft_mask[pixel]==0.0f){
                    bool reference_changed=false;
                    for(int channel=0;channel<3;++channel)
                        reference_changed|=reference[pixel*3+channel]!=
                            video.frames[frame].pixels[pixel*3+channel];
                    if(reference_changed){++affine_interpolation;++frame_affine;}
                    else{++unexplained;++frame_unexplained;}
                }else if(changed&&pixel_maximum<=1){
                    ++composite_rounding;++frame_composite;
                }else if(changed){
                    ++upstream_quantization;++frame_upstream;
                }
            }
            final_frames.push_back(std::move(result));
            std::cout<<std::setprecision(10)<<name
                     <<" chunk="<<chunk<<" local_index="<<index
                     <<" differing_pixels="<<frame_pixels
                     <<" differing_channels="<<frame_channels
                     <<" max_delta="<<frame_maximum
                     <<" mean_delta="<<static_cast<double>(frame_sum/(1024.0*1024*3))
                     <<" upstream_fp32_vae="<<frame_upstream
                     <<" affine_interpolation="<<frame_affine
                     <<" composite_rounding="<<frame_composite
                     <<" unexplained="<<frame_unexplained<<'\n';
        }
        const double composite_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-composite_started).count();
        check(unexplained==0,"final RGB contains unexplained differences");
        std::error_code error;fs::remove(argv[12],error);
        const auto encode_started=std::chrono::steady_clock::now();
        const auto encoded_mp4=vrhino::encode_mux_mp4_atomic(
            argv[6],final_frames,25,argv[8],argv[12],{},false,{16000,1,18});
        const double encode_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-encode_started).count();
        const auto probe=vrhino::probe_media_bounded(argv[6],argv[12]);
        const auto backcheck=vrhino::decode_video_rgb24(argv[6],argv[12],17);
        check(probe.video_codec.find("h264")!=std::string::npos&&
              probe.pixel_format=="yuv420p"&&probe.audio_codec.find("aac")!=std::string::npos&&
              probe.audio_sample_rate==16000&&probe.audio_channels==1&&
              backcheck.frames.size()==17,"bundled media backcheck failed");
        std::cout<<std::setprecision(10)
                 <<"LatentSync bounded workflow oracle: PASS\n"
                 <<"final_differing_pixels="<<differing_pixels<<'\n'
                 <<"final_differing_channels="<<differing_channels<<'\n'
                 <<"final_max_channel_delta="<<maximum_channel_delta<<'\n'
                 <<"final_mean_channel_delta="<<static_cast<double>(mean_sum/(17.0*1024*1024*3))<<'\n'
                 <<"attribution_upstream_fp32_vae="<<upstream_quantization<<'\n'
                 <<"attribution_affine_interpolation="<<affine_interpolation<<'\n'
                 <<"attribution_composite_rounding="<<composite_rounding<<'\n'
                 <<"final_unexplained="<<unexplained<<'\n'
                 <<"media_seconds="<<media_seconds<<'\n'
                 <<"face_analysis_alignment_seconds="<<face_seconds<<'\n'
                 <<"audio_seconds="<<audio_seconds<<'\n'
                 <<"source_total_seconds="<<source_total<<'\n'
                 <<"vae_source_seconds="<<source_seconds<<'\n'
                 <<"chunk0_sampling_seconds="<<chunk_results[0].sampling_seconds<<'\n'
                 <<"chunk1_sampling_seconds="<<chunk_results[1].sampling_seconds<<'\n'
                 <<"vae_decode_seconds="<<(chunk_results[0].decode_seconds+chunk_results[1].decode_seconds)<<'\n'
                 <<"composite_seconds="<<composite_seconds<<'\n'
                 <<"encode_seconds="<<encode_seconds<<'\n'
                 <<"mp4_bytes="<<encoded_mp4.bytes<<'\n'
                 <<"mp4_sha256="<<vrhino::product::sha256_file(argv[12])<<'\n'
                 <<"peak_device_bytes="<<backend.peak_device_bytes()<<'\n';
        return 0;
    }catch(const std::exception&error){std::cerr<<"LatentSync bounded workflow oracle: FAIL: "<<error.what()<<'\n';return 1;}
}
