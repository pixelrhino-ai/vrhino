#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/face_mask.h"
#include "vrhino/loader.h"
#include "vrhino/semantic_segmenter.h"

namespace fs = std::filesystem;
namespace {
void check(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
std::vector<uint8_t> read_u8(const fs::path& path,std::vector<int64_t>* shape=nullptr){
    std::ifstream input(path,std::ios::binary);check(static_cast<bool>(input),"cannot open "+path.string());
    uint8_t prefix[10]{};input.read(reinterpret_cast<char*>(prefix),10);check(input&&std::memcmp(prefix,"\x93NUMPY",6)==0&&prefix[6]==1,"bad NPY");
    const uint16_t n=prefix[8]|(static_cast<uint16_t>(prefix[9])<<8);std::string h(n,'\0');input.read(h.data(),n);check(h.find("'descr': '|u1'")!=std::string::npos,"u8 NPY required");
    const size_t b=h.find('(',h.find("'shape':")),e=h.find(')',b);std::istringstream fields(h.substr(b+1,e-b-1));std::string f;std::vector<int64_t>s;int64_t count=1;
    while(std::getline(fields,f,',')){f.erase(std::remove_if(f.begin(),f.end(),[](unsigned char c){return std::isspace(c);}),f.end());if(!f.empty()){s.push_back(std::stoll(f));count*=s.back();}}
    std::vector<uint8_t> result(static_cast<size_t>(count));input.read(reinterpret_cast<char*>(result.data()),count);check(input&&input.peek()==std::char_traits<char>::eof(),"u8 NPY payload mismatch");if(shape)*shape=s;return result;
}
struct Difference{double max_abs=0,mean_abs=0,cosine=0,ref_min=1e300,ref_max=-1e300,native_min=1e300,native_max=-1e300;uint64_t nan=0,inf=0;};
Difference compare(const vrhino::Tensor&a,const vrhino::Tensor&b){check(a.device().is_host()&&b.device().is_host()&&a.dtype()==vrhino::DType::F32&&b.dtype()==vrhino::DType::F32&&a.shape()==b.shape(),"tensor contract mismatch");Difference d;long double sum=0,dot=0,aa=0,bb=0;for(int64_t i=0;i<a.numel();++i){double x=a.data_as<float>()[i],y=b.data_as<float>()[i];if(std::isnan(x))++d.nan;if(std::isinf(x))++d.inf;d.native_min=std::min(d.native_min,x);d.native_max=std::max(d.native_max,x);d.ref_min=std::min(d.ref_min,y);d.ref_max=std::max(d.ref_max,y);double e=std::abs(x-y);d.max_abs=std::max(d.max_abs,e);sum+=e;dot+=x*y;aa+=x*x;bb+=y*y;}d.mean_abs=static_cast<double>(sum/a.numel());d.cosine=static_cast<double>(dot/std::sqrt(aa*bb));return d;}
Difference qualify(const std::string&name,vrhino::CudaBackend&backend,const vrhino::Tensor&actual,const fs::path&expected,double max_abs,double cosine,bool emit=true){auto d=compare(backend.copy_to_host(actual),vrhino::test::read_npy_f32(expected.string()));if(emit)std::cout<<std::setprecision(10)<<name<<" ref=["<<d.ref_min<<','<<d.ref_max<<"] native=["<<d.native_min<<','<<d.native_max<<"] max_abs="<<d.max_abs<<" mean_abs="<<d.mean_abs<<" cosine="<<d.cosine<<" nan="<<d.nan<<" inf="<<d.inf<<'\n';check(d.nan==0&&d.inf==0&&d.max_abs<=max_abs&&d.cosine>=cosine,name+" FP32 gate failed");return d;}
std::string frame_name(int i){std::ostringstream s;s<<"frame_"<<std::setw(2)<<std::setfill('0')<<i;return s.str();}
std::array<int32_t,4> bbox(int frame){return frame<3?std::array<int32_t,4>{221,276,464,534}:std::array<int32_t,4>{221,280,464,534};}
}  // namespace

int main(int argc,char**argv){
    if(argc!=5){std::cerr<<"usage: semantic_segmenter_oracle_tests VRM FRAME_ROOT ORACLE_ROOT AUTHORITATIVE_FRAME0_ROOT\n";return 2;}
    try{
        vrhino::VrmModel component(argv[1],true);check(component.architecture_id()=="segmenter-2d","component mismatch");
        vrhino::CudaBackend backend;backend.set_execution_dtype(vrhino::DType::F32);backend.enable_weight_cache(true);
        vrhino::SemanticSegmenter2DComponentExecutor executor(backend,vrhino::WeightMap(component.bindings(component.graph())));
        const fs::path frames=argv[2],oracle=argv[3],authoritative=argv[4];
        uint64_t total_label_mismatch=0,total_alpha_mismatch=0;double worst_logits=0,worst_logits_mean=0,worst_cosine=1,worst_alpha=0,worst_alpha_mean=0;int worst_frame=0;double preprocess_ms=0,neural_ms=0,argmax_ms=0,mask_ms=0;
        for(int frame=0;frame<8;++frame){const std::string name=frame_name(frame);std::vector<int64_t>shape;auto bgr=read_u8(frames/(name+".bgr_u8.npy"),&shape);check(shape==std::vector<int64_t>({1216,704,3}),"frame shape mismatch");auto started=std::chrono::steady_clock::now();auto prep=vrhino::preprocess_face_parser_bgr_u8(bgr.data(),1216,704,bbox(frame));preprocess_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();auto reference_input=vrhino::test::read_npy_f32((oracle/name/"input_normalized.npy").string());auto pd=compare(prep.normalized_nchw,reference_input);std::cout<<name<<" preprocessing max_abs="<<pd.max_abs<<" mean_abs="<<pd.mean_abs<<" cosine="<<pd.cosine<<'\n';check(pd.max_abs<=.018&&pd.cosine>=.99998,name+" preprocessing gate failed");
            backend.synchronize();started=std::chrono::steady_clock::now();vrhino::SemanticSegmenter2DObservation observation;auto result=executor.execute(component.graph(),reference_input,frame==0?&observation:nullptr);backend.synchronize();neural_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();if(frame==0){qualify("early_backbone",backend,observation.tensors.at("early_backbone"),oracle/name/"early_backbone.npy",.01,.999999);qualify("context_feature",backend,observation.tensors.at("context_feature"),oracle/name/"context_feature.npy",.01,.999999);qualify("attention_refinement",backend,observation.tensors.at("attention_refinement"),oracle/name/"attention_refinement.npy",.003,.999999);qualify("feature_fusion",backend,observation.tensors.at("feature_fusion"),oracle/name/"feature_fusion.npy",.01,.999999);}
            auto d=qualify(name+".logits",backend,result.logits,oracle/name/"logits.npy",.01,.999999,frame==0);if(d.max_abs>worst_logits){worst_logits=d.max_abs;worst_logits_mean=d.mean_abs;worst_frame=frame;}worst_cosine=std::min(worst_cosine,d.cosine);
            auto host_logits=backend.copy_to_host(result.logits);started=std::chrono::steady_clock::now();auto labels=vrhino::semantic_argmax_first(host_logits);argmax_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();auto ref_labels=read_u8(oracle/name/"labels.npy");uint64_t label_mismatch=0;for(size_t i=0;i<ref_labels.size();++i)label_mismatch+=labels.labels[i]!=ref_labels[i];total_label_mismatch+=label_mismatch;std::cout<<name<<" raw_label_mismatch="<<label_mismatch<<'\n';check(label_mismatch<=128,name+" label-map divergence");
            vrhino::JawMaskObservation stages;started=std::chrono::steady_clock::now();auto alpha=vrhino::build_musetalk_jaw_alpha_mask(labels,bbox(frame),prep.crop_box,90,90,.5,&stages);mask_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();auto compare_u8=[&](const std::vector<uint8_t>&a,const fs::path&p,const std::string&stage){auto b=read_u8(p);check(a.size()==b.size(),"mask shape mismatch");uint64_t mismatch=0;int delta=0;long double sum=0;for(size_t i=0;i<a.size();++i){int e=std::abs(int(a[i])-int(b[i]));mismatch+=e!=0;delta=std::max(delta,e);sum+=e;}std::cout<<name<<' '<<stage<<" mismatch="<<mismatch<<" max_abs="<<delta<<" mean_abs="<<double(sum/a.size())<<'\n';return std::array<double,3>{double(delta),double(sum/a.size()),double(mismatch)};};auto c1=compare_u8(stages.class_one_region,oracle/name/"class_one_region.npy","class-one");auto di=compare_u8(stages.dilated,oracle/name/"dilated.npy","dilation");auto er=compare_u8(stages.eroded,oracle/name/"eroded.npy","erosion");auto se=compare_u8(stages.selected,oracle/name/"selected.npy","selection");auto lo=compare_u8(stages.lower_half,oracle/name/"lower_half.npy","lower-half");check(c1[2]<=32&&di[2]<=32&&er[2]<=16&&se[2]<=16&&lo[2]<=16,name+" mask-stage divergence");auto ad=compare_u8(alpha.values,oracle/name/"alpha.npy","alpha");worst_alpha=std::max(worst_alpha,ad[0]);worst_alpha_mean=std::max(worst_alpha_mean,ad[1]);total_alpha_mismatch+=static_cast<uint64_t>(ad[2]);check(ad[0]<=3&&ad[1]<=.012,name+" alpha blur divergence");std::cout<<name<<" preprocess_max_abs="<<pd.max_abs<<" label_mismatch="<<label_mismatch<<" alpha_max_abs="<<ad[0]<<" alpha_mean_abs="<<ad[1]<<'\n';
            if(frame==0){auto auth=vrhino::test::read_npy_f32((authoritative/"raw_logits.npy").string());auto authd=compare(host_logits,auth);std::cout<<"authoritative_frame0_logits max_abs="<<authd.max_abs<<" mean_abs="<<authd.mean_abs<<" cosine="<<authd.cosine<<'\n';auto auth_labels=read_u8(authoritative/"argmax_19class.npy");uint64_t am=0;for(size_t i=0;i<auth_labels.size();++i)am+=labels.labels[i]!=auth_labels[i];std::cout<<"authoritative_frame0_label_mismatch="<<am<<'\n';auto auth_alpha=read_u8(authoritative.parent_path()/"blend"/"final_alpha.npy");uint64_t aam=0;int aad=0;long double aas=0;for(size_t i=0;i<auth_alpha.size();++i){int e=std::abs(int(alpha.values[i])-int(auth_alpha[i]));aam+=e!=0;aad=std::max(aad,e);aas+=e;}std::cout<<"authoritative_frame0_alpha_max_abs="<<aad<<" mean_abs="<<double(aas/auth_alpha.size())<<" mismatch="<<aam<<'\n';check(am<=16&&aad<=1&&double(aas/auth_alpha.size())<=.05,"authoritative frame0 endpoint gate failed");}
        }
        check(total_label_mismatch<=512&&total_alpha_mismatch<=12000,
              "eight-frame endpoint aggregate gate failed");
        std::cout<<"Semantic segmenter Native FP32 oracle tests: PASS\nworst_logits_frame="<<worst_frame<<"\nworst_logits_max_abs="<<worst_logits<<"\nworst_logits_mean_abs="<<worst_logits_mean<<"\nminimum_logits_cosine="<<worst_cosine<<"\ntotal_label_mismatch="<<total_label_mismatch<<"\nworst_alpha_max_abs="<<worst_alpha<<"\nworst_alpha_mean_abs="<<worst_alpha_mean<<"\ntotal_alpha_mismatch="<<total_alpha_mismatch<<"\npreprocess_eight_ms="<<preprocess_ms<<"\nneural_eight_ms="<<neural_ms<<"\nargmax_eight_ms="<<argmax_ms<<"\nmask_eight_ms="<<mask_ms<<"\ntotal_eight_ms="<<(preprocess_ms+neural_ms+argmax_ms+mask_ms)<<"\npeak_device_bytes="<<backend.peak_device_bytes()<<'\n';return 0;
    }catch(const std::exception&e){std::cerr<<"Semantic segmenter oracle tests: FAIL: "<<e.what()<<'\n';return 1;}
}
