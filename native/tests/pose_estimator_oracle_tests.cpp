#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
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
#include "vrhino/face_roi.h"
#include "vrhino/loader.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/pose_geometry.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;
namespace {

template <typename T> struct NpyValues {
    std::vector<int64_t> shape;
    std::vector<T> values;
};

template <typename T>
NpyValues<T> read_npy(const fs::path& path, const std::string& descriptor) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open NPY fixture: " + path.string());
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    if (!input || std::memcmp(prefix, "\x93NUMPY", 6) != 0 ||
        prefix[6] != 1 || prefix[7] != 0)
        throw std::runtime_error("unsupported NPY header: " + path.string());
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
        (static_cast<uint16_t>(prefix[9]) << 8);
    std::string header(header_size, '\0'); input.read(header.data(), header.size());
    if (!input || header.find("'descr': '" + descriptor + "'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("NPY dtype/order mismatch: " + path.string());
    const size_t key=header.find("'shape':"), begin=header.find('(',key),
                 end=header.find(')',begin);
    if(key==std::string::npos||begin==std::string::npos||end==std::string::npos)
        throw std::runtime_error("NPY shape missing: " + path.string());
    NpyValues<T> result;
    std::istringstream fields(header.substr(begin+1,end-begin-1));
    std::string field;
    int64_t elements=1;
    while(std::getline(fields,field,',')) {
        field.erase(std::remove_if(field.begin(),field.end(),
            [](unsigned char value){return std::isspace(value);}),field.end());
        if(!field.empty()){const int64_t d=std::stoll(field);result.shape.push_back(d);elements*=d;}
    }
    result.values.resize(static_cast<size_t>(elements));
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(result.values.size()*sizeof(T)));
    if(!input||input.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("NPY payload mismatch: " + path.string());
    return result;
}

struct Difference {
    double max_abs=0,mean_abs=0,cosine=0;
    double reference_min=std::numeric_limits<double>::infinity();
    double reference_max=-std::numeric_limits<double>::infinity();
    double native_min=std::numeric_limits<double>::infinity();
    double native_max=-std::numeric_limits<double>::infinity();
    uint64_t nan=0,inf=0;
};
void check(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
Difference compare(const vrhino::Tensor& actual,const vrhino::Tensor& expected){
    check(actual.device().is_host()&&expected.device().is_host()&&
          actual.dtype()==vrhino::DType::F32&&expected.dtype()==vrhino::DType::F32&&
          actual.shape()==expected.shape(),"oracle tensor contract mismatch");
    Difference d;long double sum=0,dot=0,an=0,bn=0;
    for(int64_t i=0;i<actual.numel();++i){double a=actual.data_as<float>()[i],b=expected.data_as<float>()[i];
        if(std::isnan(a))++d.nan;if(std::isinf(a))++d.inf;
        d.native_min=std::min(d.native_min,a);d.native_max=std::max(d.native_max,a);
        d.reference_min=std::min(d.reference_min,b);d.reference_max=std::max(d.reference_max,b);
        double e=std::abs(a-b);d.max_abs=std::max(d.max_abs,e);sum+=e;dot+=a*b;an+=a*a;bn+=b*b;}
    d.mean_abs=static_cast<double>(sum/actual.numel());
    d.cosine=static_cast<double>(dot/std::sqrt(an*bn));return d;
}
Difference qualify(const std::string& name,vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual,const fs::path& expected,
                   double max_abs,double cosine,bool emit=true){
    auto d=compare(backend.copy_to_host(actual),vrhino::test::read_npy_f32(expected.string()));
    if(emit)std::cout<<std::setprecision(10)<<name<<" ref=["<<d.reference_min<<','<<d.reference_max
        <<"] native=["<<d.native_min<<','<<d.native_max<<"] max_abs="<<d.max_abs
        <<" mean_abs="<<d.mean_abs<<" cosine="<<d.cosine<<" nan="<<d.nan<<" inf="<<d.inf<<'\n';
    check(d.nan==0&&d.inf==0&&d.max_abs<=max_abs&&d.cosine>=cosine,
          name+" exceeds FP32 gate");return d;
}
std::string frame_name(int frame){std::ostringstream s;s<<"frame_"<<std::setw(2)<<std::setfill('0')<<frame;return s.str();}
std::vector<float> values(const vrhino::Tensor& tensor){return {tensor.data_as<float>(),tensor.data_as<float>()+tensor.numel()};}
std::vector<vrhino::DetectorScaleHost> host_scales(vrhino::CudaBackend& backend,const vrhino::VisionDetectorResult& result){
    std::vector<vrhino::DetectorScaleHost> scales;for(const auto& scale:result.scales){auto c=backend.copy_to_host(scale.confidence_logits),l=backend.copy_to_host(scale.localization);scales.push_back({c.dim(0),c.dim(2),c.dim(3),values(c),values(l)});}return scales;
}
const std::vector<int64_t>& flip_indices(){static const std::vector<int64_t> v={
0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};return v;}
}  // namespace

int main(int argc,char** argv){
    if(argc!=7){std::cerr<<"usage: pose_estimator_oracle_tests POSE_VRM S3FD_VRM INPUT_BGR_ROOT DWPOSE_ORACLE_ROOT S3FD_ORACLE_ROOT PHASE1_FACE_ROOT\n";return 2;}
    try{
        vrhino::VrmModel pose_component(argv[1],true),detector_component(argv[2],true);
        check(pose_component.architecture_id()=="pose-estimator","pose component mismatch");
        check(detector_component.architecture_id()=="vision-detector","detector component mismatch");
        vrhino::CudaBackend backend;backend.set_execution_dtype(vrhino::DType::F32);backend.enable_weight_cache(true);
        vrhino::PoseEstimator2DComponentExecutor pose(backend,vrhino::WeightMap(pose_component.bindings(pose_component.graph())));
        vrhino::VisionDetectorComponentExecutor detector(backend,vrhino::WeightMap(detector_component.bindings(detector_component.graph())));
        const fs::path inputs=argv[3],oracle=argv[4],s3fd_oracle=argv[5],phase1=argv[6];
        const std::array<std::array<int32_t,4>,8> expected_base={{{221,276,464,524},{221,276,464,524},{221,276,464,524},{221,280,464,524},{221,280,464,524},{221,280,464,524},{221,280,464,524},{221,280,464,524}}};
        double preprocess_ms=0,neural_ms=0,decode_ms=0,geometry_ms=0,s3fd_integration_ms=0;
        double frame0_preprocess_ms=0,frame0_neural_ms=0,frame0_decode_ms=0,
               frame0_geometry_ms=0;
        double worst_x=0,worst_y=0,worst_conf=0,worst_face=0;
        int worst_point=0,worst_frame=0,worst_conf_point=0,worst_conf_frame=0;
        for(int frame=0;frame<8;++frame){
            const std::string name=frame_name(frame);const fs::path root=oracle/name;
            auto bgr=read_npy<uint8_t>(inputs/(name+".bgr_u8.npy"),"|u1");
            check(bgr.shape==std::vector<int64_t>({1216,704,3}),"source BGR shape mismatch");
            auto start=std::chrono::steady_clock::now();
            auto prep=vrhino::preprocess_topdown_pose_bgr_u8(bgr.values.data(),1216,704);
            const double current_preprocess=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            preprocess_ms+=current_preprocess;if(frame==0)frame0_preprocess_ms=current_preprocess;
            auto warped=read_npy<uint8_t>(root/"warped_bgr_u8.npy","|u1");
            check(prep.warped_bgr==warped.values,name+" affine warp differs from reference");
            auto input0=vrhino::test::read_npy_f32((root/"network_input_0.npy").string());
            check(compare(prep.normalized_nchw,input0).max_abs==0,name+" normalization differs from reference");
            std::vector<float> flipped(static_cast<size_t>(input0.numel()));
            for(int c=0;c<3;++c)for(int y=0;y<384;++y)for(int x=0;x<288;++x)
                flipped[(c*384+y)*288+x]=input0.data_as<float>()[(c*384+y)*288+(287-x)];
            vrhino::Tensor input1=vrhino::Tensor::host({1,3,384,288},vrhino::DType::F32);
            std::copy(flipped.begin(),flipped.end(),input1.data_as<float>());
            check(compare(input1,vrhino::test::read_npy_f32((root/"network_input_1.npy").string())).max_abs==0,
                  name+" flip preprocessing differs from reference");
            backend.synchronize();start=std::chrono::steady_clock::now();
            vrhino::PoseEstimator2DObservation observation;
            auto original=pose.execute(pose_component.graph(),prep.normalized_nchw,frame==0?&observation:nullptr);
            backend.synchronize();const double original_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            neural_ms+=original_ms;if(frame==0)frame0_neural_ms+=original_ms;
            if(frame==0){
                qualify("stem",backend,observation.tensors.at("stem"),root/"stem_0.npy",2e-2,0.999999);
                qualify("backbone_stage",backend,observation.tensors.at("backbone_stage"),root/"stage2_0.npy",8e-2,0.99999);
                qualify("spp",backend,observation.tensors.at("spp"),root/"spp_0.npy",2e-1,0.99999);
                qualify("backbone_final",backend,observation.tensors.at("backbone_final"),root/"backbone_final_0.npy",2e-1,0.99999);
                qualify("head_projection",backend,observation.tensors.at("head_projection"),root/"head_projection_0.npy",2e-1,0.99999);
                qualify("gau",backend,observation.tensors.at("pose_head"),root/"gau_0.npy",2e-1,0.99999);
            }
            qualify(name+".raw_x_original",backend,original.simcc_x,
                    root/"raw_x_0.npy",3e-3,0.99999,frame==0);
            qualify(name+".raw_y_original",backend,original.simcc_y,
                    root/"raw_y_0.npy",3e-3,0.99999,frame==0);
            backend.synchronize();start=std::chrono::steady_clock::now();
            auto mirrored=pose.execute(pose_component.graph(),input1);
            backend.synchronize();const double flipped_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            neural_ms+=flipped_ms;if(frame==0)frame0_neural_ms+=flipped_ms;
            qualify(name+".raw_x_flipped",backend,mirrored.simcc_x,
                    root/"raw_x_1.npy",3e-3,0.99999,frame==0);
            qualify(name+".raw_y_flipped",backend,mirrored.simcc_y,
                    root/"raw_y_1.npy",3e-3,0.99999,frame==0);
            auto ox=backend.copy_to_host(original.simcc_x),oy=backend.copy_to_host(original.simcc_y),
                 fx=backend.copy_to_host(mirrored.simcc_x),fy=backend.copy_to_host(mirrored.simcc_y);
            start=std::chrono::steady_clock::now();
            auto tta=vrhino::combine_simcc_flip_tta(ox,oy,fx,fy,flip_indices());
            auto keypoints=vrhino::decode_simcc_keypoints(
                tta.first,tta.second,prep.transform);
            const double current_decode=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            decode_ms+=current_decode;if(frame==0)frame0_decode_ms=current_decode;
            {
                Difference dx=compare(tta.first,vrhino::test::read_npy_f32((root/"tta_x.npy").string()));
                Difference dy=compare(tta.second,vrhino::test::read_npy_f32((root/"tta_y.npy").string()));
                check(dx.nan==0&&dx.inf==0&&dy.nan==0&&dy.inf==0&&
                      dx.max_abs<=3e-3&&dy.max_abs<=3e-3&&
                      dx.cosine>=.99999&&dy.cosine>=.99999,
                      name+" TTA raw output gate failed");
                if(frame==0)
                std::cout<<"tta_x max_abs="<<dx.max_abs<<" mean_abs="<<dx.mean_abs<<" cosine="<<dx.cosine<<'\n'
                         <<"tta_y max_abs="<<dy.max_abs<<" mean_abs="<<dy.mean_abs<<" cosine="<<dy.cosine<<'\n';
            }
            auto ref_points=read_npy<double>(phase1/(name+".dwpose_133.npy"),"<f8");
            auto ref_scores=read_npy<float>(phase1/(name+".dwpose_scores_133.npy"),"<f4");
            check(ref_points.shape==std::vector<int64_t>({133,2})&&ref_scores.shape==std::vector<int64_t>({133}),"Phase-1 keypoint contract mismatch");
            for(int point=0;point<133;++point){double dx=std::abs(keypoints.points[point].x-ref_points.values[point*2]),dy=std::abs(keypoints.points[point].y-ref_points.values[point*2+1]),dc=std::abs(keypoints.points[point].confidence-ref_scores.values[point]);
                if(std::max(dx,dy)>std::max(worst_x,worst_y)){worst_x=dx;worst_y=dy;worst_point=point;worst_frame=frame;}
                if(dc>worst_conf){worst_conf=dc;worst_conf_point=point;worst_conf_frame=frame;}}
            start=std::chrono::steady_clock::now();
            auto detector_input=vrhino::test::read_npy_f32((s3fd_oracle/name/"preprocessed.npy").string());
            auto detector_result=detector.execute(detector_component.graph(),detector_input);
            auto roi=vrhino::decode_multiscale_face_detector(host_scales(backend,detector_result)).selected;
            s3fd_integration_ms+=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            check(roi.has_value(),name+" Native S3FD has no fallback ROI");
            start=std::chrono::steady_clock::now();
            auto geometry=vrhino::select_musetalk_face_geometry(keypoints,roi,1216,0,10);
            const double current_geometry=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            geometry_ms+=current_geometry;if(frame==0)frame0_geometry_ms=current_geometry;
            auto ref_face=read_npy<int32_t>(phase1/(name+".face_points_68_int.npy"),"<i4");
            check(ref_face.shape==std::vector<int64_t>({68,2})&&geometry.facial_landmarks.size()==68,"68-point contract mismatch");
            for(int i=0;i<68;++i)for(int axis=0;axis<2;++axis)worst_face=std::max(worst_face,std::abs(static_cast<double>(geometry.facial_landmarks[i][axis]-ref_face.values[i*2+axis])));
            check(geometry.valid&&!geometry.used_fallback&&geometry.base_bbox==expected_base[frame]&&geometry.crop_bbox[3]==expected_base[frame][3]+10,name+" FaceGeometry differs from Phase-1");
            std::cout<<name<<" bbox=["<<geometry.base_bbox[0]<<','<<geometry.base_bbox[1]<<','<<geometry.base_bbox[2]<<','<<geometry.base_bbox[3]<<"] fallback="<<geometry.used_fallback<<'\n';
        }
        auto warm_input0=vrhino::test::read_npy_f32(
            (oracle/"frame_00"/"network_input_0.npy").string());
        auto warm_input1=vrhino::test::read_npy_f32(
            (oracle/"frame_00"/"network_input_1.npy").string());
        backend.synchronize();const auto warm_started=std::chrono::steady_clock::now();
        (void)pose.execute(pose_component.graph(),warm_input0);
        (void)pose.execute(pose_component.graph(),warm_input1);
        backend.synchronize();const double warmed_neural_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-warm_started).count();
        check(worst_x<=1e-9&&worst_y<=1e-9&&worst_conf<=.2&&worst_face==0,"decoded keypoint/face geometry parity failed");
        std::cout<<"Pose estimator Native FP32 oracle tests: PASS\n"
                 <<"worst_keypoint_frame="<<worst_frame<<"\nworst_keypoint_index="<<worst_point
                 <<"\nworst_x_abs="<<worst_x<<"\nworst_y_abs="<<worst_y
                 <<"\nworst_conf_abs="<<worst_conf<<"\nworst_conf_frame="<<worst_conf_frame
                 <<"\nworst_conf_keypoint="<<worst_conf_point
                 <<"\nworst_face_integer_abs="<<worst_face
                 <<"\npreprocess_eight_ms="<<preprocess_ms<<"\nneural_sixteen_calls_ms="<<neural_ms
                 <<"\ndecode_eight_ms="<<decode_ms<<"\ngeometry_policy_eight_ms="<<geometry_ms
                 <<"\nwarmed_pose_neural_two_calls_ms="<<warmed_neural_ms
                 <<"\ns3fd_integration_eight_ms="<<s3fd_integration_ms
                 <<"\ndwpose_geometry_one_frame_ms="<<(frame0_preprocess_ms+frame0_neural_ms+frame0_decode_ms+frame0_geometry_ms)
                 <<"\ndwpose_geometry_eight_frames_ms="<<(preprocess_ms+neural_ms+decode_ms+geometry_ms)
                 <<"\npeak_device_bytes="<<backend.peak_device_bytes()<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<"Pose estimator oracle tests: FAIL: "<<e.what()<<'\n';return 1;}
}
