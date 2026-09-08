#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "vrhino/face_mask.h"
#include "vrhino/loader.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/tflite.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {
void check(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<typename Operation>void expect(product::ModelPackageErrorCode code,Operation&& operation){try{operation();throw std::runtime_error("failure fixture succeeded");}catch(const product::ModelPackageError& failure){check(failure.code()==code,"failure fixture error code mismatch");}}

std::vector<std::array<int32_t,2>> facial_fixture() {
    std::vector<std::array<int32_t,2>> points(68,{128,128});
    for(int index=0;index<=16;++index){
        const double angle=3.141592653589793*(1.0-index/16.0);
        points[index]={static_cast<int32_t>(128+82*std::cos(angle)),
                       static_cast<int32_t>(102+94*std::sin(angle))};
    }
    points[31]={105,111};points[33]={128,119};points[35]={151,111};
    const std::array<std::array<int32_t,2>,12> mouth{{
        {103,145},{111,138},{121,136},{128,137},{135,136},{145,138},
        {153,145},{145,151},{135,154},{128,153},{121,154},{111,151}}};
    for(size_t index=0;index<mouth.size();++index)points[48+index]=mouth[index];
    return points;
}

void mask_policy_fixtures(){
    vrhino::SemanticLabelMap labels{256,256,std::vector<uint8_t>(256*256,0),
                                    "successor_segmenter_pixels"};
    // Face skin fills the face, while hair/accessory patches and body skin
    // exercise explicit exclusions and the tightly bounded inclusion rule.
    for(int y=70;y<220;++y)for(int x=45;x<211;++x)
        labels.labels[static_cast<size_t>(y)*256+x]=3;
    for(int y=70;y<115;++y)for(int x=45;x<211;++x)
        labels.labels[static_cast<size_t>(y)*256+x]=1;
    for(int y=125;y<150;++y)for(int x=160;x<190;++x)
        labels.labels[static_cast<size_t>(y)*256+x]=5;
    for(int y=180;y<240;++y)for(int x=0;x<256;++x)
        labels.labels[static_cast<size_t>(y)*256+x]=2;
    // An open-mouth hole must be restored from the configured outer-mouth
    // landmarks even though its semantic label is background.
    for(int y=137;y<155;++y)for(int x=102;x<154;++x)
        labels.labels[static_cast<size_t>(y)*256+x]=0;
    const auto landmarks=facial_fixture();
    vrhino::LandmarkConstrainedMaskObservation observation;
    const auto mask=vrhino::build_landmark_constrained_alpha_mask(
        labels,landmarks,{0,0,256,256},{},&observation);
    check(mask.width==256&&mask.height==256&&mask.values.size()==256*256&&
          mask.coordinate_space=="expanded_face_crop_pixels",
          "successor AlphaMask contract mismatch");
    check(observation.geometry_constraint[128*256+128]==255&&
          observation.geometry_constraint[80*256+128]==0,
          "DWPose lower-face constraint mismatch");
    check(observation.semantic_seed[145*256+128]==255,
          "mouth restoration did not cover the open-mouth fixture");
    check(observation.semantic_seed[130*256+175]==0,
          "accessory leaked outside mouth restoration");
    check(observation.semantic_seed[230*256+128]==0,
          "body skin leaked outside DWPose geometry");
    check(*std::max_element(mask.values.begin(),mask.values.end())==255&&
          *std::min_element(mask.values.begin(),mask.values.end())==0,
          "successor AlphaMask range mismatch");

    // Profile/edge and multiple-person safety fixtures are represented by a
    // shifted selected landmark set.  Semantic support belonging to another
    // person must remain outside the selected geometry.
    auto shifted=landmarks;for(auto& point:shifted)point[0]-=58;
    vrhino::SemanticLabelMap multiple=labels;
    for(int y=110;y<210;++y)for(int x=170;x<250;++x)
        multiple.labels[static_cast<size_t>(y)*256+x]=3;
    vrhino::LandmarkConstrainedMaskObservation shifted_observation;
    vrhino::build_landmark_constrained_alpha_mask(
        multiple,shifted,{0,0,256,256},{},&shifted_observation);
    check(shifted_observation.selected[160*256+220]==0,
          "unselected-person semantic support escaped DWPose constraint");
}
}  // namespace

int main(int argc,char** argv){
    if(argc!=4){std::cerr<<"usage: selfie_multiclass_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";return 2;}
    const fs::path source=argv[1],mapping=argv[2],output=argv[3];
    const fs::path model=source/"selfie_multiclass_256x256.tflite";
    const fs::path scratch=output.parent_path()/("selfie-multiclass-tests-"+std::to_string(getpid()));
    std::error_code error;fs::remove(output,error);fs::remove_all(scratch,error);fs::create_directories(scratch);
    try{
        mask_policy_fixtures();
        const auto inventory=product::inspect_tflite_flatbuffer(model);
        check(inventory.schema_version==3&&inventory.tensors.size()==373&&
              inventory.operators.size()==175&&inventory.buffer_count==377&&
              inventory.inputs==std::vector<int32_t>({0})&&
              inventory.outputs==std::vector<int32_t>({372}),
              "bounded SelfieMulticlass TFLite inventory mismatch");
        auto converted=product::convert_selfie_multiclass_component(source,mapping,output);
        check(converted.source_checkpoint_bytes==16371837&&
              converted.retained_tensor_bytes==16281136&&
              converted.retained_tensor_count==181,
              "SelfieMulticlass source inventory drift");
        vrhino::VrmModel component(output.string(),true);
        check(component.profile_id()=="component"&&
              component.architecture_id()=="segmenter-2d"&&
              component.tensors().size()==181&&
              component.graph().at("kind").string()=="semantic_segmenter_2d"&&
              component.graph().at("config").at("architecture").string()==
                  "xenoformer_fpn"&&
              component.graph().at("operators").array().size()==175,
              "SelfieMulticlass component contract drift");
        const fs::path repeat=scratch/"repeat.vrm";
        const auto repeated=product::convert_selfie_multiclass_component(source,mapping,repeat);
        check(repeated.output_sha256==converted.output_sha256&&
              repeated.payload_blake2b128==converted.payload_blake2b128,
              "SelfieMulticlass conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled,[&]{
            product::convert_selfie_multiclass_component(source,mapping,scratch/"cancel.vrm",[]{return true;});});
        expect(product::ModelPackageErrorCode::ArtifactMissing,[&]{
            product::convert_selfie_multiclass_component(scratch/"missing",mapping,scratch/"missing.vrm");});
        const fs::path corrupt=scratch/"corrupt";fs::create_directories(corrupt);
        std::ofstream(corrupt/"selfie_multiclass_256x256.tflite").put('\0');
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed,[&]{
            product::convert_selfie_multiclass_component(corrupt,mapping,scratch/"corrupt.vrm");});
        for(const auto& entry:fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-")==std::string::npos,
                  "SelfieMulticlass conversion left partial publication");
        fs::remove_all(scratch,error);
        std::cout<<"selfie multiclass source/converter tests pass sha256="
                 <<converted.output_sha256<<" blake2b128="
                 <<converted.payload_blake2b128<<" bytes="
                 <<converted.output_bytes<<"\n";
        return 0;
    }catch(const std::exception& failure){fs::remove_all(scratch,error);std::cerr<<"selfie multiclass tests: "<<failure.what()<<"\n";return 1;}
}
