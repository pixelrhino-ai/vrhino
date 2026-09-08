#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/pose_geometry.h"
#include "vrhino/product/converter.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;
namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Operation>
void expect(product::ModelPackageErrorCode code, Operation&& operation) {
    try { operation(); throw std::runtime_error("failure fixture succeeded"); }
    catch (const product::ModelPackageError& failure) {
        check(failure.code() == code, "failure fixture error code mismatch");
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: pose_estimator_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], mapping = argv[2], output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("pose-estimator-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error); fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        auto converted = product::convert_dwpose_keypoint_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 406878486ULL &&
              converted.retained_tensor_bytes == 134599836ULL &&
              converted.retained_tensor_count == 395,
              "pose source inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "pose-estimator" &&
              component.tensors().size() == 395,
              "pose component identity mismatch");
        const auto& graph = component.graph();
        check(graph.at("kind").string() == "pose_estimator_2d" &&
              graph.at("entry_point").string() == "execute" &&
              graph.at("config").at("keypoint_count").integer() == 133 &&
              graph.at("config").at("simcc_x_bins").integer() == 576 &&
              graph.at("config").at("simcc_y_bins").integer() == 768,
              "pose graph contract drift");
        const fs::path repeat = scratch / "repeat.vrm";
        auto repeated = product::convert_dwpose_keypoint_component(
            source, mapping, repeat);
        check(repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "pose conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_dwpose_keypoint_component(
                source, mapping, scratch / "cancel.vrm", [] { return true; });
        });
        expect(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_dwpose_keypoint_component(
                scratch / "missing", mapping, scratch / "missing.vrm");
        });
        const fs::path corrupt = scratch / "corrupt";
        fs::create_directories(corrupt);
        std::ofstream(corrupt / "dw-ll_ucoco_384.pth").put('\0');
        std::ofstream(corrupt /
            "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py").put('\0');
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_dwpose_keypoint_component(
                corrupt, mapping, scratch / "corrupt.vrm");
        });
        auto altered_mapping = [&](const std::string& before,
                                   const std::string& after,
                                   const fs::path& path) {
            std::ifstream input(mapping);
            std::string content((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
            const size_t position = content.find(before);
            check(position != std::string::npos, "mapping fixture field missing");
            content.replace(position, before.size(), after);
            std::ofstream(path) << content;
        };
        const fs::path wrong_dtype = scratch / "wrong-dtype.tsv";
        altered_mapping("\tF32\t", "\tF16\t", wrong_dtype);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_dwpose_keypoint_component(
                source, wrong_dtype, scratch / "wrong-dtype.vrm");
        });
        const fs::path wrong_shape = scratch / "wrong-shape.tsv";
        altered_mapping("\t32,3,3,3\t", "\t32,3,3,4\t", wrong_shape);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_dwpose_keypoint_component(
                source, wrong_shape, scratch / "wrong-shape.vrm");
        });
        for (const auto& entry : fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-") ==
                      std::string::npos,
                  "conversion left a partial publication artifact");

        std::vector<uint8_t> uniform(704 * 1216 * 3, 128);
        auto prep = vrhino::preprocess_topdown_pose_bgr_u8(
            uniform.data(), 1216, 704);
        check(prep.normalized_nchw.shape() ==
                  std::vector<int64_t>({1,3,384,288}) &&
              std::abs(prep.transform.center[0] - 352.0f) < 1e-6f &&
              std::abs(prep.transform.center[1] - 608.0f) < 1e-6f &&
              std::abs(prep.transform.scale[0] - 1140.0f) < 1e-6f &&
              std::abs(prep.transform.scale[1] - 1520.0f) < 1e-6f &&
              std::abs(prep.transform.matrix[0] - 0.2526315789473684) < 1e-12,
              "pose affine fixture mismatch");

        auto ox=vrhino::Tensor::host({1,133,576},vrhino::DType::F32);
        auto oy=vrhino::Tensor::host({1,133,768},vrhino::DType::F32);
        auto fx=vrhino::Tensor::host({1,133,576},vrhino::DType::F32);
        auto fy=vrhino::Tensor::host({1,133,768},vrhino::DType::F32);
        std::fill_n(ox.data_as<float>(),ox.numel(),-1.0f);
        std::fill_n(oy.data_as<float>(),oy.numel(),-1.0f);
        std::fill_n(fx.data_as<float>(),fx.numel(),-1.0f);
        std::fill_n(fy.data_as<float>(),fy.numel(),-1.0f);
        std::vector<int64_t> identity(133); for(int i=0;i<133;++i) identity[i]=i;
        ox.data_as<float>()[100]=3; fx.data_as<float>()[575-100]=5;
        oy.data_as<float>()[200]=4; fy.data_as<float>()[200]=6;
        auto combined=vrhino::combine_simcc_flip_tta(ox,oy,fx,fy,identity);
        check(combined.first.data_as<float>()[100]==4 &&
              combined.second.data_as<float>()[200]==5,
              "flip-TTA fixture mismatch");
        auto keypoints=vrhino::decode_simcc_keypoints(
            combined.first,combined.second,prep.transform);
        check(keypoints.points.size()==133 && keypoints.coordinate_space==
              "source_image_pixels", "keypoint decode contract mismatch");

        for(int i=0;i<133;++i) keypoints.points[i]={200.0+i,300.0+i,1.0f};
        auto geometry=vrhino::select_musetalk_face_geometry(
            keypoints,std::nullopt,1216,0,10);
        check(geometry.facial_landmarks.size()==68 && geometry.valid &&
              !geometry.used_fallback && geometry.facial_landmarks.front()[0]==223 &&
              geometry.facial_landmarks.back()[0]==290,
              "68-point face geometry policy mismatch");
        for(int i=23;i<91;++i) keypoints.points[i]={-10.0,100.0,1.0f};
        vrhino::FaceROI fallback{20,30,120,160,.9f,4,4};
        geometry=vrhino::select_musetalk_face_geometry(
            keypoints,fallback,165,0,10);
        check(geometry.used_fallback && geometry.valid &&
              geometry.base_bbox==std::array<int32_t,4>({20,30,120,160}) &&
              geometry.crop_bbox[3]==165,
              "S3FD fallback/lower clamp mismatch");

        fs::remove_all(scratch,error);
        std::cout << "Pose estimator CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch,error);
        std::cerr << "Pose estimator CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
