#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/json.h"
#include "vrhino/lip_sync_diffusion_workflow.h"
#include "vrhino/loader.h"
#include "vrhino/pose_estimator.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;

namespace {

void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "cannot read JSON fixture: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

template <typename T> struct Npy {
    std::vector<int64_t> shape;
    std::vector<T> values;
};

template <typename T>
Npy<T> read_npy(const fs::path& path, const std::string& descriptor) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "cannot open NPY fixture: " + path.string());
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    check(input && std::memcmp(prefix, "\x93NUMPY", 6) == 0 &&
              prefix[6] == 1 && prefix[7] == 0,
          "unsupported NPY fixture header");
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
                                 static_cast<uint16_t>(prefix[9]) << 8;
    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    check(input && header.find("'descr': '" + descriptor + "'") !=
              std::string::npos &&
              header.find("'fortran_order': False") != std::string::npos,
          "NPY dtype/order mismatch: " + path.string());
    const size_t key = header.find("'shape':");
    const size_t begin = header.find('(', key), end = header.find(')', begin);
    check(key != std::string::npos && begin != std::string::npos &&
              end != std::string::npos,
          "NPY shape missing");
    Npy<T> result;
    int64_t elements = 1;
    std::istringstream fields(header.substr(begin + 1, end - begin - 1));
    for (std::string field; std::getline(fields, field, ',');) {
        field.erase(std::remove_if(field.begin(), field.end(),
            [](unsigned char value) { return std::isspace(value); }), field.end());
        if (!field.empty()) {
            const int64_t dimension = std::stoll(field);
            result.shape.push_back(dimension);
            elements *= dimension;
        }
    }
    result.values.resize(static_cast<size_t>(elements));
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(result.values.size() * sizeof(T)));
    check(input && input.peek() == std::char_traits<char>::eof(),
          "NPY payload mismatch");
    return result;
}

std::string frame_name(int frame) {
    std::ostringstream stream;
    stream << "frame_" << std::setw(2) << std::setfill('0') << frame;
    return stream.str();
}

std::vector<float> values(vrhino::CudaBackend& backend,
                          const vrhino::Tensor& tensor) {
    const auto host = backend.copy_to_host(tensor);
    return {host.data_as<float>(), host.data_as<float>() + host.numel()};
}

const std::vector<int64_t>& flip_indices() {
    static const std::vector<int64_t> value={
0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,20,21,22,17,18,19,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,49,48,47,46,45,44,43,42,41,40,50,51,52,53,58,57,56,55,54,68,67,66,65,70,69,62,61,60,59,64,63,77,76,75,74,73,72,71,82,81,80,79,78,87,86,85,84,83,90,89,88,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111};
    return value;
}

vrhino::FaceROI reference_roi(const vrhino::Json& frame) {
    const auto& selected = frame.at("selected");
    const auto& box = selected.at("bbox_xyxy").array();
    return {static_cast<float>(box[0].number()),
            static_cast<float>(box[1].number()),
            static_cast<float>(box[2].number()),
            static_cast<float>(box[3].number()),
            static_cast<float>(selected.at("score").number()),
            selected.at("source_index").integer(), -1};
}

vrhino::Tensor host_f32(const std::vector<float>& values,
                        const std::vector<int64_t>& shape) {
    vrhino::Tensor result = vrhino::Tensor::host(shape, vrhino::DType::F32);
    check(static_cast<size_t>(result.numel()) == values.size(),
          "host tensor value count mismatch");
    std::copy(values.begin(), values.end(), result.data_as<float>());
    return result;
}

template <typename T>
double affine_delta(const vrhino::SimilarityAffine& actual,
                    const Npy<T>& expected) {
    check(expected.shape == std::vector<int64_t>({2,3}),
          "affine fixture shape mismatch");
    const auto matrix = actual.matrix();
    double result = 0.0;
    for (size_t index = 0; index < matrix.size(); ++index)
        result = std::max(result, std::abs(matrix[index] - expected.values[index]));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: latentsync_alignment_oracle_tests BLAZE_VRM "
                     "DWPOSE_VRM MEDIA_HELPER VIDEO ORACLE_ROOT\n";
        return 2;
    }
    try {
        vrhino::VrmModel blaze_model(argv[1], true), pose_model(argv[2], true);
        check(blaze_model.architecture_id() == "vision-detector" &&
              pose_model.architecture_id() == "pose-estimator",
              "face component identity mismatch");
        const fs::path root = argv[5];
        const auto decoded = vrhino::decode_video_rgb24(argv[3], argv[4], 17);
        check(decoded.frames.size() == 17 && decoded.width == 1024 &&
              decoded.height == 1024 && decoded.fps_numerator == 25 &&
              decoded.fps_denominator == 1,
              "17-frame media input contract mismatch");
        const auto blaze_manifest = vrhino::Json::parse(read_text(
            root / "alignment/blazeface/manifest.json"));
        check(blaze_manifest.at("frames").array().size() == 17,
              "BlazeFace oracle frame count mismatch");

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::VisionDetectorComponentExecutor blaze(
            backend, vrhino::WeightMap(blaze_model.bindings(blaze_model.graph())));
        vrhino::PoseEstimator2DComponentExecutor pose(
            backend, vrhino::WeightMap(pose_model.bindings(pose_model.graph())));
        vrhino::LipSyncDiffusionAlignmentState state;
        double worst_roi = 0.0, worst_point = 0.0, worst_score = 0.0;
        double worst_anchor = 0.0, worst_raw = 0.0, worst_smooth = 0.0;
        double worst_inverse = 0.0;
        uint64_t crop_differing_channels = 0;
        int crop_max_delta = 0;
        double blaze_ms = 0.0, pose_ms = 0.0, alignment_ms = 0.0;
        for (int frame = 0; frame < 17; ++frame) {
            const auto& rgb = decoded.frames[static_cast<size_t>(frame)];
            auto started = std::chrono::steady_clock::now();
            const auto blaze_input = vrhino::preprocess_dense_face_detector_rgb_u8(
                rgb.pixels.data(), rgb.height, rgb.width);
            const auto blaze_output = blaze.execute_dense(
                blaze_model.graph(),
                backend.copy_to_device(host_f32(
                    blaze_input.normalized_nchw, {1,3,128,128}),
                    vrhino::DType::F32));
            vrhino::DenseDetectorHost dense{
                values(backend, blaze_output.regressors),
                values(backend, blaze_output.classification_logits)};
            const auto stages = vrhino::decode_dense_face_detector(dense, blaze_input);
            blaze_ms += std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now() - started).count();
            check(stages.selected.has_value(), "BlazeFace selected no face");
            const auto expected_roi = reference_roi(
                blaze_manifest.at("frames").array()[static_cast<size_t>(frame)]);
            worst_roi = std::max(worst_roi, std::max<double>({
                std::abs(stages.selected->x1-expected_roi.x1),
                std::abs(stages.selected->y1-expected_roi.y1),
                std::abs(stages.selected->x2-expected_roi.x2),
                std::abs(stages.selected->y2-expected_roi.y2)}));

            std::vector<uint8_t> bgr(rgb.pixels.size());
            for (size_t pixel = 0; pixel < rgb.pixels.size()/3; ++pixel) {
                bgr[pixel*3] = rgb.pixels[pixel*3+2];
                bgr[pixel*3+1] = rgb.pixels[pixel*3+1];
                bgr[pixel*3+2] = rgb.pixels[pixel*3];
            }
            started = std::chrono::steady_clock::now();
            const auto preparation = vrhino::preprocess_topdown_pose_bgr_u8(
                bgr.data(), rgb.height, rgb.width);
            auto original = pose.execute(pose_model.graph(), preparation.normalized_nchw);
            vrhino::Tensor flipped = vrhino::Tensor::host(
                {1,3,384,288}, vrhino::DType::F32);
            for (int channel=0;channel<3;++channel)
                for (int y=0;y<384;++y) for (int x=0;x<288;++x)
                    flipped.data_as<float>()[(channel*384+y)*288+x] =
                        preparation.normalized_nchw.data_as<float>()[
                            (channel*384+y)*288+(287-x)];
            auto mirrored = pose.execute(pose_model.graph(), flipped);
            auto combined = vrhino::combine_simcc_flip_tta(
                backend.copy_to_host(original.simcc_x),
                backend.copy_to_host(original.simcc_y),
                backend.copy_to_host(mirrored.simcc_x),
                backend.copy_to_host(mirrored.simcc_y), flip_indices());
            const auto keypoints = vrhino::decode_simcc_keypoints(
                combined.first, combined.second, preparation.transform);
            pose_ms += std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now() - started).count();
            const std::string name = frame_name(frame);
            const fs::path tensors = root / "alignment/tensors";
            const auto expected_points = read_npy<float>(
                tensors / (name + "_points133.npy"), "<f4");
            const auto expected_scores = read_npy<float>(
                tensors / (name + "_scores133.npy"), "<f4");
            check(expected_points.shape == std::vector<int64_t>({133,2}) &&
                  expected_scores.shape == std::vector<int64_t>({133}),
                  "DWPose oracle shape mismatch");
            for (int point=0; point<133; ++point) {
                worst_point = std::max(worst_point, std::max(
                    std::abs(keypoints.points[point].x-
                        expected_points.values[point*2]),
                    std::abs(keypoints.points[point].y-
                        expected_points.values[point*2+1])));
                worst_score = std::max(worst_score, static_cast<double>(
                    std::abs(keypoints.points[point].confidence-
                             expected_scores.values[point])));
            }
            started = std::chrono::steady_clock::now();
            const auto aligned = vrhino::align_lip_sync_diffusion_face(
                keypoints, *stages.selected, state);
            check(aligned.has_value(), "valid DWPose alignment was rejected");
            const auto expected_anchors = read_npy<float>(
                tensors / (name + "_anchors3.npy"), "<f4");
            for (int point=0;point<3;++point) {
                worst_anchor = std::max(worst_anchor, std::max(
                    std::abs(aligned->source_anchors[point].x-
                        expected_anchors.values[point*2]),
                    std::abs(aligned->source_anchors[point].y-
                        expected_anchors.values[point*2+1])));
            }
            worst_raw = std::max(worst_raw, affine_delta(aligned->unsmoothed,
                read_npy<float>(tensors/(name+"_affine_raw.npy"),"<f4")));
            worst_smooth = std::max(worst_smooth, affine_delta(aligned->smoothed,
                read_npy<float>(tensors/(name+"_affine_smooth.npy"),"<f4")));
            worst_inverse = std::max(worst_inverse, affine_delta(aligned->inverse,
                read_npy<float>(tensors/(name+"_affine_inverse.npy"),"<f4")));
            const auto crop = vrhino::warp_lip_sync_diffusion_face(rgb,
                                                                   aligned->smoothed);
            alignment_ms += std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now() - started).count();
            const auto reference_crop = vrhino::decode_video_rgb24(
                argv[3], root/"alignment/crops512"/(name+".png"), 1).frames[0];
            check(crop.pixels.size() == reference_crop.pixels.size(),
                  "aligned crop size mismatch");
            for (size_t index=0;index<crop.pixels.size();++index) {
                const int delta = std::abs(static_cast<int>(crop.pixels[index])-
                                           reference_crop.pixels[index]);
                crop_max_delta = std::max(crop_max_delta, delta);
                crop_differing_channels += delta != 0;
            }
        }
        std::cout << std::setprecision(12)
                  << "LatentSync public-safe alignment oracle: PASS\n"
                  << "worst_roi_px=" << worst_roi << '\n'
                  << "worst_dwpose_coordinate=" << worst_point << '\n'
                  << "worst_dwpose_score=" << worst_score << '\n'
                  << "worst_anchor=" << worst_anchor << '\n'
                  << "worst_raw_affine=" << worst_raw << '\n'
                  << "worst_smoothed_affine=" << worst_smooth << '\n'
                  << "worst_inverse_affine=" << worst_inverse << '\n'
                  << "crop_max_delta=" << crop_max_delta << '\n'
                  << "crop_differing_channels=" << crop_differing_channels << '\n'
                  << "blazeface_ms=" << blaze_ms << '\n'
                  << "dwpose_ms=" << pose_ms << '\n'
                  << "alignment_ms=" << alignment_ms << '\n'
                  << "peak_device_bytes=" << backend.peak_device_bytes() << '\n';
        check(worst_roi <= 2.0 && worst_anchor <= 0.5 &&
              worst_raw <= 1.0 && worst_smooth <= 1.0 &&
              worst_inverse <= 0.5 && crop_max_delta <= 32,
              "public-safe alignment exceeds oracle gate");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LatentSync public-safe alignment oracle: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
