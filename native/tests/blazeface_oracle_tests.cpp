#include <algorithm>
#include <chrono>
#include <cmath>
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
#include "vrhino/pose_geometry.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;
namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Difference {
    double max_abs = 0.0, mean_abs = 0.0, cosine = 0.0;
    double reference_min = std::numeric_limits<double>::infinity();
    double reference_max = -std::numeric_limits<double>::infinity();
    double native_min = std::numeric_limits<double>::infinity();
    double native_max = -std::numeric_limits<double>::infinity();
    uint64_t nan = 0, inf = 0;
};

Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& expected) {
    check(actual.device().is_host() && expected.device().is_host() &&
          actual.dtype() == vrhino::DType::F32 &&
          expected.dtype() == vrhino::DType::F32 &&
          actual.shape() == expected.shape(), "oracle tensor contract mismatch");
    Difference result;
    long double error_sum = 0.0, dot = 0.0, actual_norm = 0.0, expected_norm = 0.0;
    for (int64_t index = 0; index < actual.numel(); ++index) {
        const double a = actual.data_as<float>()[index];
        const double b = expected.data_as<float>()[index];
        result.nan += std::isnan(a);
        result.inf += std::isinf(a);
        result.native_min = std::min(result.native_min, a);
        result.native_max = std::max(result.native_max, a);
        result.reference_min = std::min(result.reference_min, b);
        result.reference_max = std::max(result.reference_max, b);
        const double error = std::abs(a - b);
        result.max_abs = std::max(result.max_abs, error);
        error_sum += error;
        dot += a * b;
        actual_norm += a * a;
        expected_norm += b * b;
    }
    result.mean_abs = static_cast<double>(error_sum / actual.numel());
    result.cosine = static_cast<double>(dot / std::sqrt(actual_norm * expected_norm));
    return result;
}

Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual, const fs::path& reference,
                   double max_abs, double minimum_cosine, bool emit = true) {
    const auto result = compare(backend.copy_to_host(actual),
                                vrhino::test::read_npy_f32(reference.string()));
    if (emit) {
        std::cout << std::setprecision(10) << name << " ref=["
                  << result.reference_min << ',' << result.reference_max
                  << "] native=[" << result.native_min << ',' << result.native_max
                  << "] max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " cosine=" << result.cosine << " nan=" << result.nan
                  << " inf=" << result.inf << '\n';
    }
    check(result.nan == 0 && result.inf == 0 && result.max_abs <= max_abs &&
          result.cosine >= minimum_cosine, name + " exceeds FP32 gate");
    return result;
}

std::vector<float> tensor_values(vrhino::CudaBackend& backend,
                                 const vrhino::Tensor& tensor) {
    const auto host = backend.copy_to_host(tensor);
    return {host.data_as<float>(), host.data_as<float>() + host.numel()};
}

vrhino::Tensor host_tensor(const std::vector<float>& values,
                           const std::vector<int64_t>& shape) {
    vrhino::Tensor result = vrhino::Tensor::host(shape, vrhino::DType::F32);
    check(static_cast<size_t>(result.numel()) == values.size(),
          "host tensor value count mismatch");
    std::memcpy(result.data(), values.data(), result.bytes());
    return result;
}

std::vector<uint8_t> rgb_from_fixture(const vrhino::Tensor& tensor) {
    check(tensor.device().is_host() && tensor.dtype() == vrhino::DType::F32 &&
          tensor.ndim() == 3 && tensor.dim(2) == 3, "RGB fixture shape mismatch");
    std::vector<uint8_t> result(static_cast<size_t>(tensor.numel()));
    for (int64_t index = 0; index < tensor.numel(); ++index) {
        const float value = tensor.data_as<float>()[index];
        check(value >= 0.0f && value <= 255.0f && value == std::floor(value),
              "RGB fixture contains a non-u8 value");
        result[static_cast<size_t>(index)] = static_cast<uint8_t>(value);
    }
    return result;
}

vrhino::FaceROI reference_roi(const fs::path& path) {
    const auto tensor = vrhino::test::read_npy_f32(path.string());
    check(tensor.shape() == std::vector<int64_t>({1, 6}),
          "reference ROI shape mismatch");
    const float* value = tensor.data_as<float>();
    return {value[0], value[1], value[2], value[3], value[4],
            static_cast<int64_t>(value[5]), -1};
}

double roi_max_delta(const vrhino::FaceROI& left, const vrhino::FaceROI& right) {
    return std::max({std::abs(left.x1 - right.x1), std::abs(left.y1 - right.y1),
                     std::abs(left.x2 - right.x2), std::abs(left.y2 - right.y2)});
}

double compare_roi_stage(const std::vector<vrhino::FaceROI>& actual,
                         const fs::path& path, double coordinate_gate,
                         double score_gate) {
    const auto expected = vrhino::test::read_npy_f32(path.string());
    check(expected.ndim() == 2 && expected.dim(1) == 6 &&
          expected.dim(0) == static_cast<int64_t>(actual.size()),
          "reference ROI stage shape mismatch");
    double worst = 0.0;
    for (int64_t index = 0; index < expected.dim(0); ++index) {
        const float* value = expected.data_as<float>() + index * 6;
        const vrhino::FaceROI reference{value[0], value[1], value[2], value[3],
                                        value[4], static_cast<int64_t>(value[5]), -1};
        worst = std::max(worst, roi_max_delta(actual[static_cast<size_t>(index)],
                                              reference));
        check(roi_max_delta(actual[static_cast<size_t>(index)], reference) <=
                  coordinate_gate &&
              std::abs(actual[static_cast<size_t>(index)].confidence -
                       reference.confidence) <= score_gate &&
              actual[static_cast<size_t>(index)].source_index ==
                  reference.source_index,
              "decoded ROI stage numerical mismatch");
    }
    return worst;
}

std::string frame_name(int frame) {
    std::ostringstream stream;
    stream << "frame_" << std::setw(2) << std::setfill('0') << frame;
    return stream.str();
}

vrhino::KeypointSet keypoints(const fs::path& directory) {
    const auto scores = vrhino::test::read_npy_f32(
        (directory / "decoded_scores.npy").string());
    check(scores.shape() == std::vector<int64_t>({133}),
          "keypoint score shape mismatch");
    std::ifstream input(directory / "decoded_keypoints.npy", std::ios::binary);
    check(static_cast<bool>(input), "missing decoded keypoint fixture");
    uint8_t prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    check(input && std::memcmp(prefix, "\x93NUMPY", 6) == 0 && prefix[6] == 1 &&
          prefix[7] == 0, "unsupported decoded keypoint fixture");
    const uint16_t header_size = static_cast<uint16_t>(prefix[8]) |
                                 (static_cast<uint16_t>(prefix[9]) << 8);
    std::string header(header_size, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    check(input && header.find("'descr': '<f8'") != std::string::npos &&
          header.find("'shape': (133, 2)") != std::string::npos,
          "decoded keypoint dtype/shape mismatch");
    std::vector<double> coordinates(266);
    input.read(reinterpret_cast<char*>(coordinates.data()),
               static_cast<std::streamsize>(coordinates.size() * sizeof(double)));
    check(input && input.peek() == std::char_traits<char>::eof(),
          "decoded keypoint payload mismatch");
    vrhino::KeypointSet result;
    for (int point = 0; point < 133; ++point)
        result.points.push_back({coordinates[2 * point], coordinates[2 * point + 1],
                                 scores.data_as<float>()[point]});
    return result;
}

struct Execution {
    vrhino::BlazeFacePreprocessResult preprocessing;
    vrhino::DenseFaceDetectionStages stages;
};

Execution execute_fixture(vrhino::CudaBackend& backend,
                          vrhino::VisionDetectorComponentExecutor& executor,
                          const vrhino::Json& graph, const fs::path& directory,
                          vrhino::VisionDetectorObservation* observation,
                          double* preprocessing_ms, double* neural_ms,
                          double* postprocess_ms) {
    const auto rgb_tensor = vrhino::test::read_npy_f32((directory / "rgb.npy").string());
    const auto rgb = rgb_from_fixture(rgb_tensor);
    auto started = std::chrono::steady_clock::now();
    auto preprocessing = vrhino::preprocess_dense_face_detector_rgb_u8(
        rgb.data(), rgb_tensor.dim(0), rgb_tensor.dim(1));
    *preprocessing_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const auto expected_input = vrhino::test::read_npy_f32(
        (directory / "preprocessed.npy").string());
    const auto input_difference = compare(
        host_tensor(preprocessing.normalized_nchw, {1, 3, 128, 128}), expected_input);
    check(input_difference.max_abs == 0.0,
          directory.filename().string() + " preprocessing is not exact");
    const auto device_input = backend.copy_to_device(expected_input, vrhino::DType::F32);
    backend.synchronize();
    started = std::chrono::steady_clock::now();
    const auto output = executor.execute_dense(graph, device_input, observation);
    backend.synchronize();
    *neural_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    vrhino::DenseDetectorHost raw{tensor_values(backend, output.regressors),
                                  tensor_values(backend, output.classification_logits)};
    started = std::chrono::steady_clock::now();
    auto stages = vrhino::decode_dense_face_detector(raw, preprocessing);
    *postprocess_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return {std::move(preprocessing), std::move(stages)};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: blazeface_oracle_tests COMPONENT_VRM ORACLE_ROOT "
                     "DWPOSE_ORACLE_ROOT\n";
        return 2;
    }
    try {
        vrhino::VrmModel component(argv[1], true);
        check(component.architecture_id() == "vision-detector" &&
              component.graph().at("kind").string() ==
                  "vision_detector_dense_anchors",
              "dense detector component identity mismatch");
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::VisionDetectorComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())));
        const fs::path oracle = argv[2], pose_oracle = argv[3];
        double preprocessing_ms = 0.0, neural_ms = 0.0, postprocess_ms = 0.0;
        double cold_frame0_neural_ms = 0.0;
        double worst_roi = 0.0, worst_score = 0.0, worst_iou = 1.0;
        double worst_raw = 0.0, worst_raw_mean = 0.0, minimum_raw_cosine = 1.0;
        int worst_frame = 0;
        for (int frame = 0; frame < 8; ++frame) {
            const std::string name = frame_name(frame);
            vrhino::VisionDetectorObservation observation;
            const double neural_before = neural_ms;
            const auto result = execute_fixture(
                backend, executor, component.graph(), oracle / name,
                &observation, &preprocessing_ms,
                &neural_ms, &postprocess_ms);
            if (frame == 0) cold_frame0_neural_ms = neural_ms - neural_before;
            if (frame == 0) {
                qualify("early_backbone", backend,
                        observation.tensors.at("early_backbone"),
                        oracle / name / "early_backbone.npy", 2e-3, 0.999999);
                qualify("representative_depthwise", backend,
                        observation.tensors.at("representative_depthwise"),
                        oracle / name / "representative_depthwise.npy", 2e-3,
                        0.999999);
                qualify("deeper_feature", backend,
                        observation.tensors.at("deeper_feature"),
                        oracle / name / "deeper_feature.npy", 2e-2, 0.999999);
                qualify("final_feature", backend,
                        observation.tensors.at("final_feature"),
                        oracle / name / "final_feature.npy", 2e-2, 0.999999);
            }
            const auto reg = qualify(name + ".regressors", backend,
                observation.tensors.at("raw_regressors"),
                oracle / name / "regressors.npy", 5e-2, 0.999999, frame == 0);
            const auto cls = qualify(name + ".classification_logits", backend,
                observation.tensors.at("raw_classification_logits"),
                oracle / name / "logits.npy", 6e-2, 0.999999, frame == 0);
            if (std::max(reg.max_abs, cls.max_abs) > worst_raw) {
                worst_raw = std::max(reg.max_abs, cls.max_abs);
                worst_raw_mean = std::max(reg.mean_abs, cls.mean_abs);
                worst_frame = frame;
            }
            minimum_raw_cosine = std::min({minimum_raw_cosine, reg.cosine, cls.cosine});
            check(result.stages.selected.has_value(), name + " did not select a face");
            const double decoded_delta = compare_roi_stage(
                result.stages.decoded_candidates, oracle / name / "decoded.npy",
                5e-4, 2e-5);
            const double thresholded_delta = compare_roi_stage(
                result.stages.thresholded_candidates,
                oracle / name / "thresholded.npy", 5e-4, 2e-5);
            const auto expected = reference_roi(oracle / name / "roi.npy");
            const double box_delta = roi_max_delta(*result.stages.selected, expected);
            const double score_delta = std::abs(result.stages.selected->confidence -
                                                expected.confidence);
            check(box_delta <= 0.2 && score_delta <= 2e-5 &&
                  result.stages.selected->source_index == expected.source_index,
                  name + " decoded FaceROI mismatch");
            worst_roi = std::max(worst_roi, box_delta);
            worst_score = std::max(worst_score, score_delta);

            const auto landmarks = keypoints(pose_oracle / name);
            const auto geometry = vrhino::select_musetalk_face_geometry(
                landmarks, result.stages.selected, 1216, 0, 10);
            check(geometry.valid && !geometry.used_fallback,
                  name + " DWPose-valid geometry changed");
            vrhino::FaceROI geometry_roi{
                static_cast<float>(geometry.base_bbox[0]),
                static_cast<float>(geometry.base_bbox[1]),
                static_cast<float>(geometry.base_bbox[2]),
                static_cast<float>(geometry.base_bbox[3]), 1.0f, -1, -1};
            const double overlap = vrhino::face_roi_iou_inclusive(
                *result.stages.selected, geometry_roi);
            check(overlap >= 0.65, name + " detector/landmark overlap is unsafe");
            worst_iou = std::min(worst_iou, overlap);

            vrhino::KeypointSet invalid;
            invalid.points.assign(133, {-1.0, -1.0, 0.0f});
            const auto fallback = vrhino::select_musetalk_face_geometry(
                invalid, result.stages.selected, 1216, 0, 10);
            check(fallback.valid && fallback.used_fallback,
                  name + " invalid-DWPose fallback failed");
            std::cout << name << " candidates="
                      << result.stages.thresholded_candidates.size()
                      << " selected_index=" << result.stages.selected->source_index
                      << " roi=[" << result.stages.selected->x1 << ','
                      << result.stages.selected->y1 << ','
                      << result.stages.selected->x2 << ','
                      << result.stages.selected->y2 << "] score="
                      << result.stages.selected->confidence
                      << " reference_delta=" << box_delta
                      << " decoded_delta=" << decoded_delta
                      << " thresholded_delta=" << thresholded_delta
                      << " landmark_iou=" << overlap << '\n';
        }

        const double eight_preprocessing_ms = preprocessing_ms;
        const double eight_neural_ms = neural_ms;
        const double eight_postprocess_ms = postprocess_ms;

        auto run_named = [&](const std::string& name) {
            return execute_fixture(backend, executor, component.graph(), oracle / name,
                                   nullptr, &preprocessing_ms, &neural_ms,
                                   &postprocess_ms);
        };
        const auto no_face = run_named("no_face");
        check(!no_face.stages.selected && no_face.stages.thresholded_candidates.empty(),
              "no-face fixture did not fail safely");
        const auto small = run_named("small_face");
        check(small.stages.selected && small.stages.selected->confidence >= 0.75f,
              "small-face fixture was not detected safely");
        check(roi_max_delta(*small.stages.selected,
                            reference_roi(oracle / "small_face" / "roi.npy")) <= .2,
              "small-face reference ROI mismatch");
        const auto off_center = run_named("off_center");
        check(off_center.stages.selected && off_center.stages.selected->x1 >= 0.0f,
              "off-center fixture clipping failed");
        check(roi_max_delta(*off_center.stages.selected,
                            reference_roi(oracle / "off_center" / "roi.npy")) <= .2,
              "off-center reference ROI mismatch");
        const auto multiple = run_named("multi_face");
        check(multiple.stages.post_suppression.size() == 2 && multiple.stages.selected &&
              multiple.stages.selected->x1 < 384.0f,
              "multi-face deterministic selection failed");
        compare_roi_stage(multiple.stages.post_suppression,
                          oracle / "multi_face" / "roi.npy", .2, 2e-5);
        const auto scene_a = run_named("scene_a");
        const auto scene_b = run_named("scene_b");
        check(scene_a.stages.selected && scene_b.stages.selected &&
              scene_a.stages.selected->source_index == 177 &&
              scene_b.stages.selected->source_index == 177,
              "scene-change selection is unstable");

        std::cout << "BlazeFace Native CUDA FP32 oracle tests: PASS\n"
                  << "worst_raw_frame=" << worst_frame << '\n'
                  << "worst_raw_max_abs=" << worst_raw << '\n'
                  << "worst_raw_mean_abs=" << worst_raw_mean << '\n'
                  << "minimum_raw_cosine=" << minimum_raw_cosine << '\n'
                  << "worst_roi_coordinate_delta=" << worst_roi << '\n'
                  << "worst_score_delta=" << worst_score << '\n'
                  << "minimum_landmark_iou=" << worst_iou << '\n'
                  << "no_face=SAFE_EMPTY\nsmall_face=PASS\nmulti_face=PASS\n"
                  << "scene_cut=PASS\ndwpose_invalid_fallback=PASS\n"
                  << "cold_frame0_neural_ms=" << cold_frame0_neural_ms << '\n'
                  << "eight_preprocessing_ms=" << eight_preprocessing_ms << '\n'
                  << "eight_neural_ms=" << eight_neural_ms << '\n'
                  << "eight_postprocess_ms=" << eight_postprocess_ms << '\n'
                  << "fixture_total_neural_ms=" << neural_ms - eight_neural_ms << '\n'
                  << "peak_device_bytes=" << backend.peak_device_bytes() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "BlazeFace Native CUDA FP32 oracle tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
