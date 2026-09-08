#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "vrhino/face_roi.h"
#include "vrhino/loader.h"
#include "vrhino/pose_geometry.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/tflite.h"

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

void alter_byte(const fs::path& source, const fs::path& destination,
                uint64_t offset, uint8_t value) {
    fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
    std::fstream file(destination, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(offset));
    file.put(static_cast<char>(value));
    check(static_cast<bool>(file), "cannot create TFLite corruption fixture");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: blazeface_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], mapping = argv[2], output = argv[3];
    const fs::path model = source / "blaze_face_short_range.tflite";
    const fs::path scratch = output.parent_path() /
        ("blazeface-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error);
    fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        const auto inventory = product::inspect_tflite_flatbuffer(model);
        check(inventory.schema_version == 3 && inventory.tensors.size() == 250 &&
              inventory.operators.size() == 164 && inventory.buffer_count == 89 &&
              inventory.inputs == std::vector<int32_t>({0}) &&
              inventory.outputs == std::vector<int32_t>({175,174}),
              "bounded TFLite inventory mismatch");

        auto converted = product::convert_blazeface_short_range_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 229746 &&
              converted.retained_tensor_bytes == 405560 &&
              converted.retained_tensor_count == 74,
              "dense detector source inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "vision-detector" &&
              component.tensors().size() == 74,
              "dense detector component identity mismatch");
        const auto& graph = component.graph();
        check(graph.at("kind").string() == "vision_detector_dense_anchors" &&
              graph.at("entry_point").string() == "execute" &&
              graph.at("config").at("anchor_count").integer() == 896 &&
              graph.at("semantic_outputs").array().size() == 2,
              "dense detector graph contract drift");
        const fs::path repeat = scratch / "repeat.vrm";
        auto repeated = product::convert_blazeface_short_range_component(
            source, mapping, repeat);
        check(repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "dense detector conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_blazeface_short_range_component(
                source, mapping, scratch / "cancel.vrm", [] { return true; });
        });
        expect(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_blazeface_short_range_component(
                scratch / "missing", mapping, scratch / "missing.vrm");
        });
        const fs::path corrupt_directory = scratch / "corrupt-source";
        fs::create_directories(corrupt_directory);
        std::ofstream(corrupt_directory / "blaze_face_short_range.tflite").put('\0');
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_blazeface_short_range_component(
                corrupt_directory, mapping, scratch / "corrupt.vrm");
        });

        const fs::path unsupported = scratch / "unsupported.tflite";
        alter_byte(model, unsupported, 229723, 127);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            (void)product::inspect_tflite_flatbuffer(unsupported);
        });
        const fs::path wrong_dtype = scratch / "wrong-dtype.tflite";
        alter_byte(model, wrong_dtype, 229487, 127);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            (void)product::inspect_tflite_flatbuffer(wrong_dtype);
        });
        const fs::path wrong_shape = scratch / "wrong-shape.tflite";
        alter_byte(model, wrong_shape, 229516, 25);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            (void)product::inspect_tflite_flatbuffer(wrong_shape);
        });
        const fs::path corrupt_range = scratch / "corrupt-range.tflite";
        alter_byte(model, corrupt_range, 202544, 0xff);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            (void)product::inspect_tflite_flatbuffer(corrupt_range);
        });
        for (const auto& entry : fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-") ==
                      std::string::npos,
                  "conversion left a partial publication artifact");

        std::vector<uint8_t> pixels(4 * 4 * 3, 255);
        auto preprocessed = vrhino::preprocess_dense_face_detector_rgb_u8(
            pixels.data(), 4, 4);
        check(preprocessed.normalized_nchw.size() == 3 * 128 * 128 &&
              preprocessed.horizontal_padding == 0.0f &&
              preprocessed.vertical_padding == 0.0f &&
              preprocessed.normalized_nchw.front() == 1.0f,
              "dense detector preprocessing fixture mismatch");
        const auto anchors = vrhino::short_range_face_detector_anchors();
        check(anchors.size() == 896 &&
              std::abs(anchors[0][0] - 0.03125f) < 1e-8f &&
              std::abs(anchors[511][0] - 0.96875f) < 1e-8f &&
              std::abs(anchors[512][0] - 0.0625f) < 1e-8f,
              "dense detector anchor fixture mismatch");

        vrhino::DenseDetectorHost raw;
        raw.regressors.assign(896 * 16, 0.0f);
        raw.classification_logits.assign(896, -100.0f);
        constexpr int selected_anchor = 272;
        raw.regressors[selected_anchor * 16 + 2] = 32.0f;
        raw.regressors[selected_anchor * 16 + 3] = 32.0f;
        raw.classification_logits[selected_anchor] = std::log(9.0f);
        constexpr int overlapping_anchor = selected_anchor + 1;
        raw.regressors[overlapping_anchor * 16] = 12.8f;
        raw.regressors[overlapping_anchor * 16 + 2] = 32.0f;
        raw.regressors[overlapping_anchor * 16 + 3] = 32.0f;
        raw.classification_logits[overlapping_anchor] = std::log(4.0f);
        preprocessed.source_height = 128;
        preprocessed.source_width = 128;
        auto stages = vrhino::decode_dense_face_detector(raw, preprocessed);
        check(stages.decoded_candidates.size() == 896 &&
              stages.thresholded_candidates.size() == 2 &&
              stages.post_suppression.size() == 1 && stages.selected &&
              std::abs(stages.selected->confidence - 0.9f) < 1e-6f &&
              stages.selected->source_index == selected_anchor &&
              stages.selected->scale_index == -1 &&
              stages.selected->x1 >
                  stages.decoded_candidates[selected_anchor].x1,
              "dense detector decode/sigmoid/weighted suppression mismatch");
        raw.classification_logits[overlapping_anchor] = std::log(9.0f);
        stages = vrhino::decode_dense_face_detector(raw, preprocessed);
        check(stages.selected && stages.selected->source_index == selected_anchor,
              "dense detector equal-score tie ordering mismatch");
        raw.classification_logits.assign(896, -100.0f);
        stages = vrhino::decode_dense_face_detector(raw, preprocessed);
        check(!stages.selected && stages.post_suppression.empty(),
              "dense detector no-face fixture mismatch");

        vrhino::KeypointSet invalid_keypoints;
        invalid_keypoints.points.assign(133, {-1.0, -1.0, 0.0f});
        const vrhino::FaceROI fallback{20,30,120,160,.9f,4,-1};
        const auto geometry = vrhino::select_musetalk_face_geometry(
            invalid_keypoints, fallback, 165, 0, 10);
        check(geometry.used_fallback && geometry.valid &&
              geometry.base_bbox == std::array<int32_t,4>({20,30,120,160}) &&
              geometry.crop_bbox == std::array<int32_t,4>({20,30,120,165}),
              "dense detector FaceGeometry fallback integration mismatch");

        fs::remove_all(scratch, error);
        std::cout << "BlazeFace bounded converter/workflow CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch, error);
        std::cerr << "BlazeFace bounded converter/workflow CPU tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
