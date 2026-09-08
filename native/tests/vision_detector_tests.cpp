#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>

#include "vrhino/error.h"
#include "vrhino/face_roi.h"
#include "vrhino/loader.h"
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
        std::cerr << "usage: vision_detector_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], mapping = argv[2], output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("vision-detector-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove(output, error); fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        auto converted = product::convert_s3fd_face_detector_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 89843225ULL &&
              converted.retained_tensor_bytes == 89836440ULL &&
              converted.retained_tensor_count == 65,
              "vision detector source inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "vision-detector" &&
              component.tensors().size() == 65,
              "vision detector component identity mismatch");
        const auto& graph = component.graph();
        check(graph.at("kind").string() == "vision_detector_multiscale" &&
              graph.at("entry_point").string() == "execute" &&
              graph.at("config").at("scale_count").integer() == 6 &&
              graph.at("semantic_outputs").array().size() == 12,
              "vision detector graph contract drift");
        const fs::path repeat = scratch / "repeat.vrm";
        auto repeated = product::convert_s3fd_face_detector_component(
            source, mapping, repeat);
        check(repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "vision detector conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_s3fd_face_detector_component(
                source, mapping, scratch / "cancel.vrm", [] { return true; });
        });
        expect(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_s3fd_face_detector_component(
                scratch / "missing", mapping, scratch / "missing.vrm");
        });
        const fs::path corrupt_source = scratch / "corrupt-source";
        fs::create_directories(corrupt_source);
        std::ofstream(corrupt_source / "s3fd-619a316812.pth").put('\0');
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_s3fd_face_detector_component(
                corrupt_source, mapping, scratch / "corrupt.vrm");
        });
        auto altered_mapping = [&](const std::string& before,
                                   const std::string& after,
                                   const fs::path& path) {
            std::ifstream input(mapping);
            std::string content((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
            const size_t position = content.find(before);
            check(position != std::string::npos,
                  "tensor-map failure fixture field is missing");
            content.replace(position, before.size(), after);
            std::ofstream(path) << content;
        };
        const fs::path wrong_dtype = scratch / "wrong-dtype.tsv";
        altered_mapping("\tF32\t", "\tF16\t", wrong_dtype);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_s3fd_face_detector_component(
                source, wrong_dtype, scratch / "wrong-dtype.vrm");
        });
        const fs::path wrong_shape = scratch / "wrong-shape.tsv";
        altered_mapping("\t64,3,3,3\t", "\t64,3,3,4\t", wrong_shape);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_s3fd_face_detector_component(
                source, wrong_shape, scratch / "wrong-shape.vrm");
        });
        for (const auto& entry : fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-") ==
                      std::string::npos,
                  "conversion left a partial publication artifact");

        const uint8_t rgb[] = {104, 117, 123, 105, 119, 126};
        const auto preprocessed = vrhino::preprocess_detector_rgb_u8(rgb, 1, 2);
        check(preprocessed == std::vector<float>({0, 1, 0, 2, 0, 3}),
              "detector RGB preprocessing mismatch");
        vrhino::FaceROI a{0, 0, 9, 9, .9f, 0, 0};
        vrhino::FaceROI b{5, 5, 14, 14, .8f, 1, 0};
        check(std::abs(vrhino::face_roi_iou_inclusive(a, b) - 25.0f / 175.0f) < 1e-7f,
              "inclusive IoU mismatch");
        auto kept = vrhino::deterministic_face_nms({a, b}, 0.1f);
        check(kept.size() == 1 && kept[0].source_index == 0,
              "deterministic NMS mismatch");
        vrhino::FaceROI later{0, 0, 9, 9, .9f, 8, 0};
        vrhino::FaceROI earlier{0, 0, 9, 9, .9f, 4, 0};
        kept = vrhino::deterministic_face_nms({later, earlier}, 0.3f);
        check(kept.size() == 1 && kept[0].source_index == 4,
              "detector score-tie ordering mismatch");

        std::vector<vrhino::DetectorScaleHost> scales;
        for (int index = 0; index < 6; ++index) {
            const float face_logit = index == 0 ? std::log(9.0f) : -100.0f;
            scales.push_back({1, 1, 1, {0.0f, face_logit},
                              {0.0f, 0.0f, 0.0f, 0.0f}});
        }
        const auto stages = vrhino::decode_multiscale_face_detector(scales);
        check(stages.decoded_candidates.size() == 1 &&
              stages.thresholded_candidates.size() == 1 &&
              stages.post_nms.size() == 1 && stages.selected.has_value(),
              "detector decode/threshold fixture mismatch");
        check(std::abs(stages.selected->confidence - 0.9f) < 1e-6f &&
              std::abs(stages.selected->x1 + 6.0f) < 1e-6f &&
              std::abs(stages.selected->y1 + 6.0f) < 1e-6f &&
              std::abs(stages.selected->x2 - 10.0f) < 1e-6f &&
              std::abs(stages.selected->y2 - 10.0f) < 1e-6f,
              "detector anchor/prior decode mismatch");

        fs::remove_all(scratch, error);
        std::cout << "Vision detector CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch, error);
        std::cerr << "Vision detector CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
