#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>

#include "vrhino/face_mask.h"
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
        std::cerr << "usage: semantic_segmenter_tests SOURCE_DIR TENSOR_MAP OUTPUT_VRM\n";
        return 2;
    }
    const fs::path source = argv[1], mapping = argv[2], output = argv[3];
    const fs::path scratch = output.parent_path() /
        ("semantic-segmenter-tests-" + std::to_string(getpid()));
    std::error_code error; fs::remove(output, error); fs::remove_all(scratch, error);
    fs::create_directories(scratch);
    try {
        auto converted = product::convert_bisenet_face_parser_component(
            source, mapping, output);
        check(converted.source_checkpoint_bytes == 53289463ULL &&
              converted.retained_tensor_bytes == 52650752ULL &&
              converted.retained_tensor_count == 148,
              "semantic segmenter source inventory drift");
        vrhino::VrmModel component(output.string(), true);
        check(component.profile_id() == "component" &&
              component.architecture_id() == "segmenter-2d" &&
              component.tensors().size() == 148,
              "semantic segmenter component identity mismatch");
        const auto& graph = component.graph();
        check(graph.at("kind").string() == "semantic_segmenter_2d" &&
              graph.at("entry_point").string() == "execute" &&
              graph.at("config").at("class_count").integer() == 19 &&
              graph.at("config").at("bilinear_align_corners").boolean(),
              "semantic segmenter graph contract drift");
        const fs::path repeat = scratch / "repeat.vrm";
        auto repeated = product::convert_bisenet_face_parser_component(
            source, mapping, repeat);
        check(repeated.output_sha256 == converted.output_sha256 &&
              repeated.payload_blake2b128 == converted.payload_blake2b128,
              "semantic segmenter conversion is not deterministic");
        expect(product::ModelPackageErrorCode::Cancelled, [&] {
            product::convert_bisenet_face_parser_component(
                source, mapping, scratch / "cancel.vrm", [] { return true; });
        });
        expect(product::ModelPackageErrorCode::ArtifactMissing, [&] {
            product::convert_bisenet_face_parser_component(
                scratch / "missing", mapping, scratch / "missing.vrm");
        });
        const fs::path corrupt = scratch / "corrupt"; fs::create_directories(corrupt);
        std::ofstream(corrupt / "79999_iter.pth").put('\0');
        expect(product::ModelPackageErrorCode::SourceIntegrityFailed, [&] {
            product::convert_bisenet_face_parser_component(
                corrupt, mapping, scratch / "corrupt.vrm");
        });
        auto alter = [&](const std::string& before, const std::string& after,
                         const fs::path& path) {
            std::ifstream input(mapping);
            std::string content((std::istreambuf_iterator<char>(input)), {});
            const size_t position = content.find(before);
            check(position != std::string::npos, "mapping fixture field missing");
            content.replace(position, before.size(), after);
            std::ofstream(path) << content;
        };
        const fs::path wrong_dtype = scratch / "wrong-dtype.tsv";
        alter("\tF32\t", "\tF16\t", wrong_dtype);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_bisenet_face_parser_component(
                source, wrong_dtype, scratch / "wrong-dtype.vrm");
        });
        const fs::path wrong_shape = scratch / "wrong-shape.tsv";
        alter("\t64,3,7,7\t", "\t64,3,7,8\t", wrong_shape);
        expect(product::ModelPackageErrorCode::PackageInvalid, [&] {
            product::convert_bisenet_face_parser_component(
                source, wrong_shape, scratch / "wrong-shape.vrm");
        });
        for (const auto& entry : fs::directory_iterator(scratch))
            check(entry.path().filename().string().find(".partial-") ==
                      std::string::npos,
                  "conversion left partial publication data");

        auto logits = vrhino::Tensor::host({1, 3, 2, 2}, vrhino::DType::F32);
        const float fixture[] = {1,4,2,0, 3,4,1,0, 2,1,5,0};
        std::copy(std::begin(fixture), std::end(fixture), logits.data_as<float>());
        check(vrhino::semantic_argmax_first(logits).labels ==
                  std::vector<uint8_t>({1,0,2,0}),
              "first-max semantic argmax mismatch");

        std::vector<uint8_t> uniform(704 * 1216 * 3, 128);
        auto prep = vrhino::preprocess_face_parser_bgr_u8(
            uniform.data(), 1216, 704, {221,276,464,534});
        check(prep.crop_box == std::array<int32_t,4>({149,212,535,598}) &&
              prep.normalized_nchw.shape() ==
                  std::vector<int64_t>({1,3,512,512}),
              "face parser preprocessing contract mismatch");

        vrhino::SemanticLabelMap jaw{512,512,std::vector<uint8_t>(512*512,0)};
        for (int y=200;y<350;++y) for (int x=160;x<350;++x)
            jaw.labels[static_cast<size_t>(y)*512+x]=1;
        vrhino::JawMaskObservation observation;
        auto alpha=vrhino::build_musetalk_jaw_alpha_mask(
            jaw,{221,276,464,534},{149,212,535,598},90,90,.5,&observation);
        check(alpha.width==386&&alpha.height==386&&
              *std::max_element(alpha.values.begin(),alpha.values.end())==255&&
              std::all_of(observation.lower_half.begin(),
                          observation.lower_half.begin()+193*386,
                          [](uint8_t value){return value==0;}),
              "bounded jaw mask fixture mismatch");

        fs::remove_all(scratch,error);
        std::cout << "Semantic segmenter CPU tests: PASS\n"
                  << "component_bytes=" << converted.output_bytes << "\n"
                  << "component_sha256=" << converted.output_sha256 << "\n"
                  << "component_blake2b128=" << converted.payload_blake2b128 << "\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(scratch,error);
        std::cerr << "Semantic segmenter CPU tests: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
