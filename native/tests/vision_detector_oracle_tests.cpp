#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "npy_fixture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/face_roi.h"
#include "vrhino/loader.h"
#include "vrhino/vision_detector.h"

namespace fs = std::filesystem;
namespace {
struct Difference {
    double max_abs = 0, mean_abs = 0, cosine = 0;
    double reference_min = std::numeric_limits<double>::infinity();
    double reference_max = -std::numeric_limits<double>::infinity();
    double native_min = std::numeric_limits<double>::infinity();
    double native_max = -std::numeric_limits<double>::infinity();
    uint64_t nan = 0, inf = 0;
};
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
Difference compare(const vrhino::Tensor& actual, const vrhino::Tensor& expected) {
    check(actual.device().is_host() && expected.device().is_host() &&
          actual.dtype() == vrhino::DType::F32 && expected.dtype() == vrhino::DType::F32 &&
          actual.shape() == expected.shape(), "oracle tensor contract mismatch");
    Difference d; long double sum=0,dot=0,an=0,bn=0;
    for(int64_t i=0;i<actual.numel();++i){double a=actual.data_as<float>()[i],b=expected.data_as<float>()[i];if(std::isnan(a))++d.nan;if(std::isinf(a))++d.inf;d.native_min=std::min(d.native_min,a);d.native_max=std::max(d.native_max,a);d.reference_min=std::min(d.reference_min,b);d.reference_max=std::max(d.reference_max,b);double e=std::abs(a-b);d.max_abs=std::max(d.max_abs,e);sum+=e;dot+=a*b;an+=a*a;bn+=b*b;}
    d.mean_abs=static_cast<double>(sum/actual.numel());d.cosine=static_cast<double>(dot/std::sqrt(an*bn));return d;
}
Difference qualify(const std::string& name, vrhino::CudaBackend& backend,
                   const vrhino::Tensor& actual, const fs::path& expected,
                   double max_abs, double cosine, bool emit = true) {
    auto d=compare(backend.copy_to_host(actual),vrhino::test::read_npy_f32(expected.string()));
    if (emit)
        std::cout<<std::setprecision(10)<<name<<" ref=["<<d.reference_min<<','<<d.reference_max<<"] native=["<<d.native_min<<','<<d.native_max<<"] max_abs="<<d.max_abs<<" mean_abs="<<d.mean_abs<<" cosine="<<d.cosine<<" nan="<<d.nan<<" inf="<<d.inf<<'\n';
    check(d.nan==0&&d.inf==0&&d.max_abs<=max_abs&&d.cosine>=cosine,name+" exceeds FP32 gate");return d;
}
std::string frame_name(int frame){std::ostringstream s;s<<"frame_"<<std::setw(2)<<std::setfill('0')<<frame;return s.str();}
std::vector<float> values(const vrhino::Tensor& tensor) {
    return std::vector<float>(tensor.data_as<float>(),
                              tensor.data_as<float>() + tensor.numel());
}
std::vector<vrhino::DetectorScaleHost> host_scales(
        vrhino::CudaBackend& backend, const vrhino::VisionDetectorResult& result) {
    std::vector<vrhino::DetectorScaleHost> scales;
    for (const auto& scale : result.scales) {
        auto c=backend.copy_to_host(scale.confidence_logits);
        auto l=backend.copy_to_host(scale.localization);
        scales.push_back({c.dim(0),c.dim(2),c.dim(3),values(c),values(l)});
    }
    return scales;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: vision_detector_oracle_tests COMPONENT_VRM "
                     "INPUT_ROOT RAW_ORACLE_ROOT PHASE1_FACE_ROOT\n";
        return 2;
    }
    try {
        vrhino::VrmModel component(argv[1], true);
        check(component.architecture_id() == "vision-detector",
              "component identity mismatch");
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_weight_cache(true);
        vrhino::VisionDetectorComponentExecutor executor(
            backend, vrhino::WeightMap(component.bindings(component.graph())));
        const fs::path oracle = argv[3], authoritative = argv[4];
        check(fs::is_directory(argv[2]), "authoritative input root is missing");
        check(fs::is_directory(authoritative),
              "authoritative face oracle is missing");
        double total_ms = 0, worst_box = 0, worst_score = 0;
        double preprocessing_ms = 0, postprocess_ms = 0;
        double worst_authoritative_box = 0, worst_authoritative_score = 0;
        int worst_frame = 0, worst_authoritative_frame = 0;
        for (int frame = 0; frame < 8; ++frame) {
            const std::string name = frame_name(frame);
            auto input = vrhino::test::read_npy_f32(
                (oracle / name / "preprocessed.npy").string());
            check(input.shape() == std::vector<int64_t>({1, 3, 1216, 704}),
                  "detector input fixture shape mismatch");
            const int64_t area = input.dim(2) * input.dim(3);
            std::vector<uint8_t> rgb(static_cast<size_t>(3 * area));
            constexpr float means[3] = {104.0f, 117.0f, 123.0f};
            for (int64_t pixel = 0; pixel < area; ++pixel)
                for (int channel = 0; channel < 3; ++channel)
                    rgb[static_cast<size_t>(3 * pixel + channel)] =
                        static_cast<uint8_t>(input.data_as<float>()[channel * area + pixel] +
                                             means[channel]);
            const auto preprocess_start = std::chrono::steady_clock::now();
            const auto native_preprocessed = vrhino::preprocess_detector_rgb_u8(
                rgb.data(), input.dim(2), input.dim(3));
            preprocessing_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - preprocess_start).count();
            check(native_preprocessed.size() == static_cast<size_t>(input.numel()) &&
                  std::equal(native_preprocessed.begin(), native_preprocessed.end(),
                             input.data_as<float>()),
                  "detector preprocessing differs from frozen reference");
            vrhino::VisionDetectorObservation observation;
            backend.synchronize();
            auto start = std::chrono::steady_clock::now();
            auto result = executor.execute(component.graph(), input,
                                           frame == 0 ? &observation : nullptr);
            backend.synchronize();
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (frame > 0) total_ms += ms;
            if (frame == 0) {
                qualify("early_backbone", backend,
                        observation.tensors.at("early_backbone"),
                        oracle / name / "early_backbone.npy", 2e-4, 0.999999);
                qualify("pre_normalized_feature", backend,
                        observation.tensors.at("pre_normalized_feature"),
                        oracle / name / "pre_normalized_feature.npy", 1.5e-1,
                        0.9999999);
                qualify("normalized_feature", backend,
                        observation.tensors.at("normalized_feature"),
                        oracle / name / "normalized_feature.npy", 3e-2, 0.999999);
                qualify("localization_head", backend,
                        observation.tensors.at("first_localization_head"),
                        oracle / name / "localization_head_0.npy", 1.5e-2,
                        0.9999998);
                qualify("confidence_maxout", backend,
                        observation.tensors.at("first_confidence_maxout"),
                        oracle / name / "confidence_head_0_maxout.npy", 1.5e-2,
                        0.9999998);
            }
            for (int scale = 0; scale < 6; ++scale) {
                char suffix[16];
                std::snprintf(suffix, sizeof(suffix), "raw_%02d.npy", 2 * scale);
                qualify(name + ".confidence_" + std::to_string(scale), backend,
                        result.scales[scale].confidence_logits,
                        oracle / name / suffix, 2e-2, 0.9999995, frame == 0);
                std::snprintf(suffix, sizeof(suffix), "raw_%02d.npy", 2 * scale + 1);
                qualify(name + ".localization_" + std::to_string(scale), backend,
                        result.scales[scale].localization,
                        oracle / name / suffix, 2e-2, 0.9999995, frame == 0);
            }
            const auto postprocess_start = std::chrono::steady_clock::now();
            auto stages = vrhino::decode_multiscale_face_detector(
                host_scales(backend, result));
            postprocess_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - postprocess_start).count();
            check(stages.selected.has_value(), name + " has no selected ROI");
            auto reference = vrhino::test::read_npy_f32(
                (oracle / name / "post_nms.npy").string());
            check(reference.dim(0) == 1 && reference.dim(1) == 5,
                  "reference detection shape mismatch");
            const float* r = reference.data_as<float>();
            const auto& a = *stages.selected;
            const double box = std::max({std::abs(a.x1 - r[0]),
                std::abs(a.y1 - r[1]), std::abs(a.x2 - r[2]),
                std::abs(a.y2 - r[3])});
            const double score = std::abs(a.confidence - r[4]);
            if (box > worst_box) {
                worst_box = box; worst_score = score; worst_frame = frame;
            }
            check(box <= 0.02 && score <= 1e-6, name + " ROI parity failed");
            std::cout << name << " candidates=" << stages.decoded_candidates.size()
                      << " post_nms=" << stages.post_nms.size() << " selected=["
                      << a.x1 << ',' << a.y1 << ',' << a.x2 << ',' << a.y2
                      << ',' << a.confidence << "] reference=[" << r[0] << ','
                      << r[1] << ',' << r[2] << ',' << r[3] << ',' << r[4]
                      << "] box_abs=" << box << " ms=" << ms << '\n';

            auto phase1 = vrhino::test::read_npy_f32(
                (authoritative / (name + ".s3fd_post_nms.npy")).string());
            check(phase1.dim(0) >= 1 && phase1.dim(1) == 5,
                  "Phase-1 detection shape mismatch");
            const float* p = phase1.data_as<float>();
            const double phase1_box = std::max({std::abs(a.x1 - p[0]),
                std::abs(a.y1 - p[1]), std::abs(a.x2 - p[2]),
                std::abs(a.y2 - p[3])});
            const double phase1_score = std::abs(a.confidence - p[4]);
            if (phase1_box > worst_authoritative_box) {
                worst_authoritative_box = phase1_box;
                worst_authoritative_score = phase1_score;
                worst_authoritative_frame = frame;
            }
            check(phase1_box <= 0.03 && phase1_score <= 1e-6,
                  name + " authoritative ROI parity failed");
            check(static_cast<int>(a.x1) == static_cast<int>(p[0]) &&
                  static_cast<int>(a.y1) == static_cast<int>(p[1]) &&
                  static_cast<int>(a.x2) == static_cast<int>(p[2]) &&
                  static_cast<int>(a.y2) == static_cast<int>(p[3]),
                  name + " integer ROI decision differs from Phase-1");
            std::cout << name << " source_index=" << a.source_index
                      << " scale_index=" << a.scale_index
                      << " authoritative_box_abs=" << phase1_box << '\n';
        }
        auto timing_input = vrhino::test::read_npy_f32(
            (oracle / "frame_00" / "preprocessed.npy").string());
        backend.synchronize();
        const auto eighth_start = std::chrono::steady_clock::now();
        (void)executor.execute(component.graph(), timing_input);
        backend.synchronize();
        total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - eighth_start).count();
        std::cout << "Vision detector Native FP32 oracle tests: PASS\n"
                  << "preprocessing_eight_ms=" << preprocessing_ms
                  << "\nneural_eight_warmed_ms=" << total_ms
                  << "\npostprocess_eight_ms=" << postprocess_ms
                  << "\ntotal_eight_warmed_ms="
                  << (preprocessing_ms + total_ms + postprocess_ms)
                  << "\npeak_device_bytes="
                  << backend.peak_device_bytes() << "\nworst_frame=" << worst_frame
                  << "\nworst_box_abs=" << worst_box << "\nworst_score_abs="
                  << worst_score << "\nworst_authoritative_frame="
                  << worst_authoritative_frame << "\nworst_authoritative_box_abs="
                  << worst_authoritative_box << "\nworst_authoritative_score_abs="
                  << worst_authoritative_score << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Vision detector Native FP32 oracle tests: FAIL: "
                  << e.what() << '\n';
        return 1;
    }
}
