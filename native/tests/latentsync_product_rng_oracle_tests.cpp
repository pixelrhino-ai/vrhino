#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "npy_fixture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/lip_sync_diffusion_workflow.h"

namespace fs = std::filesystem;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Difference {
    double maximum = 0.0;
    double mean = 0.0;
};

Difference compare(vrhino::CudaBackend& backend, const vrhino::Tensor& actual,
                   const vrhino::Tensor& reference) {
    const vrhino::Tensor host = backend.copy_to_host(actual);
    check(host.shape() == reference.shape() &&
              host.dtype() == vrhino::DType::F32 &&
              reference.dtype() == vrhino::DType::F32,
          "RNG tensor contract mismatch");
    long double total = 0.0;
    Difference result;
    for (int64_t index = 0; index < host.numel(); ++index) {
        const double delta = std::abs(
            static_cast<double>(host.data_as<float>()[index]) -
            static_cast<double>(reference.data_as<float>()[index]));
        result.maximum = std::max(result.maximum, delta);
        total += delta;
    }
    result.mean = static_cast<double>(total / host.numel());
    return result;
}

vrhino::Tensor generated(vrhino::CudaBackend& backend, uint64_t seed,
                         uint64_t offset, const std::vector<int64_t>& shape) {
    vrhino::RngState state{seed, offset, "pytorch_compat.v1"};
    vrhino::Tensor result = backend.rng_normal(state, shape, vrhino::DType::F32);
    check(state.offset == offset + 4, "RNG state offset advance mismatch");
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: rng-oracle-tests ORACLE-ROOT");
        const fs::path root = fs::path(argv[1]) / "rng";
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        double worst = 0.0;
        auto qualify = [&](const std::string& name, uint64_t seed,
                           uint64_t offset,
                           const std::vector<int64_t>& shape,
                           const fs::path& path) {
            const vrhino::Tensor reference =
                vrhino::test::read_npy_f32(path.string());
            const Difference difference = compare(
                backend, generated(backend, seed, offset, shape), reference);
            worst = std::max(worst, difference.maximum);
            std::cout << name << " max_abs=" << difference.maximum
                      << " mean_abs=" << difference.mean << '\n';
            check(difference.maximum <= 4.0e-6,
                  name + " independent RNG vector mismatch");
        };

        qualify("scalar", 5701, 0, {1},
                root / "vectors/seed_5701_offset_0_1.npy");
        qualify("length2", 5701, 0, {2},
                root / "vectors/seed_5701_offset_0_2.npy");
        qualify("odd", 5701, 0, {7},
                root / "vectors/seed_5701_offset_0_7.npy");
        qualify("offset4", 99, 4, {3, 5},
                root / "vectors/seed_99_offset_4_3x5.npy");
        qualify("rank3-offset12", 0xfedcba9876543210ULL, 12, {2, 3, 4},
                root / "vectors/seed_18364758544493064720_offset_12_2x3x4.npy");

        const std::vector<vrhino::LipSyncDiffusionRngBranch> branches = {
            vrhino::LipSyncDiffusionRngBranch::MaskedSource,
            vrhino::LipSyncDiffusionRngBranch::ReferenceSource};
        for (int64_t frame = 0; frame < 17; ++frame) {
            for (const auto branch : branches) {
                const bool masked =
                    branch == vrhino::LipSyncDiffusionRngBranch::MaskedSource;
                vrhino::RngState state =
                    vrhino::lip_sync_diffusion_rng(1247, frame, branch);
                const fs::path path = root /
                    (masked ? "masked_source" : "reference_source") /
                    ("frame_" + (frame < 10 ? std::string("0") : std::string()) +
                     std::to_string(frame) + ".npy");
                qualify((masked ? "masked." : "reference.") +
                            std::to_string(frame),
                        state.seed, state.offset, {1, 4, 64, 64}, path);
            }
        }
        for (const int64_t start : {0, 16}) {
            vrhino::RngState state = vrhino::lip_sync_diffusion_rng(
                1247, start,
                vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
            qualify("initial." + std::to_string(start), state.seed, state.offset,
                    {1, 4, 1, 64, 64},
                    root / "initial_noise" /
                        ("chunk_" + (start < 10 ? std::string("0") : std::string()) +
                         std::to_string(start) + "_base.npy"));
        }

        // Derivation is pure: evaluating branches in any order yields the same
        // state, and seed/index/branch changes remain separated.
        const auto masked_a = vrhino::lip_sync_diffusion_rng(
            1247, 3, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        const auto reference = vrhino::lip_sync_diffusion_rng(
            1247, 3, vrhino::LipSyncDiffusionRngBranch::ReferenceSource);
        const auto chunk = vrhino::lip_sync_diffusion_rng(
            1247, 16, vrhino::LipSyncDiffusionRngBranch::InitialDiffusionNoise);
        const auto masked_b = vrhino::lip_sync_diffusion_rng(
            1247, 3, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        const auto other_seed = vrhino::lip_sync_diffusion_rng(
            1248, 3, vrhino::LipSyncDiffusionRngBranch::MaskedSource);
        check(masked_a.seed == masked_b.seed && masked_a.offset == masked_b.offset,
              "semantic stream depends on derivation call order");
        check(masked_a.seed != reference.seed && masked_a.seed != chunk.seed &&
                  masked_a.seed != other_seed.seed,
              "semantic RNG streams are not separated");
        std::cout << "call_order_independence=PASS branch_separation=PASS "
                  << "seed_sensitivity=PASS worst_max_abs=" << worst << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LatentSync generated-RNG oracle tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
