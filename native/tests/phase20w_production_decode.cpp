#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/component_tiling.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/runtime.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path);
    vrhino::require(static_cast<bool>(stream), "Cannot open config: " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 4,
            "usage: vrhino-phase20w-production-decode MODEL LATENT PROFILE");
        vrhino::VrmModel model(argv[1], false);
        const vrhino::TensorBundle fixture = vrhino::read_bundle(argv[2]);
        const vrhino::Tensor& latent = fixture.at("latent");
        vrhino::require(latent.device().is_host() &&
                            latent.dtype() == vrhino::DType::BF16 &&
                            latent.shape() ==
                                std::vector<int64_t>({1, 16, 33, 68, 120}),
                        "Phase 20W production latent contract mismatch");
        const vrhino::Json profile = vrhino::Json::parse(read_text(argv[3]));
        const vrhino::ComponentExecutionConfig execution =
            vrhino::component_execution_config_from_json(
                profile.at("component_execution"));

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        backend.configure_memory_runtime(
            vrhino::MemoryBudget{64ULL << 30, 2ULL << 30, 64ULL << 30,
                                 2ULL << 30, 1ULL << 30},
            vrhino::MemoryRuntimeOptions{true, false, false});
        backend.set_vrm_mapped_bytes(model.file_size());
        auto architecture = vrhino::create_architecture(model);
        const vrhino::PrecisionPolicy policy =
            vrhino::PrecisionPolicy::unqualified_default(vrhino::DType::BF16);
        vrhino::NativeRuntime runtime(backend, policy, execution);

        cudaDeviceSynchronize();
        const auto started = std::chrono::steady_clock::now();
        const vrhino::Tensor video_device = runtime.decode_component(
            *architecture, latent, fixture);
        backend.synchronize();
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const vrhino::Tensor video = backend.copy_to_host(video_device);
        vrhino::require(video.dtype() == vrhino::DType::BF16 &&
                            video.shape() ==
                                std::vector<int64_t>({1, 3, 129, 544, 960}),
                        "Phase 20W production video contract mismatch");
        uint64_t nan = 0;
        uint64_t inf = 0;
        for (int64_t index = 0; index < video.numel(); ++index) {
            const float value = bf16_to_float(video.data_as<uint16_t>()[index]);
            nan += std::isnan(value);
            inf += std::isinf(value);
        }
        vrhino::require(nan == 0 && inf == 0,
                        "Phase 20W production video is non-finite");
        const auto& stats = runtime.component_execution_stats();
        vrhino::require(stats.mode == vrhino::ComponentExecutionMode::Tiled &&
                            stats.tiling.temporal_tiles == 3 &&
                            stats.tiling.spatial_tiles_per_temporal == 15 &&
                            stats.tiling.graph_executions == 45,
                        "Phase 20W production executor telemetry mismatch");

        std::cout << "status=PASS\nexecutor_mode="
                  << vrhino::component_execution_mode_name(stats.mode)
                  << "\ntemporal_tiles=" << stats.tiling.temporal_tiles
                  << "\nspatial_tiles_per_temporal="
                  << stats.tiling.spatial_tiles_per_temporal
                  << "\ngraph_executions=" << stats.tiling.graph_executions
                  << "\noutput_shape=[1,3,129,544,960]"
                  << "\nvae_wall_seconds=" << wall_seconds
                  << "\npeak_device_bytes=" << backend.peak_device_bytes()
                  << "\nnan=" << nan << "\ninf=" << inf << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20w_production_decode: " << error.what() << '\n';
        return 1;
    }
}
