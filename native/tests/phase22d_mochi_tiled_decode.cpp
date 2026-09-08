#include <bit>
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
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

vrhino::Tensor zero_latent() {
    vrhino::Tensor result = vrhino::Tensor::host(
        {1, 12, 1, 60, 106}, vrhino::DType::BF16);
    std::memset(result.data(), 0, result.bytes());
    return result;
}

void require_plan(const std::vector<vrhino::ComponentTileRegion>& actual,
                  const std::vector<std::pair<int64_t, int64_t>>& expected,
                  const char* name) {
    vrhino::require(actual.size() == expected.size(),
                    std::string(name) + " plan count mismatch");
    for (size_t index = 0; index < expected.size(); ++index)
        vrhino::require(actual[index].start == expected[index].first &&
                            actual[index].stop == expected[index].second,
                        std::string(name) + " plan interval mismatch");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 3,
            "usage: vrhino-phase22d-mochi-tiled-decode MODEL CONFIG");
        vrhino::VrmModel model(argv[1], false);
        const vrhino::ComponentExecutionConfig execution =
            vrhino::component_execution_config_from_json(
                vrhino::Json::parse(read_text(argv[2])));

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        backend.configure_memory_runtime(
            vrhino::MemoryBudget{21ULL << 30, 2ULL << 30, 64ULL << 30,
                                 2ULL << 30, 1ULL << 30},
            vrhino::MemoryRuntimeOptions{true, false, false});
        backend.set_vrm_mapped_bytes(model.file_size());
        auto architecture = vrhino::create_architecture(model);
        const vrhino::PrecisionPolicy policy =
            vrhino::PrecisionPolicy::unqualified_default(vrhino::DType::BF16);
        vrhino::NativeRuntime runtime(backend, policy, execution);
        const vrhino::Tensor latent = zero_latent();

        const auto started = std::chrono::steady_clock::now();
        const vrhino::Tensor first_device = runtime.decode_component(
            *architecture, latent, {});
        backend.synchronize();
        const vrhino::Tensor first = backend.copy_to_host(first_device);
        const double first_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();

        const auto repeat_started = std::chrono::steady_clock::now();
        const vrhino::Tensor second_device = runtime.decode_component(
            *architecture, latent, {});
        backend.synchronize();
        const vrhino::Tensor second = backend.copy_to_host(second_device);
        const double repeat_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - repeat_started).count();

        vrhino::require(first.dtype() == vrhino::DType::BF16 &&
                            first.shape() ==
                                std::vector<int64_t>({1, 3, 1, 480, 848}) &&
                            first.bytes() == second.bytes(),
                        "Mochi recursive tiled short decode output mismatch");
        vrhino::require(std::memcmp(first.data(), second.data(), first.bytes()) == 0,
                        "Mochi recursive tiled short decode is not deterministic");
        uint64_t nan = 0;
        uint64_t inf = 0;
        for (int64_t index = 0; index < first.numel(); ++index) {
            const float value = bf16_to_float(first.data_as<uint16_t>()[index]);
            nan += std::isnan(value);
            inf += std::isinf(value);
        }
        vrhino::require(nan == 0 && inf == 0,
                        "Mochi recursive tiled short decode is non-finite");

        const auto& stats = runtime.component_execution_stats();
        vrhino::require(stats.mode == vrhino::ComponentExecutionMode::Tiled &&
                            stats.tiling.temporal_tiles == 1 &&
                            stats.tiling.spatial_tiles_per_temporal == 8 &&
                            stats.tiling.graph_executions == 8,
                        "Mochi recursive tiled execution telemetry mismatch");
        require_plan(stats.tiling.height_plan,
                     {{0, 34}, {26, 60}}, "height");
        require_plan(stats.tiling.width_plan,
                     {{0, 32}, {24, 57}, {49, 81}, {73, 106}}, "width");

        std::cout << "status=PASS\nexecutor_mode=TILED"
                  << "\ntemporal_tiles=1\nspatial_tiles=8"
                  << "\ngraph_executions=8"
                  << "\noutput_shape=[1,3,1,480,848]"
                  << "\nfirst_decode_seconds=" << first_seconds
                  << "\nrepeat_decode_seconds=" << repeat_seconds
                  << "\npeak_device_bytes=" << backend.peak_device_bytes()
                  << "\ndeterministic=PASS\nnan=" << nan
                  << "\ninf=" << inf << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase22d_mochi_tiled_decode: " << error.what() << '\n';
        return 1;
    }
}
