#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/component_tiling.h"
#include "vrhino/error.h"
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"

namespace {

class IdentityArchitecture final : public vrhino::Architecture {
public:
    std::unique_ptr<vrhino::Denoiser> create_denoiser(
            vrhino::Backend&, const vrhino::PrecisionPolicy&,
            const vrhino::TensorBundle&) override {
        throw vrhino::Error("Identity Architecture has no denoiser");
    }
    vrhino::SamplingProgram create_program(
            const vrhino::TensorBundle&) override {
        throw vrhino::Error("Identity Architecture has no Sampling Program");
    }
    vrhino::Tensor decode(vrhino::Backend&, const vrhino::PrecisionPolicy&,
                          const vrhino::Tensor& latent,
                          const vrhino::TensorBundle&) override {
        ++calls;
        return latent;
    }
    uint64_t calls = 0;
};

vrhino::ComponentExecutionConfig tiled_config() {
    return vrhino::component_execution_config_from_json(vrhino::Json::parse(R"({
        "mode": "TILED",
        "tiling": {
            "temporal_tile": 3,
            "temporal_stride": 2,
            "spatial_tile_height": 4,
            "spatial_tile_width": 4,
            "spatial_stride_height": 3,
            "spatial_stride_width": 3,
            "temporal_overlap": 0.3333333333333333,
            "spatial_overlap_height": 0.25,
            "spatial_overlap_width": 0.25,
            "output_scale": [1, 1, 1],
            "temporal_context": 1,
            "temporal_boundary": "PRESERVE_FIRST_FRAME_CAUSAL",
            "crop_policy": "STRIDE_CORE",
            "blend_policy": "LINEAR"
        }
    })"));
}

void expect_failure(const std::string& json, const std::string& name) {
    bool failed = false;
    try {
        static_cast<void>(vrhino::component_execution_config_from_json(
            vrhino::Json::parse(json)));
    } catch (const std::exception&) {
        failed = true;
    }
    vrhino::require(failed, name + " did not fail closed");
}

}  // namespace

int main() {
    try {
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        const vrhino::PrecisionPolicy policy =
            vrhino::PrecisionPolicy::unqualified_default(vrhino::DType::F32);
        const vrhino::Tensor host = vrhino::host_f32(
            {1, 1, 7, 7, 6}, std::vector<float>(294, 0.25f));
        const vrhino::Tensor device =
            backend.copy_to_device(host, vrhino::DType::F32);

        IdentityArchitecture untiled_architecture;
        vrhino::NativeRuntime untiled_runtime(backend, policy);
        const vrhino::Tensor untiled = untiled_runtime.decode_component(
            untiled_architecture, device, {});
        const auto& untiled_stats = untiled_runtime.component_execution_stats();
        vrhino::require(untiled_architecture.calls == 1 &&
                            untiled_stats.mode ==
                                vrhino::ComponentExecutionMode::Untiled &&
                            untiled_stats.tiling.graph_executions == 1,
                        "UNTILED production selector mismatch");

        IdentityArchitecture tiled_architecture;
        vrhino::NativeRuntime tiled_runtime(backend, policy, tiled_config());
        const vrhino::Tensor tiled = tiled_runtime.decode_component(
            tiled_architecture, device, {});
        backend.synchronize();
        const auto& tiled_stats = tiled_runtime.component_execution_stats();
        vrhino::require(tiled_stats.mode ==
                                vrhino::ComponentExecutionMode::Tiled &&
                            tiled_stats.tiling.temporal_tiles == 3 &&
                            tiled_stats.tiling.spatial_tiles_per_temporal == 6 &&
                            tiled_stats.tiling.graph_executions == 18 &&
                            tiled_architecture.calls == 18,
                        "TILED production selector mismatch");
        const vrhino::Tensor untiled_host = backend.copy_to_host(untiled);
        const vrhino::Tensor tiled_host = backend.copy_to_host(tiled);
        vrhino::require(untiled_host.bytes() == tiled_host.bytes() &&
                            std::memcmp(untiled_host.data(), tiled_host.data(),
                                        untiled_host.bytes()) == 0,
                        "TILED identity execution changed output");

        expect_failure(R"({"mode":"TILED"})", "missing tiling config");
        expect_failure(
            R"({"mode":"UNTILED","tiling":{}})",
            "unexpected UNTILED config");
        expect_failure(R"({"mode":"UNKNOWN"})", "unknown execution mode");
        expect_failure(R"({"mode":"TILED","tiling":{}})",
                       "incomplete tiling config");

        std::cout << "untiled=PASS\ntiled=PASS\n"
                  << "tiled_temporal_tiles="
                  << tiled_stats.tiling.temporal_tiles << '\n'
                  << "tiled_spatial_tiles_per_temporal="
                  << tiled_stats.tiling.spatial_tiles_per_temporal << '\n'
                  << "tiled_graph_executions="
                  << tiled_stats.tiling.graph_executions << '\n'
                  << "fail_closed=PASS\nstatus=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20w_component_execution_binding_tests: "
                  << error.what() << '\n';
        return 1;
    }
}
