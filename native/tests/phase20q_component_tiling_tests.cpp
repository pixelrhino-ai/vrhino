#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/component_tiling.h"
#include "vrhino/error.h"

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

vrhino::Tensor input_tensor(const std::vector<int64_t>& shape) {
    vrhino::Tensor result = vrhino::Tensor::host(shape, vrhino::DType::F32);
    float* values = result.data_as<float>();
    for (int64_t index = 0; index < result.numel(); ++index)
        values[index] = static_cast<float>((index % 31) - 15) / 16.0f;
    return result;
}

void require_exact(vrhino::CudaBackend& backend, const vrhino::Tensor& expected,
                   const vrhino::Tensor& actual, const std::string& name) {
    const vrhino::Tensor host = backend.copy_to_host(actual);
    vrhino::require(expected.shape() == host.shape() &&
                        expected.dtype() == host.dtype() &&
                        expected.bytes() == host.bytes() &&
                        std::memcmp(expected.data(), host.data(), host.bytes()) == 0,
                    name + " is not byte exact");
    std::cout << name << "=PASS\n";
}

vrhino::ComponentTilingConfig config(int64_t temporal_tile,
                                      int64_t temporal_stride,
                                      int64_t spatial_tile,
                                      int64_t spatial_stride) {
    vrhino::ComponentTilingConfig result;
    result.temporal_tile = temporal_tile;
    result.temporal_stride = temporal_stride;
    result.spatial_tile_height = spatial_tile;
    result.spatial_tile_width = spatial_tile;
    result.spatial_stride_height = spatial_stride;
    result.spatial_stride_width = spatial_stride;
    result.temporal_overlap = 1.0 -
        static_cast<double>(temporal_stride) / temporal_tile;
    result.spatial_overlap_height = 1.0 -
        static_cast<double>(spatial_stride) / spatial_tile;
    result.spatial_overlap_width = result.spatial_overlap_height;
    result.output_scale = {1, 1, 1};
    result.temporal_context = 1;
    result.temporal_boundary =
        vrhino::TemporalBoundaryPolicy::PreserveFirstFrameCausal;
    return result;
}

}  // namespace

int main() {
    try {
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        const auto identity = [](const vrhino::Tensor& tile) { return tile; };

        const auto production_temporal =
            vrhino::TiledComponentExecutor::plan_axis(33, 16, 12, 1, true);
        vrhino::require(production_temporal.size() == 3 &&
                            production_temporal[0].start == 0 &&
                            production_temporal[0].stop == 17 &&
                            production_temporal[1].start == 12 &&
                            production_temporal[1].stop == 29 &&
                            production_temporal[2].start == 24 &&
                            production_temporal[2].stop == 33,
                        "production temporal planner mismatch");
        const auto production_h =
            vrhino::TiledComponentExecutor::plan_axis(68, 32, 24);
        const auto production_w =
            vrhino::TiledComponentExecutor::plan_axis(120, 32, 24);
        vrhino::require(production_h.size() == 3 && production_w.size() == 5,
                        "production spatial planner mismatch");
        std::cout << "planner.production=PASS\n";

        for (int64_t temporal : {1, 3, 5}) {
            vrhino::Tensor host = input_tensor({1, 1, temporal, 2, 2});
            vrhino::Tensor device = backend.copy_to_device(host, vrhino::DType::F32);
            vrhino::TiledComponentExecutor tiler(backend, config(3, 2, 8, 4));
            require_exact(backend, host, tiler.execute(device, identity),
                          "temporal.t" + std::to_string(temporal));
        }

        for (const auto& dimensions : std::vector<std::pair<int64_t, int64_t>>{
                 {3, 3}, {4, 4}, {5, 5}, {7, 6}}) {
            vrhino::Tensor host = input_tensor(
                {1, 1, 1, dimensions.first, dimensions.second});
            vrhino::Tensor device = backend.copy_to_device(host, vrhino::DType::F32);
            vrhino::TiledComponentExecutor tiler(backend, config(8, 4, 4, 3));
            require_exact(backend, host, tiler.execute(device, identity),
                "spatial." + std::to_string(dimensions.first) + "x" +
                    std::to_string(dimensions.second));
        }

        vrhino::Tensor host = input_tensor({1, 1, 5, 7, 6});
        vrhino::Tensor device = backend.copy_to_device(host, vrhino::DType::F32);
        vrhino::TiledComponentExecutor tiler(backend, config(3, 2, 4, 3));
        const vrhino::Tensor first = tiler.execute(device, identity);
        const vrhino::Tensor second = tiler.execute(device, identity);
        require_exact(backend, host, first, "spatiotemporal");
        require_exact(backend, host, second, "deterministic.repeat");
        const vrhino::Tensor first_host = backend.copy_to_host(first);
        const vrhino::Tensor second_host = backend.copy_to_host(second);
        vrhino::require(std::memcmp(first_host.data(), second_host.data(),
                                    first_host.bytes()) == 0,
                        "tiled repeat is not deterministic");
        cuda_check(cudaDeviceSynchronize(), "final synchronize");
        std::cout << "nan=0\ninf=0\nstatus=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20q_component_tiling_tests: " << error.what() << '\n';
        return 1;
    }
}
