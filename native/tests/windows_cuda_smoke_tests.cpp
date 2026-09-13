#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/backend/cudnn_conv.h"
#include "vrhino/tensor_util.h"
#include "windows_cuda_resource_observer.h"

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Partial construction and exceptional paths retain ownership until cleanup.
struct Resources {
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (event) cudaEventDestroy(event);
        if (device) cudaFree(device);
        if (stream) cudaStreamDestroy(stream);
    }
    void close() {
        check(cudaStreamSynchronize(stream));
        check(cudaEventDestroy(event)); event = nullptr;
        check(cudaFree(device)); device = nullptr;
        check(cudaStreamDestroy(stream)); stream = nullptr;
    }
};

void cycle(bool interrupt) {
    Resources resources;
    check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&resources.event, cudaEventDisableTiming));
    constexpr size_t count = 1024 * 256;
    check(cudaMalloc(&resources.device, count * sizeof(float)));
    if (interrupt) throw std::runtime_error("injected host interruption");
    std::vector<float> input(count, 3.0f), output(count);
    check(cudaMemcpyAsync(resources.device, input.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice, resources.stream));
    check(cudaMemcpyAsync(output.data(), resources.device, count * sizeof(float),
                          cudaMemcpyDeviceToHost, resources.stream));
    check(cudaEventRecord(resources.event, resources.stream));
    check(cudaEventSynchronize(resources.event));
    check(cudaEventQuery(resources.event));
    require(output == input, "stream transfer mismatch");
    resources.close();

    vrhino::CudaBackend backend;
    require(backend.device_count() > 0, "backend device query failed");
    const auto host = vrhino::host_f32({4}, {1, 2, 3, 4});
    const auto device = backend.copy_to_device(host, vrhino::DType::F32);
    const auto result = backend.copy_to_host(backend.add(device, device));
    for (int index = 0; index < 4; ++index)
        require(result.data_as<float>()[index] == 2 * (index + 1), "backend add mismatch");
    const auto fence = backend.create_fence();
    backend.record_fence(fence);
    backend.wait_fence(fence);
    backend.synchronize();
    require(backend.query_fence(fence), "backend fence incomplete");
    backend.destroy_fence(fence);
    // Leave one completed fence for backend shutdown ownership to release.
    const auto shutdown_fence = backend.create_fence();
    backend.record_fence(shutdown_fence);
    backend.synchronize();
}

DWORD handles() {
    DWORD count = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &count) != FALSE, "handle query failed");
    return count;
}

void descriptor_boundaries() {
    struct Case { int64_t x, k, pad, dilation, stride, y, x_step; bool backend, legacy; };
    const Case cases[] = {
        {3, 3, 1, 1, 1, 3, 1, true, true},
        {INT_MAX, 1, 0, 1, 1, INT_MAX, 1, true, true},
        {int64_t{INT_MAX} + 1, 1, 0, 1, 1, int64_t{INT_MAX} + 1, 1, true, false},
        {INT64_MAX, 1, 0, 1, 1, INT64_MAX, 1, true, false},
        {INT64_MAX, 1, 0, 1, 1, INT64_MAX, 2, false, false},
        {INT64_MAX, 1, INT64_MAX / 2, 1, 2, INT64_MAX, 1, true, false},
        {INT64_MAX, 1, INT64_MAX / 2, 1, 1, INT64_MAX, 1, false, false},
        {1, 2, 0, INT64_MAX, 1, 1, 1, false, false},
        {INT64_MAX, 4, INT64_MAX / 2, INT64_MAX, 1, 1, 1, false, false},
        {7, 3, 0, 2, 2, 2, 1, true, true},
        {7, 3, 0, 2, 2, 3, 1, false, false},
    };
    for (const auto& item : cases) {
        vrhino::CudnnConvDescriptor value;
        value.spatial_rank = 2;
        value.x_dimensions = {1, 1, item.x, 1};
        value.w_dimensions = {1, 1, item.k, 1};
        value.y_dimensions = {1, 1, item.y, 1};
        value.x_strides = {1, 1, item.x_step, 1};
        value.w_strides = value.y_strides = {1, 1, 1, 1};
        value.padding = {item.pad, 0};
        value.dilation = {item.dilation, 1};
        value.convolution_strides = {item.stride, 1};
        require(vrhino::cudnn_backend_conv_admitted(value) == item.backend,
                "INT64 convolution admission mismatch");
        require(vrhino::cudnn_legacy_conv_admitted(value) == item.legacy,
                "legacy convolution admission mismatch");
    }
    std::cout << "descriptor_boundaries=22 PASS\n";
}
}

int main() {
    try {
        descriptor_boundaries();
        int devices = 0, runtime = 0, driver = 0;
        check(cudaGetDeviceCount(&devices));
        require(devices > 0, "no CUDA device");
        check(cudaSetDevice(0));
        check(cudaFree(nullptr));
        cuda_observer::Imports observation;
        observation.start();
        check(cudaRuntimeGetVersion(&runtime));
        check(cudaDriverGetVersion(&driver));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        std::cout << "device=" << properties.name << " compute=" << properties.major << '.'
                  << properties.minor << " runtime=" << runtime << " driver=" << driver << '\n';
        for (int index = 0; index < 3; ++index) cycle(false);
        size_t free_before = 0, total = 0, free_after = 0;
        check(cudaMemGetInfo(&free_before, &total));
        const DWORD handles_before = handles();
        for (int index = 0; index < 32; ++index) {
            cycle(false);
            try { cycle(true); }
            catch (const std::runtime_error& error) {
                require(std::strcmp(error.what(), "injected host interruption") == 0,
                        "unexpected interruption failure");
            }
        }
        check(cudaDeviceSynchronize());
        check(cudaGetLastError());
        check(cudaMemGetInfo(&free_after, &total));
        const DWORD handles_after = handles();
        require(handles_after <= handles_before, "Windows HANDLE growth");
        require(cuda_observer::errors == 0, "CUDA resource pairing or cleanup error");
        const char* resource_names[] = {"allocation", "stream", "event", "pool", "pinned_host"};
        for (size_t kind = 0; kind < cuda_observer::kind_count; ++kind) {
            require(cuda_observer::live[kind].empty(), "unreleased CUDA resource");
            std::cout << "resource=" << resource_names[kind] << " created=" << cuda_observer::created[kind]
                      << " outstanding=" << cuda_observer::live[kind].size() << '\n';
        }
        require(cuda_observer::created[cuda_observer::allocation] > 64 &&
                cuda_observer::created[cuda_observer::stream] > 64 &&
                cuda_observer::created[cuda_observer::event] > 64,
                "resource observation did not cover backend calls");
        // WDDM free memory includes other applications. Report it as context;
        // executable-owned CUDA resource pairs are checked above.
        std::cout << "success_cycles=32 interruption_cycles=32 handles_before=" << handles_before
                  << " handles_after=" << handles_after << " free_before=" << free_before
                  << " free_after=" << free_after << '\n';
        check(cudaDeviceReset());
        std::cout << "windows_cuda_smoke=PASS clean_shutdown=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "windows CUDA smoke: " << error.what() << '\n';
        return 1;
    }
}
