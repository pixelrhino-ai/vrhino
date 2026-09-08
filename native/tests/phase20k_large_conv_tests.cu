#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/backend/cudnn_conv.h"
#include "vrhino/bundle.h"

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

void cudnn_check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudnnGetErrorString(status));
}

std::vector<int64_t> dense_strides(const std::vector<int64_t>& dimensions) {
    std::vector<int64_t> strides(dimensions.size(), 1);
    for (int index = static_cast<int>(dimensions.size()) - 2; index >= 0; --index)
        strides[static_cast<size_t>(index)] =
            strides[static_cast<size_t>(index + 1)] *
            dimensions[static_cast<size_t>(index + 1)];
    return strides;
}

vrhino::CudnnConvDescriptor descriptor(
    int device, int compute_major, vrhino::DType dtype,
    std::vector<int64_t> x, std::vector<int64_t> weight,
    std::vector<int64_t> output, std::vector<int64_t> padding,
    std::vector<int64_t> stride, std::vector<int64_t> dilation,
    int64_t groups = 1) {
    vrhino::CudnnConvDescriptor result;
    result.device = device;
    result.compute_major = compute_major;
    result.io_dtype = dtype;
    result.compute_dtype = vrhino::DType::F32;
    result.spatial_rank = static_cast<int>(x.size()) - 2;
    result.x_dimensions = std::move(x);
    result.x_strides = dense_strides(result.x_dimensions);
    result.w_dimensions = std::move(weight);
    result.w_strides = dense_strides(result.w_dimensions);
    result.y_dimensions = std::move(output);
    result.y_strides = dense_strides(result.y_dimensions);
    result.padding = std::move(padding);
    result.convolution_strides = std::move(stride);
    result.dilation = std::move(dilation);
    result.groups = groups;
    result.mode = CUDNN_CROSS_CORRELATION;
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void print_admission(const char* name,
                     const vrhino::CudnnConvDescriptor& value,
                     bool expected_legacy, bool expected_backend) {
    std::string legacy_reason;
    std::string backend_reason;
    const bool legacy = vrhino::cudnn_legacy_conv_admitted(value, &legacy_reason);
    const bool backend = vrhino::cudnn_backend_conv_admitted(value, &backend_reason);
    require(legacy == expected_legacy,
            std::string(name) + " legacy admission mismatch: " + legacy_reason);
    require(backend == expected_backend,
            std::string(name) + " backend admission mismatch: " + backend_reason);
    std::cout << "boundary." << name << ".legacy=" << (legacy ? "ADMITTED" : "NOT_ADMITTED")
              << " reason=\"" << legacy_reason << "\"\n";
    std::cout << "boundary." << name << ".backend=" << (backend ? "ADMITTED" : "NOT_ADMITTED")
              << " reason=\"" << backend_reason << "\"\n";
}

template <typename T>
std::vector<T> filled(size_t count, float value);

template <>
std::vector<float> filled<float>(size_t count, float value) {
    return std::vector<float>(count, value);
}

template <>
std::vector<__nv_bfloat16> filled<__nv_bfloat16>(size_t count, float value) {
    return std::vector<__nv_bfloat16>(count, __float2bfloat16(value));
}

template <typename T>
float as_float(T value);

template <>
float as_float<float>(float value) { return value; }

template <>
float as_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
void small_candidate_test(cudnnHandle_t handle, int device, int compute_major,
                          int spatial_rank, vrhino::DType dtype,
                          const char* label) {
    std::vector<int64_t> x = spatial_rank == 2
        ? std::vector<int64_t>{1, 2, 3, 4}
        : std::vector<int64_t>{1, 2, 2, 3, 4};
    std::vector<int64_t> weight = spatial_rank == 2
        ? std::vector<int64_t>{3, 2, 1, 1}
        : std::vector<int64_t>{3, 2, 1, 1, 1};
    std::vector<int64_t> output = x;
    output[1] = 3;
    const std::vector<int64_t> parameters(static_cast<size_t>(spatial_rank), 1);
    const std::vector<int64_t> padding(static_cast<size_t>(spatial_rank), 0);
    auto desc = descriptor(device, compute_major, dtype, x, weight, output,
                           padding, parameters, parameters);
    const int64_t x_count = desc.x_strides[0] * desc.x_dimensions[0];
    const int64_t w_count = desc.w_strides[0] * desc.w_dimensions[0];
    const int64_t y_count = desc.y_strides[0] * desc.y_dimensions[0];
    std::vector<T> hx = filled<T>(static_cast<size_t>(x_count), 1.0f);
    std::vector<T> hw = filled<T>(static_cast<size_t>(w_count), 1.0f);
    std::vector<T> hy(static_cast<size_t>(y_count));
    T* dx = nullptr;
    T* dw = nullptr;
    T* dy = nullptr;
    cuda_check(cudaMalloc(&dx, hx.size() * sizeof(T)), "cudaMalloc small X");
    cuda_check(cudaMalloc(&dw, hw.size() * sizeof(T)), "cudaMalloc small W");
    cuda_check(cudaMalloc(&dy, hy.size() * sizeof(T)), "cudaMalloc small Y");
    cuda_check(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(T), cudaMemcpyHostToDevice),
               "copy small X");
    cuda_check(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(T), cudaMemcpyHostToDevice),
               "copy small W");
    vrhino::CudnnConvPlanCache cache(handle);
    const auto execution = cache.execute(desc, dx, dw, dy);
    require(execution.executed, std::string(label) + " execution failed: " + execution.reason);
    cuda_check(cudaDeviceSynchronize(), "small convolution synchronize");
    cuda_check(cudaMemcpy(hy.data(), dy, hy.size() * sizeof(T), cudaMemcpyDeviceToHost),
               "copy small Y");
    float maximum_error = 0.0f;
    for (T value : hy)
        maximum_error = std::max(maximum_error, std::abs(as_float(value) - 2.0f));
    require(maximum_error == 0.0f, std::string(label) + " reference mismatch");
    std::cout << "candidate." << label << "=PASS max_abs_error=" << maximum_error
              << " workspace=" << execution.workspace_bytes << "\n";
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(dx);
}

__global__ void audit_bf16(const __nv_bfloat16* values, int64_t count,
                           unsigned long long* nonzero,
                           unsigned long long* nan_count,
                           unsigned long long* inf_count) {
    unsigned long long local_nonzero = 0;
    unsigned long long local_nan = 0;
    unsigned long long local_inf = 0;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float value = __bfloat162float(values[index]);
        local_nonzero += value != 0.0f;
        local_nan += isnan(value);
        local_inf += isinf(value);
    }
    if (local_nonzero) atomicAdd(nonzero, local_nonzero);
    if (local_nan) atomicAdd(nan_count, local_nan);
    if (local_inf) atomicAdd(inf_count, local_inf);
}

void exact_large_test(cudnnHandle_t handle, int device, int compute_major) {
    auto desc = descriptor(
        device, compute_major, vrhino::DType::BF16,
        {1, 512, 35, 274, 482}, {512, 512, 3, 3, 3},
        {1, 512, 33, 272, 480}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1});
    std::string reason;
    require(!vrhino::cudnn_legacy_conv_admitted(desc, &reason),
            "exact descriptor unexpectedly admitted by legacy cuDNN");
    require(vrhino::cudnn_backend_conv_admitted(desc, &reason),
            "exact descriptor rejected by Backend candidate: " + reason);
    const int64_t x_count = desc.x_strides[0];
    const int64_t w_count = desc.w_strides[0] * desc.w_dimensions[0];
    const int64_t y_count = desc.y_strides[0];
    __nv_bfloat16* x = nullptr;
    __nv_bfloat16* weight = nullptr;
    __nv_bfloat16* output = nullptr;
    cuda_check(cudaMalloc(&x, static_cast<size_t>(x_count) * sizeof(*x)), "cudaMalloc exact X");
    cuda_check(cudaMalloc(&weight, static_cast<size_t>(w_count) * sizeof(*weight)), "cudaMalloc exact W");
    cuda_check(cudaMalloc(&output, static_cast<size_t>(y_count) * sizeof(*output)), "cudaMalloc exact Y");
    cuda_check(cudaMemset(x, 0, static_cast<size_t>(x_count) * sizeof(*x)), "zero exact X");
    cuda_check(cudaMemset(weight, 0, static_cast<size_t>(w_count) * sizeof(*weight)), "zero exact W");
    cuda_check(cudaMemset(output, 0x7f, static_cast<size_t>(y_count) * sizeof(*output)), "poison exact Y");

    vrhino::CudnnConvPlanCache cache(handle);
    const auto cold = cache.execute(desc, x, weight, output);
    require(cold.executed, "exact cold execution failed: " + cold.reason);
    cuda_check(cudaDeviceSynchronize(), "exact cold synchronize");

    cudaEvent_t start{};
    cudaEvent_t finish{};
    cuda_check(cudaEventCreate(&start), "create warm start event");
    cuda_check(cudaEventCreate(&finish), "create warm finish event");
    cuda_check(cudaEventRecord(start), "record warm start");
    const auto warm = cache.execute(desc, x, weight, output);
    require(warm.executed && warm.cache_hit, "exact warm cache execution failed: " + warm.reason);
    cuda_check(cudaEventRecord(finish), "record warm finish");
    cuda_check(cudaEventSynchronize(finish), "exact warm synchronize");
    float warm_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&warm_ms, start, finish), "exact elapsed time");
    cudaEventDestroy(finish);
    cudaEventDestroy(start);

    unsigned long long* counters = nullptr;
    cuda_check(cudaMalloc(&counters, 3 * sizeof(*counters)), "allocate audit counters");
    cuda_check(cudaMemset(counters, 0, 3 * sizeof(*counters)), "zero audit counters");
    audit_bf16<<<65535, 256>>>(output, y_count, counters, counters + 1, counters + 2);
    cuda_check(cudaGetLastError(), "launch exact output audit");
    unsigned long long host_counters[3]{};
    cuda_check(cudaMemcpy(host_counters, counters, sizeof(host_counters),
                          cudaMemcpyDeviceToHost), "copy audit counters");
    require(host_counters[0] == 0 && host_counters[1] == 0 && host_counters[2] == 0,
            "zero trusted reference or finite check failed");
    require(cache.build_count() == 1 && cache.cache_hit_count() == 1,
            "plan cache did not reuse exact descriptor plan");
    std::cout << "exact.legacy=NOT_ADMITTED\n"
              << "exact.backend=ADMITTED\n"
              << "exact.plan=PASS plan_build_seconds=" << cold.plan_build_seconds << "\n"
              << "exact.execute=PASS synchronize=PASS\n"
              << "exact.workspace_bytes=" << cold.workspace_bytes << "\n"
              << "exact.warm_execution_ms=" << warm_ms << "\n"
              << "exact.reference=PASS reference=zeros max_abs_error=0\n"
              << "exact.nonzero=" << host_counters[0]
              << " nan=" << host_counters[1]
              << " inf=" << host_counters[2] << "\n"
              << "exact.cache_builds=" << cache.build_count()
              << " cache_hits=" << cache.cache_hit_count() << "\n";
    cudaFree(counters);
    cudaFree(output);
    cudaFree(weight);
    cudaFree(x);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool run_exact = false;
        std::string zero_latent_path;
        std::string audit_bundle_path;
        for (int index = 1; index < argc; ++index) {
            if (std::string(argv[index]) == "--exact") run_exact = true;
            else if (std::string(argv[index]) == "--write-zero-latent" && index + 1 < argc)
                zero_latent_path = argv[++index];
            else if (std::string(argv[index]) == "--audit-bundle" && index + 1 < argc)
                audit_bundle_path = argv[++index];
            else throw std::runtime_error("unknown argument: " + std::string(argv[index]));
        }
        int device = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        cudnnHandle_t handle{};
        cudnn_check(cudnnCreate(&handle), "cudnnCreate");

        auto below = descriptor(device, properties.major, vrhino::DType::BF16,
                                {1, 1, 1, INT_MAX - 1LL}, {1, 1, 1, 1},
                                {1, 1, 1, INT_MAX - 1LL}, {0, 0}, {1, 1}, {1, 1});
        auto boundary = descriptor(device, properties.major, vrhino::DType::BF16,
                                   {1, 1, 1, INT_MAX}, {1, 1, 1, 1},
                                   {1, 1, 1, INT_MAX}, {0, 0}, {1, 1}, {1, 1});
        auto above = descriptor(device, properties.major, vrhino::DType::BF16,
                                {1, 1, 1, static_cast<int64_t>(INT_MAX) + 1},
                                {1, 1, 1, 1},
                                {1, 1, 1, static_cast<int64_t>(INT_MAX) + 1},
                                {0, 0}, {1, 1}, {1, 1});
        auto large_stride = descriptor(device, properties.major, vrhino::DType::BF16,
                                       {1, 1, 3, 3}, {1, 1, 1, 1},
                                       {1, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1});
        large_stride.x_strides[0] = static_cast<int64_t>(INT_MAX) + 1;
        auto large_extent = descriptor(device, properties.major, vrhino::DType::BF16,
                                       {2, 1, 1, 1}, {1, 1, 1, 1},
                                       {2, 1, 1, 1}, {0, 0}, {1, 1}, {1, 1});
        large_extent.x_strides[0] = INT_MAX;
        large_extent.y_strides[0] = INT_MAX;
        auto exact = descriptor(device, properties.major, vrhino::DType::BF16,
                                {1, 512, 35, 274, 482}, {512, 512, 3, 3, 3},
                                {1, 512, 33, 272, 480}, {0, 0, 0},
                                {1, 1, 1}, {1, 1, 1});
        auto conv2d = descriptor(device, properties.major, vrhino::DType::F32,
                                 {1, 2, 8, 8}, {4, 2, 3, 3}, {1, 4, 6, 6},
                                 {0, 0}, {1, 1}, {1, 1});
        auto conv3d = descriptor(device, properties.major, vrhino::DType::F32,
                                 {1, 2, 5, 8, 8}, {4, 2, 3, 3, 3}, {1, 4, 3, 6, 6},
                                 {0, 0, 0}, {1, 1, 1}, {1, 1, 1});
        print_admission("below_2g", below, true, true);
        print_admission("at_2g", boundary, true, true);
        print_admission("above_2g", above, false, true);
        print_admission("large_stride_small_count", large_stride, false, true);
        print_admission("large_addressable_extent", large_extent, false, true);
        print_admission("phase20j_exact", exact, false, true);
        print_admission("small_conv2d", conv2d, true, true);
        print_admission("small_conv3d", conv3d, true, true);

        small_candidate_test<float>(handle, device, properties.major, 2,
                                    vrhino::DType::F32, "conv2d_fp32");
        small_candidate_test<__nv_bfloat16>(handle, device, properties.major, 2,
                                            vrhino::DType::BF16, "conv2d_bf16");
        small_candidate_test<float>(handle, device, properties.major, 3,
                                    vrhino::DType::F32, "conv3d_fp32");
        small_candidate_test<__nv_bfloat16>(handle, device, properties.major, 3,
                                            vrhino::DType::BF16, "conv3d_bf16");
        if (run_exact) exact_large_test(handle, device, properties.major);
        if (!zero_latent_path.empty()) {
            vrhino::Tensor latent = vrhino::Tensor::host(
                {1, 16, 33, 68, 120}, vrhino::DType::BF16);
            std::memset(latent.data(), 0, latent.bytes());
            vrhino::write_bundle(zero_latent_path, {{"latent", latent}});
            std::cout << "zero_latent_fixture=PASS path=\"" << zero_latent_path
                      << "\" shape=[1,16,33,68,120] dtype=bfloat16\n";
        }
        if (!audit_bundle_path.empty()) {
            const vrhino::TensorBundle bundle = vrhino::read_bundle(audit_bundle_path);
            const auto found = bundle.find("video");
            require(found != bundle.end(), "audit bundle has no video tensor");
            const vrhino::Tensor& video = found->second;
            require(video.dtype() == vrhino::DType::BF16,
                    "audit video is not bfloat16");
            uint64_t nan_count = 0;
            uint64_t inf_count = 0;
            long double sum_squares = 0.0L;
            float absolute_maximum = 0.0f;
            const __nv_bfloat16* values = video.data_as<__nv_bfloat16>();
            for (int64_t index = 0; index < video.numel(); ++index) {
                const float value = __bfloat162float(values[index]);
                nan_count += std::isnan(value);
                inf_count += std::isinf(value);
                if (std::isfinite(value)) {
                    sum_squares += static_cast<long double>(value) * value;
                    absolute_maximum = std::max(absolute_maximum, std::abs(value));
                }
            }
            require(nan_count == 0 && inf_count == 0,
                    "audit video contains NaN or Inf");
            std::cout << "video_audit=PASS shape=[";
            for (size_t index = 0; index < video.shape().size(); ++index) {
                if (index) std::cout << ',';
                std::cout << video.shape()[index];
            }
            std::cout << "] dtype=bfloat16 nan=" << nan_count
                      << " inf=" << inf_count
                      << " rms=" << std::sqrt(static_cast<double>(
                             sum_squares / video.numel()))
                      << " abs_max=" << absolute_maximum << "\n";
        }
        cudnnDestroy(handle);
        std::cout << "phase20k=PASS exact=" << (run_exact ? "EXECUTED" : "SKIPPED") << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "phase20k=FAIL reason=\"" << error.what() << "\"\n";
        return EXIT_FAILURE;
    }
}
