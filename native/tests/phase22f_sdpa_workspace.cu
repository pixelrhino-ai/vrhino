#include <cmath>
#include <cstdint>
#include <iostream>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "vrhino/backend/cudnn_sdpa.h"

namespace {

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    constexpr int64_t batch = 2;
    constexpr int64_t tokens = 44776;
    constexpr int64_t heads = 24;
    constexpr int64_t width = 128;
    const size_t tensor_bytes = static_cast<size_t>(batch * tokens * heads * width) *
        sizeof(__nv_bfloat16);
    void* query = nullptr;
    void* key = nullptr;
    void* value = nullptr;
    void* output = nullptr;
    int32_t* query_lengths = nullptr;
    int32_t* key_lengths = nullptr;
    require_cuda(cudaMalloc(&query, tensor_bytes), "allocate Q");
    require_cuda(cudaMalloc(&key, tensor_bytes), "allocate K");
    require_cuda(cudaMalloc(&value, tensor_bytes), "allocate V");
    require_cuda(cudaMalloc(&output, tensor_bytes), "allocate O");
    require_cuda(cudaMalloc(&query_lengths, batch * sizeof(int32_t)), "allocate Q lengths");
    require_cuda(cudaMalloc(&key_lengths, batch * sizeof(int32_t)), "allocate K lengths");
    const int32_t lengths[batch] = {static_cast<int32_t>(tokens),
                                    static_cast<int32_t>(tokens)};
    require_cuda(cudaMemcpy(query_lengths, lengths, sizeof(lengths),
                            cudaMemcpyHostToDevice), "upload Q lengths");
    require_cuda(cudaMemcpy(key_lengths, lengths, sizeof(lengths),
                            cudaMemcpyHostToDevice), "upload K lengths");

    cudnnHandle_t handle = nullptr;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return 1;
    vrhino::CudnnSdpaPlanCache cache(handle);
    vrhino::CudnnSdpaDescriptor descriptor;
    descriptor.batch = batch;
    descriptor.query_tokens = tokens;
    descriptor.key_tokens = tokens;
    descriptor.heads = heads;
    descriptor.head_width = width;
    descriptor.scale = 1.0f / std::sqrt(static_cast<float>(width));
    descriptor.has_padding_mask = true;
    const vrhino::CudnnSdpaExecution execution = cache.execute(
        descriptor, query, key, value, nullptr, query_lengths, key_lengths,
        output);
    require_cuda(cudaDeviceSynchronize(), "synchronize SDPA");
    std::cout << "executed=" << (execution.executed ? "PASS" : "FAIL")
              << "\nworkspace_bytes=" << execution.workspace_bytes
              << "\nqkv_shape=[2,44776,24,128]"
              << "\nqkv_bytes_each=" << tensor_bytes
              << "\nscore_elements="
              << static_cast<long double>(batch) * heads * tokens * tokens
              << "\nreason=" << execution.reason << '\n';
    cudaFree(key_lengths);
    cudaFree(query_lengths);
    cudaFree(output);
    cudaFree(value);
    cudaFree(key);
    cudaFree(query);
    cudnnDestroy(handle);
    return execution.executed ? 0 : 1;
}
