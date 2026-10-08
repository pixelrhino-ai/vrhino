#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace vrhino;
struct Shape { int rank; int64_t dimensions[8]{}, strides[8]{}; };
Shape describe(const Tensor& tensor) {
    Shape result{};
    result.rank = static_cast<int>(tensor.ndim());
    for (int i = 0; i < result.rank; ++i) {
        result.dimensions[i] = tensor.dim(i);
        result.strides[i] = tensor.strides()[i];
    }
    return result;
}
// Frozen pre-optimization GPU addressing/arithmetic, independent of the new
// flat-period dispatch. Check every result bit, including BF16 rounding.
__device__ int64_t reference_offset(int64_t index, Shape output, Shape input) {
    int64_t offset = 0;
    for (int axis = output.rank - 1; axis >= 0; --axis) {
        const int64_t coordinate = index % output.dimensions[axis];
        index /= output.dimensions[axis];
        const int input_axis = axis - (output.rank - input.rank);
        if (input_axis >= 0 && input.dimensions[input_axis] != 1)
            offset += coordinate * input.strides[input_axis];
    }
    return offset;
}
template <typename T>
__global__ void reference(const T* a, const T* b, T* output, int64_t count,
                          Shape out_shape, Shape a_shape, Shape b_shape, int kind) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const float av = float(a[reference_offset(i, out_shape, a_shape)]);
        const float bv = float(b[reference_offset(i, out_shape, b_shape)]);
        float value;
        if (kind == 0) value = av + bv;
        else if (kind == 1) value = av * bv;
        else if (kind == 2) value = av / bv;
        else value = fmaxf(av, bv);
        output[i] = T(value);
    }
}
Tensor input(CudaBackend& backend, const std::vector<int64_t>& shape, DType dtype, int seed) {
    Tensor host = Tensor::host(shape, DType::F32);
    const int64_t count = host.numel();
    for (int64_t i = 0; i < count; ++i)
        host.data_as<float>()[i] = float((i * 17 + seed) % 255 - 127) / 128.0f;
    const float special[] = {0.0f, -0.0f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (int64_t i = 0; i < count && i < 5; ++i)
        host.data_as<float>()[i] = special[(i + seed) % 5];
    return backend.copy_to_device(host, dtype);
}
Tensor invoke(CudaBackend& backend, int kind, const Tensor& a, const Tensor& b) {
    if (kind == 0) return backend.add(a, b);
    if (kind == 1) return backend.mul(a, b);
    if (kind == 2) return backend.div(a, b);
    return backend.maximum(a, b);
}
void check_case(CudaBackend& backend, DType dtype, const char* label,
                const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
                const std::vector<int64_t>& output_shape, bool benchmark) {
    const Tensor a = input(backend, a_shape, dtype, 1), b = input(backend, b_shape, dtype, 3);
    Tensor expected = backend.allocate_device(output_shape, dtype);
    const int grid = int(std::min<int64_t>((expected.numel() + 255) / 256, 65535));
    for (int kind = 0; kind < 4; ++kind) {
        const auto launch_reference = [&] {
            BackendProfileRegion region(backend, "reference.binary");
            if (dtype == DType::BF16)
                reference<<<grid, 256>>>(a.data_as<__nv_bfloat16>(), b.data_as<__nv_bfloat16>(),
                    expected.data_as<__nv_bfloat16>(), expected.numel(), describe(expected), describe(a), describe(b), kind);
            else reference<<<grid, 256>>>(a.data_as<float>(), b.data_as<float>(),
                    expected.data_as<float>(), expected.numel(), describe(expected), describe(a), describe(b), kind);
            require(cudaGetLastError() == cudaSuccess, "Reference binary launch failed");
        };
        Tensor actual = invoke(backend, kind, a, b);
        launch_reference();
        if (benchmark) {
            (void)backend.profile_stats();
            for (int repeat = 0; repeat < 5; ++repeat) actual = invoke(backend, kind, a, b);
            for (int repeat = 0; repeat < 5; ++repeat) launch_reference();
            for (const auto& [name, stat] : backend.profile_stats())
                if (name == "reference.binary" || name.find("elementwise.") == 0)
                    std::cout << label << ',' << dtype_name(dtype) << ',' << kind << ','
                              << name << ',' << stat.calls << ',' << stat.mean_milliseconds << '\n';
        }
        const Tensor host_actual = backend.copy_to_host(actual), host_expected = backend.copy_to_host(expected);
        require(actual.shape() == output_shape && actual.dtype() == dtype &&
                    host_actual.bytes() == host_expected.bytes() &&
                    std::memcmp(host_actual.data(), host_expected.data(), host_actual.bytes()) == 0,
                std::string("Binary output changed: ") + label + "/" + dtype_name(dtype) + "/" + std::to_string(kind));
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--benchmark"), "Invalid test arguments");
        const bool benchmark = argc == 2;
        CudaBackend backend;
        backend.enable_profiling(benchmark);
        for (DType dtype : {DType::F32, DType::BF16}) {
            backend.set_execution_dtype(dtype);
            check_case(backend, dtype, "full", {1,32760,1536}, {1,32760,1536}, {1,32760,1536}, benchmark);
            check_case(backend, dtype, "suffix", {1,32760,1536}, {1,1,1536}, {1,32760,1536}, benchmark);
            check_case(backend, dtype, "scalar_first", {}, {3,257}, {3,257}, false);
            check_case(backend, dtype, "scalar_last", {3,257}, {1,1}, {3,257}, false);
            check_case(backend, dtype, "suffix_first", {7}, {3,5,7}, {3,5,7}, false);
            check_case(backend, dtype, "suffix_matrix", {3,7}, {2,3,7}, {2,3,7}, false);
            check_case(backend, dtype, "internal_singleton", {3,1,7}, {1,5,1}, {3,5,7}, false);
            check_case(backend, dtype, "rank8", {1,1,1,1,1,1,257,3}, {257,3}, {1,1,1,1,1,1,257,3}, false);
        }
        bool rejected = false;
        try { (void)backend.add(input(backend, {1,1,1,1,1,1,1,1,1}, DType::BF16, 1), scalar_f32(1)); }
        catch (const Error& error) { rejected = std::string(error.what()) == "CUDA primitive rank exceeds 8"; }
        require(rejected, "Flat binary path bypassed rank validation");
        rejected = false;
        try { (void)backend.mul(input(backend, {2,3}, DType::BF16, 1), input(backend, {2,4}, DType::BF16, 3)); }
        catch (const Error&) { rejected = true; }
        require(rejected, "Binary path accepted incompatible shapes");
        backend.synchronize();
        std::cout << "CUDA binary indexing: PASS (64 byte-exact cases; rank and shape rejection)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA binary indexing: FAIL: " << error.what() << '\n';
        return 1;
    }
}
