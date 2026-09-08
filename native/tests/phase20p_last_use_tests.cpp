#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/components.h"
#include "vrhino/error.h"

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

uint16_t bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

vrhino::Tensor host_bf16(const std::vector<int64_t>& shape, float value) {
    vrhino::Tensor result = vrhino::Tensor::host(shape, vrhino::DType::BF16);
    uint16_t* values = result.data_as<uint16_t>();
    for (int64_t index = 0; index < result.numel(); ++index)
        values[index] = bf16(value + static_cast<float>(index % 7));
    return result;
}

vrhino::Tensor pooled_source(vrhino::CudaBackend& backend,
                             const std::vector<int64_t>& shape) {
    const vrhino::Tensor host = host_bf16(shape, 1.0f);
    const vrhino::Tensor direct = backend.copy_to_device(host, vrhino::DType::BF16);
    return backend.activation(direct, vrhino::Activation::Silu);
}

void require_equal(vrhino::CudaBackend& backend, const vrhino::Tensor& lhs,
                   const vrhino::Tensor& rhs, const std::string& message) {
    const vrhino::Tensor left = backend.copy_to_host(lhs);
    const vrhino::Tensor right = backend.copy_to_host(rhs);
    vrhino::require(left.shape() == right.shape() && left.dtype() == right.dtype() &&
                        left.bytes() == right.bytes() &&
                        std::memcmp(left.data(), right.data(), left.bytes()) == 0,
                    message);
}

}  // namespace

int main() {
    try {
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        // The tiny cache ceiling exercises the same oversize one-shot path as
        // the production 32 GiB activation without a large allocation.
        backend.configure_memory_runtime(
            vrhino::MemoryBudget{1ULL << 30, 64, 1ULL << 30,
                                 64, 64ULL << 20},
            vrhino::MemoryRuntimeOptions{true, false, false});

        vrhino::Tensor weight = vrhino::Tensor::host(
            {1, 1, 3, 3, 3}, vrhino::DType::BF16);
        std::memset(weight.data(), 0, weight.bytes());
        weight.data_as<uint16_t>()[13] = bf16(1.0f);
        vrhino::WeightMap weights(
            std::map<std::string, const vrhino::Tensor*>{{"conv.weight", &weight}});
        vrhino::ComponentExecutor executor(backend, weights);

        // Borrowed reference: the caller is a live fan-out consumer, so its
        // storage cannot be handed off to the convolution output.
        vrhino::Tensor borrowed_source = pooled_source(backend, {1, 1, 3, 4, 4});
        void* borrowed_pointer = borrowed_source.data();
        const vrhino::Tensor& borrowed = borrowed_source;
        const vrhino::Tensor reference = executor.causal_conv3d(
            borrowed, "conv", vrhino::PadMode::Replicate);
        vrhino::require(borrowed_source.defined() &&
                            reference.data() != borrowed_pointer,
                        "fan-out source was released or overwritten");
        std::cout << "fan_out=PASS\n";

        // Mutable last-use: pad is enqueued, source ownership is transferred,
        // and the exact-size Conv result reuses the released block.
        vrhino::Tensor source = pooled_source(backend, {1, 1, 3, 4, 4});
        void* source_pointer = source.data();
        vrhino::Tensor output = executor.causal_conv3d(
            source, "conv", vrhino::PadMode::Replicate);
        vrhino::require(!source.defined(), "consumed source handle remains live");
        vrhino::require(output.data() == source_pointer,
                        "Conv output did not reuse consumed source storage");
        require_equal(backend, reference, output,
                      "stream-ordered reuse changed Conv output");
        std::cout << "simple_chain=PASS\npad_conv_reuse=PASS\n";

        // A residual alias retains the allocation through branch execution.
        vrhino::Tensor residual = pooled_source(backend, {1, 1, 3, 4, 4});
        vrhino::Tensor branch = residual;
        void* residual_pointer = residual.data();
        vrhino::Tensor branch_output = executor.causal_conv3d(
            branch, "conv", vrhino::PadMode::Replicate);
        vrhino::require(!branch.defined() && residual.defined() &&
                            branch_output.data() != residual_pointer,
                        "residual alias was prematurely reused");
        const vrhino::Tensor residual_output = backend.add(residual, branch_output);
        backend.synchronize();
        vrhino::require(residual_output.defined(), "residual result is undefined");
        std::cout << "residual=PASS\n";

        // A reshape view shares the same Storage and therefore blocks reuse.
        vrhino::Tensor view_source = pooled_source(backend, {1, 1, 3, 4, 4});
        vrhino::Tensor view = backend.reshape(view_source, {1, 1, 3, 2, 8});
        void* view_pointer = view_source.data();
        vrhino::Tensor view_output = executor.causal_conv3d(
            view_source, "conv", vrhino::PadMode::Replicate);
        vrhino::require(view.defined() && view.data() == view_pointer &&
                            view_output.data() != view_pointer,
                        "view alias was prematurely reused");
        std::cout << "alias_view=PASS\n";

        // Slice is materialized by the backend; it remains valid while its
        // independent source block can be consumed and reused.
        vrhino::Tensor slice_source = pooled_source(backend, {1, 1, 3, 4, 4});
        vrhino::Tensor slice = backend.slice(slice_source, 2, 0, 1);
        void* slice_source_pointer = slice_source.data();
        vrhino::Tensor slice_output = executor.causal_conv3d(
            slice_source, "conv", vrhino::PadMode::Replicate);
        vrhino::require(slice.defined() &&
                            slice_output.data() == slice_source_pointer,
                        "materialized slice incorrectly retained its source");
        const vrhino::Tensor slice_host = backend.copy_to_host(slice);
        vrhino::require(slice_host.defined(), "slice became invalid after reuse");
        std::cout << "slice=PASS\n";

        const vrhino::Tensor concat = backend.concat({slice, slice}, 2);
        const vrhino::Tensor padded = backend.pad(
            concat, {1, 1, 1, 1, 1, 0}, 0.0f, vrhino::PadMode::Replicate);
        backend.synchronize();
        vrhino::require(concat.defined() && padded.defined(),
                        "concat/pad regression");
        std::cout << "concat=PASS\npad=PASS\nconv=PASS\n";

        const vrhino::MemoryRuntimeStats stats = backend.memory_runtime_stats();
        vrhino::require(stats.stream_ordered_handoff_releases > 0 &&
                            stats.stream_ordered_handoff_reuses > 0 &&
                            stats.largest_stream_ordered_handoff_bytes >=
                                output.bytes(),
                        "stream-ordered handoff telemetry missing");
        cuda_check(cudaDeviceSynchronize(), "final synchronize");
        std::cout << "stream_safety=PASS\n"
                  << "temporary_pool_reuses=" << stats.temporary_pool_reuses << '\n'
                  << "handoff_releases="
                  << stats.stream_ordered_handoff_releases << '\n'
                  << "handoff_reuses="
                  << stats.stream_ordered_handoff_reuses << '\n'
                  << "last_handoff_bytes="
                  << stats.last_stream_ordered_handoff_bytes << '\n'
                  << "largest_handoff_bytes="
                  << stats.largest_stream_ordered_handoff_bytes << '\n'
                  << "status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase20p_last_use_tests: " << error.what() << '\n';
        return 1;
    }
}
