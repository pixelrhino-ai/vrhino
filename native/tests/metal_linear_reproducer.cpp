#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "phase13_safetensors.h"
#include "vrhino/backend/metal_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"

namespace {

std::string dimensions(const std::vector<int64_t>& values) {
    std::ostringstream stream;
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << values[index];
    }
    return stream.str() + ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 6 || argc == 8,
            "usage: metal-linear-reproducer INPUT_BUNDLE INPUT_KEY WEIGHTS WEIGHT_KEY OUTPUT_BUNDLE [ROW OUTPUT_CHANNEL]");
        const std::string input_path = argv[1];
        const std::string input_key = argv[2];
        const std::string weight_path = argv[3];
        const std::string weight_key = argv[4];
        const std::string output_path = argv[5];

        const vrhino::TensorBundle inputs = vrhino::read_bundle(input_path);
        const auto input_item = inputs.find(input_key);
        vrhino::require(input_item != inputs.end(), "Missing isolated Linear input tensor");
        const vrhino::Tensor& input = input_item->second;
        auto assets = vrhino::phase13::SafeTensorSet::single(weight_path);
        const vrhino::WeightMap weights = assets.weights();
        const vrhino::Tensor* weight_pointer = weights.find(weight_key);
        vrhino::require(weight_pointer != nullptr, "Missing isolated Linear weight tensor");
        const vrhino::Tensor& weight = *weight_pointer;

        vrhino::require(input.dtype() == vrhino::DType::F32,
                        "Isolated Linear input must be FP32");
        vrhino::require(input.ndim() >= 1 && weight.ndim() == 2 &&
                            input.dim(-1) == weight.dim(1),
                        "Isolated Linear shape mismatch");

        vrhino::MetalBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        vrhino::MemoryBudget budget{21ULL << 30, 2ULL << 30, 64ULL << 30,
                                    2ULL << 30, 1ULL << 30};
        vrhino::MemoryRuntimeOptions options;
        options.enabled = true;
        options.host_staging = false;
        options.prefetch = false;
        backend.configure_memory_runtime(budget, options);
        backend.set_vrm_mapped_bytes(assets.mapped_bytes());
        backend.enable_profiling(true);

        const int64_t k = input.dim(-1);
        const int64_t n = weight.dim(0);
        const int64_t m = input.numel() / k;
        const bool use_mps = k >= 16 && n >= 16 &&
            (k * static_cast<int64_t>(sizeof(float))) % 16 == 0 &&
            (n * static_cast<int64_t>(sizeof(float))) % 16 == 0;
        vrhino::Tensor output = backend.linear(input, weight, nullptr, vrhino::DType::F32);
        backend.synchronize();
        output = backend.copy_to_host(output);
        vrhino::TensorBundle observations{{"isolated.output", output}};
        int64_t observed_row = -1;
        int64_t observed_column = -1;
        if (argc == 8) {
            observed_row = std::stoll(argv[6]);
            observed_column = std::stoll(argv[7]);
            vrhino::require(observed_row >= 0 && observed_row < m &&
                                observed_column >= 0 && observed_column < n,
                            "Isolated Linear observation coordinate is out of range");
            vrhino::Tensor input_matrix = backend.reshape(
                backend.copy_to_device(input, vrhino::DType::F32), {m, k});
            vrhino::Tensor input_vector = backend.slice(
                input_matrix, 0, observed_row, observed_row + 1);
            vrhino::Tensor weight_vector = backend.slice(
                weight, 0, observed_column, observed_column + 1);
            backend.synchronize();
            observations.emplace("isolated.input_vector",
                                 backend.copy_to_host(input_vector));
            observations.emplace("isolated.weight_vector",
                                 backend.copy_to_host(weight_vector));
        }
        vrhino::write_bundle(output_path, observations);

        std::cout << "status=PASS"
                  << ",m=" << m << ",n=" << n << ",k=" << k
                  << ",input_shape=" << dimensions(input.shape())
                  << ",input_strides=" << dimensions(input.strides())
                  << ",weight_shape=" << dimensions(weight.shape())
                  << ",weight_strides=" << dimensions(weight.strides())
                  << ",output_shape=" << dimensions(output.shape())
                  << ",output_strides=" << dimensions(output.strides())
                  << ",input_dtype=" << vrhino::dtype_name(input.dtype())
                  << ",weight_dtype=" << vrhino::dtype_name(weight.dtype())
                  << ",output_dtype=" << vrhino::dtype_name(output.dtype())
                  << ",input_row_bytes=" << k * sizeof(float)
                  << ",weight_row_bytes=" << k * sizeof(float)
                  << ",output_row_bytes=" << n * sizeof(float)
                  << ",kernel_variant="
                  << (use_mps ? "mps_matrix_multiplication" : "matmul_f32")
                  << ",transpose_left=false,transpose_right=true,alpha=1,beta=0"
                  << ",observed_row=" << observed_row
                  << ",observed_output_channel=" << observed_column << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
