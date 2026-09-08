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
        vrhino::require(argc == 7,
            "usage: metal-rms-norm-reproducer INPUT_BUNDLE INPUT_KEY WEIGHTS WEIGHT_KEY EPS OUTPUT_BUNDLE");
        const std::string input_path = argv[1];
        const std::string input_key = argv[2];
        const std::string weight_path = argv[3];
        const std::string weight_key = argv[4];
        const float eps = std::stof(argv[5]);
        const std::string output_path = argv[6];

        const vrhino::TensorBundle inputs = vrhino::read_bundle(input_path);
        const auto input_item = inputs.find(input_key);
        vrhino::require(input_item != inputs.end(), "Missing isolated RMSNorm input tensor");
        const vrhino::Tensor& input = input_item->second;
        auto assets = vrhino::phase13::SafeTensorSet::single(weight_path);
        const vrhino::WeightMap weights = assets.weights();
        const vrhino::Tensor* weight_pointer = weights.find(weight_key);
        vrhino::require(weight_pointer != nullptr, "Missing isolated RMSNorm weight tensor");
        const vrhino::Tensor& weight = *weight_pointer;

        vrhino::require(input.dtype() == vrhino::DType::F32,
                        "Isolated RMSNorm input must be FP32");
        vrhino::require(input.ndim() >= 1 && weight.ndim() == 1 &&
                            input.dim(-1) == weight.numel(),
                        "Isolated RMSNorm shape mismatch");
        vrhino::require(weight.dtype() == vrhino::DType::BF16,
                        "Isolated RMSNorm audit requires BF16 stored weight");
        vrhino::require(eps > 0.0f, "Isolated RMSNorm epsilon must be positive");

        vrhino::MetalBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        backend.enable_profiling(true);
        vrhino::Tensor output = backend.rms_norm(input, &weight, eps);
        backend.synchronize();
        output = backend.copy_to_host(output);
        vrhino::write_bundle(output_path, {
            {"isolated.output", output},
            {"isolated.weight_stored", weight},
        });

        std::cout << "status=PASS"
                  << ",input_shape=" << dimensions(input.shape())
                  << ",input_strides=" << dimensions(input.strides())
                  << ",weight_shape=" << dimensions(weight.shape())
                  << ",weight_strides=" << dimensions(weight.strides())
                  << ",output_shape=" << dimensions(output.shape())
                  << ",output_strides=" << dimensions(output.strides())
                  << ",input_dtype=" << vrhino::dtype_name(input.dtype())
                  << ",weight_stored_dtype=" << vrhino::dtype_name(weight.dtype())
                  << ",compute_dtype=float32,output_dtype="
                  << vrhino::dtype_name(output.dtype())
                  << ",eps=" << eps
                  << ",axis=-1,kernel_variant=rms_norm_f32"
                  << ",rows=" << input.numel() / input.dim(-1)
                  << ",width=" << input.dim(-1) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
