#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/tensor_util.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open precision policy: " + path);
    std::ostringstream output;
    output << stream.rdbuf();
    vrhino::require(stream.good() || stream.eof(),
                    "Cannot read precision policy: " + path);
    return output.str();
}

vrhino::Tensor host_tensor(vrhino::Backend& backend, const vrhino::Tensor& tensor) {
    return tensor.device().is_host() ? tensor : backend.copy_to_host(tensor);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 7,
            "usage: vrhino-precision-denoiser-calibration "
            "MODEL.vrm INPUT.bundle FROZEN_REFERENCE.bundle OUTPUT.bundle "
            "{fp32|bf16} BF16_POLICY.json");
        const std::string mode = argv[5];
        vrhino::require(mode == "fp32" || mode == "bf16",
                        "Calibration precision must be fp32 or bf16");
        const vrhino::DType dtype = mode == "fp32"
            ? vrhino::DType::F32 : vrhino::DType::BF16;
        const vrhino::PrecisionPolicy policy = mode == "fp32"
            ? vrhino::PrecisionPolicy::fp32()
            : vrhino::PrecisionPolicy::from_json(
                vrhino::Json::parse(read_text(argv[6])));

        vrhino::VrmModel model(argv[1]);
        vrhino::TensorBundle input = vrhino::read_bundle(argv[2]);
        input.insert_or_assign("audit_trace", vrhino::scalar_i64(1));
        input.insert_or_assign("audit_trace_block", vrhino::scalar_i64(0));
        const vrhino::TensorBundle frozen = vrhino::read_bundle(argv[3]);
        const vrhino::Tensor& latent = frozen.at("initial_noise");
        const vrhino::Tensor& timestep = frozen.at("step.0.timestep");
        vrhino::require(latent.dtype() == vrhino::DType::F32,
                        "Calibration source latent must be canonical FP32");

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(dtype);
        backend.enable_weight_cache(true);
        auto architecture = vrhino::create_architecture(model);
        auto denoiser = architecture->create_denoiser(backend, policy, input);
        std::vector<vrhino::Tensor> predictions = denoiser->evaluate(latent, timestep);
        vrhino::TensorBundle trace = denoiser->take_trace();
        backend.synchronize();

        vrhino::TensorBundle output;
        output.emplace("input.initial_noise", latent);
        output.emplace("input.timestep", timestep);
        for (size_t index = 0; index < predictions.size(); ++index)
            output.emplace("step.0.prediction." + std::to_string(index),
                           host_tensor(backend, predictions[index]));
        for (const auto& [name, tensor] : trace)
            output.emplace("trace." + name, host_tensor(backend, tensor));
        vrhino::write_bundle(argv[4], output);
        std::cout << "status=PASS\n"
                  << "precision=" << mode << "\n"
                  << "prediction_count=" << predictions.size() << "\n"
                  << "sampling_runtime=0\nvae=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-precision-denoiser-calibration: "
                  << error.what() << '\n';
        return 1;
    }
}
