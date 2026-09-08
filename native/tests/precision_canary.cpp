#include <chrono>
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
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open precision policy: " + path);
    std::ostringstream output;
    output << stream.rdbuf();
    vrhino::require(stream.good() || stream.eof(), "Cannot read precision policy: " + path);
    return output.str();
}

uint64_t peak_cpu_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) != 0) continue;
        std::istringstream stream(line.substr(6));
        uint64_t kib = 0;
        stream >> kib;
        return kib * 1024;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 5,
            "usage: vrhino-precision-canary POLICY.json MODEL.vrm INPUT.bundle OUTPUT.bundle");
        const vrhino::Json document = vrhino::Json::parse(read_text(argv[1]));
        const vrhino::PrecisionPolicy policy = vrhino::PrecisionPolicy::from_json(document);
        vrhino::require(policy.requested_dtype() == vrhino::DType::BF16,
                        "Precision canary requires a BF16 policy");

        vrhino::VrmModel model(argv[2]);
        const auto input_started = std::chrono::steady_clock::now();
        const vrhino::TensorBundle input = vrhino::read_bundle(argv[3]);
        const double input_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - input_started).count();

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::BF16);
        backend.enable_weight_cache(true);
        auto architecture = vrhino::create_architecture(model);
        vrhino::RuntimeResult result = vrhino::NativeRuntime(backend, policy).execute(
            *architecture, input);
        result.outputs.emplace("metric.vrm_load_seconds",
                               vrhino::scalar_f32(static_cast<float>(model.load_seconds())));
        result.outputs.emplace("metric.peak_cpu_bytes",
                               vrhino::scalar_i64(static_cast<int64_t>(peak_cpu_bytes())));
        vrhino::write_bundle(argv[4], result.outputs);

        std::cout << "status=ok\n"
                  << "vrm_load_seconds=" << model.load_seconds() << "\n"
                  << "input_preparation_seconds=" << input_seconds << "\n"
                  << "sampling_seconds=" << result.sampling_seconds << "\n"
                  << "decode_seconds=" << result.decode_seconds << "\n"
                  << "execution_seconds=" << result.execution_seconds << "\n"
                  << "weight_upload_seconds=" << result.upload_seconds << "\n"
                  << "weight_upload_bytes=" << result.upload_bytes << "\n"
                  << "peak_device_bytes=" << result.peak_device_bytes << "\n"
                  << "peak_cpu_bytes=" << peak_cpu_bytes() << "\n"
                  << "execution_dtype=" << vrhino::dtype_name(backend.execution_dtype()) << "\n"
                  << "residual_state_dtype=" << vrhino::dtype_name(
                         policy.persistent_state_dtype(vrhino::PrecisionSemantic::ResidualState)) << "\n"
                  << "sampling_state_dtype=" << vrhino::dtype_name(
                         policy.persistent_state_dtype(vrhino::PrecisionSemantic::SamplingState)) << "\n";
        std::cout << "denoiser_output_dtype=" << vrhino::dtype_name(
                         policy.boundary_dtype(vrhino::PrecisionSemantic::DenoiserOutput)) << "\n";
        std::cout << "denoiser_producer_output_dtype=" << vrhino::dtype_name(
                         policy.producer_output_dtype(
                             vrhino::PrecisionOperation::Linear,
                             vrhino::PrecisionSemantic::DenoiserOutput)) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-precision-canary: " << error.what() << '\n';
        return 1;
    }
}
