#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#if VRHINO_USE_METAL_BACKEND
#include "vrhino/backend/metal_backend.h"
#else
#include "vrhino/backend/cuda_backend.h"
#endif
#include "vrhino/bundle.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"

namespace {
#if VRHINO_USE_METAL_BACKEND
using NativeBackend = vrhino::MetalBackend;
#else
using NativeBackend = vrhino::CudaBackend;
#endif

uint64_t peak_cpu_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) != 0) continue;
        std::istringstream stream(line.substr(6)); uint64_t kib = 0; stream >> kib;
        return kib * 1024;
    }
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "inspect") {
            vrhino::VrmModel model(argv[2]);
            std::cout << "profile=" << model.profile_id() << "\n"
                      << "architecture=" << model.architecture_id() << "\n"
                      << "tensors=" << model.tensors().size() << "\n"
                      << "bytes=" << model.file_size() << "\n"
                      << "load_seconds=" << model.load_seconds() << "\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "device") {
            NativeBackend backend;
            const auto capability = backend.device_capability();
            std::cout << "backend=" << backend.name() << "\ndevices=" << backend.device_count()
                      << "\ndevice_name=" << capability.name
                      << "\ndevice_family=" << capability.family
                      << "\ndevice_memory_bytes=" << capability.device_memory_bytes << "\n";
            return 0;
        }
        if ((argc >= 5 && argc <= 7) && std::string(argv[1]) == "run") {
            vrhino::VrmModel model(argv[2]);
            const auto input_started = std::chrono::steady_clock::now();
            vrhino::TensorBundle input = vrhino::read_bundle(argv[3]);
            const double input_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - input_started).count();
            NativeBackend backend;
            const std::string dtype = argc >= 6 ? argv[5] : "fp32";
            if (dtype == "bf16") backend.set_execution_dtype(vrhino::DType::BF16);
            else if (dtype != "fp32") throw vrhino::Error("run dtype must be fp32 or bf16");
            const std::string mode = argc == 7 ? argv[6] : "";
            const bool profile = mode == "profile" || mode == "profile-nocache";
            const bool cache = mode != "nocache" && mode != "profile-nocache";
            if (argc == 7 && mode != "profile" && mode != "nocache" && mode != "profile-nocache")
                throw vrhino::Error("seventh run argument must be profile, nocache, or profile-nocache");
            backend.enable_profiling(profile);
            backend.enable_weight_cache(cache);
            auto architecture = vrhino::create_architecture(model);
            vrhino::RuntimeResult result = vrhino::NativeRuntime(backend).execute(*architecture, input);
            result.outputs.emplace("metric.vrm_load_seconds", vrhino::scalar_f32(static_cast<float>(model.load_seconds())));
            result.outputs.emplace("metric.peak_cpu_bytes", vrhino::scalar_i64(static_cast<int64_t>(peak_cpu_bytes())));
            vrhino::write_bundle(argv[4], result.outputs);
            std::cout << "status=ok\n"
                      << "vrm_load_seconds=" << model.load_seconds() << "\n"
                      << "mmap_setup_seconds=" << model.mmap_setup_seconds() << "\n"
                      << "checksum_seconds=" << model.checksum_seconds() << "\n"
                      << "metadata_parse_seconds=" << model.metadata_parse_seconds() << "\n"
                      << "input_preparation_seconds=" << input_seconds << "\n"
                      << "sampling_seconds=" << result.sampling_seconds << "\n"
                      << "decode_seconds=" << result.decode_seconds << "\n"
                      << "execution_seconds=" << result.execution_seconds << "\n"
                      << "weight_upload_seconds=" << result.upload_seconds << "\n"
                      << "weight_upload_bytes=" << result.upload_bytes << "\n"
                      << "peak_device_bytes=" << result.peak_device_bytes << "\n"
                      << "peak_cpu_bytes=" << peak_cpu_bytes() << "\n";
            std::cout << "execution_dtype=" << vrhino::dtype_name(backend.execution_dtype()) << "\n";
            std::cout << "weight_cache_hits=" << backend.weight_cache_hits() << "\n"
                      << "weight_cache_misses=" << backend.weight_cache_misses() << "\n"
                      << "weight_cache_resident_bytes=" << backend.weight_cache_resident_bytes() << "\n"
                      << "weight_cache_capacity_bytes=" << backend.weight_cache_capacity_bytes() << "\n";
            std::cout << "prepared_tensor_prepare_hits="
                      << result.prepared_tensors.prepare_hits << "\n"
                      << "prepared_tensor_prepare_misses="
                      << result.prepared_tensors.prepare_misses << "\n"
                      << "prepared_tensor_reuses="
                      << result.prepared_tensors.reuses << "\n"
                      << "prepared_tensor_host_materializations="
                      << result.prepared_tensors.host_materializations << "\n"
                      << "prepared_tensor_h2d_transfers="
                      << result.prepared_tensors.host_to_device_transfers << "\n"
                      << "prepared_tensor_h2d_bytes="
                      << result.prepared_tensors.host_to_device_bytes << "\n"
                      << "prepared_tensor_resident_bytes="
                      << result.prepared_tensors.resident_bytes << "\n";
            if (profile) {
                for (size_t index = 0; index < result.denoiser_call_seconds.size(); ++index)
                    std::cout << "denoiser_call." << index << "_seconds="
                              << result.denoiser_call_seconds[index] << "\n";
                std::cout << "scheduler_seconds=" << result.scheduler_seconds << "\n";
            }
            if (profile) for (const auto& [name, stat] : backend.profile_stats())
                std::cout << "profile=" << name << ",calls=" << stat.calls
                          << ",device_ms=" << stat.device_milliseconds << "\n";
            return 0;
        }
        std::cerr << "usage: vrhino-native {inspect MODEL.vrm|device|run MODEL.vrm INPUT.bundle OUTPUT.bundle [fp32|bf16] [profile|nocache|profile-nocache]}\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-native: " << error.what() << "\n";
        return 1;
    }
}
