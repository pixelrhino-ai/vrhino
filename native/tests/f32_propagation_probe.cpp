// Research-only fixed-input operator replay. No model/config/package knowledge.
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include <iostream>

using namespace vrhino;
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: f32-propagation-probe INPUT.bundle OUTPUT.bundle");
        auto source = read_bundle(argv[1]);
        CudaBackend device;
        device.set_execution_dtype(DType::F32);
        Backend& backend = device;
        TensorBundle operands, output;
        for (const auto& [name, tensor] : source)
            if (tensor.dtype() == DType::F32)
                operands[name] = backend.copy_to_device(tensor, DType::F32);
        for (const auto& [name, kind] : source) {
            const std::string suffix = ".kind";
            if (name.size() < suffix.size() || name.substr(name.size() - suffix.size()) != suffix) continue;
            require(kind.dtype() == DType::I64 && kind.numel() == 1, "Invalid operator kind");
            const auto prefix = name.substr(0, name.size() - suffix.size());
            auto get = [&](const char* key) -> const Tensor& { return operands.at(prefix + "." + key); };
            const auto& x = get("input");
            Tensor result;
            switch (*kind.data_as<int64_t>()) {
                case 0: {
                    const auto w = operands.find(prefix + ".weight"), b = operands.find(prefix + ".bias");
                    const auto& eps = source.at(prefix + ".eps");
                    require(eps.dtype() == DType::F32 && eps.numel() == 1, "Invalid epsilon");
                    result = backend.layer_norm(x, w == operands.end() ? nullptr : &w->second,
                        b == operands.end() ? nullptr : &b->second, *eps.data_as<float>());
                    break;
                }
                case 1:
                    result = backend.add(backend.mul(x, backend.add(get("one"), get("scale"))), get("shift"));
                    break;
                case 2: result = backend.add(x, get("other")); break;
                case 3: result = backend.add(x, backend.mul(get("other"), get("gate"))); break;
                default: throw Error("Unsupported replay operator");
            }
            output[prefix] = result;
        }
        require(!output.empty(), "Empty operator replay");
        backend.synchronize();
        TensorBundle host;
        for (const auto& [name, tensor] : output) host[name] = backend.copy_to_host(tensor);
        backend.synchronize();
        write_bundle(argv[2], host);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
