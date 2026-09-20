// Research/qualification probe only; uses the unchanged production RNG primitive.
#include <charconv>
#include <iostream>
#include <string>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/tensor_util.h"

namespace {
uint64_t integer(const char* value) {
    const std::string text(value);
    uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("invalid unsigned integer");
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::runtime_error("usage: rng-replay-probe COUNT SEED f32|bf16 OUTPUT.bundle");
        const auto count = integer(argv[1]);
        const auto seed = integer(argv[2]);
        if (!count || count > 4 * 1024 * 1024)
            throw std::runtime_error("probe count must be between 1 and 4194304");
        const std::string precision(argv[3]);
        if (precision != "f32" && precision != "bf16")
            throw std::runtime_error("unsupported probe storage dtype");
        const auto dtype = precision == "f32" ? vrhino::DType::F32 : vrhino::DType::BF16;
        vrhino::CudaBackend backend;
        backend.set_execution_dtype(dtype);
        vrhino::RngState state{seed, 0, "pytorch_compat.v1"};
        const std::vector<int64_t> shape{static_cast<int64_t>(count)};
        vrhino::TensorBundle out;
        out.emplace("first", backend.copy_to_host(backend.rng_normal(state, shape, dtype)));
        out.emplace("offset_after_first", vrhino::scalar_i64(static_cast<int64_t>(state.offset)));
        out.emplace("second", backend.copy_to_host(backend.rng_normal(state, shape, dtype)));
        out.emplace("offset_after_second", vrhino::scalar_i64(static_cast<int64_t>(state.offset)));
        state.offset = 0;
        out.emplace("replay", backend.copy_to_host(backend.rng_normal(state, shape, dtype)));
        backend.synchronize();
        vrhino::write_bundle(argv[4], out);
        std::cout << "native RNG probe completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
