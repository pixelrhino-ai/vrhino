#include <functional>
#include <iostream>
#include <string>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {
void rejects(const std::string& name, const std::function<void()>& operation, int& count) {
    try { operation(); } catch (const vrhino::Error&) { ++count; return; }
    throw vrhino::Error("negative case did not fail loudly: " + name);
}
}

int main() {
    try {
        vrhino::CudaBackend backend; int count = 0;
        rejects("unsupported dtype", [&] { (void)backend.cast(vrhino::host_f32({1}, {1}), vrhino::DType::F16); }, count);
        rejects("linear shape", [&] { (void)backend.linear(vrhino::host_f32({2, 3}, {1,2,3,4,5,6}), vrhino::host_f32({2, 2}, {1,2,3,4})); }, count);
        rejects("attention shape", [&] { (void)backend.attention(
            vrhino::host_f32({1,2,1,2}, {1,2,3,4}),
            vrhino::host_f32({1,3,1,2}, {1,2,3,4,5,6}),
            vrhino::host_f32({1,2,1,2}, {1,2,3,4}), nullptr, false, 0); }, count);
        rejects("convolution channels", [&] { (void)backend.conv3d(
            vrhino::host_f32({1,2,1,1,1}, {1,2}),
            vrhino::host_f32({1,1,1,1,1}, {1}), nullptr,
            {1,1,1}, {0,0,0}, {1,1,1}, 1); }, count);
        rejects("bad RNG algorithm", [&] { vrhino::RngState state{1,0,"unknown"};
            (void)backend.rng_normal(state, {2}, vrhino::DType::F32); }, count);
        rejects("bad reshape", [&] { (void)backend.reshape(vrhino::host_f32({2}, {1,2}), {3}); }, count);
        rejects("indexed gather bounds", [&] { (void)backend.indexed_gather(
            vrhino::host_f32({2,2}, {1,2,3,4}), vrhino::host_i64({1}, {2})); }, count);
        rejects("attention bias broadcast", [&] {
            const vrhino::Tensor qkv = vrhino::host_f32({1,2,1,2}, {1,2,3,4});
            const vrhino::Tensor bias = vrhino::host_f32({2,2,2}, {0,0,0,0,0,0,0,0});
            (void)backend.attention(qkv, qkv, qkv, nullptr, false, 0, &bias);
        }, count);
        vrhino::require(count == 8, "negative test count mismatch");
        std::cout << "native negative backend cases=" << count << " pass\n"; return 0;
    } catch (const std::exception& error) {
        std::cerr << "native negative tests: " << error.what() << "\n"; return 1;
    }
}
