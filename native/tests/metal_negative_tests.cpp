#include <functional>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "vrhino/backend/metal_backend.h"
#include "vrhino/error.h"

namespace {

void reject(const char* name, const std::function<void()>& operation, int& cases) {
    try {
        operation();
        throw std::runtime_error(std::string(name) + " was accepted");
    } catch (const vrhino::Error&) {
        ++cases;
    } catch (const std::invalid_argument&) {
        ++cases;
    }
}

}  // namespace

int main() {
    try {
        vrhino::MetalBackend backend;
        int cases = 0;
        reject("unsupported execution dtype", [&] { backend.set_execution_dtype(vrhino::DType::BF16); }, cases);
        reject("invalid allocation shape", [&] { (void)backend.allocate_device({-1}, vrhino::DType::F32); }, cases);
        reject("invalid device index", [&] { (void)backend.device_capability(1); }, cases);
        reject("invalid fence device", [&] { (void)backend.create_fence(1); }, cases);
        reject("invalid fence", [&] { backend.wait_fence({}); }, cases);
        auto fence = backend.create_fence();
        reject("wait before record", [&] { backend.wait_fence(fence); }, cases);
        backend.record_fence(fence); backend.wait_fence(fence); backend.destroy_fence(fence);
        reject("double fence destroy", [&] { backend.destroy_fence(fence); }, cases);
        auto duplicate = backend.create_fence(); backend.record_fence(duplicate);
        reject("double fence record", [&] { backend.record_fence(duplicate); }, cases);
        backend.wait_fence(duplicate); backend.destroy_fence(duplicate);
        reject("foreign accelerator buffer", [&] {
            auto storage = std::make_shared<vrhino::Storage>();
            float value = 0.0f; storage->data = &value; storage->bytes = sizeof(value);
            storage->device = vrhino::DeviceId::accelerator();
            vrhino::Tensor tensor(storage, 0, {1}, vrhino::DType::F32);
            (void)backend.copy_to_host(tensor);
        }, cases);
        reject("invalid convolution layout", [&] {
            auto x = vrhino::Tensor::host({1, 1, 2}, vrhino::DType::F32);
            auto w = vrhino::Tensor::host({1, 1, 1, 1}, vrhino::DType::F32);
            (void)backend.conv2d(x, w, nullptr, {1, 1}, {0, 0}, {1, 1}, 1);
        }, cases);
        reject("invalid copy shape", [&] {
            auto source = vrhino::Tensor::host({2}, vrhino::DType::F32);
            auto destination = backend.allocate_device({3}, vrhino::DType::F32);
            (void)backend.copy(source, destination);
        }, cases);
        reject("unsupported quantized primitive", [&] {
            uint8_t packed[2] = {0, 0};
            vrhino::Tensor weight = vrhino::Tensor::borrowed(packed, sizeof(packed), {1, 2}, vrhino::DType::U8);
            auto info = std::make_shared<vrhino::QuantizationInfo>();
            info->type = vrhino::QuantType::INT8Symmetric;
            info->logical_dtype = vrhino::DType::F32;
            info->compute_dtype = vrhino::DType::F32;
            info->accumulation_dtype = vrhino::DType::F32;
            info->group_size = 2; info->axis = 1; info->symmetric = true;
            info->granularity = "per_channel"; info->zero_point_mode = "none";
            info->packing_layout = "byte_twos_complement";
            info->scales = vrhino::Tensor::host({1, 1}, vrhino::DType::F32);
            weight.set_quantization(info);
            auto input = vrhino::Tensor::host({1, 2}, vrhino::DType::F32);
            (void)backend.linear(input, weight, nullptr, vrhino::DType::F32);
        }, cases);
        reject("indexed gather out of range", [&] {
            auto table = vrhino::Tensor::host({2, 1}, vrhino::DType::F32);
            auto indices = vrhino::Tensor::host({1}, vrhino::DType::I64);
            const int64_t invalid = 2;
            std::memcpy(indices.data(), &invalid, sizeof(invalid));
            (void)backend.indexed_gather(table, indices);
        }, cases);
        reject("attention additive bias broadcast", [&] {
            auto q = vrhino::Tensor::host({1, 1, 1, 1}, vrhino::DType::F32);
            auto k = vrhino::Tensor::host({1, 2, 1, 1}, vrhino::DType::F32);
            auto v = vrhino::Tensor::host({1, 2, 1, 1}, vrhino::DType::F32);
            auto bias = vrhino::Tensor::host({3}, vrhino::DType::F32);
            (void)backend.attention(q, k, v, nullptr, false, 1.0f, &bias);
        }, cases);
        if (cases != 14) throw std::runtime_error("Metal negative case count mismatch");
        std::cout << "native Metal negative backend cases=14 pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Metal negative tests: " << error.what() << "\n";
        return 1;
    }
}
