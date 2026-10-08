#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

using namespace vrhino;

namespace {
Tensor host_i32(std::vector<int64_t> shape, const std::vector<int32_t>& values) {
    Tensor result = Tensor::host(std::move(shape), DType::I32);
    require(result.numel() == static_cast<int64_t>(values.size()), "Index fixture size mismatch");
    std::memcpy(result.data(), values.data(), result.bytes());
    return result;
}

void exact(const Tensor& a, const Tensor& b) {
    require(a.shape() == b.shape() && a.dtype() == b.dtype() && a.bytes() == b.bytes() &&
            std::memcmp(a.data(), b.data(), a.bytes()) == 0, "Gather output changed");
}

Tensor table(std::vector<int64_t> shape, DType dtype) {
    Tensor value = Tensor::host(shape, dtype);
    for (int64_t i = 0; i < value.numel(); ++i) {
        const float f = i % 11 == 0 ? -0.0f : std::sin(static_cast<float>(i) * 0.17f);
        switch (dtype) {
            case DType::F32: value.data_as<float>()[i] = f; break;
            case DType::BF16: value.data_as<uint16_t>()[i] = std::bit_cast<uint32_t>(f) >> 16; break;
            case DType::F16: {
                const uint16_t bits[] = {0x8000, 0x3c01, 0xbc00, 0x0400, 0x7bff};
                value.data_as<uint16_t>()[i] = bits[i % 5]; break;
            }
            case DType::I64: value.data_as<int64_t>()[i] = i - 30; break;
            case DType::I32: value.data_as<int32_t>()[i] = i - 30; break;
            case DType::U8: value.data_as<uint8_t>()[i] = i % 251; break;
            case DType::Bool: value.data_as<uint8_t>()[i] = i % 2; break;
            default: throw Error("Unsupported test table dtype");
        }
    }
    return value;
}

void selective_and_fallback() {
    int cases = 0;
    for (DType execution : {DType::F32, DType::BF16}) {
        CudaBackend backend;
        backend.set_execution_dtype(execution);
        MemoryRuntimeOptions options; options.enabled = true;
        options.host_staging = false; options.prefetch = false;
        backend.configure_memory_runtime({64ULL << 20, 8ULL << 20, 128ULL << 20,
                                          8ULL << 20, 8ULL << 20}, options);
        for (DType source : {DType::F32, DType::BF16, DType::F16, DType::I64,
                            DType::I32, DType::U8, DType::Bool})
            for (bool batched : {false, true}) for (bool i32 : {false, true}) {
                Tensor host = table(batched ? std::vector<int64_t>{2, 17, 7}
                                           : std::vector<int64_t>{17, 7}, source);
                Tensor ids = i32 ? host_i32({2, 3}, {16, 0, 2, 2, 1, 15})
                                 : host_i64({2, 3}, {16, 0, 2, 2, 1, 15});
                Tensor device = backend.copy_to_device(host, execution);
                Tensor expected = backend.copy_to_host(backend.indexed_gather(device, ids));
                for (bool borrowed : {false, true}) for (bool device_indices : {false, true}) {
                    Tensor input = borrowed ? Tensor::borrowed(host.data(), host.bytes(), host.shape(), source) : host;
                    Tensor indices = device_indices ? backend.copy_to_device(ids, ids.dtype()) : ids;
                    const size_t uploaded = backend.weight_upload_bytes();
                    Tensor actual = backend.indexed_gather(input, indices);
                    const size_t transferred = backend.weight_upload_bytes() - uploaded;
                    require(transferred == 0 || transferred == 6 * 7 * dtype_size(source),
                            "Selective gather uploaded the table or indices");
                    exact(expected, backend.copy_to_host(actual));
                    require(backend.weight_cache_resident_bytes() == 0, "Selective gather cached a borrowed table");
                    ++cases;
                }
            }
        // A nonselective host gather keeps the full-table GPU implementation.
        auto host = table({3, 2}, DType::F32);
        auto ids = host_i64({2, 2}, {2, 0, 1, 2});
        exact(backend.copy_to_host(backend.indexed_gather(host, ids)),
              backend.copy_to_host(backend.indexed_gather(backend.copy_to_device(host, execution), ids)));
        // Rank-1 batched selectors choose one row from each batch.
        auto batches = table({2, 17, 7}, DType::F32);
        auto batch_ids = host_i64({2}, {16, 0});
        exact(backend.copy_to_host(backend.indexed_gather(batches, batch_ids)),
              backend.copy_to_host(backend.indexed_gather(
                  backend.copy_to_device(batches, execution), batch_ids)));
        // Dense byte-offset views are supported as well as base allocations.
        auto backing = std::make_shared<Storage>();
        auto owner = table({18, 7}, DType::F32);
        backing->data = owner.data(); backing->bytes = owner.bytes();
        Tensor view(backing, 7 * sizeof(float), {17, 7}, DType::F32);
        auto selected = host_i32({2}, {16, 0});
        exact(backend.copy_to_host(backend.indexed_gather(view, selected)),
              backend.copy_to_host(backend.indexed_gather(backend.copy_to_device(view, execution), selected)));
        Tensor pending;
        {
            auto temporary = table({17, 7}, DType::F32);
            pending = backend.indexed_gather(temporary, selected);
        }
        auto reference = table({17, 7}, DType::F32);
        exact(backend.copy_to_host(pending), backend.copy_to_host(
            backend.indexed_gather(backend.copy_to_device(reference, execution), selected)));
        // Quantized tables still use their existing dequantization path.
        Tensor packed = Tensor::host({4, 4}, DType::U8);
        for (int i = 0; i < 16; ++i) packed.data_as<uint8_t>()[i] = i;
        auto quant = std::make_shared<QuantizationInfo>();
        quant->type = QuantType::INT8Symmetric; quant->axis = 1; quant->group_size = 4;
        quant->granularity = "per_group"; quant->zero_point_mode = "none";
        quant->packing_layout = "byte_twos_complement";
        quant->scales = host_f32({4, 1}, {0.5f, 0.25f, 1.0f, 2.0f});
        packed.set_quantization(quant);
        auto quant_indices = host_i64({2}, {3, 0});
        exact(backend.copy_to_host(backend.indexed_gather(packed, quant_indices)),
              backend.copy_to_host(backend.indexed_gather(
                  backend.copy_to_device(packed, execution), quant_indices)));
        backend.release_cached_device_memory();
        for (int64_t bad : {-1, 17}) {
            CudaBackend invalid_backend;
            invalid_backend.set_execution_dtype(execution);
            bool rejected = false;
            try { (void)invalid_backend.indexed_gather(view, host_i64({1}, {bad})); }
            catch (const Error&) { rejected = true; }
            require(rejected && invalid_backend.weight_upload_bytes() == 0 &&
                    invalid_backend.resource_session_terminal(),
                    "Gather bounds check occurred after upload");
        }
        // Malformed shapes and nonintegral selectors must fail before upload
        // or host row access, just as on the existing device path.
        for (int invalid = 0; invalid < 4; ++invalid) {
            CudaBackend invalid_backend;
            auto invalid_table = table(invalid == 0 ? std::vector<int64_t>{17}
                : invalid == 3 ? std::vector<int64_t>{2, 17, 7}
                               : std::vector<int64_t>{17, 7}, DType::F32);
            auto invalid_ids = invalid == 1 ? host_i64({}, {0})
                : invalid == 2 ? host_f32({1}, {0}) : host_i64({1}, {0});
            bool rejected = false;
            try { (void)invalid_backend.indexed_gather(invalid_table, invalid_ids); }
            catch (const Error&) { rejected = true; }
            require(rejected && invalid_backend.weight_upload_bytes() == 0 &&
                    invalid_backend.resource_session_terminal(),
                    "Malformed gather bypassed admission checks");
        }
        // An empty index list is not newly supported. Keep the original
        // GPU fallback's rejection for a nonempty table/batch.
        {
            CudaBackend invalid_backend;
            bool rejected = false;
            try {
                (void)invalid_backend.indexed_gather(
                    Tensor::host({1, 3, 2}, DType::F32), host_i64({1, 0}, {}));
            } catch (const Error&) { rejected = true; }
            require(rejected && invalid_backend.resource_session_terminal(),
                    "Empty gather bypassed the original rejection path");
        }
    }
    std::cout << "CUDA_INDEXED_GATHER=PASS selective_cases=" << cases
              << " fallback=PASS quantized=PASS views=PASS lifetime=PASS bounds=PASS shapes=PASS empty_rejection=PASS\n";
}
}  // namespace

int main() {
    try { selective_and_fallback(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
