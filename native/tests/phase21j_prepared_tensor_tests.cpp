#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/precision.h"
#include "vrhino/tensor_util.h"

namespace {

bool finite_bf16(const vrhino::Tensor& tensor) {
    vrhino::require(tensor.device().is_host() &&
                        tensor.dtype() == vrhino::DType::BF16,
                    "Prepared tensor finite check requires host BF16");
    for (int64_t index = 0; index < tensor.numel(); ++index) {
        const uint32_t bits =
            static_cast<uint32_t>(tensor.data_as<uint16_t>()[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) return false;
    }
    return true;
}

void require_exact(const vrhino::Tensor& left, const vrhino::Tensor& right,
                   const std::string& message) {
    vrhino::require(left.shape() == right.shape() &&
                        left.dtype() == right.dtype() &&
                        left.bytes() == right.bytes() &&
                        std::memcmp(left.data(), right.data(), left.bytes()) == 0,
                    message);
}

}  // namespace

int main() {
    try {
        using namespace vrhino;
        CudaBackend backend;
        backend.set_execution_dtype(DType::BF16);
        const PrecisionPolicy policy =
            PrecisionPolicy::unqualified_default(DType::BF16);
        const std::vector<float> coordinate_values = {
            0.0f, 0.32f, 0.64f, 0.96f,
            0.0f, 32.0f, 64.0f, 96.0f,
            0.0f, 32.0f, 64.0f, 96.0f};
        const Tensor coordinates = host_f32({1, 3, 4}, coordinate_values);
        const std::vector<float> maximum = {20.0f, 2048.0f, 2048.0f};

        PreparedTensorCache cache;
        const PreparedTensorHandle first_handle = prepare_fractional_rope(
            cache, backend, policy, coordinates, maximum, 12, 10000.0f,
            true);
        const std::vector<Tensor>& first = cache.reuse(first_handle);
        require(first.size() == 2, "Prepared fractional RoPE arity changed");
        const void* first_cosine_identity = first[0].data();
        const void* first_sine_identity = first[1].data();

        const PreparedTensorHandle duplicate_handle = prepare_fractional_rope(
            cache, backend, policy, coordinates, maximum, 12, 10000.0f,
            true);
        const std::vector<Tensor>& duplicate = cache.reuse(duplicate_handle);
        require(duplicate[0].data() == first_cosine_identity &&
                    duplicate[1].data() == first_sine_identity,
                "Prepared fractional RoPE did not reuse device tensors");

        auto [direct_cosine, direct_sine] = fractional_rope(
            backend, policy, coordinates, maximum, 12, 10000.0f, true);
        const Tensor prepared_cosine = backend.copy_to_host(first[0]);
        const Tensor prepared_sine = backend.copy_to_host(first[1]);
        const Tensor direct_cosine_host = backend.copy_to_host(direct_cosine);
        const Tensor direct_sine_host = backend.copy_to_host(direct_sine);
        require_exact(prepared_cosine, direct_cosine_host,
                      "Prepared cosine changed numerical output");
        require_exact(prepared_sine, direct_sine_host,
                      "Prepared sine changed numerical output");
        require(finite_bf16(prepared_cosine) && finite_bf16(prepared_sine),
                "Prepared fractional RoPE produced NaN/Inf");

        std::vector<float> changed_values = coordinate_values;
        changed_values[1] += 0.125f;
        const Tensor changed_coordinates =
            host_f32({1, 3, 4}, changed_values);
        const PreparedTensorHandle changed_handle = prepare_fractional_rope(
            cache, backend, policy, changed_coordinates, maximum, 12,
            10000.0f, true);
        const std::vector<Tensor>& changed = cache.reuse(changed_handle);
        require(changed[0].data() != first_cosine_identity,
                "Prepared tensor content change did not invalidate the key");

        const PreparedTensorCacheStats& stats = cache.stats();
        require(stats.prepare_hits == 1 && stats.prepare_misses == 2 &&
                    stats.reuses == 3 && stats.host_materializations == 2 &&
                    stats.host_to_device_transfers == 4 &&
                    stats.host_to_device_bytes == 768 &&
                    stats.resident_entries == 2 &&
                    stats.resident_tensors == 4,
                "Prepared tensor telemetry mismatch");
        std::cout << "status=PASS\n"
                  << "operation=positional.fractional_rope.v1\n"
                  << "prepare_hits=" << stats.prepare_hits << "\n"
                  << "prepare_misses=" << stats.prepare_misses << "\n"
                  << "reuses=" << stats.reuses << "\n"
                  << "host_materializations="
                  << stats.host_materializations << "\n"
                  << "h2d_transfers="
                  << stats.host_to_device_transfers << "\n"
                  << "h2d_bytes=" << stats.host_to_device_bytes << "\n"
                  << "byte_exact=PASS\ncontent_invalidation=PASS\n"
                  << "nan_inf=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase21j prepared tensor: " << error.what() << '\n';
        return 1;
    }
}
