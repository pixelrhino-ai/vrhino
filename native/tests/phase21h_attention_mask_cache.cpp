#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"

namespace {

uint64_t calls(const std::map<std::string, vrhino::ProfileStat>& profile,
               const std::string& name) {
    const auto found = profile.find(name);
    return found == profile.end() ? 0 : found->second.calls;
}

vrhino::Tensor prefix_mask(int64_t batch, int64_t keys,
                           const std::vector<int64_t>& lengths) {
    std::vector<uint8_t> values(static_cast<size_t>(batch * keys), 0);
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t k = 0; k < lengths.at(static_cast<size_t>(b)); ++k)
            values[static_cast<size_t>(b * keys + k)] = 1;
    return vrhino::host_bool({batch, 1, keys}, values);
}

}  // namespace

int main() {
    try {
        using namespace vrhino;
        constexpr int64_t batch = 2, queries = 64, keys = 128;
        constexpr int64_t heads = 2, width = 64;
        const std::vector<int64_t> q_shape = {batch, queries, heads, width};
        const std::vector<int64_t> kv_shape = {batch, keys, heads, width};
        std::vector<float> q(static_cast<size_t>(shape_numel(q_shape)));
        std::vector<float> k(static_cast<size_t>(shape_numel(kv_shape)));
        std::vector<float> v(k.size());
        for (size_t index = 0; index < q.size(); ++index)
            q[index] = static_cast<float>(static_cast<int>(index % 37) - 18) / 64.0f;
        for (size_t index = 0; index < k.size(); ++index) {
            k[index] = static_cast<float>(static_cast<int>(index % 41) - 20) / 128.0f;
            v[index] = static_cast<float>(static_cast<int>(index % 53) - 26) / 32.0f;
        }

        CudaBackend backend;
        backend.set_execution_dtype(DType::BF16);
        const Tensor q_device = backend.copy_to_device(host_f32(q_shape, q), DType::BF16);
        const Tensor k_device = backend.copy_to_device(host_f32(kv_shape, k), DType::BF16);
        const Tensor v_device = backend.copy_to_device(host_f32(kv_shape, v), DType::BF16);
        Tensor mask_device = backend.copy_to_device(
            prefix_mask(batch, keys, {128, 97}), DType::Bool);

        setenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION", "force", 1);
        backend.enable_profiling(true);
        const Tensor first = backend.attention(
            q_device, k_device, v_device, &mask_device, false, 0.0f);
        const Tensor second = backend.attention(
            q_device, k_device, v_device, &mask_device, false, 0.0f);
        const auto initial = backend.profile_stats();
        const Tensor first_host = backend.copy_to_host(first);
        const Tensor second_host = backend.copy_to_host(second);
        require(std::memcmp(first_host.data(), second_host.data(),
                            first_host.bytes()) == 0,
                "cached Attention output changed");
        require(calls(initial, "attention.mask_cache.hit") == 1 &&
                calls(initial, "attention.mask_cache.miss") == 1 &&
                calls(initial,
                      "attention.mask_cache.canonicalization_build") == 1 &&
                calls(initial, "attention.mask_cache.descriptor_build") == 1 &&
                calls(initial, "attention.mask_cache.d2h") == 1,
                "initial cache telemetry mismatch");

        const Tensor changed_host = prefix_mask(batch, keys, {128, 65});
        backend.copy(changed_host, mask_device);
        const Tensor changed = backend.attention(
            q_device, k_device, v_device, &mask_device, false, 0.0f);
        const Tensor changed_repeat = backend.attention(
            q_device, k_device, v_device, &mask_device, false, 0.0f);
        const auto mutated = backend.profile_stats();
        unsetenv("VRHINO_CUDA_ATTENTION_SDPA_ADMISSION");
        const Tensor changed_host_output = backend.copy_to_host(changed);
        const Tensor changed_repeat_host = backend.copy_to_host(changed_repeat);
        require(std::memcmp(changed_host_output.data(), changed_repeat_host.data(),
                            changed_host_output.bytes()) == 0,
                "post-invalidation repeat changed");
        require(std::memcmp(first_host.data(), changed_host_output.data(),
                            first_host.bytes()) != 0,
                "mask mutation did not affect Attention output");
        require(calls(mutated, "attention.mask_cache.invalidation") == 1 &&
                calls(mutated, "attention.mask_cache.hit") == 1 &&
                calls(mutated, "attention.mask_cache.miss") == 1 &&
                calls(mutated,
                      "attention.mask_cache.canonicalization_build") == 1 &&
                calls(mutated, "attention.mask_cache.descriptor_build") == 1 &&
                calls(mutated, "attention.mask_cache.d2h") == 1,
                "mutation invalidation telemetry mismatch");

        for (int64_t index = 0; index < changed_host_output.numel(); ++index) {
            const uint16_t bits = changed_host_output.data_as<uint16_t>()[index];
            const uint32_t widened = static_cast<uint32_t>(bits) << 16;
            float value = 0.0f;
            std::memcpy(&value, &widened, sizeof(value));
            require(std::isfinite(value), "Attention cache produced NaN/Inf");
        }
        std::cout << "initial_hits=1\ninitial_misses=1\ninitial_d2h=1\n"
                  << "initial_descriptor_builds=1\n"
                  << "post_mutation_invalidations=1\n"
                  << "post_mutation_hits=1\npost_mutation_misses=1\n"
                  << "phase21h_attention_mask_cache=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase21h Attention mask cache: " << error.what() << '\n';
        return 1;
    }
}
