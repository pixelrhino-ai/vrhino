// Exercise the production position preparation at a full video token grid.
#define make_wan_architecture make_wan_position_test_architecture
#define bound_denoiser bound_wan_position_test_denoiser
#define realize realize_wan_position_test_graph
#define ExecutionDefinition PositionTestExecutionDefinition
#include "../src/architectures/wan.cpp"
#undef ExecutionDefinition
#undef realize
#undef bound_denoiser
#undef make_wan_architecture
#include "vrhino/backend/cuda_backend.h"
#include <cstring>
#include <iostream>
#include <memory>

namespace {
void require_exact(const vrhino::Tensor& actual, const vrhino::Tensor& expected,
                   const std::string& message) {
    vrhino::require(actual.shape() == expected.shape() &&
                        actual.dtype() == expected.dtype() &&
                        actual.bytes() == expected.bytes() &&
                        std::memcmp(actual.data(), expected.data(), actual.bytes()) == 0,
                    message);
}
}  // namespace

int main() {
    try {
        using namespace vrhino;
        CudaBackend backend;
        backend.set_execution_dtype(DType::BF16);
        const auto policy = PrecisionPolicy::unqualified_default(DType::BF16);
        wan_family::Config config;
        config.dim = 1536;
        config.heads = 12;
        config.rope_theta = 10000.0;
        const DType dtype = policy.operation_contract(
            PrecisionOperation::Modulation,
            PrecisionSemantic::TemporaryCompute).temporary_dtype;
        PositionGridCache cache;
        prepare_position_grid(backend, policy, config, dtype, 21, 30, 52, cache);
        const void* first_cosine = cache.cosine.data();
        const void* first_sine = cache.sine.data();
        for (int step = 0; step < 100; ++step)
            prepare_position_grid(backend, policy, config, dtype, 21, 30, 52, cache);
        require(cache.cosine.data() == first_cosine &&
                    cache.sine.data() == first_sine,
                "Position grid was rebuilt for unchanged shape");

        std::vector<std::vector<float>> coordinates;
        for (int64_t t = 0; t < 21; ++t) for (int64_t h = 0; h < 30; ++h)
            for (int64_t w = 0; w < 52; ++w)
                coordinates.push_back({float(t), float(h), float(w)});
        const int d = int(config.dim / config.heads);
        auto [cosine, sine] = standard_rope(backend, policy, coordinates,
            {d - 4 * (d / 6), 2 * (d / 6), 2 * (d / 6)},
            config.rope_theta, true);
        cosine = backend.cast(cosine, dtype);
        sine = backend.cast(sine, dtype);
        const Tensor cached_cosine = backend.copy_to_host(cache.cosine);
        const Tensor cached_sine = backend.copy_to_host(cache.sine);
        const Tensor direct_cosine = backend.copy_to_host(cosine);
        const Tensor direct_sine = backend.copy_to_host(sine);
        require_exact(cached_cosine, direct_cosine, "Cached cosine changed numerical values");
        require_exact(cached_sine, direct_sine, "Cached sine changed numerical values");

        // Configuration changes belong to a different endpoint/cache, never to
        // an existing endpoint. Keep the first cache live to detect sharing.
        auto other_config = config;
        other_config.rope_theta = 1000.0;
        auto [other_cosine, other_sine] = standard_rope(backend, policy, coordinates,
            {d - 4 * (d / 6), 2 * (d / 6), 2 * (d / 6)},
            other_config.rope_theta, true);
        other_cosine = backend.cast(other_cosine, dtype);
        other_sine = backend.cast(other_sine, dtype);
        const Tensor other_cosine_host = backend.copy_to_host(other_cosine);
        const Tensor other_sine_host = backend.copy_to_host(other_sine);
        require(std::memcmp(other_cosine_host.data(), direct_cosine.data(),
                            direct_cosine.bytes()) != 0,
                "Changed theta did not exercise different positions");
        const Tensor expected_rope = backend.copy_to_host(
            backend.rope_nd(cosine, other_cosine, other_sine));
        for (bool fail : {false, true}) {
            std::weak_ptr<const void> cosine_lease, sine_lease;
            Tensor queued_rope;
            bool caught = false;
            try {
                PositionGridCache other_cache;
                prepare_position_grid(backend, policy, other_config, dtype,
                                      21, 30, 52, other_cache);
                require(other_cache.cosine.data() != first_cosine &&
                            other_cache.sine.data() != first_sine,
                        "Independent caches shared position tensors");
                require_exact(backend.copy_to_host(other_cache.cosine), other_cosine_host,
                              "New cache reused cosine from a different configuration");
                require_exact(backend.copy_to_host(other_cache.sine), other_sine_host,
                              "New cache reused sine from a different configuration");
                cosine_lease = other_cache.cosine.storage_lease();
                sine_lease = other_cache.sine.storage_lease();
                queued_rope = backend.rope_nd(cosine, other_cache.cosine, other_cache.sine);
                if (fail) throw Error("Injected position cache owner failure");
            } catch (const Error& error) {
                require(fail && std::string(error.what()) == "Injected position cache owner failure",
                        "Unexpected position cache failure");
                caught = true;
            }
            require(caught == fail && cosine_lease.expired() && sine_lease.expired(),
                    "Position tensor ownership survived cache destruction");
            // Submit the next preparation before explicitly waiting for RoPE;
            // safe teardown relies on the Backend's existing release rules.
            PositionGridCache next_cache;
            prepare_position_grid(backend, policy, config, dtype, 21, 30, 52, next_cache);
            require(next_cache.cosine.data() != first_cosine &&
                        next_cache.sine.data() != first_sine,
                    "Independent caches shared tensors for identical parameters");
            require_exact(backend.copy_to_host(queued_rope), expected_rope,
                          "Cache teardown corrupted queued GPU use");
            require_exact(backend.copy_to_host(next_cache.cosine), direct_cosine,
                          "Subsequent cache retained previous configuration");
            require_exact(backend.copy_to_host(next_cache.sine), direct_sine,
                          "Subsequent cache retained previous configuration");
        }

        prepare_position_grid(backend, policy, config, dtype, 21, 30, 51, cache);
        require(cache.width == 51 && cache.cosine.data() != first_cosine &&
                    cache.sine.data() != first_sine,
                "Position grid shape change did not invalidate cache");
        backend.synchronize();
        std::cout << "Wan position grid cache: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Wan position grid cache: FAIL: " << error.what() << '\n';
        return 1;
    }
}
