// This same translation unit is also compiled against pristine baseline sources
// by step_execution_baseline_compare.py. Do not use new execution APIs here.
#include <iomanip>
#include <iostream>

#include "step_execution_test_support.h"

using namespace vrhino;
namespace st = vrhino::step_test;

int main() {
    try {
        for (int kind = 0; kind < 4; ++kind) for (bool initial : {false, true})
            for (auto mode : {GuidanceMode::CFG, GuidanceMode::Linear}) {
                st::Backend backend;
                st::LegacyDenoiser denoiser(backend, scalar_f32(0.375f));
                auto p = st::program(kind);
                p.guidance_mode = mode;
                if (mode == GuidanceMode::Linear) p.guidance_coefficients = {0.25f, 0.75f};
                SamplingRuntime runtime(backend);
                const auto result = initial ? runtime.run_with_initial_state(denoiser, p, host_f32({1,2}, {0.2f,-0.3f}))
                                            : runtime.run(denoiser, p);
                std::cout << "case " << kind << ' ' << initial << ' ' << static_cast<int>(mode) << '\n';
                for (const auto& [name, t] : result.trace) {
                    std::cout << name << ' ' << dtype_name(t.dtype()) << ' ';
                    for (auto dim : t.shape()) std::cout << dim << ',';
                    std::cout << ' ';
                    const auto* data = static_cast<const unsigned char*>(t.data());
                    for (size_t i = 0; i < t.bytes(); ++i)
                        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]);
                    std::cout << std::dec << '\n';
                }
                std::cout << "rng " << backend.rng_calls << ' ' << result.rng_after_initialization.seed << ' '
                          << result.rng_after_initialization.offset << ' ' << result.rng_after_initialization.algorithm_id << '\n';
                for (const auto& [name, count] : runtime.primitives().calls()) std::cout << name << '=' << count << '\n';
                for (const auto& call : backend.calls) std::cout << call << ',';
                std::cout << '\n';
                const auto& stats = result.prepared_tensors;
                std::cout << "prepared " << stats.prepare_hits << ' ' << stats.prepare_misses << ' ' << stats.reuses << ' '
                          << stats.host_materializations << ' ' << stats.host_to_device_transfers << ' '
                          << stats.host_to_device_bytes << ' ' << stats.resident_entries << ' '
                          << stats.resident_tensors << ' ' << stats.resident_bytes << '\n';
            }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
