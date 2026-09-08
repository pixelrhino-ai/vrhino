#include "neural_graph.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"

#include <cuda_runtime_api.h>
#include <sys/resource.h>
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using namespace vrhino;
namespace ng = vrhino::neural_graph;

// Observation only, in this serial test executable. Never substitutes completion.
size_t sync_calls = 0;
cudaEvent_t observed_event = nullptr;
cudaError_t event_before = cudaSuccess, event_after = cudaErrorNotReady;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void cuda_check(cudaError_t result) { check(result == cudaSuccess, cudaGetErrorString(result)); }

ng::Graph graph(int stages, bool multiple = false) {
    ng::Description d;
    d.schema = ng::schema_v1;
    d.values = {{{2, 3}}, {{2, 3}}, {{2, 3}}};
    d.inputs = {0, 1};
    if (stages == 1) {
        d.nodes = {{2, ng::Add{0, 1}}}; d.outputs = {2};
    } else {
        d.inputs.push_back(2);
        d.values.push_back({{2, 3}}); d.values.push_back({{2, 3}});
        d.nodes = {{3, ng::Add{0, 1}}, {4, ng::Mul{3, 2}}}; d.outputs = {4};
        if (stages == 3) {
            d.values.push_back({{3, 2}});
            d.nodes.push_back({5, ng::Reshape{4, {3, 2}}});
            d.outputs = multiple ? std::vector<ng::ValueId>{5, 3}
                                 : std::vector<ng::ValueId>{5};
        }
    }
    return ng::admit(d);
}

struct Fixture {
    std::vector<float> x, y, z;
    std::vector<float> expected(int stages) const {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            // Explicit binary32 node boundary, independently evaluated on host.
            volatile float sum = x[i] + y[i];
            result[i] = stages == 1 ? sum : sum * z[i];
        }
        return result;
    }
    std::vector<Tensor> inputs(Backend& backend, int stages) const {
        std::vector<Tensor> result;
        for (const auto* values : {&x, &y, &z}) {
            if (stages == 1 && values == &z) break;
            result.push_back(backend.copy_to_device(host_f32({2, 3}, *values), DType::F32));
        }
        return result;
    }
};

std::vector<Tensor> run(Backend& backend, const ng::Graph& g,
                        const std::vector<Tensor>& inputs) {
    const size_t before = sync_calls;
    auto outputs = ng::evaluate(backend, g, inputs, {});
    // Check completion before D2H or profiling can hide a missing evaluator sync.
    check(sync_calls == before + 1, "evaluate must perform exactly one real CUDA sync");
    cuda_check(cudaStreamQuery(nullptr));
    check(outputs.size() == g.description().outputs.size(), "output count/order contract");
    return outputs;
}

std::vector<float> verify(Backend& backend, const Tensor& tensor,
                         const std::vector<int64_t>& shape,
                         const std::vector<float>& expected, const std::string& name) {
    check(tensor.shape() == shape && tensor.numel() == static_cast<int64_t>(expected.size()),
          "output shape/element count mismatch");
    check(tensor.dtype() == DType::F32 && tensor.logical_dtype() == DType::F32 &&
          !tensor.is_quantized(), "output dtype mismatch");
    check(tensor.device() == DeviceId::accelerator() &&
          tensor.memory_domain() == MemoryDomain::DeviceLocal, "output is not a CUDA tensor");
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, tensor.data()));
    check(attributes.type == cudaMemoryTypeDevice, "output lacks real device storage");
    Tensor host = backend.copy_to_host(tensor);
    std::vector<float> actual(host.data_as<float>(), host.data_as<float>() + host.numel());
    double maximum = 0, sum = 0, relative = 0;
    size_t finite = 0, relative_count = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isfinite(actual[i])) ++finite;
        const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
        maximum = std::max(maximum, error); sum += error;
        if (expected[i] != 0) {
            relative = std::max(relative, error / std::abs(static_cast<double>(expected[i])));
            ++relative_count;
        }
    }
    const bool exact = std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
    std::cout << "NUM\t" << name << '\t' << actual.size() << '\t' << maximum << '\t'
              << sum / actual.size() << '\t' << relative << '\t' << relative_count << '\t'
              << finite << '\t' << actual.size() - finite << '\t' << exact << '\n';
    check(finite == actual.size() && exact, "F32 bitwise reference mismatch");
    return actual;
}

void dispatch(Backend& backend, uint64_t add, uint64_t mul, uint64_t reshape) {
    const auto stats = backend.profile_stats();
    const auto calls = [&](const char* key) {
        auto found = stats.find(key); return found == stats.end() ? 0 : found->second.calls;
    };
    check(calls("elementwise.add") == add && calls("elementwise.mul") == mul &&
          calls("reshape") == reshape, "production CUDA dispatch counts mismatch");
    std::cout << "DISPATCH\t" << add << '\t' << mul << '\t' << reshape << '\n';
}

void CUDART_CB delayed_completion(void*) {
    // A bounded asynchronous host callback on a nonblocking CUDA stream; no kernel.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void completion_probe(Backend& backend, const ng::Graph& g, const std::vector<Tensor>& inputs) {
    cudaStream_t stream{}; cudaEvent_t event{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    try {
        cuda_check(cudaLaunchHostFunc(stream, delayed_completion, nullptr));
        cuda_check(cudaEventRecord(event, stream));
        check(cudaEventQuery(event) == cudaErrorNotReady, "completion probe did not start pending");
        observed_event = event;
        auto outputs = run(backend, g, inputs);
        observed_event = nullptr;
        check(event_before == cudaErrorNotReady && event_after == cudaSuccess,
              "real evaluator sync did not drain observed pending CUDA work");
        cuda_check(cudaEventQuery(event));
        std::cout << "COMPLETION\tpending_before_real_sync\tcomplete_after_real_sync\tPASS\n";
    } catch (...) {
        observed_event = nullptr;
        cudaStreamSynchronize(stream); cudaEventDestroy(event); cudaStreamDestroy(stream);
        throw;
    }
    cuda_check(cudaEventDestroy(event)); cuda_check(cudaStreamDestroy(stream));
}

size_t minimum_free = SIZE_MAX;
void resource(const char* phase, size_t backend_peak = 0) {
    size_t free = 0, total = 0; cuda_check(cudaMemGetInfo(&free, &total));
    minimum_free = std::min(minimum_free, free);
    rusage usage{}; check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    std::cout << "RESOURCE\t" << phase << '\t' << free << '\t' << total - free << '\t'
              << total - minimum_free << '\t' << backend_peak << '\t' << usage.ru_maxrss << '\n';
}
}  // namespace

extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++sync_calls;
    if (observed_event) event_before = cudaEventQuery(observed_event);
    const cudaError_t status = __real_cudaDeviceSynchronize();
    if (observed_event) event_after = cudaEventQuery(observed_event);
    return status;
}

int main() {
    const auto start = std::chrono::steady_clock::now();
    try {
        check(std::fegetround() == FE_TONEAREST, "host reference requires round-to-nearest");
        std::cout << std::setprecision(17);
        resource("before_backend");
        const Fixture exact{{1, -2, .5f, 4, -8, .25f}, {2, .5f, 1.5f, -2, 4, .75f},
                            {2, -2, .25f, .5f, -.5f, 4}};
        const Fixture nontrivial{{.1f, -.7f, 1.234567f, -3.141592f, .00012345f, 123.4567f},
                                 {.2f, .123456f, -.987654f, 2.718281f, -.00002346f, -.76543f},
                                 {.3f, -1.11111f, 7.34567f, -.333333f, 1234.567f, .0012345f}};
        {
            CudaBackend cuda;
            Backend& backend = cuda;
            backend.set_execution_dtype(DType::F32); backend.enable_profiling(true);
            std::cout << "BACKEND\t" << backend.name() << '\n';
            resource("backend_initialized");
            uint64_t adds = 0, muls = 0, reshapes = 0;
            for (int stages = 1; stages <= 3; ++stages) {
                auto outputs = run(backend, graph(stages), exact.inputs(backend, stages));
                verify(backend, outputs[0], stages == 3 ? std::vector<int64_t>{3, 2}
                                                       : std::vector<int64_t>{2, 3},
                       exact.expected(stages), "exact_stages_" + std::to_string(stages));
                ++adds; muls += stages >= 2; reshapes += stages == 3;
                dispatch(backend, adds, muls, reshapes);
                adds = muls = reshapes = 0;  // profile_stats consumes resolved counters.
            }
            // Returned aliases must remain valid after local graph/inputs disappear
            // and while subsequent evaluations allocate their own temporaries.
            auto held = run(backend, graph(3, true), exact.inputs(backend, 3));
            ++adds; ++muls; ++reshapes;
            verify(backend, held[0], {3, 2}, exact.expected(3), "multi_output_0");
            verify(backend, held[1], {2, 3}, exact.expected(1), "multi_output_1");
            const auto g = graph(3); const auto inputs = nontrivial.inputs(backend, 3);
            std::vector<float> first;
            for (int repeat = 0; repeat < 20; ++repeat) {
                auto outputs = run(backend, g, inputs);
                auto actual = verify(backend, outputs[0], {3, 2}, nontrivial.expected(3),
                                     "nontrivial_repeat_" + std::to_string(repeat));
                if (repeat == 0) first = actual;
                check(std::memcmp(first.data(), actual.data(), actual.size() * sizeof(float)) == 0,
                      "repeated evaluation is not bitwise deterministic");
                ++adds; ++muls; ++reshapes;
                resource("repeat", backend.peak_device_bytes());
            }
            verify(backend, held[0], {3, 2}, exact.expected(3), "held_after_repeats_0");
            verify(backend, held[1], {2, 3}, exact.expected(1), "held_after_repeats_1");
            dispatch(backend, adds, muls, reshapes);
            backend.enable_profiling(false);
            completion_probe(backend, g, inputs);
            std::vector<Tensor> failed_outputs;
            const size_t before = sync_calls;
            bool rejected = false;
            try { failed_outputs = ng::evaluate(backend, g, {}, {}); }
            catch (const ng::GraphError& error) {
                rejected = error.code == ng::Code::InvalidInputBinding && error.phase == ng::Phase::Bind;
            }
            check(rejected && failed_outputs.empty() && sync_calls == before,
                  "binding rejection returned partial success or submitted work");
            auto recovered = run(backend, g, inputs);
            verify(backend, recovered[0], {3, 2}, nontrivial.expected(3), "after_binding_failure");
            resource("primary_end", backend.peak_device_bytes());
        }
        resource("primary_destroyed");
        for (int instance = 0; instance < 3; ++instance) {
            CudaBackend backend;
            Fixture distinct = nontrivial; distinct.x[0] += static_cast<float>(instance + 1);
            auto outputs = run(backend, graph(3), distinct.inputs(backend, 3));
            verify(backend, outputs[0], {3, 2}, distinct.expected(3),
                   "serial_instance_" + std::to_string(instance));
            resource("serial_instance", backend.peak_device_bytes());
        }
        resource("after_all_backends");
        std::cout << "PASS\t20_bitwise_repeats\t3_fresh_serial_instances\tordered_alias_lifetime"
                  << "\tbinding_failure_no_partial_outputs\tbackend_failure_NOT_TESTED\n";
        std::cout << "WALL_SECONDS\t" << std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural graph CUDA E2: " << error.what() << '\n'; return 1;
    }
}
