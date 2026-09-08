#include "vrhino/components.h"
#include "vrhino/component_tiling.h"
#include "neural_graph_test_backend.h"

#include <iostream>
#include <type_traits>

using namespace vrhino;

// Including both headers in one translation unit also prevents the original
// duplicate external class identity from silently returning.
static_assert(!std::is_same_v<ComponentExecutor, ComponentGraphExecutor>);

namespace {
template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Expected component failure");
}

void cycle() {
    neural_graph::test::TinyBackend backend;
    auto weight = Tensor::host({1}, DType::F32);
    weight.data_as<float>()[0] = 3.0f;
    const WeightMap weights({{"weight", &weight}});
    auto numerical = std::make_unique<ComponentExecutor>(backend, weights);
    auto graph = std::make_unique<ComponentGraphExecutor>(
        backend, ComponentExecutionConfig{});
    auto input = Tensor::host({1, 1, 1, 1, 1}, DType::F32);
    input.data_as<float>()[0] = 2.0f;
    auto decode = [&](const Tensor& tile) {
        ComponentExecutor local(backend, weights);
        require(&local.weight("weight") == &weight, "Borrowed weight identity");
        return backend.add(tile, local.weight("weight"));
    };
    auto output = graph->execute(input, decode);
    require(output.data_as<float>()[0] == 5.0f, "Component callback output");
    require(graph->stats().tiling.graph_executions == 1, "Execution statistics");
    rejects([&] { graph->execute(input, {}); });
    rejects([&] {
        graph->execute(input, [&](const Tensor&) -> Tensor {
            ComponentExecutor local(backend, weights);
            return local.weight("missing");
        });
    });
    require(graph->execute(input, decode).data_as<float>()[0] == 5.0f,
            "Executor reuse after callback failure");
    ComponentExecutionConfig invalid;
    invalid.mode = ComponentExecutionMode::Tiled;
    rejects([&] { ComponentGraphExecutor bad(backend, invalid); });
    graph.reset();
    require(&numerical->weight("weight") == &weight, "Independent ownership");
    numerical.reset();
    require(weight.data_as<float>()[0] == 3.0f, "Borrowed tensor lifetime");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const int repetitions = argc == 2 ? std::stoi(argv[1]) : 1;
        require(repetitions > 0, "Positive repetition count required");
        for (int i = 0; i < repetitions; ++i) cycle();
        std::cout << "PASS component type identity, callback execution, ownership, "
                     "error cleanup; repetitions=" << repetitions << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
