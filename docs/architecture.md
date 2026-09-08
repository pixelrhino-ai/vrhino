# Production architecture

VRhino executes model computation in its own C++ tensor runtime. Architecture descriptions and model tensor mappings sit above that runtime; official Python pipelines are reference tools outside the production dependency graph.

```text
VRM container and model/component metadata
  -> architecture graph and Sampling declarations
  -> shared DiT / Flow runtime and bounded NeuralGraph
  -> tensor, memory and precision contracts
  -> generic backend operators
```

`native/src/runtime/` owns execution, prepared tensors and memory planning. `native/src/runtime/neural_graph_*` validates and evaluates a bounded typed DAG. Inputs, parameters, shapes and dtypes must satisfy the graph contract before backend execution. Graph evaluation owns intermediate values and drains backend work safely on failure. This is not a general Python program executor.

Wan and LTX production self-attention use NeuralGraph composition. Their graph construction/binding lives under `native/src/architectures/`; it reuses validation, evaluation and backend operators. Test-only imperative reference implementations are kept under `native/tests/`. They are not production fallback paths.

| Architecture source | Shared facilities reused | Architecture-specific work | Release meaning |
|---|---|---|---|
| Wan | tensor operators, NeuralGraph, attention, normalization, positional operations, memory, Sampling | graph/parameter binding and architecture configuration | consult the historical release model list |
| LTX | the same shared runtime and backend foundation | graph/parameter binding and architecture configuration | consult the historical release model list |
| HunyuanVideo | tensor operators, attention, normalization, positional operations, memory, Sampling and Component infrastructure | architecture composition/configuration | source present; not added to v0.6 support |
| Mochi | shared tensor, attention, normalization, memory, Sampling and Component infrastructure | architecture composition/configuration | consult the historical release model list |
| CogVideoX | shared tensor/runtime/backend capabilities | architecture canary composition | source canary; no full release-support claim |

This publication adds zero runtime operators, zero backend kernels and zero model-specific runtime implementations. It publishes the existing foundation. For later architectures, contributions must record shared operators reused, new generic operators, shared runtime changes and architecture-specific graph/config work. Maintenance should track architecture families, not model count.

The Backend interface in `native/include/vrhino/backend.h` is generic. CUDA source lives in `native/src/backend/cuda/`; vendor libraries are external build/runtime dependencies. CPU host tests use bounded test backends. Precision policy in `native/src/precision/` and its public header governs semantic dtype behavior. F32, BF16/Cast and INT8 qualification are separate validation cases, not model-specific implementations or private tolerances.

Sampling programs describe schedules, state updates and denoiser calls. `native/src/sampling/` executes the shared declarations. Component execution and tiling in `native/src/components/` compose generic operators for compatible text/audio/vision/VAE structures. A component numerical executor and a component graph executor have distinct type and lifetime ownership; synthetic tests cover that boundary.

Converter code in `native/src/product/` maps upstream tensor formats into VRM and validates package metadata without importing or spawning Python. Product, application and API code own request validation, acquisition/cache, jobs, progress, media subprocesses and transport. Those responsibilities do not enter the shared tensor runtime. Neural execution never uses a model-specific official pipeline.

See [VRM format](../spec/vrm-v0.1.md), [model package](../spec/model-package-v0.md), [precision/container extensions](source-contracts.md), [API](api/native-api-v1.md) and [packaging](source-packaging.md).
