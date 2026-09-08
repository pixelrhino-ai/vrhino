# Contributing to VRhino

Public main is the canonical production source. Start with [source build and tests](docs/source-build.md) and [architecture](docs/architecture.md). Open an issue or pull request with the concrete problem, changed behavior and relevant validation.

A new architecture describes a genuinely different graph or parameterization and reuses the shared tensor runtime, precision contracts, memory system and backends. A new backend implements generic tensor capabilities. Neither change creates a model-specific runtime or a wrapper around an official Python pipeline.

Keep NeuralGraph semantics bounded and typed. Precision policy is architecture-neutral: the validation matrix may grow, the implementation matrix must not. Do not add model, block, workload or fixed-shape precision exceptions. Runtime changes should include meaningful synthetic positive and rejection tests, relevant host/CUDA qualification, and an explanation of shared operators reused or added.

Default tests must work offline with synthetic data and redistributable repository inputs. Official PyTorch/Diffusers/Transformers environments may generate external reference artifacts only; they must never enter Native configuration, build, tests claiming Native execution or production execution. Model/oracle tests are optional and take user-supplied inputs.

Keep generated outputs, weights, private media and qualification archives outside Git. Preserve upstream licenses; identify copied or adapted code and its origin. By submitting project-owned contributions, you submit them under the project Apache-2.0 license unless explicitly stated otherwise. See LICENSE and THIRD_PARTY_NOTICES.md.

Future production work belongs on Public main. Private work is a research, evidence, unreleased experiment and license-sensitive overlay. Source presence alone does not establish release support; do not retroactively expand historical release claims.
