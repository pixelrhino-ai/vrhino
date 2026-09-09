# Contributing to VRhino

Public main is the canonical production source. Start with [source build and tests](docs/source-build.md) and [architecture](docs/architecture.md). Open an issue or pull request with the concrete problem, changed behavior and relevant validation.

## Production contribution workflow

1. Fetch the Public repository and update local main with a fast-forward:
   `git fetch origin`, `git switch main`, then `git merge --ff-only origin/main`.
   Here `origin` must be `https://github.com/pixelrhino-ai/vrhino.git`; fork contributors should use their Public upstream remote for these commands.
2. Create a topic branch from that main, for example `git switch -c feature/<topic>`.
3. Make a focused production change and keep research history outside the Public repository.
4. Run relevant local validation. At minimum, run `python3 tools/validate_source_hygiene.py` and `python3 tools/validate_public_contract.py` with the developer prerequisites in [source build and tests](docs/source-build.md). Add the relevant synthetic host tests and, only when applicable to the change, CUDA qualification. Report unavailable validation accurately.
5. Push the topic branch and open a Pull Request targeting Public `main`. Describe the problem, resulting behavior and validation.
6. Wait for the required `Linux host and offline tokenizer build` check from the `Public source` workflow. If main advances, update the branch with current Public main and wait for CI again.
7. Squash merge only after required CI passes. No external reviewer or approval is required for the solo maintainer. Delete the merged topic branch when it is no longer needed.

Recommended branch names are `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `ci/<topic>` and `release/<version>`. These are lightweight conventions. `research/*` is not a normal Public production branch namespace.

Main requires PRs, current-base CI and linear history; force pushes and deletion are blocked. See [repository governance](docs/repository-governance.md) for the exact settings and emergency policy.

## Architecture and validation

Preserve these boundaries:

- new checkpoint != new Runtime
- new architecture != new Runtime
- new backend != architecture implementation
- new architecture != new Precision Policy

A new architecture describes a genuinely different graph or parameterization and reuses the shared tensor runtime, precision contracts, memory system and backends. A new backend implements generic tensor capabilities. Neither change creates a model-specific runtime or a wrapper around an official Python pipeline.

Keep NeuralGraph semantics bounded and typed. Precision policy is architecture-neutral: the validation matrix may grow, the implementation matrix must not. Do not add model, block, workload or fixed-shape precision exceptions. Runtime changes should include meaningful synthetic positive and rejection tests, relevant host/CUDA qualification, and an explanation of shared operators reused or added.

Default tests must work offline with synthetic data and redistributable repository inputs. Official PyTorch/Diffusers/Transformers environments may generate external reference artifacts only; they must never enter Native configuration, build, tests claiming Native execution or production execution. Model/oracle tests are optional and take user-supplied inputs.

Keep generated outputs, weights, private media and qualification archives outside Git. Preserve upstream licenses; identify copied or adapted code and its origin. By submitting project-owned contributions, you submit them under the project Apache-2.0 license unless explicitly stated otherwise. See LICENSE and THIRD_PARTY_NOTICES.md.

Future production work belongs on Public main. Private work is a research, evidence, unreleased experiment and license-sensitive overlay. Source presence alone does not establish release support; do not retroactively expand historical release claims.

Private research starts from a recorded Public main commit SHA. Research history is not production history: reduce any research promoted to production to a clean production diff on a fresh Public topic branch, then use the PR and CI workflow above. Keep research, evidence, artifacts, builds and archive workspaces separate from Public source.

## Published releases

Published tags and releases are immutable by project policy, including the frozen `v0.5.0-alpha`, `v0.6.0-alpha` and `v0.7.0-alpha` releases. Never move a published tag, replace published binary assets, force-update release refs or reuse an existing release version. Fixes require a new version.
