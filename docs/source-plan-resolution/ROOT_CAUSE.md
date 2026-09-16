# Declared source-plan resolution: root cause and caller audit

## Failure mechanism

Linux v0.8 packaged remote-import qualification at product commit
`606bffc177344919b2324bb0e2c8f5767c7796eb` failed with
`SOURCE_INVALID: source plan field must be a non-empty string: model_reference`.
Local-source conversion and real inference passed because they never called the
remote source-plan resolver. Those results do not qualify remote import.

The failing tree is `share/vrhino/converters` in
`vrhino-linux-x86_64-cuda-v0.8.0-alpha-final-candidate-606bffc.tar.gz`
(SHA256 `aa4bf5f5819b8c204150163c28bf058aec64067fb8b2ac3674f67988f376212f`).
It contains the same 20 source-plan documents as `native/specs`: ten model plans
(including historical identities) and ten component provenance documents.

`vrhino import MODEL hf://REPOSITORY@REVISION` passed the entire packaged converter
root to `load_source_artifact_plan`. That function recursively visited every file
named `source-plan.json`, called the strict model parser, and only **after parsing**
compared `model_reference`. A valid unrelated component therefore aborted the
whole search, even if the requested model had already been found.

The first component visited in the observed source and extracted package trees
was `bisenet_face_parser/source-plan.json`. It has `schema_version: 1`,
`component: semantic_segmenter_2d`, `graph_identity: semantic_segmenter_2d.v1`,
`reference`, and `sources`. It records the BiSeNet checkpoint and upstream
provenance for a reusable component, not a runnable model or model acquisition
request; it legitimately has no `model_reference` or `requested_source`.
`schema_version: 1` is not a universal schema discriminator.
Filesystem iteration order is not an API guarantee; every component collision
would cause the same failure. Regression fixtures create a component first and
place it under a lexically earlier `00-component` directory, plus a nested
`model/component-a` document.

This is shared Product acquisition code, not a Linux-specific issue. All five
public successor identities are affected by whole-tree remote import. Existing
pull and doctor calls scan only the selected pull-plan directory, whose current
successor subtrees contain no components, so they do not currently hit this exact
collision. They nevertheless rediscover a resource already explicitly declared
and could fail or disagree when unrelated plans are added to that subtree.

## Resolution and fail-closed boundaries

- `load_source_artifact_plan_file(path, expected_model_reference)` strictly parses
  exactly one model plan and enforces its identity. Schema, source provider,
  immutable revisions, artifact paths, sizes, hashes and duplicate checks remain.
- `load_pull_source_artifact_plan(plan)` validates the relative declaration,
  rejects portable traversal/drive paths and symlink escapes, loads that exact
  resource, and verifies its immutable source binding against the pull plan.
- Pull, remote import and doctor share that resolver. CLI import also passes the
  selected manifest explicitly to conversion, preserving successor profiles.
- Missing, malformed, wrong-identity or unsafe explicitly declared plans fail;
  they never fall back to discovery. Doctor no longer suppresses declared-plan
  validation failures, including for an otherwise installed healthy model.
- Legacy discovery remains for callers without a pull declaration. It recognizes
  component/component_kind provenance discriminators before the model parser.
  A document carrying model_reference or requested_source is never skipped as a
  component. Malformed model documents and duplicate matching model plans fail.
- No component resource, model spec, preset, version, runtime, architecture,
  backend, CUDA kernel or precision policy is changed.

## Caller audit

| Caller before fix | Class | Resolution after fix |
|---|---|---|
| `pull_runnable_model`, including multi-component repair | A | Shared exact declared loader; validate before removing repair descriptors |
| `doctor.cpp: descriptor_manifest` | A | Same exact loader, propagate invalid declared plan |
| CLI `import hf://...` | D | A when pull plan exists; B only for legacy models without a pull plan |
| `native_source_acquisition_tests` standalone acquisition harness | C/B | Retains component-aware model discovery for transport fixtures |
| `native_source_acquisition_tests` synthetic lookup cases | C | Discovery coverage retained |
| `native_pull_orchestration_tests` | C | Exact, discovery, collision, strict rejection and real loopback transport coverage |
| `product_schema_successor_orchestration_tests` | C/A | Exact pull declaration plus whole packaged-tree discovery for all five identities |
| `wan_product_integration_tests` | C/B | Historical model-reference discovery |
| `mochi_product_integration_tests` | C/B | Historical model-reference discovery |
| `public_musetalk_product_tests` | C/B | Existing public source declaration coverage |
| `public_latentsync_product_tests` | C/B | Existing public source declaration coverage |

No additional source-plan resolver call exists in successor conversion or repair;
repair goes through `pull_runnable_model`. A = pull-plan declaration; B = genuine
model-reference discovery; C = test-only; D = previously ambiguous production use.

## All public model paths

Paths below are relative to `native/specs` or packaged `share/vrhino/converters`.
The machine-readable [audit](PUBLIC_SOURCE_PLAN_AUDIT.json) includes every resolved
path and component document schema/role. Every declaration below is exactly
`source-plan.json`, resolved beside its pull plan.

| Model | Pull-plan path | Components under package root / selected directory | Pre-fix root collision |
|---|---|---:|---|
| `vrhino/ltx-video-v0.9.1:1.1.1` | `ltx_v0_9_1/successors/1.1.1/pull-plan.json` | 10 / 0 | yes |
| `vrhino/wan2.1-t2v-1.3b:1.0.1` | `wan2_1_t2v_1_3b/successors/1.0.1/pull-plan.json` | 10 / 0 | yes |
| `vrhino/mochi-1-preview:1.0.1` | `mochi_1_preview/successors/1.0.1/pull-plan.json` | 10 / 0 | yes |
| `vrhino/musetalk-v1.5:1.0.1` | `public_musetalk_v15/successors/1.0.1/pull-plan.json` | 10 / 0 | yes |
| `vrhino/latentsync-1.6:1.0.1` | `public_latentsync_16/successors/1.0.1/pull-plan.json` | 10 / 0 | yes |

`PUBLIC_SOURCE_PLAN_AUDIT=PASS`

## Validation and release consequence

Focused native regressions exercise exact loading despite unrelated component
plans, discovery duplicates, wrong identity, missing/malformed declared documents,
invalid schema/hash/path/revision/source binding, and symlink escape when the host
permits symlinks. A loopback server verifies a real Native HTTP acquisition with
no source-cache reuse; pull then reaches conversion. Doctor regressions use the
real doctor implementation and ensure invalid declarations cannot become READY.
The public CI runs resolution/transport/successor tests on Linux and Windows.

MuseTalk 1.0.1 execution profile remains
`a5e1592fec66c5fb2a69b0e965e2dde8a7a4d45319535123712ca5c7b04d4a68`;
LatentSync 1.0.1 remains
`f867e628bf4d2d588e265f480c30d0c60e9727d2e341a4300cc7c3081ce85b81`.
The existing successor resource regressions remain required.

Windows rc8 is immutable historical qualification evidence with qualification
PASS, but is superseded as a release asset by this shared implementation fix.
New Windows rc9 and Linux candidates must use the exact post-merge main commit
and still report `v0.8.0-alpha`. No release tag or publication is authorized here.
Packaged real remote acquisition and bounded cross-platform runtime qualification
remain separate release gates; local-source qualification cannot substitute.
Mochi memory admission and deferred objective lip-sync quality policy are unchanged.

`SOURCE_ACQUISITION_RESOLUTION_DELTA=1`; architecture, runtime, backend, CUDA,
kernel, precision, model-spec and model-preset semantic deltas are all zero.
