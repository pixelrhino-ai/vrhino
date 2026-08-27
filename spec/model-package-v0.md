# VRhino Runnable Model Package v0

Status: product-layer contract v0. Runtime contract `cuda-v1` and `.vrm` format
`0.1`/metadata schema `1` remain frozen and unchanged.

## Boundary

A `.vrm` is the Native Runtime execution artifact. A Runnable Model Package is
the distribution unit that makes prompt-to-video execution self-contained:

```text
Runnable Model Package
  manifest
  runtime .vrm
  tokenizer assets
  conditioning graph and weights
  precision policy
  user-facing default preset
  source and license provenance
        |
        v
ResolvedRunnableModel
        |
        v
Native product orchestration -> frozen Native Runtime
```

The package is not a launcher for another model framework and may not reference
Python, PyTorch, Diffusers, a provider cache, or an absolute machine path at
execution time. Package installation, resolution, and Native execution are
self-contained.

## Identity

The canonical exact reference is:

```text
namespace/name:version
```

Each segment contains only lowercase ASCII letters, digits, `.`, `_`, and `-`.
All three segments are required. There is deliberately no implicit `latest` in
v0. Example:

```text
vrhino/ltx-video-v0.9.1:1.0.0
```

`name + version` is immutable within a namespace. Multiple versions coexist.
The package version describes this distribution and is independent from the
upstream checkpoint revision and `.vrm` schema version.

## Manifest

Every source package has a root `vrhino-model.json`. The authoritative machine
schema is [vrhino-model-v0.schema.json](vrhino-model-v0.schema.json). The v0
fields are:

| Field | Meaning |
|---|---|
| `schema_version` | Runnable package schema, currently integer `1` |
| `identity` | namespace, name, version, architecture declaration, publisher |
| `compatibility` | required Runtime contract and `.vrm` format/schema |
| `artifacts` | logical id/role, normalized relative source path, size, SHA256, required flag |
| `entrypoint` | runtime artifact and generic component declarations |
| `defaults` | default preset and user-level preset inputs |
| `hardware` | per-preset admission/qualification evidence; unknown values may be null |
| `source` | upstream repository/revision and converter version |
| `license` | identifier, license artifact, and upstream notice |

An abbreviated manifest has this shape:

```json
{
  "schema_version": 1,
  "identity": {
    "namespace": "vrhino",
    "name": "ltx-video-v0.9.1",
    "version": "1.0.0",
    "architecture": "ltx_v0_9_1",
    "publisher": "VRhino"
  },
  "compatibility": {
    "runtime_contract": "cuda-v1",
    "vrm_schema": {"format_major": 0, "format_minor": 1, "metadata_schema": 1}
  },
  "artifacts": [
    {
      "id": "runtime",
      "role": "runtime.vrm",
      "path": "model/model.vrm",
      "size": 5717174080,
      "sha256": "267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d",
      "required": true
    }
  ],
  "entrypoint": {
    "runtime_artifact": "runtime",
    "components": [],
    "default_preset": "default"
  },
  "defaults": {
    "default_preset": "default",
    "presets": {
      "default": {"profile_artifact": "default-profile", "inputs": {}}
    }
  },
  "hardware": {"presets": {"default": {"minimum_vram_bytes": null}}},
  "source": {
    "repository": "Lightricks/LTX-Video",
    "revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
    "converter_version": "vrhino-native-converter-1"
  },
  "license": {
    "identifier": "LicenseRef-LTX-Video-0.9.1-RAIL-M",
    "artifact": "license",
    "upstream_notice": "https://huggingface.co/Lightricks/LTX-Video"
  }
}
```

## Artifacts and components

Artifact roles are generic, open product vocabulary rather than architecture
dispatch. v0 examples include:

- `runtime.vrm`
- `conditioning.weights.index`
- `conditioning.weights.shard`
- `conditioning.graph`
- `tokenizer.model` and `tokenizer.config`
- `execution.profile` and `execution.precision_policy`
- `legal.license`

An artifact `id` is package-local. Components bind logical groups of artifact
ids using a generic `kind`, for example `conditioning.text_encoder`. The cache
resolver does not interpret architecture names or component kinds; it returns
validated paths and declarations to product orchestration.

All assets needed for the package's declared prompt-to-video entrypoint must be
present as required artifacts. Optional files may add documentation or future
features, but may not be required by the default entrypoint.

## Architecture composition

External tokenizer, conditioning, execution-profile, precision-policy, and
license/provenance components are declared in the manifest rather than
hardcoded in the package resolver. Architecture differences remain package
data; there is no architecture-specific package resolver or package loader.

## Defaults, presets, and hardware

A preset contains user inputs such as width, height, frames, fps, steps, CFG,
and precision plus an `execution.profile` artifact reference. It does not
redeclare internal SamplingProgram, RoPE, tiling, or graph semantics. The
architecture/profile declaration remains authoritative for those semantics.

Hardware metadata is evidence, not guessed precision. Unknown minimums are
`null`. Qualification entries may record the tested GPU and exact scope. A
future preflight may consume minimum/recommended VRAM and compute capability;
the Runtime contains no `if model` admission rule.

## Compatibility and provenance

The v0 Native resolver accepts exactly:

- package schema `1`;
- Runtime contract `cuda-v1`;
- `.vrm` format `0.1`, metadata schema `1`.

It rejects incompatible declarations before execution. Source repository,
immutable upstream revision, converter version, publisher, license identifier,
license text artifact, and upstream notice are mandatory.

## Resolved API

`vrhino::product::LocalModelCache::resolve()` returns
`ResolvedRunnableModel`, containing:

- parsed and compatibility-checked manifest;
- installed manifest path;
- CAS path of the runtime `.vrm`;
- map from every available artifact id to its declaration and CAS path;
- generic component declarations and default preset.

The resolver interface has no CUDA, tokenizer, Python, PyTorch, or Diffusers
dependency.

Remote artifact URLs are deliberately absent from this installed manifest.
They live in the separate Registry v0 descriptor documented in
[registry-v0.md](../docs/protocol/registry-v0.md), preserving package portability and Package
schema 1.

## Native product-run declaration

Package schema 1 remains unchanged. A package selected by
`vrhino run` must point its preset at an `execution.profile` whose generic
`run` declaration fully binds tokenizer assets/specification, conditioning
components and outputs, Runtime inputs, precision policy, and output tensor
contract. The complete contract and fail-closed behavior are documented in
[run-v0.md](../docs/cli/run-v0.md).

The qualified Public Alpha package identity is
`vrhino/ltx-video-v0.9.1:1.1.0`.
