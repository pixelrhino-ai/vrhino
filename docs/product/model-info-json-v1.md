# VRhino Model Info JSON v1

`vrhino info NAMESPACE/NAME:VERSION --json` emits a stable, machine-readable
projection of an installed Runnable Model Package. It is intended for CLI
automation, community UI discovery, and reuse by the Native API model-detail
route. It is not an execution request, workflow description, or API error
protocol.

The document envelope version is the integer:

```json
{"schema_version":1}
```

This envelope version is distinct from the embedded Product contract identity
`vrhino.product.input-schema.v1`.

## Lookup and execution boundary

The command preserves the existing `info` lookup contract:

- the exact package version must already be installed in the local model cache;
- the installed manifest and artifact presence/size admission must succeed;
- no Registry request, pull, source acquisition, conversion, telemetry, or
  automatic repair occurs;
- no Backend, CUDA context, cuBLAS/cuDNN state, model weights, neural execution,
  media processing, or inference is constructed.

A missing or invalid package fails with a nonzero exit and a human-readable
diagnostic on stderr. No partial JSON or JSON error envelope is written.

## Stable projection

The v1 top-level fields are:

| Field | Source and meaning |
|---|---|
| `schema_version` | Model Info JSON envelope version, exactly `1` |
| `model` | structured package namespace, name, version, reference, architecture, publisher |
| `product` | family/status/workflow and the canonical Product contract |
| `compatibility` | package schema, Runtime contract, VRM schema and default preset |
| `admission` | enforced minimum/recommended VRAM metadata; null when undeclared |
| `qualification` | optional separately-authored observations already present in package metadata |
| `distribution` | existing public distribution mode |
| `source` | public-safe upstream repository and revision |
| `license` | existing identifier and artifact/upstream notice references |
| `artifacts` | logical bytes and declared/present/component counts |
| `installation` | privacy-safe installed state |

Qualification observations are not admission requirements. Distribution,
source, and license data are not Product parameters.

No local manifest path, CAS path, cache root, source-development path,
credential, token, graph, tensor list, workflow body, execution profile, or
Backend state is emitted. Human `vrhino info MODEL` retains its existing local
diagnostic path and formatting.

## Product contract

For schema-backed successor packages, `product.input_schema` and
`product.frozen_profile` are copied directly from the already parsed and
verified `vrhino-model.json`. The CLI does not maintain model-name branches,
defaults, declaration tables, validation tables, or a second Product schema.

Product declarations therefore preserve exactly:

- names and group order;
- v1 semantic types;
- requiredness;
- defaults;
- bounded validation;
- decimal-string uint64 bounds;
- rational FPS objects;
- curated read-only frozen facts.

Frozen facts remain read-only. They are never merged into `parameters`, and no
field is reflected automatically from execution or workflow metadata.

For a legacy package with no ProductInputSchema, both fields are JSON null:

```json
{
  "product": {
    "input_schema": null,
    "frozen_profile": null
  }
}
```

Null means successful inspection of a legacy package whose contract is absent.
Malformed or unsupported schema metadata instead causes command failure.

## Serialization and streams

Success writes one compact JSON document followed by one newline to stdout.
Nothing else is written to stdout or stderr. Object fields are serialized in
stable lexical order from ordered maps; arrays retain package order. There are
no timestamps, generated IDs, or locale-dependent formatting. Strings are
escaped and validated as UTF-8.

Integer values that exceed signed 64-bit JSON storage are represented as
canonical unsigned decimal strings. Existing ProductInputSchema uint64 bounds
remain strings, so JavaScript clients do not lose precision. Rational values
remain numerator/denominator objects.

Failure writes no JSON to stdout, exits nonzero, and writes the existing
human-readable diagnostic to stderr.

## Reuse boundary

One transport-neutral projection is shared by the CLI and
`GET /api/v1/models/{model}`. Clients can use either surface without parsing
human CLI text. Job submission, UI layout, SDKs, and Website generation are
outside Model Info JSON v1.
