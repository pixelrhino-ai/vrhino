# VRhino Static HTTPS Registry v0

Status: product distribution contract v0. It does not change Runnable Model
Package schema 1, `.vrm` 0.1/schema 1, or CUDA Runtime contract `cuda-v1`.

## Source-backed public catalog entries

Public models that are converted locally may ship a fixed declarative catalog
entry with `pull-plan.json`, `source-plan.json`, and `vrhino-model.json`.
These entries resolve before the ready-package network registry and point only
to official upstream model sources. They contain sizes and hashes but no model
weights, converted `.vrm` payloads, credentials, or executable scripts.

`source_backed` covers an ordinary source package;
`multi_component_source_backed` covers one immutable package assembled from
multiple generic component roles. Both use the same Native acquisition, hash
verification, conversion, CAS, and transactional publication infrastructure.
They are distribution metadata, not Runtime or Backend dispatch.

## Exact references and endpoint

The ready-package Registry v0 fallback resolves only the reproducible canonical reference:

```text
namespace/name:version
```

There is no implicit namespace, semver range, or `latest` resolution. A client
requests:

```http
GET {registry}/v1/models/{namespace}/{name}/{version}/index.json
```

The hierarchy is intentionally static-host/CDN compatible. It needs no
database or model-semantic service.

## Descriptor

`index.json` is distribution metadata, separate from the portable installed
`vrhino-model.json`:

```json
{
  "registry_schema_version": 1,
  "identity": "vrhino/ltx-video-v0.9.1:1.0.0",
  "manifest": {
    "url": "https://objects.example/v1/models/vrhino/ltx-video-v0.9.1/1.0.0/vrhino-model.json",
    "size": 4587,
    "sha256": "<64 lowercase hex>"
  },
  "artifacts": [
    {
      "id": "runtime",
      "url": "https://objects.example/blobs/267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d"
    }
  ]
}
```

Artifact ids must refer to package-manifest declarations. The downloaded
package manifest remains byte-for-byte portable and contains no remote URL.
The registry knows distribution identity, bytes and URLs; it does not know
SamplingProgram, CUDA kernels, graphs, tensor names, or architecture execution
semantics.

## Immutability

An exact `namespace/name:version` is immutable. Its manifest hash and every
artifact hash must never change. New bytes require a new package version. The
client treats manifest and artifact SHA256 declarations as the authority and
fails closed before package publication on any mismatch.

## Transport and configuration

The CLI reads `--registry URL`, then `VRHINO_REGISTRY`. No placeholder public
service is hardcoded while no production registry exists. HTTPS certificate
and hostname verification are enabled by default. `--ca-file` admits a private
test/publisher CA without disabling verification.

Plain HTTP is rejected. `--allow-http` is a development-only exception limited
to loopback URLs. Redirects are limited to five and remain restricted to the
configured protocol policy.

The native implementation uses libcurl for mature HTTPS, certificate stores,
Range, redirects, proxy support and timeouts. It never invokes `curl`, `wget`,
Python, PyTorch, Diffusers, or an official model environment.
