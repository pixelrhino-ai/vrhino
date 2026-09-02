# Unified VRhino Model CLI v0

The first public product entrypoint is `vrhino`:

```text
vrhino [--cache-root PATH] [--registry URL] [--ca-file PATH] pull namespace/name:version
vrhino [--cache-root PATH] list
vrhino [--cache-root PATH] info namespace/name:version [--json]
vrhino [--cache-root PATH] rm namespace/name:version
vrhino [--cache-root PATH] run namespace/name:version --prompt TEXT [RUN OPTIONS]
vrhino [--cache-root PATH] run namespace/name:version --video PATH --audio PATH [--output PATH]
```

All v0 references require namespace and exact version. This avoids hidden
resolution state; a future default-version endpoint may add a shorthand without
changing installed identity.

## Commands

`pull` resolves Registry v0, downloads/verifies missing artifacts, reuses CAS
content, and atomically publishes the exact package. It alone requires network.

`list` scans installed manifests rather than blobs and does not rehash multi-GB
files. Columns are `NAME`, `VERSION`, `ARCHITECTURE`, `SIZE`, and `INSTALLED`.

`info` resolves the local package and reports identity, architecture
declaration, package/Runtime/VRM compatibility, default preset, hardware
minimum/recommended metadata, source revision, license, logical size and local
CAS completeness. Internal tensor and SamplingProgram details are omitted.
Without an output option it retains the existing human-readable local
diagnostic format.

`info MODEL --json` emits the stable
[Model Info JSON v1](../product/model-info-json-v1.md) projection.
Schema-backed packages project their canonical
`product.input_schema` and `product.frozen_profile` directly; legacy packages
emit null for both fields rather than inferring controls. Success writes only
one JSON document to stdout. Failures remain nonzero with human-readable stderr.
The JSON projection omits local cache/manifest paths and requires neither
network access nor GPU/Backend construction.

`rm` deletes only the exact immutable package reference and reports `Blobs
retained`. Automatic CAS garbage collection is intentionally outside v0.

`run` resolves an already-installed package and executes its typed Native
Product entrypoint. It never pulls implicitly. Text-to-video Products accept
`--prompt`; lip-sync Products accept `--video` and `--audio`. `--output`,
`--seed`, `--overwrite`, and `--debug` are bounded common controls where the
Product schema admits them. The package Product contract and profile, rather
than the CLI, own requiredness, defaults, geometry, timing, sampling,
precision, and component execution policy.

## Configuration and offline behavior

`--cache-root` overrides `VRHINO_HOME` and `~/.vrhino`. `--registry` overrides
`VRHINO_REGISTRY`. `--ca-file` selects a trusted private CA for controlled
registries. Installed `list`, `info`, and `rm` are completely offline.

The binary links the product package/cache layer, Native Runtime, and native
libcurl. It has no Python, PyTorch, Diffusers, official repository, external
`curl`, or `wget` runtime dependency. MP4 encoding uses a separately bundled
native `vrhino-ffmpeg` executable; this is the only child process in `run`.
