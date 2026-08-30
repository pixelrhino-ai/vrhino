# Unified VRhino Model CLI v0

The first public product entrypoint is `vrhino`:

```text
vrhino [--cache-root PATH] [--registry URL] [--ca-file PATH] pull namespace/name:version
vrhino [--cache-root PATH] list
vrhino [--cache-root PATH] info namespace/name:version
vrhino [--cache-root PATH] rm namespace/name:version
vrhino [--cache-root PATH] run namespace/name:version --prompt TEXT [RUN OPTIONS]
vrhino [--cache-root PATH] run namespace/name:version --video PATH --audio PATH --output PATH
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

`rm` deletes only the exact immutable package reference and reports `Blobs
retained`. Automatic CAS garbage collection is intentionally outside v0.

`run` resolves an already-installed package and executes its typed Native
product entrypoint. It never pulls implicitly. Text-to-video products accept
`--prompt`; lip-sync products accept `--video`, `--audio`, and `--output`.
Both may accept the bounded common controls `--seed`, `--overwrite`, and
`--debug` where supported. A lip-sync product rejects prompt input. The package
profile, rather than the CLI, owns internal workflow and component policy.

## Configuration and offline behavior

`--cache-root` overrides `VRHINO_HOME` and `~/.vrhino`. `--registry` overrides
`VRHINO_REGISTRY`. `--ca-file` selects a trusted private CA for controlled
registries. Installed `list`, `info`, and `rm` are completely offline.

The binary links the product package/cache layer, Native Runtime, and native
libcurl. It has no Python, PyTorch, Diffusers, official repository, external
`curl`, or `wget` runtime dependency. MP4 encoding uses a separately bundled
native `vrhino-ffmpeg` executable; this is the only child process in `run`.
