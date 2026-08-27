# VRhino Immutable Local Model Cache v0

Status: product-layer contract v0. Network pull is a cache producer and
preserves this cache layout and immutable publication contract.

## Root and layout

The default root is `~/.vrhino`. `VRHINO_HOME` or the CLI's explicit
`--cache-root` selects another root for testing or managed installations.

```text
~/.vrhino/
  models/
    <namespace>/
      <name>/
        <version>/
          vrhino-model.json
  blobs/
    sha256/
      <first-two-hex>/
        <full-64-hex-sha256>
  tmp/
    downloads/<sha256>.partial
    manifests/<sha256>.json
    locks/package-*.lock
    locks/artifact-*.lock
```

Version directories are immutable references. Artifact bytes live once in the
minimal content-addressed store. This avoids duplicating shared 10–20GB text
encoder weights without introducing OCI layers or a registry protocol.

## Offline source package

An offline source is a directory containing `vrhino-model.json` and all
artifact relative paths declared by it. Paths must be normalized, relative,
non-symlink files contained by the source directory. Absolute paths, `..`,
missing required files, and symlink escapes fail closed.

Current command surface:

```text
vrhino-model [--cache-root PATH] install PACKAGE_DIR
vrhino-model [--cache-root PATH] list
vrhino-model [--cache-root PATH] info NAMESPACE/NAME:VERSION
vrhino-model [--cache-root PATH] resolve NAMESPACE/NAME:VERSION [--verify]
vrhino-model [--cache-root PATH] rm NAMESPACE/NAME:VERSION
```

`vrhino-model install` remains the offline publisher/development primitive.
The `vrhino pull/list/info/rm` commands reuse this exact cache implementation;
the cache contract remains independent from Registry v0.

## Immutable atomic install

Install is a single-writer operation:

1. Canonicalize the package root and parse the bounded manifest.
2. Validate package schema, exact identity, Runtime/VRM compatibility, unique
   ids and paths, entrypoint/default references, source, license, and hardware
   objects.
3. For every artifact, reject symlinks/escapes, check size, stream it through
   SHA256 using bounded memory, and compare the declared digest.
4. Write a new blob to `.incoming-*` in its final CAS directory, flush it,
   mark it read-only, and atomically rename it to the digest name. An existing
   digest is reused after the package source itself is verified.
5. Write and flush the manifest in a unique directory under `tmp/`.
6. Atomically rename that directory to
   `models/<namespace>/<name>/<version>/` and sync the parent directory.

The installed version is invisible until step 6. An existing exact version is
never overwritten. Failure preserves all installed versions. A verified blob
published before a later failure may remain orphaned; that is safe and is the
documented v0 policy.

The implementation has bounded memory usage: the real 24.77GB LTX install used
an 8MiB transfer buffer and approximately 12MiB process RSS.

## Resolution and integrity

Resolution requires an exact `namespace/name:version` reference. It validates
the installed manifest, compatibility, identity/path match, required blob
existence, regular-file status, and size, then returns CAS paths.

CAS blobs are named by the digest and made read-only after full install-time
verification. `resolve --verify` additionally rehashes all installed artifacts
for an explicit deep integrity audit. Normal resolution avoids reading tens of
gigabytes on every run.

Supported stable error codes are:

- `MODEL_NOT_FOUND`
- `PACKAGE_INVALID`
- `PACKAGE_VERSION_UNSUPPORTED`
- `ARTIFACT_MISSING`
- `CHECKSUM_MISMATCH`
- `INSTALL_FAILED`
- `CACHE_ERROR`

## List, info, remove, and GC

`list` scans only the stable `models/` identity tree. `info` reports identity,
architecture declaration, compatibility, default preset, provenance, license,
and counts. `resolve` additionally emits every resolved artifact path.

`rm` deletes only the exact version's manifest/reference directory. It refuses
directories containing unexpected data. It never deletes blobs, so shared
content cannot be damaged. Automatic reference-counted garbage collection is
out of scope for v0; orphan blobs are retained. A later explicit `gc` may scan
all manifests and reclaim unreferenced digests.

Readers may run concurrently with an install because publication is atomic.
Remote pull uses per-package and per-artifact advisory locks plus atomic rename,
so concurrent identical pulls cannot overwrite partials, CAS blobs, or an
installed version. Offline `install` remains fail-closed on an existing exact
version. Concurrent `rm`/future `gc` orchestration is outside v0.

## Security and dependency boundary

The cache contract provides path containment, SHA256 integrity, atomic
publication, immutable identities, and no implicit executable hooks. Registry
v0 adds verified HTTPS/Range input but cannot bypass CAS admission. Package
installation, pull, and resolution are fully Native and require neither Python
nor PyTorch.

The cache library is a separate `vrhino_model_package` target. It does not
alter or link a hardware backend. `.vrm` semantic interpretation remains the
frozen Native Runtime's responsibility.
