# VRhino Pull v0

`vrhino pull` reliably creates one exact Runnable Model Package from its fixed
declarative source plan, or obtains a ready package from Registry v0, and
publishes it into the immutable Local Cache v0. Pull is product orchestration;
Runtime, CUDA, architecture and `.vrm` semantics are untouched.

## Transaction

```text
exact reference
  -> HTTPS registry descriptor
  -> HTTPS package manifest
  -> manifest size + SHA256
  -> package schema/compatibility/identity validation
  -> CAS lookup for each artifact
  -> disk-space admission for missing bytes
  -> resumable HTTPS downloads
  -> artifact size + SHA256
  -> atomic CAS publication
  -> verify all required CAS blobs exist
  -> atomic package-manifest publication
```

Until the last rename, `vrhino list` cannot see the package. Failure may leave
a verified orphan CAS blob or a resumable partial, but never a partially
installed package.

## Resume and partial identity

Partials are content-bound, not filename-bound:

```text
~/.vrhino/tmp/downloads/<artifact-sha256>.partial
```

The current byte length becomes `Range: bytes=<length>-`. A valid `206`
continues append-only. If a server ignores Range and returns `200`, the client
truncates and restarts once. A partial larger than the declared artifact is
discarded. Interrupted and truncated transfers retain safe partial bytes for a
subsequent retry/pull.

Completion requires exact declared size followed by streaming SHA256. A wrong
digest is deleted and returns `CHECKSUM_MISMATCH`; it never enters `blobs/`.

## Network policy

- connect timeout: 30 seconds;
- low-speed read timeout: 60 seconds at less than one byte/second;
- transient retry count: three after the initial attempt;
- retry backoff: bounded 100 ms increments;
- redirect limit: five;
- certificate and hostname verification: mandatory by default.

Network interruption preserves partial bytes. HTTP errors, certificate errors,
unsupported schema, size/hash mismatch, and insufficient disk fail closed.
A single Ctrl+C aborts the active transfer, preserves its partial, prevents
later artifacts and conversion from starting, and exits with status 130.

## CAS and concurrency

An existing size-valid immutable CAS blob is reused without a network request
or full rehash. SHA256 was verified on CAS admission; explicit deep verification
remains available through the package resolver.

Two process-local filesystem locks protect publication:

```text
tmp/locks/package-<namespace-name-version>.lock
tmp/locks/artifact-<sha256>.lock
```

Different artifacts/packages can progress independently. Identical pulls
serialize safely; the second observes the installed package or verified blob.
All final publication uses same-filesystem atomic rename.

## Disk admission and progress

Before each missing artifact, available cache-filesystem bytes are compared to
the declared remaining bytes after a valid partial. Failure returns
`INSUFFICIENT_DISK_SPACE` before transferring that artifact.

Interactive terminals receive one aggregate in-place acquisition progress line
with human-readable bytes, a progress bar, and percentage. Native conversion
uses the same display, driven by actual bytes validated, written, and admitted.
Redirected output contains no carriage returns or ANSI controls and reports
only sparse percentage milestones.

CAS admission and package publication use a separate `Finalizing` display,
also driven by actual verified bytes. The completion summary reports
resolution, acquisition, conversion, and installation/finalization time
separately.

## Source data after conversion

Downloaded source data is retained after network, conversion, or installation
failure so a retry does not force a full redownload. Only after the runnable
package is atomically published and resolves successfully does pull remove the
model's materialized source tree, matching completed partials, and source blobs
proven unshared. Installed package CAS and shared source blobs remain intact.
Pull reports a reclaimed-byte count when cleanup releases data.

## Cache and network environment

Cache-root precedence is:

```text
--cache-root PATH
VRHINO_HOME
$HOME/.vrhino
```

`VRHINO_HOME` changes model/cache storage, not the extracted VRhino binary
installation. `HF_TOKEN` is optional; when set, it is sent only as Bearer
authorization for Hugging Face artifact requests and is neither printed nor
persisted. The native HTTPS stack honors standard `HTTPS_PROXY` and `NO_PROXY`
settings.

## Stable errors

Pull adds these codes to Model Package v0 errors:

- `REGISTRY_UNAVAILABLE`
- `DOWNLOAD_FAILED`
- `DOWNLOAD_RESUME_FAILED`
- `NETWORK_ERROR`
- `INSUFFICIENT_DISK_SPACE`
- `CANCELLED` (process exit status 130)

Existing `PACKAGE_INVALID`, `PACKAGE_VERSION_UNSUPPORTED`, `ARTIFACT_MISSING`,
`CHECKSUM_MISMATCH`, `INSTALL_FAILED`, and `CACHE_ERROR` remain authoritative.
