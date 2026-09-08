# Hermetic tokenizer build sources

This directory is the complete, repository-local source closure for the
production tokenizer build. Exact upstream identities, licenses, archive
hashes, local paths, and integrity inventories are recorded in
`SOURCE-MANIFEST.json`.

`tokenizers-cpp/` contains unmodified upstream Git-archive bytes expanded with
its exact SentencePiece and msgpack gitlinks. The qualified `Cargo.lock` was not
tracked by upstream tokenizers-cpp, so VRhino freezes its exact bytes as a build
input. Small VRhino-owned `.gitignore` exceptions keep that lock, the local
Abseil include bridge, and checksum-covered Cargo source fixtures visible to
Git; they do not change compiled dependency source.
`abseil-cpp/` is the exact commit to which SentencePiece tag `20260107.1`
resolved. `cargo-vendor/` was produced with `cargo vendor --locked --offline`;
Cargo's per-crate `.cargo-checksum.json` files are preserved.

VRhino CMake verifies every regular source file before configuring the
dependency. It adds local Abseil targets and selects SentencePiece's supported
package-provider branch, so SentencePiece's module-provider FetchContent path
is inactive. A generated build-local Cargo home replaces crates.io with
`cargo-vendor/`, and the Cargo wrapper always supplies `--locked --offline`.

Missing or changed local source is a hard configure error. There is no network,
system-package, user Cargo cache, or Git fallback. These build sources are not
copied into the runtime package; only applicable legal texts are packaged.

Public source-only projection: see `SOURCE-PROJECTION.md` for the three omitted upstream WebAssembly binary outputs and the refreshed integrity inventory. Compiled dependency source and license bytes are unchanged.
