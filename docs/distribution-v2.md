# Distribution v2 — v0.9.2-alpha

The v0.9.2-alpha Linux release promotes the qualified RC1 split archives as
exactly two artifacts: `vrhino-core` and `vrhino-nvidia-runtime`. The archives
retain their original v0.9.1-alpha build identity and bytes. This is a
distribution change; executable, model, CUDA, and media bytes are unchanged.
The v0.9.1-alpha monolithic release remains available and immutable.

## Artifact provenance

The audited split inventory assigned every original regular file and symlink
to one of the two artifacts. The split process verified the monolithic outer
SHA and every source file against its internal `SHA256SUMS`. The three original
launchers and checksum file remain under core `provenance/`; the split launchers
resolve the private dependency path. The formal release uploads the qualified
RC1 archives unchanged. The final catalog is generated from those exact
archives and their embedded manifests, then independently pinned by the
versioned bootstrap.

## Offline install and upgrade

```sh
sh tools/install_distribution_v2.sh --bundle-dir /path/to/output
```

The default store is `${XDG_DATA_HOME:-$HOME/.local/share}/vrhino-v2` with
`runtime/VERSION`, `dependencies/EXACT_ID`, and an atomically replaced
`current` symlink. Override with `--prefix` and `--bin-dir` for testing. A
catalog may also contain `core_url` and `runtime_url` HTTPS fields and be
passed with `--catalog`. A versioned online catalog can be produced with
`tools/distribution_v2_catalog.py` after freezing both archive SHA256 values.
Online installation uses `--catalog-url HTTPS_URL --catalog-sha256 SHA256`;
`--expected-catalog-version VERSION` rejects a stale or different entry;
`--ca-file PEM` is available for an isolated HTTPS qualification source. The
catalog SHA must come from a trusted release entry or installer, independently
of the catalog URL. The default `install.sh` pins the published v0.9.2-alpha
catalog SHA and the qualified installer SHA. Offline users can put both frozen
archives, the exact installer script, and a copy of the final catalog named
`catalog.txt` in one directory, then use the versioned `install.sh --bundle-dir DIR`.

The installer verifies the outer SHA before extraction, the internal SHA and
symlink targets after extraction, exact runtime ID, runtime API, and core
dependency SHA. It runs `--version` and `doctor` before switching `current`.
An existing dependency with the same archive SHA is verified and reused, so
an upgrade with the same dependency downloads only the core archive.

Core launchers set `LD_LIBRARY_PATH` to the exact dependency `lib/` and core
`lib/` roots, check every declared NVIDIA SONAME before execution, and clear
`LD_PRELOAD` and `LD_AUDIT`. They do not append the ambient path. This preserves the existing
binary `RUNPATH=$ORIGIN/../lib`; the NVIDIA Driver remains host provided.
Direct execution of `libexec` bypasses the launcher contract.

To see which installed core versions reference a dependency:

```sh
sh tools/distribution_v2_references.sh /path/to/store EXACT_RUNTIME_ID
```

Do not garbage collect a dependency while this command reports a core version.
Automatic garbage collection is outside this prototype.

## v0.9.1 monolithic handoff and rollback

The installer recognizes the two v0.9.1-alpha launchers by their installer
marker and confirms that the existing CLI reports v0.9.1-alpha. It leaves the
monolithic tree in place. It downloads and verifies the exact v2 NVIDIA runtime
artifact instead of assuming that the old embedded libraries have matching
bytes. It then saves byte-identical copies of both old launchers under
`legacy/v0.9.1-alpha/bin`, installs stable v2 wrappers, and atomically switches
`current` to the new core. An error before that switch restores the old
launchers. A subsequent installer run repairs an interrupted launcher pair or
stale dead-process install lock before proceeding. Existing users do not need
to uninstall v0.9.1-alpha.

To switch a migrated installation back to the preserved monolithic release:

```sh
sh tools/rollback_distribution_v2_legacy.sh
```

This switches `current` to `legacy/v0.9.1-alpha`. The old monolithic files and
v2 store remain available. Re-running the pinned v2 installer can activate v2
again without redownloading the same verified NVIDIA runtime.

## Publication trust boundary

The versioned catalog binds an exact core archive SHA and an exact NVIDIA
runtime archive SHA/ID. Online installation requires a catalog SHA supplied
through an independent trusted channel; HTTPS alone does not establish
catalog immutability. The published versioned `install.sh` embeds the exact
catalog URL, SHA, and expected catalog version, and verifies the downloaded
installer script's SHA. The pin is delivered through project-controlled
release tag source, not read from the catalog's own `.sha256` companion.
Future releases must use new versioned tags and pins; published assets under
this tag must not be replaced.

## Qualification boundary

The test suite `python3 tools/test_distribution_v2.py` covers offline install,
reuse, bad/missing archives, wrong identity, interrupted transfer, preflight
failure, legacy migration, and rollback. A separate clean system root on the
same GPU host has qualified the frozen candidate through real HTTPS and a
Wan2.1 Product smoke. The four-model comparison is recorded in the release
qualification report. The independently operated clean host completed RC1
qualification. A short external smoke from the formal
v0.9.2-alpha URL remains a post-publication check. No numerical qualification
status changes because of this split.
