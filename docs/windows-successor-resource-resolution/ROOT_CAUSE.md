# Successor execution-profile resource resolution

## Failure and cause

The rc7 Windows clean-host MuseTalk installation acquired and verified all
12 source payloads, then failed during static resource finalization with
`CHECKSUM_MISMATCH: local CAS source size/SHA256 mismatch: execution-profile`.
The selected manifest correctly identified `vrhino/musetalk-v1.5:1.0.1`.
The converter nevertheless supplied the family-root `execution.json` (1.0.0)
to `LocalModelCache::admit_local_blob` with the successor artifact declaration.
The size and SHA256 check correctly rejected those different bytes.

`converter_musetalk_package.cpp` and `converter_latentsync_package.cpp` both
kept `package_spec` at the family root while allowing `manifest_path` to select
a successor manifest. Both constructed the execution-profile source as
`package_spec / "execution.json"`. LatentSync has the same proven source-level
mismatch; its rc7 clean-host installation was not run and remains HOLD.
This is shared converter logic, not a Windows-specific decoding or CAS bug.

The root and successor profiles differ in the existing `output.required` value.
Neither profile, declaration, model identity, nor admission rule is changed.

| Package | Version | Bytes | Execution-profile SHA256 |
| --- | --- | ---: | --- |
| MuseTalk v1.5 | 1.0.0 | 347 | `c770d7ecbfe3c6cf21617176ebdf054a3c021c2bfc787f024ef8d919a5ca3cde` |
| MuseTalk v1.5 | 1.0.1 | 348 | `a5e1592fec66c5fb2a69b0e965e2dde8a7a4d45319535123712ca5c7b04d4a68` |
| LatentSync 1.6 | 1.0.0 | 486 | `aedba58531ffbba784f8d9ac7bc38666dae756c6f07e62616d49dbb6acbc7332` |
| LatentSync 1.6 | 1.0.1 | 487 | `f867e628bf4d2d588e265f480c30d0c60e9727d2e341a4300cc7c3081ce85b81` |

## Complete public successor audit

All five public successor manifests and their converter dispatch paths were
examined against source baseline `9936af0545ce4287abeb7679311551af87e6b744`.
Artifact comparisons include both size and SHA256, not just filenames.

| Public model | Root -> successor | Artifact declarations | Actual mismatches |
| --- | --- | ---: | --- |
| `vrhino/ltx-video-v0.9.1` | 1.1.0 -> 1.1.1 | 9 | None |
| `vrhino/wan2.1-t2v-1.3b` | 1.0.0 -> 1.0.1 | 8 | None |
| `vrhino/mochi-1-preview` | 1.0.0 -> 1.0.1 | 11 | None |
| `vrhino/musetalk-v1.5` | 1.0.0 -> 1.0.1 | 13 | Execution profile only |
| `vrhino/latentsync-1.6` | 1.0.0 -> 1.0.1 | 17 | Execution profile only |

Static resources retain the following scopes. Neural component conversion
and runtime artifacts are outside this path-selection change.

| Converter | Manifest/version scoped | Package-family scoped | Source-tree scoped |
| --- | --- | --- | --- |
| MuseTalk | `execution-profile`, sibling of selected manifest | `workflow-config` under `musetalk_v15_workflow_v2`; notices and OpenRAIL/Apache/MIT licenses under `public_musetalk_v15` | `whisper-preprocessor` |
| LatentSync | `execution-profile`, sibling of selected manifest | `workflow-config` under `latentsync_16_workflow`; BF16 precision policy, notices and OpenRAIL++/Apache/MIT licenses under `public_latentsync_16` | `whisper-preprocessor`, `mel-filters`, `unet-config`, `scheduler-config`, `fixed-mask` |
| Wan | No distinct successor static payload | UMT5 index, conditioning graph, precision policy, default profile under `wan2_1_t2v_1_3b` | Tokenizer and license |
| LTX | No distinct successor static payload | Conditioning graph, precision policy, default profile under `ltx_v0_9_1` | T5 index/shards, tokenizer and license |
| Mochi | No distinct successor static payload | Conditioning graph, precision policy, default profile under `mochi_1_preview` | T5 index/shards, tokenizer and license |

Wan, LTX and Mochi use family-root static paths, but all those declarations
are identical across the audited versions and match the existing family
resources. No speculative converter refactoring is needed there. Workflows,
policies, notices and licenses for MuseTalk/LatentSync also retain identical
declarations across the two public versions.

## Bounded fix and validation contract

The two affected converters use a shared internal helper to resolve
`execution.json` beside the actual selected manifest. It contains no model
name or version switch. Root manifests still select root profiles; successor
manifests select their sibling profiles. The manifest remains authoritative
for model identity and artifact size/hash.

The helper canonicalizes paths and requires both the selected manifest and
resolved profile to remain within the package-family root, including symlink
resolution. Missing/non-file profiles fail closed. The existing CAS admission
and manifest publication checks remain unchanged. Family-level and source-tree
resources keep their existing locations. The same helper runs on Windows and
POSIX; no platform-specific execution path is added.

The existing successor orchestration test now exercises the real converters
and finalizers using tiny private neural-component fixtures. Execution profile
bytes and their expected size/SHA256 are frozen constants. Coverage includes
both public versions of both models, wrong-root/wrong-hash/malformed/missing
profiles, authoritative manifest identity, a non-hardcoded nested manifest
directory, and containment rejection. Symlink escape rejection is also exercised
where the host permits creating symlinks. Fixture-only neural and unrelated
static artifacts do not establish real model installation evidence; rc8 must
separately pass fresh conversion/install/doctor for all four requested models.

Native Windows MSVC 19.38.33145 validation reproduces the checksum mismatch
for both 1.0.1 packages with the baseline converters, while both 1.0.0 cases
pass. With the fix, all 18 ordinary cases and the five-model dispatch test pass.
Two symlink cases are explicitly skipped on an unprivileged Windows host;
the same tests exercise those cases on hosts permitting symlink creation.
The test target now selects `/W4 /WX /utf-8` under MSVC instead of passing
unsupported GCC warning flags. Its existing POSIX flags remain unchanged.

Architecture, Runtime, Backend, CUDA, kernel, precision, model-spec and preset
semantic deltas are zero. The intended converter resource-resolution delta is
one. rc7 remains permanently frozen HOLD evidence; a new rc8 is required and
must pass local distribution and installation gates before a separate clean-host
qualification. This fix does not claim that rc7 or an unqualified rc8 is ready
for release.
