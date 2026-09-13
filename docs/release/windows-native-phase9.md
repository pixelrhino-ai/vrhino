# Windows native Phase 9 closure

Phase 9 is **PASS** for the frozen run below. The reviewed portability changes
are ready for commit after resolution of the component-archive Unicode blocker.
This closure does not claim Windows release publication.

## Frozen real-model result

The authoritative run used base commit
`76f9746c28aa00eef6c7ea14d7267b22b817953e` on
`feature/windows-native-v1`, plus the reviewed Windows portability changes.
It ran once on native Windows with an RTX 3090 (24 GiB, SM 8.6), NVIDIA
driver 610.47, CUDA 12.8 and cuDNN 9.8.

| Input | Frozen value |
|---|---|
| Model | `vrhino/wan2.1-t2v-1.3b:1.0.1` |
| Prompt | `a rhinoceros walking through a snowy forest` |
| Preset / seed | `default` / `5701` |
| Precision | BF16 |
| Video | 832 x 480, 81 frames, 16 FPS, no audio |
| Sampling | 50 steps, CFG 5.0, flow shift 5.0 |

Run start: `2026-09-13T04:38:31.7855800Z`.
Run end: `2026-09-13T04:48:17.0879221Z`.
Exit code: **0**. Final stage: **Done**.
All stages were present in order, including Sampling 1/50 through 50/50.

| Measurement | Result |
|---|---:|
| Sampling | 512.273 s |
| Decode | 13.1966 s |
| Encode | 10.3717 s |
| Product total | 583.905 s |
| Peak device allocation reported by product | 13.09 GiB |
| MP4 size | 1,684,771 bytes |
| Full decoded RGB24 size | 97,044,480 bytes |

| Frozen artifact | SHA256 |
|---|---|
| `vrhino.exe` (10,876,416 bytes) | `93d56c16937f52c0a4ce8c32fbd8f0b94cf65c19a560c8a4332456e7662e92cc` |
| `vrhino-ffmpeg.exe` (316,416 bytes) | `e4d614f3555fe4b03820be6476966e420989477986995a00835fbd372e6dd632` |
| `wan21-windows-smoke.mp4` | `1e76b15316990620010b46710e673553082d119b32ef81726e63df0db53cecc5` |
| Canonical and CAS preset (2,685 bytes each) | `867fcde4a9b41d3ced026e34340cc7b0aa83b68d800ddbe82a70415eb93e48e3` |
| Authoritative smoke report | `a8c88a183b01131ae08af97d408996ac040930bf053a67c1a5ee20bb6421c334` |

All eight model artifacts matched their manifest sizes and hashes before and
after inference. Both executable hashes remained unchanged. Bundled FFmpeg
probed and decoded all 81 frames, with no audio, decoder error or truncation,
using a Windows-only PATH with no system FFmpeg.

Doctor exited 0 before and after the run: package, runtime, components and
media readiness passed. Its overall **WARNING** is expected because
`minimum_vram_bytes` remains null/undeclared. Neither this declaration nor the
frozen runtime memory budget was changed.

No stale product/helper process or output staging residue remained. The
existing CUDA cleanup test passed with zero outstanding resources, stable
handles and unchanged free device memory. Its counters are separate-test
measurements, not internal counters from the product inference.

The historical `CACHE_ERROR: Invalid JSON number` was not reproduced. Its root
cause is unproven. No cache-corruption, parser or old-binary explanation is
claimed. Current JSON parsing and cache integrity pass.

## Review scope and invariants

The retained source changes cover generic read-only Windows safetensors
mapping/ownership, Windows doctor inspection, and exact MSVC frame-count
arithmetic. Build changes cover Windows compiler options, a locked/offline
Cargo launcher, CRT selection, verified Git symlink placeholders and a
build-tree include alias, plus the platform executable suffix for FFmpeg.

There is no Windows-specific Wan execution implementation, model-specific
backend, precision-policy fork, CUDA kernel change, or preset/admission change.
The non-Windows/non-MSVC lines of the four initially changed production files
match the base after selecting platform branches and normalizing whitespace.
The subsequent archive fix is confined to the existing Windows implementation;
the Linux/POSIX archive path remains unchanged. The Unix
Cargo launcher, pinned inventories, source hashes, warning policy and genuine
symlink requirement are retained. This is structural comparison, not a new
Linux execution qualification.

The initial closure added this note and the ignore rule for local smoke evidence.
The subsequent archive encoding fix was validated in an isolated native test
tree. The frozen CUDA product and inference evidence remain unchanged; the
583-second Wan inference was not rerun.

## Closure regressions and resolved archive blocker

The isolated native Windows Release regression build passed allocation,
mapping, stable verification, cache publication, process/media and product CLI
suites. The cache loopback download/resume/cancellation/integrity and real
cross-volume publication checks also passed. The CLI suite made 100 invocations
and checked relocation, Unicode paths and operation without developer PATH.

Focused JSON, safetensors mapping/ownership/rejection, wide frame-count
arithmetic, tokenizer inventory verification, nine tokenizer source cases,
actual Wan tokenizer probing, CUDA resource cleanup, product help/version/
doctor, and bundled media encode/decode checks passed.

The component archive regression initially failed on its Unicode-member fixture:

```text
fixture valid PASS
FAIL: COMPONENT_INVALID: component archive: missing UTF-8 member name
native exit code: 0x1
```

At reproduction, the extractor and fixture sources were unchanged from base.
The isolated test uses the same `archive.dll` as the frozen product, SHA256
`25cbf03a74d3656becee9b4889524a1c2429a66ebf655b8f229c3fc8ed132592`,
from the installed libarchive 3.6.2 SDK. Diagnosis confirmed valid PAX UTF-8 bytes
for `vrhino-media/目录/媒体 文件.txt`. Under `LC_CTYPE=C`, narrow pathname
conversion returned null while the wide API returned the correct UTF-16 name.

The bounded fix uses wide archive entry/link APIs and strict UTF-16 to UTF-8
conversion before the unchanged validators. All **58 archive fixtures PASS**,
including Unicode hardlinks, supplementary-plane names and existing security
rejections. Staging, validation, publication and cleanup checks pass; archive
handles remain **93 -> 93**. The product CLI regression also passes all **100
invocations**. No global locale change or dependency replacement was introduced.
Linux/POSIX archive code remains structurally unchanged; Linux execution was
not rerun. The archive blocker is resolved and **commit readiness is GO**.
This does not change the successful cached Wan inference or bundled encoder result.

## Evidence retention

Private evidence remains in the ignored repository-local directory
`windows-wan21-smoke/evidence-20260913-123605/`, including
`WINDOWS_WAN21_SMOKE_REPORT.md`, full logs, artifact hashes, telemetry and
decode results. The video remains at
`windows-wan21-smoke/wan21-windows-smoke.mp4`.
Closure inventory, commands and regression logs remain under the ignored
`build-windows-phase9-close/` directory. No local evidence was deleted.

Binaries, DLLs, model/cache data, private logs, videos and local dependency
trees are excluded from the proposed source commit. The existing archive/cache
test fixture generators use Python only as a test tool; the product has no
Python/PyTorch/Diffusers/Transformers or WSL runtime dependency.

Architecture, runtime, backend, CUDA, kernel, precision and Wan
preset/admission semantic deltas are all **0**.
