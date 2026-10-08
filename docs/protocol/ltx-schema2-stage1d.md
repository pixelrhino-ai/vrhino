# LTX Schema2 integration consolidation evidence

Stage 1D initially returned **HOLD for PR qualification**. Stage 1D-R resumes
under CTO authorization to treat a proven unchanged baseline F32 failure as a
known limitation while preserving its FAIL status. The final Product smoke now
passes; normal required CI and CTO review remain the merge boundary. Approved main
is `722ab78df28870833e09ebfa87ae00a9250aade7`. Original Stage 1B/1C source, patches,
test results and manifests are preserved. The production candidate is a clean cumulative patch against that main; no public
package, release, tag or numerical gate changed. The sections below retain the
original Stage 1D qualification result; the resumed result is recorded separately.

## Review and candidate validation

The cumulative implementation has correct Architecture/Product/Shared Sampling
ownership. No model-specific Runtime, Backend, CUDA, PrecisionPolicy, arbitrary
workflow or implicit Schema1 execution path was found. Consolidation removes an
unused generic Product field, preserves managed mapping state across moves on all
platforms, checks Windows backing timestamps, and verifies request resources from
the exact consumed bytes. New test names identify their capability; small offline
contracts, synthetic SentencePiece/shards/masks and existing Wan admission/wiring
are registered in normal public host CTest. New sources are formatted and contract
owners documented. Numerical forward/decoder/sampling expressions are unchanged
from the accepted Stage 1C implementation.

| Final candidate check | Result |
| --- | --- |
| Full Linux CUDA/tokenizer build | PASS, GCC 11.4 / CUDA 12.8 / SM89 |
| Host CTest profile | 58/58 PASS |
| Additional host/Legacy tests | 8/8 PASS; includes real SentencePiece fixture |
| Windows package source tests on Linux | 16/16 PASS; no Windows compiler/GPU claim |
| CUDA regression subset | 9/10 PASS; one historical numerical failure below |
| Real package CPU admission | PASS; 715 denoiser slots, 297 decoder slots, 219 T5 index entries, two verified shards, masks, geometry and 40 typed steps; 13 negative cases |
| FlowEuler declaration negatives | 56 PASS, including unsupported reader capability |
| Resource/tokenizer/mask/owner negatives | 16 PASS with synthetic and external SentencePiece inputs |
| Old program reader | Exact main parser body rejects `programs.v2` as unsupported |
| Frozen Legacy snapshot | 26,432 bytes, SHA256 `88f4784259b45abd044da9315560dedfdf8999bb99b47e7388b034e27ce02c9b`, byte-identical to pristine main |
| Source hygiene / immutable public contract | PASS |
| Required GitHub CI | NOT RUN; no PR created while qualification is HOLD |

Stage 1B's 21 matrix entries, 55 declaration rejection invocations and 8 full
execution rejection invocations remain preserved. Stage 1C's 21 matrix entries,
15 host and 13 real preparation negatives remain preserved. Matrix entries include
comparisons/static checks; they are not counts of independent executable tests.
Stage 1D reruns affected checks, rather than re-generating the unchanged 5.7 GB
full-topology synthetic fixture or mechanically repeating long model runs.

The failing `vrhino-neural-graph-layernorm-f32-offset-cuda-tests` has 40 witnesses,
20 failures, including nonfinite captures and fixed independent F32 gate failures.
A fresh independent build of the exact approved main produces the same transcript,
SHA256 `2e078b287e697a667f85007209c4b0ffac486ddae4ec5e7d9b37575239940861`.
Backend/CUDA/Precision and test source are unchanged. This is the already recorded
[generic F32 finite-offset limitation](../release/v0.9.1-alpha.md), not an observed
new numerical regression. It is still FAIL; no tolerance, expectation or CTest label
was changed. Generic BF16 numerical qualification remains HOLD as well.

## Retained real-model proof and provenance

Stage 1C executed normal Native `pull/run` with full text conditioning, all 40
sampling steps, Native decoding and complete media output on RTX 4090 D. Both
Schema1 and Schema2 processes exited zero. Frozen inputs were prompt
`A red cube rests on a table in soft daylight.`, seed 5703, CFG 3, 704×480,
121 frames at 25 FPS, and negative prompt
`worst quality, inconsistent motion, blurry, jittery, distorted`.

| Identity | SHA256 |
| --- | --- |
| Exact Stage 1C smoke CLI | `09d6b1202eeb5d92398c4abb79a63f0a464aaa3fc524e06e1e825609dc03b9f3` |
| Stage 1C final CLI after missing-owner guard | `647f04e5a8d10bd5756decd2c206e38e7d0291bd570eac9cc67adb1152ae8305` |
| Final Stage 1D candidate CLI | `f66762d3e43b43243905841b21879d9c44410a45b00ab3f4f4c120cc3a60d9ef` |
| Frozen Schema1 VRM | `267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d` |
| Schema2 VRM, identical 1012 parameter payloads | `202689bec8f52a1490b60e8d8308de027dc169ada177f4a837476a7f466c69f4` |
| Frozen 40-step programs | `3e087be341d6a4210c4a06e1d28fe342c7ee34e0d3406432056c3e9a66993ee7` |
| Both complete MP4 outputs | `f47d33546e0c7369f71cc54030235df1d04e57167f854a916d73e3d4dba39534` |

The LTX source revision is `8984fa25007f376c1a299016d0957a37a2f797bb`
from Lightricks/LTX-Video. T5/tokenizer revision is
`b89adadeccd9ead2adcb9fa2825d3fabec48d404` from
PixArt-alpha/PixArt-XL-2-1024-MS. Exact per-artifact sizes, hashes and upstream
paths are in the [frozen source contract](../../native/specs/ltx_schema2_v1/source-contract.json).
Stage 1D real CPU admission re-verifies the resource identities; no large download
or weight mutation occurred.

Full MP4 bytes (42,869 bytes) and all 121 decoded frames are identical. Separate
comparison proved all 40 schedule/timestep/delta bits, Native conditioning, and
the first two steps' noise, predictions, CFG and latents bitwise. It did not prove
all 40 intermediate tensors or raw decoder float bitwise identity. The successful
comparison harness SHA256 is
`7e1c8cbf8b7846c0fefe418fd6ccccd076ee4659dbd52c30ec3ea30ac2a5c8b4`.
The production encoder was the existing VRhino FFmpeg; a separate existing FFmpeg
only observed decoded frames. No Python/PyTorch inference dependency or fallback
was used.

The Stage 1C exact CLI and final CLI differ by a verified missing-conditioning-owner
admission guard. The Stage 1D CLI additionally includes resource-boundary cleanup.
Its real CPU admission is verified, but no full GPU smoke was run with this final
binary: work stopped at the unresolved numerical qualification check. Historical
Stage 1C execution PASS must not be described as a final Stage 1D binary PASS.
The uncommitted CLI's embedded Git HEAD alone cannot identify its source; external
qualification evidence also freezes the full patch and per-file source hashes.

## Admission decision

| Level | Status |
| --- | --- |
| Synthetic second-family capability | PASS retained; affected host checks PASS |
| Final candidate structural Product admission | PASS |
| Real Product execution | Stage 1C PASS retained; final Stage 1D binary NOT RUN |
| Generic F32 finite-offset / generic BF16 numerical qualification | FAIL / HOLD unchanged |
| Production release readiness | NOT GRANTED |
| Draft PR / formal merge qualification | HOLD; no commit, push, PR or CI bypass |

The only proposed next step is CTO adjudication of the known baseline F32 failure's
role in this PR's admission scope, without relabeling or weakening its gate. If
consolidation is authorized to resume, bind one bounded Product smoke to the final
candidate binary, then create the clean topic commit/Draft PR and wait for normal
required CI. Any new regression, ownership ambiguity or Runtime/Backend/CUDA/
Precision architecture compromise still stops that work. No numerical repair or
qualification promotion is part of this candidate.


## Stage 1D-R: final binary verification and review admission

`BASELINE_F32_LAYERNORM=FAIL_UNCHANGED`. Product/tokenizer build switches were
aligned to ON in both candidate and pristine builds, using GCC 11.4, CUDA 12.8.93,
Release, SM89 and identical numerical-target compile/link flags. The same RTX 4090 D
(UUID `GPU-e4303f5d-252c-15c6-2c73-27eabd57a21a`) and NVIDIA driver 580.105.08 ran
both existing offset tests with the same `f32-offset` command/input/gate. All 40
coordinates and error classifications match: 20 FAIL and 20 PASS. Both full
transcripts retain the SHA256 above. No numerical repair or gate change occurred.

The unique test CLI retains SHA256
`f66762d3e43b43243905841b21879d9c44410a45b00ab3f4f4c120cc3a60d9ef`.
It was built from full indexed source tree
`cd3564182d57e3168a8f05501880a93b5c217d31` at the approved main plus dirty patch
SHA256 `131708c0399c690a2c13e16b63883c9e1741b486dcb8db4969082fc5e8417056`.
The native changed-file SHA256 map fingerprint is
`901162e9324f5869e7a96746299f3474a9ac17140348f8912fc7588c71eb375b`.
All native source contents remain identical after verification; only this
qualification document is updated to record resumed outcomes. The external
archive includes a SHA256 manifest of the complete tracked source, CMake cache,
executable and shared-dependency identities. The embedded baseline Git HEAD alone
is not asserted to identify a committed PR revision.

Runtime dependencies: CUDA runtime 12.8, cuBLAS API version 120804, cuDNN 9.8.0,
NVIDIA CUDA driver API 13.0. The production media encoder is the existing VRhino
FFmpeg 4.4.2; independent FFmpeg 7.0.2-static only observes the resulting media.
The package/weight/program identities and frozen inputs are those listed above.
No weights were downloaded or modified.

One final normal Native `pull/run` smoke completed with positive/negative Native
SentencePiece/T5 conditioning, all 40 typed FlowEuler steps, Native decoding,
704×480, 121 frames and 25 FPS. Both processes exited zero; GPU memory returned to
0 MiB. Final MP4 is byte-identical to the frozen Schema1 reference, 42,869 bytes,
with the SHA256 above. All 121 decoded RGB24 frames compare byte for byte; SHA256
of their concatenation is
`278c36111ee794173d54ec47476f885a8573a40370df05614ef449b9c00b2734`.
Duration is 4.84 seconds. This is new evidence from the final Stage 1D binary, not
inherited execution PASS from the Stage 1C binary.

The initial observation harness counted sampling log messages as steps and
expected 40. Native emits one sampling-stage entry plus 40 completion callbacks,
so both historical and final logs correctly contain 41 messages. The observation
failure is preserved, the counting check is corrected against the unchanged event
implementation, and the model was not rerun. Native itself checks completed steps,
callback count and program completion before decoding. No execution test, CI check
or numerical threshold was weakened.

Historical prefix comparisons remain independent Stage 1C evidence. The final
smoke proves MP4/RGB output identity, not all 40 intermediate state tensors or raw
decoder float bitwise identity. Generic F32 offset remains FAIL; Generic BF16
Numerical HOLD remains; no full numerical or release qualification is granted.
Windows real Product qualification remains unavailable.

The original 41-file scope is retained: six Architecture files, thirteen generic
Product files, two shared Sampling files, two Native tokenizer files, eight
converter/spec files, six test files, one CMake file and three documents.
Model-specific Runtime/Backend/CUDA/Precision semantic deltas remain zero.
A separate Release build with CUDA OFF and Native tokenizers ON completed all
52 public host CTests successfully. The earlier CUDA-enabled build's 58 host
checks, additional eight host/Legacy checks and nine passing CUDA regressions
remain applicable because no Native source changed. Source hygiene and immutable
public contract validation pass. Required GitHub CI still runs normally on the PR.

The CTO-authorized next boundary is a Draft PR with normal required CI followed
by architecture review. Nothing here authorizes merging, changing branch
protection, bypassing tests, or publishing a release.
