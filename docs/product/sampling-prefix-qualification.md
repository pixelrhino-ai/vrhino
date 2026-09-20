# Bounded sampling qualification

`vrhino sampling-step-check MANIFEST LOCAL-RESOURCES REQUEST.json OPTIONS.json OUTPUT-DIR`
executes real native tokenization and text conditioning, admits the complete package
execution/sampling declarations, then executes a bounded sampling prefix using
the normal architecture factory, native denoisers, Backend and SamplingRuntime.
It does not decode or encode video. It does not grant full-run admission.

Options are a closed declaration:

```json
{
  "schema": "vrhino.sampling-step-check.v1",
  "precision": "fp32",
  "completed_step_limit": 1,
  "device_budget_bytes": 64424509440,
  "host_budget_bytes": 214748364800
}
```

The v1 declaration remains restricted to exactly one transition. Opt-in
`vrhino.sampling-step-check.v2` accepts integer limits 1–3. For bounded original-
schedule transition studies, opt-in `vrhino.sampling-step-check.v3` accepts 1–32
and records per-step admitted IDs, cache residency/hits/misses/evictions and device
memory in `step-resources.json`. These observations never choose a component or
change cache policy. Older declaration limits remain unchanged. All require a strict
prefix of the original declared schedule, never a shortened schedule.

F32 is the explicit qualification mode, not a model-selected fallback. This entry
accepts no binding, guidance, schedule, precision-role or numerical-gate override.
The latent is bounded to 1024 elements. Output directories must not already exist.
Resources and package identity use the same fail-closed checks as product preflight.
Device budget admission and observed peaks do not guarantee total inference memory.

## Shared execution boundary

`SamplingRuntime::run_prefix` admits the entire unchanged SamplingProgram and
ExecutionProgram, initializes RNG normally, and stops after the requested number
of completed transitions. It uses the same loop and numerical recurrence as `run`.
The original schedule length still controls multistep solver order and observer
metadata. No model identity or new solver numerical operation is introduced.

`SamplingResult.completed_steps` and `program_complete` distinguish a prefix from
a complete program. Prefix output is not a resumable solver checkpoint: rerunning
it restarts initialization. Qualification covers only transitions actually executed: a three-step prefix can
exercise solver history, but does not prove a component switch unless that switch
occurs in those original schedule rows. It does not qualify long-chain stability
or full video generation.

Host synthetic tests compare every prefix trace to the corresponding full-run
trace for Euler, Flow multistep, V and Epsilon prediction. Malformed declarations
after the requested prefix must still fail before RNG or denoiser execution.

## Evidence and qualification

The command records tokens, real conditioning, initial noise, branch predictions,
CFG result, solver controls/history and next latent. `status=PASS` reports finite
bounded execution, not reference numerical qualification. Reference comparison is
an independent research step using frozen gates and the same input bytes.

`reference_numerical_qualified=false` and `production_numerically_qualified=false`
remain explicit in raw execution evidence. Full-run package qualification and
Generic BF16 HOLD are not changed by this command or by a bounded F32 comparison.
Python reference utilities are never imported or invoked by the native product.

## Controlled complete sampling and media

`vrhino sampling-video-check MANIFEST LOCAL-RESOURCES REQUEST.json OPTIONS.json OUTPUT-DIR`
uses the same normal Product preparation and existing `SamplingRuntime::run`, then
passes its completed host latent to the native architecture decoder and bundled
media encoder. Each device session is destroyed before the next component starts.
There is no external latent input and no prefix can enter this decode path.

Its separate closed declaration is:

```json
{
  "schema": "vrhino.sampling-video-check.v1",
  "precision": "fp32",
  "max_steps": 40,
  "device_budget_bytes": 64424509440,
  "host_budget_bytes": 214748364800,
  "fps": 8,
  "media_range": [-1, 1]
}
```

`max_steps` is a safety bound (2–64), never a replacement schedule length. The
complete declared schedule must fit this bound. Latent size is limited to 1024
elements and decoded video to 1048576 elements. Media rate/range are explicit
Product output declarations, not model or precision semantics. Output must be a
new directory; the encoder is checked before numerical execution.

Evidence includes the sampling trace, the exact generated `latent.vrt`, decoded
F32 `decoded.vrt`, `qualification.mp4`, component memory statistics and per-step
binding/cache observations. `program_complete` and `decode_executed` describe
actual completion; later failure yields HOLD without discarding earlier evidence.
A command PASS establishes finite execution/media creation for that request only.
Independent reference and media readback qualification are required separately.
General package numerical admission, BF16 HOLD and release status remain unchanged.
Older prefix declarations and their bounds remain valid and unchanged.
