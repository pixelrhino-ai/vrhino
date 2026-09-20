# Resource-bounded Product evaluation

`vrhino evaluate MANIFEST LOCAL-RESOURCES REQUEST.json OPTIONS.json OUTPUT-DIR`
is an explicit, unqualified evaluation entry. It uses native Product admission,
tokenization, conditioning, execution/sampling declarations, the current default
Backend, and (in video mode) decoding and media output. It does not promote a
package, override ordinary `run` admission, or establish numerical qualification.
No reference implementation or research numerical route is used by this entry.

Version 1 requires Linux, a CUDA/tokenizer-enabled build, NVML monitoring, and an
already declared BF16 PrecisionPolicy artifact in the local resource map. The
artifact uses existing policy validation; this command cannot change policy
roles, numerical gates, bindings, guidance, or the timestep schedule.

## Closed options declaration

```json
{
  "schema": "vrhino.product-evaluation.v1",
  "precision": "bf16",
  "precision_policy_artifact": "qualification-precision",
  "mode": "prefix",
  "max_steps": 1,
  "device_budget_bytes": 81604378624,
  "host_budget_bytes": 214748364800,
  "max_output_bytes": 1073741824,
  "max_latent_elements": 2096640,
  "max_video_elements": 97044480,
  "timeout_seconds": 1800,
  "fps": 16,
  "media_range": [-1, 1]
}
```

All shown fields are required; unknown fields fail closed. One optional field,
`weight_cache_budget_bytes`, sets a positive weight-cache cap no greater than
`device_budget_bytes` minus the existing 2 GiB workspace and 1 GiB safety reserves.
Omitting it preserves the legacy cache capacity. It does not change the device
observation ceiling, temporary pool envelope, workspace reserve, execution dtype,
or numerical dispatch. This is a generic cache limit, not a binding/model rule.
Request geometry is admitted
against explicit element limits before GPU execution. `prefix` executes exactly
`max_steps` of the original declared schedule, which must have more steps; it
does not generate a new short schedule or decode a video. `video` executes the
entire schedule, whose length must not exceed `max_steps`, then decodes and
encodes it. Budget limits are study controls, not numerical acceptance gates.
Values in this example are not a promise that a configuration fits or completes.

## Supervision and evidence

A separate process supervises the native worker, so cancellation or a deadline
can terminate a stalled GPU call. Timeout includes source identity verification,
preparation, execution, and media encoding. The supervisor polls every 100 ms:

- process-group RSS, including descendants (shared pages can be counted twice);
- the sum of NVML used bytes on **all physical GPUs**, including unrelated work;
- aggregate bytes under the new output directory.

Use an otherwise idle device/host for interpretable measurements. This is sampled
monitoring, **not a hard total GPU/host allocation guarantee**: peaks between
samples and allocation overshoot are possible. A per-file size limit supplements
the aggregate output check. Missing monitoring fails closed. Existing output
directories are rejected. Worker errors, cancellation, timeout and resource
limits stop execution and retain partial evidence; descendant processes are
terminated. This supervisor is intended for the single-threaded CLI entry.

`supervision.json` is authoritative for completion. `worker.json` may describe
the last completed phase when a worker was interrupted; `worker-error.json`
records caught worker exceptions. `steps.json` records resolved instance/binding
IDs, step progress and resource statistics. Completed phases emit token,
conditioning, initial/final latent and decoded-output bundles. Internal tensor
and solver snapshots are disabled to bound diagnostic overhead; arithmetic,
solver state and execution selection are unchanged. Final phase outputs undergo
existing finite checks; this is not instrumentation of every kernel intermediate.

`memory-timeline.jsonl` records current device usage and process-group RSS with
monotonic elapsed seconds. Supervision includes peaks, byte-seconds, and
time-weighted means computed by trapezoidal integration over the first-to-last
recorded samples. These whole-run means include loading and preparation; they
must not be compared with sampling-only means. The timeline itself counts toward
the output quota. Sampled averages do not eliminate peak capacity requirements.

Success is named `EXECUTED_UNQUALIFIED`, with production/reference numerical
qualification still false. Prefix success does not imply full-chain success,
video quality, speed parity, a general memory guarantee, or release readiness.
Report preparation, conditioning, sampling, decoding, and media times separately
before comparing against another implementation.
