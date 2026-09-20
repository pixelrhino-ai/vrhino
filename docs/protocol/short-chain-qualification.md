# Continuous short-chain qualification

This research procedure supplements `numerical-boundary-contract.md`; it does
not change precision policy, selection semantics, or solver arithmetic.

Freeze 3–8 transitions before execution: initial-state identity, seed, model
timesteps, sigmas, component instance IDs, guidance, and comparison roles. Use
one admitted ExecutionProgram and SamplingProgram per chain. Native execution
must call `SamplingRuntime::run_with_initial_state` once. The reference must
retain one scheduler object and its evolving state across all evaluations.
An isolated denoiser worker may release its weights between evaluations; it
must neither own nor reset the reference scheduler.

## Strict boundaries and retained diagnostics

Use the existing F32 mixed elementwise gate, `2e-5 + 2e-5*abs(reference)`, for
each denoiser prediction branch, guided prediction, next latent, and final
latent. Report the reference and native trajectories using their own previous
outputs. Never substitute reference latents into the native evolving chain.
Initial supplied input bytes and all integer controls must agree exactly.

Keep every internal architecture checkpoint, converted prediction, history
entry, and corrected sample with its raw numerical metrics. Those internal
values have the diagnostic role defined by the layered contract. Nonfinite
values, shape/dtype disagreement, unknown/missing trace keys, failed operator
qualification, and observable failures cannot be waived as diagnostic drift.
The denoiser-only reference and sampling trace use different names for branch
returns. Validate any redundant reference aliases exactly before normalizing
these names; retain the original raw evidence and record the normalization.

## Structural continuity is exact

For each trajectory independently, check bitwise that the next iteration
receives the preceding next-latent tensor. Check each history entry against
the converted prediction that produced it, and check before-history against
the preceding after-history. Validate actual history lengths, predictor order,
previous corrector order, warmup counter, and primitive call counts. Exercise
both directions of component selection and a return to a previously used
instance. Guidance is an independent per-step declaration; use a different
scale on return to the same instance to expose accidental binding-derived CFG.

`SamplingRuntime::set_solver_trace_enabled(true)` enables observation of
actual state. It defaults off and does not add numerical operations. The
trace's `state_id` is an invocation-local opaque label, not a global identity
and not standalone proof of continuity. History provenance, one run, and
operation counts supply that proof. Observation adds host snapshots and
synchronization overhead and is unsuitable for performance measurement.

For externally supplied input, validate zero native RNG draws and unchanged
seed/offset throughout; compare unchanged reference CPU and device generator
states. This does not qualify cross-implementation random-number generation.
A separate generic initialized-state regression checks one initialization
and no draws on component transitions.

## Limits

A controlled short chain may stop at a nonzero sigma. It does not qualify a
complete production schedule, terminal-sigma arithmetic, another dtype, large
inputs, video quality, or peak full-inference resource requirements. Opaque
selection sequences used for stress qualification are experiment declarations,
not replacements for product execution policy.
