# Generic BF16 operator acceptance v3

Schema: `generic.bf16.operator.acceptance.v3`. This is a research acceptance
contract. It changes no production dtype, kernel selection or PrecisionPolicy
role. Earlier v1/v2 runs and failures remain historical evidence.

Freeze scope, checkpoints, classification reasons, operands, policy bytes,
reference implementation and these gates before a qualification run. Record
their hashes. No per-run tolerance override or reclassification after observing
a failure is allowed. A demonstrated reference semantic error requires a new,
identified reference run; preserve the failed run and its localization evidence.

## Strict gates

For identical-input BF16-output operators, let `a` be native and `r` reference.
Define `R(x)` as half the larger distance to the neighboring BF16 values.
At zero/subnormal values this is `2^-134`; at a finite endpoint use the finite
neighbor spacing. Evaluate comparisons and statistics in F64. Require **both**:

```
abs(a[i] - r[i]) <= 0.005 + R(a[i]) + R(r[i])  for every element
RMS(a - r)      <= 0.005 + 2^-8 * RMS(r)
```

The absolute floor is retained from the earlier experimental gate. The added
pointwise allowance describes two quantization cells, rather than choosing a
constant from a failing tensor's maximum error. BF16 normal unit roundoff is
`2^-8`. Different valid F32 reductions can straddle a BF16 midpoint. Adjacent
stored results are not sufficient to pass: the RMS guard rejects dense,
systematic one-spacing shifts. Both gates are empirical acceptance criteria,
not a proof of correct reduction or a universal error bound. Composition can
still fail, including near zero or under cancellation; investigate rather than
increase the gate. Underflow, overflow, nonfinite values and dtype/shape changes
cannot be excused as quantization drift.

Same-input F32 semantic operator outputs retain the all-element gate:
`abs(a-r) <= 2e-5 + 2e-5*abs(r)`.

Architecture observable outputs on the BF16 path retain the **unchanged**
all-element gate: `abs(a-r) <= 0.005 + 0.005*abs(r)`, including when their
semantic storage is F32. The quantization-aware operator gate does not apply
to these observable outputs. Passing operators does not imply passing their
composition or the architecture output.

Shape, actual and declared dtype, complete checkpoint sets, and finiteness are
mandatory in every category. Softmax additionally requires nonnegative values
at most one and each native row sum within `2^-8 + 2e-5` of one. Exact controls
use exact equality, including integer values beyond F64's exact integer range.

## Boundary classification

| Boundary | Strict qualification | Diagnostic use inside a larger computation |
|---|---|---|
| Linear | Same-input operator output; BF16 operator gate or F32 semantic gate according to declared dtype | Accumulator and pre-round bias sum; retain all metrics |
| Attention | Same Q/K/V, scale, layout, mask semantics and roles; complete operator output strict | QK, score, exponentials, denominator, PV accumulator |
| Softmax | Standalone output strict plus probability/row-mass invariants | Attention internal softmax statistics |
| LayerNorm / RMS norm | Same-input output strict; epsilon, affine order and temporary rounding explicit | Mean, variance, inverse norm and normalized temporary |
| FFN | Standalone two-linear/activation composition output strict; never replaced by local replay alone | Linear1, activation and linear2 traces inside an architecture scope |
| Residual | Same-input F32 semantic operation strict under its F32 gate | Propagated residual checkpoint may be diagnostic if declared before the run |
| Block output | Architecture observable, unchanged strict output gate | Internal branch traces do not replace it |
| Denoiser output | Every declared conditional/unconditional branch and final result strict | Selected block/internal traces remain visible |
| CFG output / sampling latent | Each CFG result and next latent strict, unchanged observable gate | Intermediate arithmetic recorded for localization |
| IDs, schedules, branch order, history provenance, RNG state | Exact control equality | No numeric diagnostic waiver |

A tensor's role follows the scope, not its name. A strict standalone FFN result
cannot be relabeled as diagnostic because it failed. Diagnostics retain max/
mean absolute error, pointwise relative error, relative L2, distribution and
old-gate exceedances. Only their numeric exceedance is non-gating; invalid
metadata or nonfinite values still fail.

`native/tests/bf16_acceptance_contract.py` validates the declaration and complete
tensor sets. An operator scope must contain an `operator_output`; an architecture
scope must contain an `observable_output`. Empty or diagnostic-only scopes,
unknown schema/roles/operators, missing/extra checkpoints, per-run tolerances
and precision mismatches fail closed. The declared outputs must cover the
requested experiment; a passing scope is not permission to omit required
branches or steps from the experiment's manifest.

## Independent reference precision contract

Reference math represents the existing operation/semantic roles, not arbitrary
library autocast defaults. Round heavy operands to BF16 first. Accumulate in
F32 with TF32 and reduced-precision BF16 reductions disabled. Linear applies
the BF16-rounded bias to the F32 accumulator before one BF16 output rounding.
Representing already-rounded BF16 operands in F32 in the research oracle does
not restore discarded operand bits or alter the native BF16 compute path.

Opaque BF16 library Linear calls are not assumed to implement that epilogue for
every shape. Verify it independently: `x=[1,1]`, `w=[1,1/256]`, `bias=1/256`
must yield BF16 `1+1/128`; rounding the matrix product before adding bias yields
`1`. This semantic distinction is tested, not granted a larger tolerance.

Normalization uses F32 statistics with the existing affine and intermediate
rounding order. Attention consumes BF16 Q/K/V and uses F32 statistics and PV
accumulation before its output cast. Residual, timestep/modulation and RoPE
roles remain those of the selected policy and semantic contract. Sampling
state is **not assumed to be F32**: preserve its declared role (BF16 in the
qualified policy). Reference code is never a production dependency or fallback.

## Retry acceptance and scope

1. Record frozen inputs, independent reference revision, policy digest, dtype/
   device, canonical layout and strict/diagnostic declaration before execution.
2. Pass same-input strict operators across the declared corpus, including the
   known failing fixtures; local downstream replay is localization only.
3. Compare each block observable and both component denoiser outputs using the
   unchanged architecture output gate. Require independent full reference
   propagation; native intermediates cannot seed a passing reference result.
4. Compare each branch, CFG output, solver transition and next latent in the
   declared short chain. Require exact input replay, selected instance IDs,
   timestep/sigma/guidance/branch order, solver history provenance and RNG
   continuity. Solver numeric state uses the predeclared observable gate.
5. Retain every diagnostic, finite check, memory/identity/cleanup observation
   and failed attempt. Any strict failure keeps the experiment on HOLD.

The operator contract was tested on multiple hidden/head geometries, operand
scales and reduction lengths, with existing family and generic regressions.
Qualification covers the recorded device/library versions and test inputs.
It is not a certificate for arbitrary shapes, every architecture, full-chain
numerics, or a different precision policy. Those observables require their
own measurements before acceptance.
