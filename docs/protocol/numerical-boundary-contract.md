# Layered F32 numerical qualification

This research contract classifies observations before executing a qualification
run. It does not modify production PrecisionPolicy, operator arithmetic, the
solver or execution selection. It applies independently of architecture and
parameter identity.

## Frozen scalar tolerance

For finite F32 observations, the strict comparison is elementwise:

`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.

Keep the same absolute and relative constants at all strict boundaries. Report
max/mean absolute error, relative L2, max relative error (denominator floor
1e-8), exceedance count and tensor shape. Do not replace a strict gate with an
average-error check. Integer control values and supplied identical inputs are
checked exactly. Shape/dtype disagreement or nonfinite values fail even in
diagnostic traces.

## Layer 1: operator qualification — strict

Compare an operator using identical operands, parameters and attributes on both
sides. Attention, linear, normalization, activation, modulation and residual
probes use this layer. When operands were captured from a larger computation,
replay both recorded operand sets separately; verify native replay against its
captured result before attributing a discrepancy to propagation.

An end-to-end internal checkpoint may differ because its operands differ.
Retain that comparison alongside the same-input operator result; do not label
the propagated comparison bitwise equal or erase its exceedance count.

## Layer 2: internal diagnostic checkpoints — diagnostic-only numerically

Keep every declared checkpoint and its raw metrics, including a failed legacy
all-checkpoint gate. These observations localize floating-point drift and do
not independently determine the overall qualification result. They are not
claims that each internal operator invocation has been individually qualified.

Investigate new error patterns with the causal replay and propagation methods
in `f32-compositional-qualification.md`. Evidence of a local operator defect,
wrong parameters/layout/dtype or nonfinite values cannot be waived by calling
the checkpoint diagnostic. Passing the observable output does not waive an
already failed operator qualification either.

## Layer 3: architecture observable outputs — strict

Declare outputs according to the interface being tested: block return values,
denoiser prediction branches, guided prediction and sampling transition outputs.
All declared outputs must meet the frozen scalar tolerance. An operator or
internal propagation explanation cannot waive a failing observable output.

The same tensor may be a strict output in an isolated block test and an internal
observation inside a denoiser test. Freeze the tested interface and complete role
map before the run; never move an output to diagnostics after observing failure.

## Evidence and failure behavior

Require operator qualification, observable qualification and all identity /
semantic checks to pass. Require exact agreement on the trace key set; do not
drop failed or missing diagnostics. Record the contract digest and all metrics.
Unknown roles, missing observable outputs or empty observable sets fail closed.

`numerical_observable_compare.py` compares a predeclared observable set while
retaining all other metrics. It does not provide operator qualification; that
evidence is mandatory and separate. The older strict all-checkpoint comparator
is unchanged, and historical results retain their original meaning.

F64 research diagnostics may distinguish rounding from defects; they are not a
production precision change or a silent replacement of the primary F32 oracle.
Qualification of a small finite-input fixture is not a universal error bound,
full-chain qualification or qualification of another execution dtype.
