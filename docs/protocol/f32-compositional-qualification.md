# F32 compositional numerical qualification

This is a research qualification contract, not a Runtime precision policy.
It distinguishes observations of the same program at different boundaries.
It neither modifies production arithmetic nor widens the existing elementwise
gate: `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` for these fixtures.
Nonfinite values, shape/dtype disagreement and semantic/layout disagreement fail.

## Declare observation roles before a new qualification run

1. **Observable boundary:** an output in the interface of the object under test
   (operator, block, denoiser or sampling transition). Apply the frozen gate to
   every element. An internal explanation cannot waive a failing boundary.
2. **Primitive qualification boundary:** hold *all* operands, parameters and
   attributes fixed and compare the implementations of that operator. Apply
   the same frozen gate, including when diagnosing an internal checkpoint.
3. **Internal diagnostic checkpoint:** record both the original end-to-end gate
   result and error metrics. Implementations may reach this point with different
   rounded inputs. An end-to-end difference is not by itself a local operator
   error, and passing a primitive gate does not bound arbitrary compositions.

Roles follow the tested interface and graph dependencies, not observed errors,
model names, binding IDs, a selected FFN or a favorable final result. The same
tensor is an observable output when its operator is independently under test
and an internal diagnostic when that operator is inside a larger block.
Declare the role map and the input/shape/dtype/parameter identity before future
runs. Retain historical reports made under an earlier all-checkpoint gate; do
not rewrite their failures as passes.

## Required investigation of an internal exceedance

An internal exceedance remains unresolved until all of the following hold:

- Replay identical inputs/weights/dtype and reproduce the reported error.
- Identify the first bitwise divergence separately from the first gate failure.
- Check finite values, shape, layout, casts, parameter identity and branch order.
- Hold each implicated operator's inputs fixed to both recorded operand sets.
  Native replay must reproduce the captured native output. Both same-input
  comparisons must pass the unchanged primitive gate. Use an independent,
  higher-accuracy research oracle when cancellation or reference rounding is
  in question; do not silently replace the primary oracle.
- Account for propagation using the operation's mathematical dependence on its
  inputs. Report local rounding separately from propagated differences. Bounds
  must be derived from the operation and existing local contracts, not fitted
  to the observed failed elements. A very loose bound alone is insufficient.
- Every declared observable boundary still passes its unchanged gate, and
  relevant operator/architecture regressions pass.

Missing evidence, unexplained error, failed local qualification or a failed
observable boundary means HOLD. A passing final output alone is insufficient:
it could hide a broken operator behind a small gain or cancellation.

## Linear propagation example

For fixed `W,b`, let `F(x)=x W^T+b` in real arithmetic. If implementations reach
the linear operator with rounded operands `x_N,x_R`, then

```text
y_N - y_R = (x_N - x_R) W^T + e_N - e_R
e_N = y_N - F(x_N)
e_R = y_R - F(x_R)
```

Evaluate this identity with independent higher-accuracy diagnostic arithmetic.
Require each local error to meet the original primitive contract; for example
`|e_N| <= T(F(x_N))` and `|e_R| <= T(F(x_R))`, where `T(z)=2e-5+2e-5|z|`.
An elementwise envelope then follows from the triangle inequality:

```text
|y_N-y_R| <= |x_N-x_R| |W|^T + T(F(x_N)) + T(F(x_R))
```

This is a diagnostic propagation envelope, not a new acceptance tolerance for
the block's output. Check the signed propagation identity as well as the
envelope. Numerical identity closure in an F64 diagnostic is evidence for the
tested finite fixture, not an exact-real proof for all floating-point inputs.
For nonlinear operators, use fixed-input replay and their own justified
sensitivity analysis; do not apply a linear formula across the nonlinearity.

## Scope and unchanged boundaries

F32 storage does not prescribe one reduction tree, fused expression, or exact
sequence of internal rounding. All such differences still require evidence
under the above checks. This contract is not permission to change precision,
ignore cast/layout errors, choose model tolerances or skip failing checkpoints.

The default comparison tool remains strict over every supplied checkpoint.
This document adds an explicit compositional interpretation and evidence
requirements; it does not silently change that tool's exit status. Any future
role-aware runner must enforce the complete role map and fail closed on missing
or unsupported roles and missing causal evidence.

An analysis PASS qualifies this method for the demonstrated fixture. It does
not imply full-denoiser, dual-binding, sampler, BF16 or end-to-end qualification.
