# Generic BF16 rounding and composition qualification v2

This is a qualification contract, not a change to production PrecisionPolicy.
Freeze a `generic.bf16.boundaries.v2` declaration, scope, checkpoint categories,
classification reasons, input bytes and reference implementation before running
a new qualification. Archive its digest with results. Never relabel an old
failed strict output or retrospectively change a previous result.

| Category | Requirement |
|---|---|
| Operator output | Identical operands and precision roles; strict frozen operator gate; independently localize arithmetic, accumulation and rounding on failure. |
| Observable output | Strict frozen output gate for each declared block, denoiser branch, CFG result and sampling transition. Passing internal diagnostics cannot substitute for this comparison. |
| Internal diagnostic | Retain all differences and exceedance counts. Propagated numerical drift alone does not fail an architecture scope. Shape, dtype and finiteness remain mandatory. |
| Exact control | Identical inputs, step/instance IDs, schedule, branch order, history provenance and RNG continuity. |

Category is defined by the test's boundary, not by a tensor's spelling. A
two-linear-layer composition can be the strict result of an operator test and
an internal diagnostic in a larger architecture test. Its earlier strict
failure stays recorded. An operator scope must have an operator output; an
architecture scope must have an observable output. Unknown roles, missing
checkpoints, arbitrary tolerances and diagnostic-only claims fail closed.

This revision changes no numeric threshold: BF16 path comparisons retain
`0.005 + 0.005*abs(reference)`; same-input F32 semantic boundaries retain
`2e-5 + 2e-5*abs(reference)`. These are experimental acceptance gates, not
universal error bounds implied by the dtype. Production rounding/role behavior
must be verified separately. A failed gate cannot be waived by this document.

BF16 nearest rounding has unit roundoff `2^-8` on normal values. Independent
F32 reductions can lie on opposite sides of a BF16 midpoint, even when both
have small pre-round errors. Their stored results can then differ by one BF16
spacing. An allclose gate below that spacing does not, by itself, identify a
semantic or dtype bug. Conversely, a one-spacing difference is not automatic
permission to pass: bad arithmetic can also produce adjacent results.

Observe the unchanged numerical path and verify observer output against the
production BF16 result bitwise. Verify final rounding independently. Trace QK,
scale, exponential/rescaling, denominator, PV accumulator and pre-round output.
Do not replace production computation with an F32-output policy or a different
kernel variant to obtain a passing result. Reference algebra can legitimately
use a different reduction order; it must retain the same operands and roles.

For localization, a sequential F32 recurrence with at most n rounded operations
has the standard conservative factor `gamma_n = n*u/(1-n*u)`, `u=2^-24`,
when n*u < 1 and no overflow/underflow invalidates the analysis. A weighted
sum can be checked against `gamma_n * sum(abs(weight*value))`. Apply denominator
and numerator bounds separately. If weights are captured exponential/rescaling
values, this is a CONDITIONAL accumulation bound: it does not qualify exp,
score computation or the complete attention function. F64 evaluation is only
a diagnostic oracle for the stated subproblem, not a production fallback.

For composition, replay downstream operations from the same captured upstream
intermediate to separate a local implementation mismatch from propagated
input drift. This replay cannot stand in for independent end-to-end reference.
Keep cancellation, conditioning and quantization effects visible. A locally
passing linear output does not imply the same gate bounds its full composition.

Classify rounding/reduction/propagation observations as a generic contract gap
unless evidence establishes an arithmetic or semantic defect. Any proposed
new rounding-aware operator acceptance needs its own prospective version,
independent error analysis, adversarial tests and cross-family regressions.
Do not derive a tolerance from observed failures. Numerical equivalence remains
unqualified until the strict observable outputs have actually been tested.
