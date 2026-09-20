# BF16 reference qualification

Compare two implementations of the same operation and storage contract.
BF16 versus F32 is a separate sensitivity measurement, not a BF16 reference
acceptance test. Fix operands, semantic roles, operation dtype, accumulation,
output rounding, policy bytes and discrete controls before measurements.

The research oracle uses independent PyTorch math with BF16 heavy operands,
F32 matrix accumulation, TF32 disabled, reduced-precision BF16 reductions
disabled and deterministic algorithms. It is not an unmodified upstream
autocast pipeline. Architecture forward topology alone does not define all
precision boundaries. Record every reference lowering and preserve any failed
oracle attempts when correcting a demonstrated semantic mistake.

Rounding points matter: an F32 affine output can follow a normalized temporary
rounded to BF16; an F32 residual can consume a BF16 projection. Preserve these
existing operation boundaries instead of introducing an all-F32 fallback.
F64 calculations may diagnose distance to rounding midpoints but must not
replace the frozen F32-accumulator oracle or change the pass criterion.

Before testing, serialize a qualification contract and digest. The current
experiment selects `abs(actual-reference) <= 0.005 + 0.005*abs(reference)` for
BF16 operation results. This is borrowed from an existing BF16 attention
implementation comparison; its extension to composed operators and observable
outputs is a prospective qualification gate, not an already proven universal
BF16 error bound. Same-input F32 semantic operations retain the generic
`2e-5 + 2e-5*abs(reference)` gate. Shape, dtype and finiteness are strict for
all checkpoints; discrete controls and within-chain provenance are exact.

Keep internal diagnostics, including their exceedance counts, separate from
strict operator and observable outputs. Do not relabel a failed strict output
as diagnostic after measuring it. Passing component operations does not imply
that their composition passes the same numerical bound. Replaying later
operations from an earlier native intermediate localizes propagation; it is
not an independent end-to-end reference and cannot qualify that composition.

If strict operators fail, retain HOLD and explicitly mark downstream denoiser
and chain reference comparisons as blocked/not run. Earlier native finite,
repeatability or resource tests cannot substitute for reference agreement.
Adjacent BF16 values may straddle a rounding midpoint after different legal
F32 reduction orders; documenting that fact does not waive an exceedance.
Resolve an accumulation defect or review a generic, independently validated
rounding/propagation contract before restarting qualification. Never choose a
threshold from one model's observed maximum error.

Reference code is test infrastructure only. Qualification must not change the
production precision policy, operation dispatch, resource ownership, or add
an architecture-specific numerical rule. Re-run resource transaction, binding
identity and precision regressions, and record fresh-session repeatability and
teardown observations separately from numerical acceptance.
