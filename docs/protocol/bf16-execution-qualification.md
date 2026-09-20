# First BF16 execution qualification

This scope establishes execution and resource correctness under an existing
operation/semantic-role PrecisionPolicy. It does not establish BF16 numerical
equivalence to F32 or an end-to-end quality bound.

Freeze the policy bytes, input bytes, schedule, selection and guidance before
execution. Do not substitute unqualified defaults for an existing policy.
The policy may prescribe F32 statistics, accumulation, residual storage and
small temporary transforms alongside BF16 heavy operands and sampling state.
Validate actual trace dtypes against those roles. A F32 role already prescribed
by the shared policy is not an exception keyed to an architecture.

Strict checks include admitted shapes/dtypes, exact discrete inputs, finiteness
of all recorded floating values, correct initial-state quantization, exact
within-trajectory latent/history links, solver order/warmup/counts, RNG
continuity, and independent per-step guidance. Compare isolated instances
against interleaved execution, including returning to a previously selected
instance. Require identical outputs on repeat with identical state and inputs;
retain separate parameter backing identities. Observe process RSS, mapped-file
RSS, device usage, cache residency and transfers, including teardown baseline.

F32-to-BF16 comparisons report max/mean absolute error, relative L2, relative
error near zero, output magnitude and per-step latent RMS. Keep every trace;
nonfinite diagnostics are failures. These cross-precision metrics do not use
the F32-to-F32 gate as an acceptance bound. The existing BF16 attention
implementation-comparison gate (`5e-3 + 5e-3*abs(reference)`) remains a strict
gate only for its original operator comparison. Its exceedance counts may also
be shown as a clearly labeled diagnostic marker on larger computations, but
must not be presented as passed end-to-end equivalence or silently omitted.
No generic end-to-end BF16-vs-F32 error bound is currently established here.

Classify failures before changing production: a research caller must honor the
same policy-prescribed temporary/input boundaries as the production caller.
Keep failed harness attempts and their cleanup observations. Never resolve a
numerical or dtype failure by a model branch, a special precision override, or
a tolerance chosen after observing the error.

Sampling measurements on one device and a small latent are not a total-device
memory guarantee. Cache budget limits resident cache, not all allocations;
source upload counters can count F32 transfer bytes while resident parameters
use BF16. Host RSS can include large mmap-backed file pages without an eager
anonymous copy. Report these distinctions explicitly. Returning device usage
to baseline is an observation, not proof of every resource handle's lifecycle.
