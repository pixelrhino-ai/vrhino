# F32 attention accumulation qualification

The CUDA F32 default implements `softmax(scale * QK^T + bias) V` on contiguous
BSHD tensors. Boolean masking and causal masking keep their existing meaning.
This qualification does not change the operator interface or PrecisionPolicy.

## Arithmetic contract

- Q/K/V, scores, output and accumulation state use F32. QK uses pedantic cuBLAS
  F32, without TF32 or implicit conversion to BF16.
- Ordered attention retains its bounded key tiles and online maximum rescaling.
- A compensated F32 sum is maintained for each softmax denominator and each
  weighted-value output lane. Corrections survive tile boundaries and are
  rescaled with the running maximum. Explicit rounded F32 additions/subtractions
  prevent contraction from removing the correction. F32 FMA recovers product
  rounding residuals; no F64 operation is required by the production algorithm.
- The same rule applies to every F32 ordered attention invocation. There is no
  architecture, checkpoint, component, phase, or input-content dispatch.
- BF16 kernels and their selection rules are unchanged. The explicit research
  `baseline` variant remains an old numerical candidate, not the newly qualified
  F32 default.

## Qualification gate

For the fixed finite-input fixture matrix, require every element to satisfy
`abs(actual - reference) <= 2e-5 + 2e-5 * abs(reference)` and reject nonfinite
results. Reference math is evaluated independently in F64 from the exact F32
input bytes; an unfused F32 PyTorch oracle with TF32 disabled is also measured.
A less accurate F32 oracle is not treated as exact truth under cancellation.
The gate is fixed across all fixtures, not selected by model identity or fitted
to observed errors. It is not a universal error bound for arbitrary ill-conditioned
floating-point inputs.

The supported qualification cases have at least one visible key per query,
finite Q/K/V and bias, positive dimensions, and legal BSHD/broadcast contracts.
Fully masked rows and nonfinite inputs are not newly specified by this change.

Capture QK, online probability factors, denominator, accumulation and output
using the actual private CUDA kernels in the research executable. Its final
result must be bitwise equal to `Backend::attention` before those observations
are used for attribution. Compare softmax on fixed native scores, P×V on fixed
native effective probabilities, and output projection on fixed native output to
separate propagated error from each local arithmetic error.

## Memory and compatibility

Additional F32 scratch is `sizeof(float) * B * Q * H * (D + 1)`, independent of
full key count. `ordered_f32_workspace_bytes` includes it. This is an accounting
estimate, not a total-device-memory guarantee. Corrections are normal Backend
owned temporary tensors, with existing allocation failure and lifecycle behavior.
No complete Q×K matrix is allocated by production attention.

F32 results may change in low bits because reduction accuracy is improved.
Compatibility means the same attention semantics, dtype, branch/graph order and
fixed numerical gates; it does not mean bitwise preservation of the faulty sum.
Retain existing generic attention, mask, long-key, LTX graph and Wan-family graph
regressions. Python and the stage-capture executable are research/test tooling,
not production dependencies.

## Reproduction tools

- `vrhino-f32-attention-accumulation-tests INPUT.bundle OUTPUT.bundle`
- `native/tests/f32_attention_reference.py generate --output FIXTURES`
- `native/tests/f32_attention_reference.py analyze --input INPUT.bundle --actual OUTPUT.bundle --output METRICS.json`

The generator can accept an optional `--capture` containing opaque fixed Q/K/V
arrays; the Backend does not consume fixture names or source provenance.

Run the complete synthetic matrix with `qualify --binary PATH_TO_EXECUTABLE --output EVIDENCE_DIRECTORY`; optional `--capture` adds fixed external Q/K/V cases. Every stage metric and both F32/F64 reference comparisons must pass the same gate.
