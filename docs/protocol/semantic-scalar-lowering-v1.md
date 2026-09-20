# Semantic scalar lowering v1

This is an explicit operation contract, independent of architecture, component,
model, backend identity and execution phase. It changes no numerical acceptance
gate. A coefficient role is never inferred from a name or a single-element shape.

`PrecisionPolicy::with_scalar_roles` declares roles for C++ callers. JSON callers
use `vrhino.precision.policy.v2` and `effective.scalar_roles`:

```json
{"contract":"semantic-scalar.v1","roles":["SOLVER_COEFFICIENT","SIGMA"]}
```

All previous policy fields remain required. The JSON loader remains a BF16
qualification-policy loader; typed C++ policies also support FP32. Version 1
without this declaration is unchanged. A scalar declaration under version 1,
unknown contract, empty role list, unknown or duplicate role fails closed. Old
readers that only accept policy v1 reject policy v2 at their schema check.
Existing product declarations and frozen policy files are not migrated implicitly.

| Role | Scalar source | Arithmetic | Result storage |
|---|---|---|---|
| GuidanceCoefficient | F32 | F32 | admitted input storage |
| SolverCoefficient | F32 | F32 | admitted input storage |
| Sigma | F32 | F32 | admitted input storage |
| TimestepScale | F32 | F32; input coordinate must also be F32 | F32 |
| ModulationCoefficient | F32 | F32 | F32 temporary |
| NormalizationParameter | F32 | F32 | admitted input storage (statistics normally F32) |
| DataOperand | admitted input dtype | existing dtype operands, backend element arithmetic | admitted input dtype |

F32/BF16 input storage is accepted. The coefficient is a finite rank-0 or rank-1
single-element tensor; divide rejects zero. Host/device placement is irrelevant
for a declared role. Discrete timestep IDs remain discrete and use their existing
explicit embedding conversion; they are not floating scalar operands.

For a declared F32 coefficient `c`, a BF16 value `x` is widened exactly to F32,
then `x OP c` executes, then the result is cast once to the declared output
storage. Add, multiply and divide are supported. Each existing elementary
operation keeps its own rounding boundary: no fusion, reassociation, CFG branch
reordering or solver recurrence change. Already quantized BF16 coefficient
sources are rejected instead of being widened and misrepresented as F32 sources.
DataOperand remains available for explicitly low-precision data, even if its
shape is scalar; this is not an all-scalar FP32 fallback.

Undeclared roles call the original binary operation with the original operands.
Raw Backend binary rules remain unchanged, including their historical operand
order and host-only dtype behavior. Legacy Euler retains its original host delta
expression. Declared Euler preserves F32 delta/sign calculation before applying
it to the prediction. Flow-to-x0 uses Sigma, CFG/linear guidance use
GuidanceCoefficient, and the other solver coefficients use SolverCoefficient.

SamplingPrimitives stores a copy of the policy; SamplingRuntime passes its policy
to the primitives and continues to admit state/output storage as before. Scalar
lowering preserves that admitted state storage; it does not globally change the
latent/history dtype. There is one solver chain, one RNG, and no role-dependent
component selection. LTX's existing timestep scale consumes TimestepScale; absent
that role its old expression is reproduced. Existing F32 modulation/normalization
and heavy Linear data contracts are not replaced by scalar roles.

The implementation uses existing Backend copy/cast/binary operations. Device
scalar validation currently copies the scalar to host, and coefficient arithmetic
can allocate F32 temporaries. This is a correctness implementation, not a
performance or peak-memory guarantee. No Backend/CUDA kernel or model routing
change is required.

Qualification separates scalar semantics from reduction error. F32 timestep GEMM
geometry and BF16 Linear accumulation/output rounding can still differ after
scalar lowering. Same-input operator gates and strict block/denoiser/latent gates
remain those in the generic BF16 V3 contract. A passing scalar/solver replay does
not waive failed architecture observables, and a frozen-prediction solver replay
is not a closed-loop model qualification.
