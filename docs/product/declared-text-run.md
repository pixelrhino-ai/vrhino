# Declared text Product execution

A schema2 text Product with a program-backed frozen sampling profile can reference
a required artifact through `product.execution_artifact`. Its role must be
`product.execution`; its size and SHA256 are checked against the bytes parsed.
The declaration is bounded to 64 KiB and uses this closed schema:

```json
{
  "schema": "vrhino.product-text-run.v1",
  "negative_prompt": "blurry, distorted",
  "precision_artifact": "policy",
  "memory": {
    "device_budget_bytes": 81604378624,
    "host_budget_bytes": 214748364800,
    "workspace_bytes": 2147483648,
    "safety_margin_bytes": 1073741824,
    "weight_cache_budget_bytes": 32212254720
  },
  "media_range": [-1, 1]
}
```

These memory values illustrate a local deployment, not a universal requirement or
default. Cache cap is optional; when present it must be positive and fit the
existing device envelope after reserves. Request resources may only tighten it.
The budgets remain declarations, not a guarantee of full inference peak memory.

`precision_artifact` resolves a required `precision.policy` artifact. Structural
lowering verifies and parses its immutable bytes through the existing
PrecisionPolicy parser, without device work or qualification promotion. Research
environment overrides remain rejected. No model identity selects a precision rule.

The run document owns negative conditioning defaults, resource budgets and media
range. ProductInputSchema owns prompt/seed/output admission; the frozen profile
owns geometry/FPS. Canonical Sampling and Execution Programs retain timestep,
guidance and component selection. Those fields cannot be overridden in the run
document. Resource paths stay in the existing local resolver or installed cache.

Ordinary-envelope dry-run performs lowering and native tokenization only. The
normal run adapter shares the prepared native executor and media encoder, with
cancellation and phase events. **Schema2 numerical admission is still closed.**
Both the normal service dispatcher and the adapter enforce that guard before
output preflight, Backend construction or execution. A valid declaration, passing
dry-run or loadable precision policy does not authorize ordinary schema2 inference.
The guarded execution branch therefore still requires end-to-end qualification
after numerical admission is legitimately available.

Schema1 profiles and their existing numerical admission are unchanged. No BF16
reference gates, tolerance, PrecisionPolicy implementation, or numerical operator
implementation are changed by this declaration.
