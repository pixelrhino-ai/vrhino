# Declared precision for Product qualification

The explicit `sampling-step-check` and `sampling-video-check` commands can
consume an existing package PrecisionPolicy artifact. This extends evidence
collection through normal Product preparation and native execution. It grants
no ordinary `run` admission, numerical qualification, or release status.

Legacy step-check v1/v2/v3 and video-check v1 retain their FP32 declarations.
Step-check v4 and video-check v2 require `precision: "bf16"` and one additional
field, `precision_policy_artifact`, containing a logical artifact ID. All other
options and bounds remain the same (prefix at most 32 transitions; complete
schedule at most 64 steps; existing tiny latent/decoded output bounds). A bound
does not rewrite the admitted schedule, bindings, guidance, or solver.

The artifact must be required, have role `precision.policy`, and declare its
size and SHA256 in the package inventory. The local resource resolver maps its
logical ID to a path; the options cannot inject a policy pathname. Admission
rechecks a bounded document's exact bytes before parsing with the existing
`PrecisionPolicy::from_json`. Its current supported requested mode is BF16.
No policy roles, operation contracts, kernels, or tolerance gates are changed.
The result records the artifact ID, digest, and complete policy declaration.

Qualification rejects environment overrides that select research attention or
BF16 implementations, including capture/early-exit switches. It uses the linked
Backend's default path. Build/source/library identity still must be recorded
by the qualification runner: rejecting environment overrides does not prove
the binary was built from an unmodified Backend.

The normal native tokenizer/conditioning, prepared execution, Execution and
Sampling Programs, decoder, and media path remain the owners of execution.
Prefix execution does not decode. Existing component-check v1 remains FP32.
No model identity is used to select policy or execution behavior.

The existing conditioning executor receives the Backend execution dtype, not
the complete PrecisionPolicy object. These entrypoints do not establish that
every internal conditioning operation implements every declared semantic role.
That mapping, and the component's independent numerical reference, must be
audited before claiming whole-product policy conformance.

An execution result `status: PASS` means the bounded execution and its finite /
control checks completed. `production_numerically_qualified` and
`reference_numerical_qualified` remain false; `generic_bf16_status` remains HOLD.
An independent comparison against frozen gates is required. Even a successful
comparison does not automatically promote package admission or release status.
