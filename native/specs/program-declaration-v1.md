# Generic product program declaration v1

This wire contract belongs to Product / Converter lowering. It is separate from
the graph schema v2 binding catalog, which only declares what exists. No model
name, source path, device, precision or residency rule is accepted here.

`vrhino.programs.v1` requires exactly these fields:

- `schema`: `vrhino.programs.v1`
- `required_capabilities`: `execution.per_step.v1`, `sampling.flow_sigma_cfg.v1`
- `execution`: `kind=per_step` and a nonempty `instance_ids` array
- `sampling`: `prediction=flow`, `solver=multistep_predictor_corrector`,
  `maximum_order=2`, `schedule=flow_sigma`,
  `branch_order=unconditional_then_conditional`, and `transitions`

Each transition contains exactly `model_timestep` (nonnegative int64), `sigma`,
`next_sigma`, and `guidance_scale` (finite, nonnegative). Flow transition chains
are validated by the existing ScheduleContract. Selection and guidance lengths
must match. Every selected instance must exist and share an admitted interface.
This initial wire version conservatively requires equal graph declarations for
selectable instances; broadening compatibility requires separate qualification.

`admit_program_declaration` validates caller-supported capabilities and constructs
existing B-3 ExecutionProgram / SamplingProgram / GuidanceSchedule objects. It
does not execute them. Request geometry, seed, conditioning and actual component
factories remain to be supplied and admitted by native load qualification.
Changing the step count requires re-lowering the tables, not silently reusing them.

The package inventory remains graph schema v2 with `binding_catalog.v1`. v0.8
readers reject that schema. New product consumers must also call program admission
and reject unsupported program capabilities before creating an executable request.
This change does not make the existing single-binding product runner support
these packages; Stage C must qualify loader wiring. There is no fallback instance.

Binary VRM remains v0.1. The local structural package sidecar has schema
`vrhino.structural-package.v1` and a stable list of relative resource paths, sizes
and SHA256 values. It is an isolated qualification artifact, not a registry or
release manifest. Execution-bearing data is in the separate program declaration;
source provenance may retain human-readable upstream identifiers.
