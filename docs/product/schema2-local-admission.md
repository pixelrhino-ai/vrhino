# Schema2 local product admission

Product schema2 adds a closed `admission` object to the existing product manifest.
VRM **binary format and metadata envelope remain v0.1 / schema1**; the embedded
**package graph schema is 2**. These version numbers are independent.

`admission` contains required artifact IDs `graph_artifact`, `metadata_artifact`,
`programs_artifact`, `source_manifest_artifact`; `structural_only: true`;
`required_capabilities: ["binding_catalog.v1", "execution.per_step.v1", "sampling.flow_sigma_cfg.v1"]`;
and `resources` mapping `conditioning_declaration`, `conditioning_index`,
`conditioning_weights`, `tokenizer` to required artifact IDs. The default preset
references a structural profile with only `latent_shape` (five integers) and `seed`.
Graph/program/metadata resources must equal the authenticated embedded declarations.
Product identity and primary repository/revision must match embedded provenance.
Each declared resource has an exact size and SHA256, including the source manifest.

Machine-local resolution is separate:

```json
{"schema":"vrhino.local-resources.v1","resources":{"artifact-id":"/local/file"}}
```

Every artifact must have exactly one map entry. Relative paths are resolved against
the local map directory, independent of cwd; absolute paths never enter graph semantics.
Missing/extra IDs, invalid types, missing files, sizes and hashes fail closed.
No resource is downloaded, installed into CAS, rewritten or copied.

Use the regular product CLI:

```sh
vrhino preflight /path/vrhino-model.json /path/local-resources.json
# An already installed schema2 package uses the same admission:
vrhino --cache-root /path/cache preflight namespace/name:version
```

This host-only command verifies SHA256 by streaming and VRM internal checksums,
retains the stable descriptor-backed mmap, admits all canonical parameter slots,
constructs the family adapter and declared schedule, and inspects conditioning
resource references. It uploads no weights and performs no text encoding, sampling,
or decoding. Its host resource estimate describes mapped backing, not inference RSS
or GPU residency. File identity must be rechecked for a future run; preflight is
not a reusable authorization for resources changed afterward.

The current family capability supports one canonical shared graph, any positive
number of declared bindings/instances, and the existing per-step flow/CFG program.
Unsupported families/topologies/program versions fail before numerical execution.
Schema1 continues through the unchanged caller-retained legacy factory. Schema2
requires the owning factory overload; single-denoiser fallback and legacy scalar
sampling overrides are rejected. Program selection belongs to generic execution
admission, never to model-named conditions.

`structurally_runnable=true` means the above declared structural request is admitted.
It does **not** mean text conditioning/decode have been numerically exercised,
full video wiring is complete, production numerical qualification has passed,
or inference memory fits. `numerically_qualified=false`, numerical status `HOLD`,
and no total-device-memory guarantee are reported separately. Both the CLI run
entry and product run service reject unqualified schema2 before numerical execution.
No research numerical route is enabled by this admission feature.

Both resource resolvers enter `preflight_resolved_product`. It reparses the
manifest, rejects a manifest changed since resolution, checks the complete
resolved artifact inventory and declarations, and verifies that the runtime
path identifies the declared runtime artifact. Cache resolution does not skip
SHA256 or mmap verification. This adds no new installation or download behavior;
local maps remain suitable for large existing assets without copying them to CAS.

A schema2 package can expose the standard Product prompt/seed/output declaration
without replacing its program with constant guidance. Its frozen sampling profile
uses `{"program_artifact":"RESOURCE_ID"}` as the sole sampling field. The ID must
equal `admission.programs_artifact`; normal artifact integrity and canonical
program admission remain mandatory. This is metadata/request qualification,
not permission to execute an ordinary numerical run. A fixed Product geometry
also constrains prepared text requests; profiles are not request overrides.
