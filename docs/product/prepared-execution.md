# Prepared Product execution

`execute_prepared_product` is Product orchestration over existing generic
Backend, Architecture declarations, ExecutionProgram and SamplingRuntime APIs.
It is shared by bounded prefix/complete qualification. It does not grant numerical
admission and is not yet connected to unrestricted schema2 `run`.

The Product caller supplies a prepared request, its existing PrecisionPolicy and
a Backend factory that declares resource budgets. The factory is separate from
execution intent and from architecture lowering. The executor creates and destroys
three independent device/cache sessions:

1. Native conditioning produces owned host inputs.
2. Complete program admission selects instances; one SamplingRuntime performs the
   continuous chain. Final latent and diagnostic results own host storage.
3. Architecture decode consumes the final latent and returns a validated host video.

No Runtime chooses an expert, derives guidance or reads a model identity. Selection
and guidance remain declarations. The adapter constructs endpoints and declares
numerical topology; the Product executor owns phase lifetimes. All sources remain
leased until the corresponding Backend drains and is destroyed.

Controls are limited to cancellation, read-only phase/step observation, optional
solver snapshots and an explicit qualification-only prefix bound. A prefix admits
the whole original program, never changes its schedule/order and never decodes.
Ordinary execution defaults to the complete program. No external latent injection,
model identity, scalar override or precision override is accepted by this control.

Phase observers run after Backend teardown. Step observers receive only generic
instance/binding IDs and resource statistics; they do not control selection.
Exceptions retire/drain the device session and propagate; cancelled requests do not
start a later phase. Undefined/nonfinite outputs and incorrect latent/video shapes
fail before media encoding. Returned tensors outlive both the prepared source owner
and device sessions.

Ordinary CLI/service admission remains fail closed for unqualified schema2 packages.
A successfully executed qualification command does not establish reference numerical
agreement, ordinary-run qualification or release readiness. Full ordinary-run
connection and a qualified run declaration remain separate outstanding work.
