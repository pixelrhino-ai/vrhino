# Controlled native product component qualification

VRhino targets multiple video architecture families. The current text-conditioned
diffusion product adapter is one capability set, not a universal contract for
UNet/hybrid, autoregressive, or future architectures. New families can declare
different component interfaces and execution intents without teaching the shared
Runtime, Backend, or PrecisionPolicy their model identities.

For admitted schema2 text products, the ordinary CLI offers a bounded component
check before denoiser qualification:

```sh
vrhino component-check MANIFEST LOCAL-RESOURCES REQUEST.json OPTIONS.json OUTPUT-DIR
```

The manifest, local resolver and text request use the same path as `dry-run`.
Options are a closed declaration, for example:

```json
{
  "schema":"vrhino.component-check.v1",
  "precision":"fp32",
  "actions":["conditioning","decoder"],
  "device_budget_bytes":64424509440,
  "host_budget_bytes":214748364800,
  "fps":8,
  "latent_bundle":"/local/explicit-small-latent.vrt",
  "media_range":[-1,1]
}
```

`precision` explicitly selects an existing FP32 qualification mode for this
bounded command. It is not a model-specific fallback or a production policy
override. This version does not qualify BF16. Unsupported actions/modes and
malformed declarations fail closed. The evidence directory must not exist.
Decoder output is limited to 1,048,576 elements; the latent fixture must match the
admitted request and contain exactly one finite F32 tensor named `latent`.

Ownership is explicit:

| Capability | Owner |
|---|---|
| Source identities and logical resources | Product/package admission |
| Token IDs, masks, component output interfaces | Prepared product request |
| Declared pre-norm Transformer execution | Existing ConditioningComponentExecutor |
| Canonical graph, bindings, component endpoints | Existing Architecture factory/adapter |
| Selection and guidance | Admitted Execution/Sampling declarations |
| Per-phase backend lifetime and result export | Product orchestration |
| Tensor operations and cache implementation | Existing generic Backend/CUDA |
| Precision roles and execution dtype rules | Existing PrecisionPolicy |

`execute_text_conditioning` consumes the prepared graph and real IDs/masks,
retains source-owner leases in the backend, returns validated finite host results,
and binds them to the declared targets. It does not choose a model or alter the
program. `admit_conditioned_sampling` verifies that conditioning did not modify
seed/shape intent or inject extra fields, constructs the existing component
catalog and admits its selection. It never evaluates a denoiser, advances RNG,
creates solver state or enters a sampling loop.

Decoder checking reuses the existing architecture decoder endpoint and shared
component executor. The legacy `Architecture::decode` API still expresses
family-specific decoder topology imperatively; this command does not introduce
another execution engine or move that topology into Product/Runtime/Backend.
A future fully declarative decoder migration must be reviewed separately rather
than silently treating this legacy API as the final universal architecture boundary.

Each device phase owns its backend/cache and completes before the next phase.
The result records source leases, cache/upload observations and device memory
after teardown. Cache planning budgets and source-backing admission are not a
guarantee of complete inference peak memory. Qualification runners should record
independent process/GPU observations and explicit resource stop limits.

Conditioning emits `tokens.vrt` and `conditioning.vrt`. Independent decoding emits
`decoded.vrt` and a clearly named `synthetic-decoder-check.mp4`, through the existing
native media encoder path. This clip is a component fixture, not sampled video.
One phase's failure is recorded as HOLD while other independent phases may finish.

The command's PASS means native execution, finite/interface checks and relevant
plumbing passed. Reference comparisons are separate research evidence and must
pin source, input, dtype and existing gates. No Python is invoked by the product.
Neither this command nor its reference results release the generic BF16 HOLD,
grant full sampling admission, or establish release readiness.
