# Schema2 text request preparation

The normal product CLI supports host-only request preparation:

```sh
vrhino dry-run MANIFEST.json LOCAL-RESOURCES.json REQUEST.json
# Equivalent preparation of an installed schema2 package:
vrhino --cache-root CACHE dry-run NAMESPACE/NAME:VERSION REQUEST.json
```

This requires a build with `VRHINO_ENABLE_TOKENIZERS=ON`. Without native tokenizer
support the command fails closed. It verifies the complete admitted package, runs
the native tokenizer on real prompt text, and prepares conditioning input requests.
It does not encode text, evaluate the denoiser, decode video, or permit numerical
execution of a package on HOLD.

Installed and machine-local packages share the same verified admission and
request lowering. Both retain backing owners and preserve declared selection,
schedule and guidance. Neither command grants ordinary `run` qualification.

The closed request schema is:

```json
{"schema":"vrhino.product-text-request.v1","prompt":"A red panda","negative_prompt":"blurry","seed":5701,"width":32,"height":32,"frames":1}
```

Seed is a nonnegative signed 64-bit JSON integer or a canonical decimal string
covering the full uint64 domain (no signs or leading zeros). Dimensions must fit the declared
decoder stride/origin geometry and the admitted architecture patch contract.
Step count, guidance, binding, and precision overrides are rejected. Execution
selection and guidance come from the package's admitted programs.

Packages with a program-backed Product profile and an immutable
`product.execution_artifact` also accept the ordinary Product request envelope:

```json
{"model":"namespace/name:version","inputs":{"prompt":"A red panda"},"parameters":{"seed":5701},"resources":{"weight_cache_budget_bytes":32212254720}}
```

The envelope is validated by the same ProductInputSchema mapper as the API.
Its model reference must match the resolved package. Frozen geometry and FPS come
from the Product profile; seed defaults come from the input schema. There is no
fallback to constant guidance or a single binding. See
[declared text execution](declared-text-run.md) for the immutable run declaration.

The manifest's optional `admission.request_artifact` references a required,
size/SHA256-qualified resource. Existing manifests without it remain valid for
structural preflight; text preparation requires it. Example wiring:

```json
{
  "schema":"vrhino.text-product-wiring.v1",
  "tokenizer_spec":{"format":"huggingface_json","max_length":512,"pad_id":0,"suffix_ids":[1]},
  "tokenizer_config_artifact":"tokenizer-config",
  "geometry":{"spatial_scale":8,"temporal_scale":4,"temporal_origin":1},
  "conditioning_bindings":[
    {"component_id":"positive-conditioning","text_source":"prompt","target":"positive"},
    {"component_id":"negative-conditioning","text_source":"negative_prompt","target":"negative"}
  ]
}
```

These values are declarations, not defaults selected by model identity. Token
length must equal the canonical graph bound. Pad/EOS IDs must match the verified
tokenizer and tokenizer config. Geometry must agree with the embedded decoder
contract. This version supports BCTHW, positive spatial/temporal strides, and the
declared `origin+scale*(F-1)` temporal relation. Other contracts fail closed.
External tokenizer metadata may contain large numeric sentinels: only that
descriptive config opts into parsing oversized integers as finite doubles. It
cannot supply the admitted sequence bound; package/request integer parsing stays
strict. Neither token assets nor model files are rewritten.

`PreparedTextProductRequest` retains the admitted package, mmap, conditioning
asset, and architecture owners. It provides token IDs, masks, conditioning graph,
expected hidden shapes, borrowed weight access, and an output binding function.
That function checks conditioning result count, host layout, shape, and dtype
before creating the architecture's input bundle. It does not execute an encoder
or prove numerical correctness. Positive/negative hidden states are never fabricated
by dry-run. Tests may use explicitly labeled shape-only tensors.

Dry-run emits token IDs and the admitted per-step instance/binding, timestep, and
guidance trace. It reports numerical HOLD separately. Its JSON contains input text
derived token data and should be handled as request evidence.

VRM integrity checks retain streaming SHA256 and internal BLAKE2b-128 verification.
The latter now uses 8 MiB reads from the owned descriptor rather than touching all
mapped tensor pages. This bounds checksum working memory without omitting payload
bytes. Mmap/borrowed tensor ownership remains unchanged. Host RSS during this
command is not a GPU or full inference memory guarantee. Future execution must
revalidate mutable resources; a saved dry-run is not permission to execute later
modified files.

Real text encoding, its output numerical qualification, production sampling,
decoder execution, and media output remain separate work packages.
