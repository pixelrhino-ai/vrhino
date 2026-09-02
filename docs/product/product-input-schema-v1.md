# VRhino ProductInputSchema v1

Status: canonical Runnable Model Package Product contract.

## Boundary

`ProductInputSchema v1` describes only values a user may provide or control
when invoking a runnable Product. It is embedded in the verified
`vrhino-model.json` Product declaration:

```text
Runnable Model Package / Product contract
  product.family
  product.input_schema
  product.frozen_profile
        |
        v
bounded Product orchestration
        |
        v
workflow / Component Graph / Sampling Program / Runtime / Backend
```

It is not a workflow language, graph, script, plugin interface, UI layout,
Runtime input table, PrecisionPolicy, qualification record, acquisition plan,
or license policy.

The exact schema identity is:

```text
vrhino.product.input-schema.v1
```

## Groups and declarations

The schema contains exactly three groups:

```json
{
  "schema": "vrhino.product.input-schema.v1",
  "inputs": [],
  "parameters": [],
  "outputs": []
}
```

Each declaration contains only:

- `name`: stable Product semantic name;
- `type`: one v1 type;
- `required`: whether the user must supply the value;
- optional `default`: admitted only on optional declarations;
- optional `validation`: closed, type-specific declarative constraints.

Names are unique across all three groups. A required declaration cannot have a
default. Unknown declaration fields and validation semantics fail closed.

## V1 types

The exact v1 type set is:

| Type | Admitted group | Current semantic |
|---|---|---|
| `text` | `inputs` | text-to-video prompt |
| `integer` | `parameters` | unsigned 64-bit seed |
| `media.video` | `inputs` | source video |
| `media.audio` | `inputs` | driving audio |
| `media.mp4` | `outputs` | output MP4 destination |

No image, boolean, float, enum, array, object, tensor, or directory type is
part of v1. Package presets remain the existing `defaults.presets` package
selector contract; they are not duplicated as a ProductInputSchema field.

## Validation semantics

Validation is a closed vocabulary:

| Type | Admitted keys |
|---|---|
| `text` | `min_length` |
| `integer` | `minimum`, `maximum` |
| `media.video` | `regular_file`, `decodable`, exact rational `fps` |
| `media.audio` | `regular_file`, `decodable`, `minimum_duration_ms` |
| `media.mp4` | `parent_creatable_and_writable` |

Rational FPS is `{ "numerator": N, "denominator": D }`, with positive
integers. Boolean constraint values, when present, are `true`.

Integer bounds admit non-negative JSON integers or canonical unsigned decimal
strings. Values outside the interoperable JSON number range use decimal
strings. The uint64 maximum is therefore
`"18446744073709551615"`. Parsing rejects signs, non-digits, leading zeroes,
overflow, and `minimum > maximum`.

No callback, expression, formula, conditional script, or cross-field language
exists. Audio-derived frame count remains bounded workflow behavior rather
than a schema expression.

## Defaults and precedence

For schema-backed packages:

```text
user request override
>
product.input_schema declaration default
```

`seed` and `output` are optional. The Product contract is the canonical
user-visible default source. Existing execution metadata may retain the seed
for execution compatibility, but package qualification requires exact
equality. A disagreement fails closed. `RunOptions` does not independently own
`output.mp4`; schema-backed packages resolve it from the Product contract.

Legacy packages without the schema retain their historical execution seed and
`output.mp4` fallback. Absence is legacy compatibility, not an inferred v1
schema.

## Frozen profile

`product.frozen_profile` is a sibling read-only Product profile. It is required
whenever `input_schema` is present and has a bounded vocabulary:

- `output`: optional fixed width, height, and frames; required exact FPS,
  duration semantic (`fixed` or `audio_derived`), and audio semantic (`none` or
  `driving_audio`);
- optional `sampling`: deliberately curated method, prediction, steps,
  guidance scale, and eta;
- optional `temporal`: deliberately curated chunk frame count.

Frozen fields are never request parameters. Execution/workflow data may be
checked for equality but is never automatically projected. Alignment EMA,
detector/TTA policy, tensor bindings, latent layout, RNG streams, memory
planning, precision mechanics, and Backend details remain internal.

## Family contracts

`product.family` remains the only Product capability discriminator.

`text_to_video` v1 has required `prompt`, optional `seed`, and optional
`output`. Its frozen profile records qualified fixed output geometry/timing and
curated sampling facts.

`lip_sync` v1 has required `video` and `audio`, optional `seed`, and optional
`output`. Video is exact 25/1 FPS; driving audio must be decodable and at least
40 ms; output duration is audio-derived. Lip-sync presets are not declared.

## Compatibility and unknown semantics

- Exact known identity `vrhino.product.input-schema.v1`: parse and validate.
- Unknown schema identity/version: fail closed.
- Unknown type: fail closed.
- Unknown field inside `input_schema`, a declaration, validation, or
  `frozen_profile`: fail closed because it may carry semantics.
- Optional non-semantic additions outside these closed Product contract
  objects continue to follow the existing package-manifest additive-field
  convention.
- Missing schema on a legacy package: preserve the legacy contract.

Adding optional package metadata outside the closed objects does not change v1.
A new Product type, validation semantic, required semantic field, or changed
default precedence requires a new schema identity. Any manifest byte change
requires a new immutable package version.

## Explicit non-goals

No UI widgets, CSS, layout/order framework, arbitrary annotations, workflow
nodes, DAG, script engine, plugin parameters, HTTP/API design, SDK design,
qualification evidence, hardware observation, acquisition/source data,
license interpretation, Runtime/Backend inputs, PrecisionPolicy, or neural
execution semantics are defined here.
