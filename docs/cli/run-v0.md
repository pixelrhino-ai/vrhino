# VRhino Native Product Run v0

Status: end-to-end product-layer alpha. CUDA contract `cuda-v1`, `.vrm` format
`0.1`/schema `1`, and Runnable Model Package schema `1` are unchanged.

## Command

```text
vrhino run namespace/name:version \
  --prompt "a cat walking in snow" \
  [--output output.mp4] [--preset default] [--seed 5703] \
  [--overwrite] [--debug]
```

`run` is deliberately offline: the exact package must already be installed.
Missing packages fail as `MODEL_NOT_FOUND`; the command never contacts the
registry or Hugging Face and never performs an implicit pull.

## Generic execution flow

```text
exact package reference
  -> LocalModelCache::resolve / compatibility validation
  -> installed execution.profile and component declarations
  -> hardware preflight
  -> Native tokenizer
  -> Native conditioning component executor
  -> frozen NativeRuntime / SamplingRuntime / ComponentExecutor
  -> BCTHW RGB video in declared [0,1] range
  -> bundled vrhino-ffmpeg
  -> atomic MP4 publication
```

The product layer contains no architecture-name dispatch. Tokenizer type,
conditioning components, weight shards, input bindings, runtime inputs,
derived positional inputs, precision policy, output layout/range, and the
default preset are package/profile declarations.

## Prompt-runnable profile

Package schema 1 already makes `execution.profile` an artifact and therefore
does not change. To be prompt-runnable, the selected profile must additionally
provide a `run` declaration containing generic component specifications,
tokenizer specifications, bindings, runtime/derived inputs, memory budgets,
precision policy artifact, and output tensor contract. An installed package
without that data remains valid for low-level execution but `run` fails closed
as `PACKAGE_INVALID`.

Package identities are immutable. The earlier LTX package `1.0.0` was not
silently changed; the complete prompt-runnable profile is published as
`vrhino/ltx-video-v0.9.1:1.1.0`. The example profile is
[examples/ltx-run-profile-v1.json](../../examples/ltx-run-profile-v1.json).

## Hardware preflight

Before loading weights, the CLI compares CUDA device availability, compute
capability, total/free VRAM, driver version, and package/preset hardware
metadata. Results are:

- `SUPPORTED`: admission metadata is satisfied;
- `SUPPORTED_WITH_WARNING`: a reliable minimum is unknown or only
  qualification/recommended evidence is available;
- `INSUFFICIENT_VRAM`, `UNSUPPORTED_GPU`, or `DRIVER_INCOMPATIBLE`: fail before
  large model allocation.

There is no model-name branch. A preset declaring a 64 GiB minimum is rejected
on a 24 GiB GPU by the same generic comparison.

## Progress and cancellation

Stable product progress is `Loading model`, `Encoding prompt`, `Sampling i/N`,
`Decoding video`, `Encoding MP4`, and `Done`. SIGINT records a cancellation
request and exits at the next completed sampling operation boundary. It does
not publish an output or leave a partial MP4.

## Video output

The actual frozen Native video boundary is BCTHW RGB in `[0,1]`. Product
conversion clamps that range directly to RGB8; the old tooling declaration and
mapping for `[-1,1]` is not used. The encoder is a sibling executable named
`vrhino-ffmpeg`, invoked directly (never through a shell) with H.264/libx264,
`yuv420p`, and preset fps. Linux distribution must bundle the executable,
libraries, and applicable FFmpeg/x264 licenses; users are not expected to
install FFmpeg.

Encoding writes to an output-local unique `.partial` path and atomically
renames it to the requested `.mp4` only after success. Existing outputs fail as
`OUTPUT_EXISTS` unless `--overwrite` is explicit. Failed or cancelled inference
never publishes a successful-looking MP4.

## Errors

Product errors have a stable code and human message. Run adds
`UNSUPPORTED_GPU`, `INSUFFICIENT_VRAM`, `DRIVER_INCOMPATIBLE`, `INVALID_INPUT`,
`RUNTIME_ERROR`, `OUT_OF_MEMORY`, `OUTPUT_EXISTS`, `VIDEO_ENCODING_FAILED`, and
`CANCELLED` to the existing package/cache codes. Debug details are opt-in.

## Dependency boundary

The complete run has zero Python, PyTorch, Diffusers, Transformers-Python, or
official-reference environment dependency. The only subprocess is the bundled
native media encoder included in the Linux release.
