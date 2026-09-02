# VRhino v0.6.0-alpha

This feature Alpha adds machine-readable Product contracts and a built-in
local Native API for third-party and community integrations.

## Highlights

- inspect typed Product inputs, defaults, validation, and frozen Product facts
  with `vrhino info MODEL --json`;
- start the Native API from the primary executable with `vrhino serve`;
- submit asynchronous Jobs, inspect status, cancel individual Jobs, and stream
  bounded NDJSON progress events;
- use five immutable successor model packages; and
- benefit from substantial qualified LatentSync execution-time and memory-use
  reductions since v0.5.0-alpha.

## Native API v1

`vrhino serve` starts the foreground local server on `127.0.0.1:11435` by
default. Native API v1 supports model discovery, model detail and Product
schema inspection, run submission, run status, per-Job cancellation, and
bounded NDJSON event streaming. See the
[Native API v1 contract](docs/api/native-api-v1.md).

Native API v1 alpha is local-first. It has no authentication, TLS, or CORS and
is not intended for direct exposure to the untrusted Internet. Non-loopback
binding is explicit and emits a warning.

## Machine-readable Product contracts

[ProductInputSchema v1](docs/product/product-input-schema-v1.md) is the
authoritative declaration of required inputs, parameters and defaults,
validation, outputs, and frozen Product facts. Clients can inspect the same
contract through `vrhino info MODEL --json` without parsing human CLI output.

## Models

The exact v0.6.0-alpha Public model set is:

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

Pixel Rhino ships no original model weights or converted VRMs. Model and
content licenses remain independent from the VRhino binary license.

## Performance and runtime improvements

Post-v0.5 work substantially reduced LatentSync execution time and memory use
through source-frame demand planning, optimized mask filtering, generic BF16
policy, exact Lanczos planning, overlapped verification, sampling-state reuse,
and SHA-NI verification acceleration. These changes preserve the shared Native
Runtime and generic precision architecture.

## Packaging and build

The Linux x86_64 CUDA package remains self-contained above the compatible
NVIDIA Driver boundary and includes the required user-space runtime and media
components. It requires no Python, Node, system FFmpeg, CUDA Toolkit, or cuDNN
installation. The production build was qualified with repository-local pinned
tokenizer dependencies and zero dependency-network acquisition during the
clean Release build and package process.

## Known alpha limitations

Native API media inputs and explicit outputs use server-local absolute paths.
This alpha has no upload API, browser-direct CORS policy, authentication, TLS,
pull API, persistent scheduler, SDK, OpenAPI description, or official WebUI.
Product execution uses one worker; Job and retained event state are
process-memory only, restart-local, and bounded. See
[Alpha limitations](docs/alpha-limitations.md).

## Installation and artifact verification

Release archive:
`vrhino-linux-x86_64-cuda-v0.6.0-alpha.tar.gz`

SHA256:
`9ebeabd2741850bc6811309bbc14683d613cfdf1d4c644b2c27012d9ea3db753`

Checksum asset:
`vrhino-linux-x86_64-cuda-v0.6.0-alpha.tar.gz.sha256`

Download both assets from the v0.6.0-alpha GitHub prerelease, verify with
`sha256sum -c`, and follow the [installation guide](docs/install.md).

# VRhino v0.5.0-alpha

This feature Alpha release adds Public LatentSync 1.6 lip-sync
diffusion support.

Highlights:

- `vrhino/latentsync-1.6:1.0.0` through Mode C upstream acquisition and local
  Native conversion;
- the bounded Native `lip_sync_diffusion_workflow_v1` using the Public-safe
  BlazeFace and DWPose alignment stack;
- TemporalConditionalUNet execution through the Shared Runtime with a frozen
  20-step DDIM profile and explicit deterministic seed behavior; and
- complete CreativeML Open RAIL++-M, Apache-2.0, MIT, attribution, and
  representation-change notices.

Pixel Rhino ships no LatentSync model weights or converted VRMs. LatentSync
model use remains subject to the CreativeML Open RAIL++-M License and its
use-based restrictions. Users are responsible for lawful and consented input
media. No upstream endorsement is implied.

The qualification profile was tested on an NVIDIA GeForce RTX 4090 D, with
observed peak internal allocation around 19.9 GiB. This is not a hard minimum
or admission threshold.

Existing LTX, Wan, Mochi, and MuseTalk Public model support remains available.

Release archive: `vrhino-linux-x86_64-cuda-v0.5.0-alpha.tar.gz`

The release remains an Alpha for Linux x86_64 with NVIDIA CUDA. See
[Alpha limitations](docs/alpha-limitations.md) for the exact scope.

# VRhino v0.4.0-alpha

This feature Alpha adds the first public lip-sync product family and generic
multi-component model packages.

Highlights:

- `vrhino/musetalk-v1.5:1.0.0`;
- typed `--video`, `--audio`, and `--output` inputs;
- `vrhino pull`, `info`, `doctor`, and `run` through the bounded Native
  `lip_sync_workflow_v1`;
- local acquisition and Native conversion of fixed upstream assets;
- BlazeFace, DWPose, and Selfie Multiclass analysis components; and
- complete model-license and third-party-notice references.

Pixel Rhino ships no original or converted model weights. MuseTalk use remains
subject to CreativeML OpenRAIL-M use-based restrictions and the other listed
upstream terms. Users are responsible for lawful and consented input media. No
upstream endorsement is implied.

Existing LTX, Wan, and Mochi public model support remains available.

Release archive: `vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz`

The release remains an Alpha qualified for Linux x86_64 with NVIDIA CUDA. See
[Alpha limitations](docs/alpha-limitations.md) for the exact scope.

# VRhino v0.3.1-alpha

This maintenance Alpha improves run readiness checks and support diagnostics.

Highlights:

- `vrhino run` now checks currently available VRAM for presets with declared
  admission thresholds;
- invalid output destinations and bundled encoder failures are detected before
  expensive model execution; and
- new `vrhino doctor [MODEL]` provides privacy-safe local diagnostics for
  system, GPU, cache, media, and model package readiness.

Public model support is unchanged:

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`
- `vrhino/mochi-1-preview:1.0.0`

Release archive: `vrhino-linux-x86_64-cuda-v0.3.1-alpha.tar.gz`

The self-contained Linux x86_64 CUDA and CUDA v1 Runtime boundaries are
unchanged.

# VRhino v0.3.0-alpha

This Alpha adds Mochi 1 Preview as the third public model path.

Public models:

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`
- `vrhino/mochi-1-preview:1.0.0`

Mochi is acquired from its fixed official upstream revision and converted
locally using the native VRhino converter. The release archive contains no
model weights or converted `.vrm` model payloads.

Release archive: `vrhino-linux-x86_64-cuda-v0.3.0-alpha.tar.gz`

The self-contained Linux x86_64 CUDA boundary remains unchanged.

# VRhino v0.2.1-alpha

This maintenance Alpha improves Hugging Face model acquisition reliability.

Highlights:

- official Hugging Face remains the preferred source transport;
- automatic fallback to the third-party `hf-mirror.com` service when
  qualifying official-endpoint availability failures occur;
- resumable partial downloads remain valid across transport fallback;
- artifact size and SHA256 verification remain mandatory; and
- Hugging Face credentials are never automatically forwarded to the mirror.

Public model support is unchanged:

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`

Release archive: `vrhino-linux-x86_64-cuda-v0.2.1-alpha.tar.gz`

# VRhino v0.2.0-alpha

New to VRhino? Start with the [installation guide](docs/install.md).

This Alpha adds Wan2.1 T2V 1.3B as the second public model path.

Public models:

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`

Wan is acquired from its fixed official upstream revision and converted
locally using the native VRhino converter. The release archive contains no
model weights or converted `.vrm` model payloads.

Release archive: `vrhino-linux-x86_64-cuda-v0.2.0-alpha.tar.gz`

The self-contained Linux x86_64 CUDA boundary and CUDA v1 runtime contract are
unchanged from v0.1.1-alpha.

# VRhino v0.1.1-alpha

New to VRhino? Start with the [installation guide](docs/install.md).

This maintenance Alpha improves the `vrhino pull` experience and reliability.

Highlights:

- aggregate source acquisition, conversion, and finalization progress;
- Ctrl+C cancellation with exit status 130 and resumable partial downloads;
- hardened HTTP Range resume handling;
- configurable model/cache root with `VRHINO_HOME`;
- optional `HF_TOKEN` authentication; and
- source-only cache reclamation after verified installation.

Release archive: `vrhino-linux-x86_64-cuda-v0.1.1-alpha.tar.gz`

The supported platform, CUDA v1 runtime contract, initial LTX model package,
and fixed upstream model revision are unchanged from v0.1.0-alpha.

# VRhino Linux CUDA Public Alpha

New to VRhino? Start with the [installation guide](docs/install.md).

This Alpha provides the first ordinary-user VRhino workflow:

```text
vrhino pull MODEL
vrhino run MODEL --prompt "..."
```

Initial qualified model path:

`vrhino/ltx-video-v0.9.1:1.1.0`

The Linux x86_64 package is self-contained above the compatible NVIDIA Driver
boundary. It bundles the required CUDA/cuDNN user-space runtime and media
encoder and contains no model weights.

The release was qualified on Ubuntu 22.04 with glibc 2.35. See
[Alpha limitations](docs/alpha-limitations.md) for the exact scope.

Published release artifact:

- archive: `vrhino-linux-x86_64-cuda-alpha.tar.gz`
- SHA256: `7a3e290963fafb693201ef31f900f21c68c499a6f204733608622e24edc8273f`
- compressed size: 1,403,141,130 bytes

The archive contains no model weights. Model artifacts are downloaded from the
fixed upstream source and converted locally by `vrhino pull`.
