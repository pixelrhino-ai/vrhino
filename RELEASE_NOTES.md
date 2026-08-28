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
