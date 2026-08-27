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
