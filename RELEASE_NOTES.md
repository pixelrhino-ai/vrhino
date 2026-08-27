# VRhino Linux CUDA Public Alpha

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
`docs/alpha-limitations.md` for the exact scope.

Phase 27B produced the following private release candidate for final review:

- archive: `vrhino-linux-x86_64-cuda-alpha.tar.gz`
- SHA256: `7a3e290963fafb693201ef31f900f21c68c499a6f204733608622e24edc8273f`
- compressed size: 1,403,141,130 bytes

The archive and checksum will be attached only after an explicit publication
decision. No artifact has been published by Phase 27B.
