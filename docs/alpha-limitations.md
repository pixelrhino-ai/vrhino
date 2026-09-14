# Public Alpha Limitations

v0.8.0-alpha is an unreleased Windows-supporting line. Its new rc7 build and
exact-candidate qualification remain pending. Published v0.7.0-alpha is an
immutable Linux-scoped historical release; this source update does not change
its promises or add Windows assets to it.

The historical Linux Alpha has the following qualified scope:

- Linux x86_64 only;
- glibc 2.35 or newer;
- NVIDIA CUDA backend only;
- qualified on Ubuntu 22.04;
- compatible NVIDIA GPU and Driver required;
- ordinary-user product paths are qualified for exactly
  `vrhino/ltx-video-v0.9.1:1.1.1`,
  `vrhino/wan2.1-t2v-1.3b:1.0.1`, and
  `vrhino/mochi-1-preview:1.0.1`, plus the lip-sync products
  `vrhino/musetalk-v1.5:1.0.1` and
  `vrhino/latentsync-1.6:1.0.1`;
- model acquisition may download tens of gigabytes;
- first-time local conversion and installation can take time and require
  substantial temporary disk space; and
- Alpha interfaces, package behavior, and compatibility may change without a
  stability guarantee.

## Windows v0.8 qualification boundary

The preserved Windows baseline is Windows 10 Pro 22H2 / build 19045,
NVIDIA GeForce RTX 3090 24 GiB, driver 610.47. Frozen rc6 completed the full
clean-host Wan and LTX product paths; those exact bytes and their evidence
remain historical. They do not qualify a not-yet-built v0.8 binary.

Before publication, exact post-merge rc7 bytes must pass native conversion,
installation, doctor, real execution, full media decode, dependency isolation
and cleanup for Wan 1.0.1, LTX 1.1.1, MuseTalk 1.0.1 and LatentSync 1.0.1.
Mochi is not fully qualified on this 24 GiB host because of its unchanged
80 GiB available-device-memory admission; this is not a model failure.
See the [exact identities and release gates](release/v0.8.0-alpha.md).

The Windows product bundles its CUDA/cuDNN/NVRTC and media runtime. It requires
no separately installed Toolkit, cuDNN, Visual Studio/Build Tools, CMake,
Ninja, Python, PyTorch, Diffusers, Transformers, Conda, system FFmpeg,
MSYS2/MinGW or WSL. Developer tooling belongs on a separate build host.

MuseTalk/LatentSync technical execution and full video/audio validation are
distinct from manual visual sanity and objective lip-sync quality. Historical
developer-host manual sanity passed; rc7 outputs require their own recorded
review. No calibrated objective quality threshold has been frozen. That
research gate is deferred, not an objective PASS and not a Windows-only
technical release blocker. Do not hide a subjective quality threshold in
automated technical checks.

## Native API v1 alpha boundary

The Native API is local-first. It binds to `127.0.0.1:11435` by default and
has no authentication, TLS, or CORS policy. A non-loopback bind is explicit
and emits a warning. The server is not intended for direct exposure to the
untrusted Internet.

Media video/audio inputs and explicit output destinations are absolute paths
on the server's local filesystem. Native API v1 alpha has no multipart or
base64 upload, remote-media URL acquisition, or browser upload API.

Product execution is serialized through one worker with a bounded admission
queue. Job metadata and event replay state are process-memory only; restarting
the server invalidates Job IDs and retained event history. Successful Product
artifacts remain on disk independently. Event replay uses bounded retained
history, and an explicit stale sequence may return
`event_history_truncated`.

This alpha provides no pull API, persistent Job database, persistent client
cursor, model-residency guarantee, model scheduler, parallel Product
execution, multi-GPU scheduler, official WebUI, SDK, or OpenAPI description.
Browser-direct integration remains deferred; desktop, native, local-daemon,
CLI-adjacent, and local-bridge integrations are the intended clients.

This Alpha does not claim:

- universal Linux distribution compatibility;
- an exact minimum NVIDIA Driver version;
- support for every NVIDIA GPU;
- universal Windows compatibility or macOS support;
- completed v0.8 rc7 qualification before its exact-candidate gates pass;
- production/stable readiness;
- support for every Wan checkpoint or generation;
- support for every video-model architecture; or
- model redistribution or usage rights.

Model and content licenses are independent from the VRhino binary license.
MuseTalk is qualified on CUDA. Metal numerical qualification for this product
has not been completed. Its observed 6.27 GiB peak internal allocation on an
RTX 4090 D full-sample run is not a minimum-VRAM requirement.

Public Mode-C support for `vrhino/latentsync-1.6:1.0.0` is included in the
published v0.5.0-alpha release. Its CUDA qualification was performed on an RTX
4090 D; observed peak internal allocation was approximately 19.9 GiB for the
qualification profile. This is not a minimum or admission threshold.

For presets with a declared VRAM admission threshold, `vrhino run` checks
currently available memory on the selected GPU before expensive execution.
Invalid or unwritable output destinations and a missing or unusable bundled
media encoder also fail during preflight. These checks establish deterministic
readiness conditions; they do not guarantee that inference will complete.

The current qualified Mochi default preset is admitted only when VRhino's
planner sees at least 80 GiB of available device memory. This is the current
product support and admission threshold for that preset, not an empirically
proven absolute minimum VRAM requirement.
