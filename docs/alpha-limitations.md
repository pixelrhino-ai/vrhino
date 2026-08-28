# Public Alpha Limitations

The Public Alpha has the following qualified scope:

- Linux x86_64 only;
- glibc 2.35 or newer;
- NVIDIA CUDA backend only;
- qualified on Ubuntu 22.04;
- compatible NVIDIA GPU and Driver required;
- ordinary-user product paths are qualified for exactly
  `vrhino/ltx-video-v0.9.1:1.1.0`,
  `vrhino/wan2.1-t2v-1.3b:1.0.0`, and
  `vrhino/mochi-1-preview:1.0.0`;
- model acquisition may download tens of gigabytes;
- first-time local conversion and installation can take time and require
  substantial temporary disk space; and
- Alpha interfaces, package behavior, and compatibility may change without a
  stability guarantee.

This Alpha does not claim:

- universal Linux distribution compatibility;
- an exact minimum NVIDIA Driver version;
- support for every NVIDIA GPU;
- Windows or macOS support;
- production/stable readiness;
- support for every Wan checkpoint or generation;
- support for every video-model architecture; or
- model redistribution or usage rights.

Model and content licenses are independent from the VRhino binary license.

For presets with a declared VRAM admission threshold, `vrhino run` checks
currently available memory on the selected GPU before expensive execution.
Invalid or unwritable output destinations and a missing or unusable bundled
media encoder also fail during preflight. These checks establish deterministic
readiness conditions; they do not guarantee that inference will complete.

The current qualified Mochi default preset is admitted only when VRhino's
planner sees at least 80 GiB of available device memory. This is the current
product support and admission threshold for that preset, not an empirically
proven absolute minimum VRAM requirement.
