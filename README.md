# VRhino Public Alpha

VRhino is a self-contained Native Runtime for local DiT/Flow video generation.

```text
Video model packaged as .vrm
        -> Shared Native Runtime
        -> hardware backend
        -> video
```

The first Public Alpha targets Linux x86_64 with the NVIDIA CUDA backend. It
bundles its required CUDA/cuDNN user-space runtime and media encoder. Users do
not install Python, PyTorch, Diffusers, Conda, a CUDA Toolkit, cuDNN, or system
FFmpeg.

The proprietary Runtime source is not included. VRhino specifications and
user documentation are published separately from the closed Runtime.

## Quickstart

After extracting the release archive:

```bash
cd vrhino
./bin/vrhino device
./bin/vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
./bin/vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

`vrhino pull` downloads the fixed upstream model artifacts and converts them
locally. The VRhino binary archive contains no model weights.

## Requirements

- Linux x86_64
- glibc 2.35 or newer
- compatible NVIDIA GPU and NVIDIA Driver
- sufficient GPU VRAM and disk space
- network access for the initial model pull

VRhino was qualified on Ubuntu 22.04. Other Linux distributions have not yet
been qualified.

See [installation](docs/install.md), [Alpha limitations](docs/alpha-limitations.md),
and the [LTX v0.9.1 model notice](docs/models/ltx-video-v0.9.1.md).

## Licensing

The VRhino binary is proprietary and distributed under the
[VRhino Alpha Binary License](licenses/VRHINO-BINARY-LICENSE.txt).
Third-party components retain their own licenses; see
[THIRD_PARTY_NOTICES.txt](licenses/THIRD_PARTY_NOTICES.txt).

Model licenses are independent. VRhino grants no model or content rights.
