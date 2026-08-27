# Installing VRhino Public Alpha

## Supported initial platform

- Linux x86_64
- glibc 2.35 or newer
- NVIDIA CUDA backend
- qualified on Ubuntu 22.04

## Host requirements

Required:

- a compatible NVIDIA GPU;
- a compatible NVIDIA Driver providing `libcuda.so.1`;
- sufficient GPU VRAM and disk space; and
- network access for the first model pull.

VRhino does not yet claim an exact minimum NVIDIA Driver version. Use a
compatible current Driver.

Not required:

- CUDA Toolkit;
- system cuDNN;
- Python, PyTorch, Diffusers, Transformers, or Conda;
- system FFmpeg or x264;
- system curl executable; or
- compiler/build tools.

## Extract and verify

Download both the release archive and its `.sha256` file, then verify:

```bash
sha256sum -c vrhino-linux-x86_64-cuda-alpha.tar.gz.sha256
tar -xzf vrhino-linux-x86_64-cuda-alpha.tar.gz
cd vrhino
./bin/vrhino --version
./bin/vrhino device
```

No installation script or environment activation is required. The package is
relocatable and may be moved as a complete directory.

## Pull and run a model

```bash
./bin/vrhino pull vrhino/ltx-video-v0.9.1:1.1.0

./bin/vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

The qualified LTX source plan downloads approximately 24.77 GB. Allow
additional temporary and final storage for source caching, local conversion,
and the installed runnable package. Interrupted supported HTTP downloads are
resumable.

The release archive ships no model weights. Review the applicable upstream
model license before pulling or running a model.
