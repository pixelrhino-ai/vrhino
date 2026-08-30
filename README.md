<p align="center">
  <img src="logo.png" alt="VRhino" width="160">
</p>

# VRhino

English | [简体中文](README.zh-CN.md)

VRhino is a self-contained native runtime and model packaging system for
running AI video models locally without model-specific Python environments.

## What is VRhino?

VRhino explores a GGUF + llama.cpp-like distribution and runtime model for AI
video generation. It converts supported checkpoints into `.vrm` packages and
runs them through a shared native runtime.

```text
Model / Checkpoint
        ↓
VRhino conversion
        ↓
      .vrm
        ↓
Shared Native Runtime
        ↓
     Backend
```

The project is still an Alpha. Its model coverage and maturity are not
comparable to llama.cpp today.

## Current Alpha

The current release is **v0.4.0-alpha**.

- Platform: Linux x86_64
- Backend: NVIDIA CUDA
- Qualified public model paths:
  - `vrhino/ltx-video-v0.9.1:1.1.0`
  - `vrhino/wan2.1-t2v-1.3b:1.0.0`
  - `vrhino/mochi-1-preview:1.0.0`
  - `vrhino/musetalk-v1.5:1.0.0` (`lip_sync`)
- Qualified on Ubuntu 22.04 with glibc 2.35
- Self-contained above the compatible NVIDIA Driver boundary

You need a compatible NVIDIA GPU and NVIDIA Driver, enough VRAM, and enough
disk space. You do **not** install Python, PyTorch, Diffusers, Conda, a CUDA
Toolkit, cuDNN, or system FFmpeg.

## Install

Download both the
[release archive](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.4.0-alpha/vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz)
and its
[checksum file](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.4.0-alpha/vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz.sha256)
from [GitHub Releases](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.4.0-alpha).
With both files in `~/Downloads`, run:

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz -C "$HOME/.local/share"
printf '\nexport PATH="$HOME/.local/share/vrhino/bin:$PATH"\n' >> "$HOME/.profile"
export PATH="$HOME/.local/share/vrhino/bin:$PATH"
```

Keep the extracted `vrhino` directory intact: it contains the runtime, bundled
libraries, and media encoder. Verify the installation from any
directory:

```bash
vrhino --version
vrhino device
vrhino doctor
```

No `sudo` or repository clone is required. See the
[installation guide](docs/install.md) if your browser saves downloads
somewhere other than `~/Downloads`.

## Quick Start

Supported Public Alpha models:

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`
- `vrhino/mochi-1-preview:1.0.0`
- `vrhino/musetalk-v1.5:1.0.0`

Pull one exact model package, for example:

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
```

Then generate a video:

```bash
vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

MuseTalk uses typed video/audio inputs instead of a prompt:

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.0
vrhino run vrhino/musetalk-v1.5:1.0.0 \
  --video input.mp4 \
  --audio driving.wav \
  --output output.mp4
```

The first pull downloads the original artifacts from the model's fixed
upstream revision, converts them locally, and installs the runnable package in
the local VRhino cache. The source acquisition is about 24.77 GB for LTX,
16.36 GiB for Wan, 37.28 GiB for Mochi, and 4.01 GiB for MuseTalk. The release
archive itself contains no model weights or converted model components.

`vrhino pull` prefers the official Hugging Face endpoint and may transparently
fall back to a third-party mirror when the official endpoint is unavailable.

For a concise, privacy-safe local support report, run `vrhino doctor` or
`vrhino doctor MODEL`. It performs no telemetry upload or automatic network
diagnostic request.

Model and cache data defaults to `~/.vrhino`. To use a larger filesystem, set
`VRHINO_HOME` before pulling, for example:

```bash
export VRHINO_HOME=/mnt/large-disk/vrhino
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
```

This does not change the VRhino binary installation directory. After a
successful verified installation, pull reclaims source-only data that is no
longer needed while preserving installed and shared CAS data.

## How it works

`vrhino pull` downloads a fixed upstream model revision, verifies and caches
the source artifacts, converts them natively into the VRhino model format, and
installs an immutable local package. `vrhino run` executes that package with
the shared native runtime and writes an MP4 using the bundled media component.

## Documentation

- [Installation and system requirements](docs/install.md)
- [Model commands](docs/cli/model-cli-v0.md)
- [`pull` command](docs/cli/pull-v0.md)
- [`run` command](docs/cli/run-v0.md)
- [`doctor` diagnostics](docs/cli/doctor-v0.md)
- [LTX-Video v0.9.1 source and license notice](docs/models/ltx-video-v0.9.1.md)
- [Wan2.1 T2V 1.3B source and license notice](docs/models/wan2.1-t2v-1.3b.md)
- [Mochi 1 Preview source and license notice](docs/models/mochi-1-preview.md)
- [MuseTalk v1.5 source, use and license notice](docs/models/musetalk-v1.5.md)
- [VRM format specification](spec/vrm-v0.1.md)
- [Runnable model package specification](spec/model-package-v0.md)

## Alpha limitations

This release supports Linux x86_64, the NVIDIA CUDA backend, and the four exact
qualified model paths listed above. It does not claim support for every NVIDIA
GPU, Linux distribution, model checkpoint, or video-model architecture.
Interfaces and compatibility may change during Alpha.

See [Alpha limitations](docs/alpha-limitations.md) for details.

## License

The VRhino binary is proprietary and distributed under the
[VRhino Alpha Binary License](licenses/VRHINO-BINARY-LICENSE.txt).
Third-party components remain under their own licenses; see
[THIRD_PARTY_NOTICES.txt](licenses/THIRD_PARTY_NOTICES.txt).

Model licenses are independent. VRhino grants no rights to model weights,
inputs, outputs, or other third-party content.

For MuseTalk, Pixel Rhino distributes no model weights: users acquire the
fixed upstream models and convert them locally. Use remains subject to each
upstream license, including MuseTalk's CreativeML OpenRAIL-M use-based
restrictions. Users are responsible for lawful and consented input media. No
upstream endorsement is implied.
