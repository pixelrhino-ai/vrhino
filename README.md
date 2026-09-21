<p align="center">
  <img src="assets/branding/banner.png" alt="VRhino" width="960">
</p>

# VRhino

**Run AI video models locally with a shared native runtime — no per-model Python environment.**

English | [简体中文](README.zh-CN.md)

## Install

Linux x86_64 — install **v0.9.0-alpha** with one command (no sudo):

```bash
curl -fsSL https://raw.githubusercontent.com/pixelrhino-ai/vrhino/main/install.sh | sh
```

The installer downloads and verifies the package, then configures your command
path. Follow its final PATH instruction for the current terminal, then run:

```bash
vrhino doctor
```

[Manual download / installation options](docs/install.md) ·
[Windows v0.8 CUDA package](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.8.0-alpha)

A compatible NVIDIA GPU/driver and sufficient memory are required. Model weights
are separate; native execution needs no Python or CUDA Toolkit installation.

## Available models

| Model | Task | Entry |
|---|---|---|
| **Wan2.2 T2V A14B** | Text to video | Linux v0.9 · `evaluate` · [setup](docs/models/wan2.2-quickstart.md) |
| Wan2.1 T2V 1.3B | Text to video | `pull` / `run` · `vrhino/wan2.1-t2v-1.3b:1.0.1` |
| LTX-Video 0.9.1 | Text to video | `pull` / `run` · `vrhino/ltx-video-v0.9.1:1.1.1` |
| Mochi 1 Preview | Text to video | `pull` / `run` · `vrhino/mochi-1-preview:1.0.1` |
| MuseTalk 1.5 | Lip sync | `pull` / `run` · `vrhino/musetalk-v1.5:1.0.1` |
| LatentSync 1.6 | Lip sync | `pull` / `run` · `vrhino/latentsync-1.6:1.0.1` |

Hardware requirements and tested platforms vary by model; see the
[installation and model notes](docs/install.md). VRhino is an alpha project.

## Generate a video

```bash
vrhino pull vrhino/wan2.1-t2v-1.3b:1.0.1
vrhino run vrhino/wan2.1-t2v-1.3b:1.0.1 \
  --prompt "a red panda running through fresh snow" --output video.mp4
```

The first `pull` downloads, verifies and converts the model into a local package.
To use a larger model/cache disk, set `VRHINO_HOME` before pulling.

### Wan2.2

Follow the [Wan2.2 quick start](docs/models/wan2.2-quickstart.md) to prepare the
model and copy the bundled example. From that example's working directory:

```bash
vrhino preflight vrhino-model.json local-resources.json
vrhino evaluate vrhino-model.json local-resources.json request.json options.json ./output
```

Edit the prompt in `request.json`; the video is saved as `output/evaluation.mp4`.
Use a new output directory. Wan2.2 currently uses `evaluate`, rather than ordinary
`run`. See [tested settings and alpha notes](docs/release/v0.9-wan2.2-known-limitations.md).

### Lip sync

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.1
vrhino run vrhino/musetalk-v1.5:1.0.1 \
  --video input.mp4 --audio driving.wav --output lipsync.mp4
```

LatentSync uses the same video/audio command shape with its package ID above.

## More

- [Installation and cache configuration](docs/install.md)
- [Architecture](docs/architecture.md) · [Build from source](docs/source-build.md)
- [Local Native API — v0.6.0-alpha contract](docs/api/native-api-v1.md) (`vrhino serve`)
- [v0.9 release and checksums](docs/release/v0.9.0-alpha.md) · [Windows v0.8 scope](docs/release/v0.8.0-alpha.md)
- [Contributing](CONTRIBUTING.md)

Project-owned source: [Apache-2.0](LICENSE). Distributed binaries retain their
[binary license](licenses/VRHINO-BINARY-LICENSE.txt); dependencies and models retain
their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and
[model license notes](docs/install.md#linux-shell-pull-and-run-examples).
