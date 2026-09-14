# Install VRhino

## Release status

The current source version is **v0.8.0-alpha, unreleased**. It prepares native
Windows CUDA support, pending a fresh rc7 and four-path clean-host qualification.
The latest downloadable release is still
[v0.7.0-alpha](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.7.0-alpha),
which remains Linux-scoped and immutable. Do not rename a historical Windows
rc6 archive as v0.8 or attach it to that release.
See the [v0.8 preparation status](release/v0.8.0-alpha.md).

## Requirements

- Published Linux package: x86_64, glibc 2.35 or newer
- Planned Windows package: native Windows x64 CUDA; qualification baseline
  Windows 10 Pro 22H2 / build 19045, RTX 3090 24 GiB, NVIDIA driver 610.47
- compatible NVIDIA GPU and NVIDIA Driver
- sufficient GPU VRAM and disk space
- network access for the initial model pull

Historical Linux packages were qualified on Ubuntu 22.04. Frozen Windows rc6
passed Wan and LTX on the stated Windows baseline. Exact v0.8 rc7 qualification
for Wan, LTX, MuseTalk and LatentSync remains pending. The baseline driver is
not a qualified minimum; universal Windows/GPU coverage and macOS are not claimed.

You do not install a CUDA Toolkit, standalone cuDNN, Python, PyTorch, Diffusers,
Transformers, Conda, system FFmpeg/x264, system curl, Visual Studio/Build Tools,
CMake, Ninja, MSYS2/MinGW or WSL for the Windows product. VRhino
bundles the required user-space runtime and media components. The host NVIDIA
Driver remains required.

## Published Linux download

Download these two files from the
[v0.7.0-alpha release](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.7.0-alpha):

- `vrhino-linux-x86_64-cuda-v0.7.0-alpha-final-candidate-5a07d09.tar.gz`
- `vrhino-linux-x86_64-cuda-v0.7.0-alpha-final-candidate-5a07d09.tar.gz.sha256`

If both files are in `~/Downloads`, run:

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.7.0-alpha-final-candidate-5a07d09.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-v0.7.0-alpha-final-candidate-5a07d09.tar.gz -C "$HOME/.local/share"
printf '\nexport PATH="$HOME/.local/share/vrhino/bin:$PATH"\n' >> "$HOME/.profile"
export PATH="$HOME/.local/share/vrhino/bin:$PATH"
```

If the files are elsewhere, change to that directory before running the
checksum and extraction commands.

The full installation remains under `~/.local/share/vrhino`. Do not copy only
`bin/vrhino`: the launcher resolves the bundled libraries and media encoder
relative to the intact installation directory.

Verify from any directory:

```bash
vrhino --version
vrhino device
vrhino doctor
```

No `sudo`, repository clone, environment activation, or manual
`LD_LIBRARY_PATH` configuration is required.

## Planned Windows ZIP installation

There is no published v0.8 rc7 download yet. After qualification and a separate
release decision, use the exact Windows ZIP and checksum identified by that
release. The planned candidate name is
`vrhino-windows-x86_64-cuda-v0.8.0-alpha-<commit7>-rc7.zip`; `<commit7>` is a
placeholder for the post-version-merge source commit, not an available asset.

Compare `Get-FileHash -LiteralPath <downloaded-ZIP> -Algorithm SHA256` with
the published checksum before extracting it into a fresh directory. Keep
`vrhino.exe`, `vrhino-ffmpeg.exe`, all 39 application-local DLLs, `share/`,
licenses/notices, provenance and the rest of the package together. Do not copy
only the executable or add system CUDA/cuDNN/FFmpeg paths.

From PowerShell in the extracted package directory, the startup commands are:

```powershell
.\vrhino.exe --version
.\vrhino.exe device
.\vrhino.exe doctor
```

The new package must report `v0.8.0-alpha`. These examples also use native
PowerShell syntax (replace media paths with your own files):

```powershell
.\vrhino.exe pull vrhino/wan2.1-t2v-1.3b:1.0.1
.\vrhino.exe run vrhino/wan2.1-t2v-1.3b:1.0.1 --prompt "a rhinoceros walking through a snowy forest" --seed 5701 --output wan-output.mp4
.\vrhino.exe pull vrhino/musetalk-v1.5:1.0.1
.\vrhino.exe run vrhino/musetalk-v1.5:1.0.1 --video input.mp4 --audio driving.wav --output lipsync-output.mp4
```

The helper is discovered relative to the installation; no system FFmpeg is
needed. From another working directory, invoke the executable by its full
path using PowerShell's `&` operator. No repository clone, developer tools or
environment activation is part of installation. A doctor warning solely for
undeclared minimum VRAM does not replace actual device/component/media checks.

## Native API

The Native API introduced in v0.6.0-alpha is built into the same primary `vrhino`
executable. It requires no separate server package and no Python, FastAPI, or
Node runtime. Start the foreground local server with:

```bash
vrhino serve
# Equivalent explicit form:
vrhino serve --host 127.0.0.1 --port 11435
```

The complete command shape is:

```text
vrhino [--cache-root PATH] serve [--host HOST] [--port PORT]
```

The default is `127.0.0.1:11435`. Native API v1 alpha has no authentication,
TLS, or CORS and is not intended for direct exposure to the untrusted
Internet. See the [normative Native API contract](api/native-api-v1.md).

The exact successor package identities introduced in v0.6.0-alpha remain:

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

This list is not a claim that all five models were qualified on Windows.
The [platform matrix](release/v0.8.0-alpha.md) separates four pending rc7 paths
from Mochi's retained historical Linux scope and 80 GiB admission requirement.

If installation, pull, or run readiness is unclear, use `vrhino doctor` for a
privacy-safe local report, or `vrhino doctor MODEL` to include installed
package health and preset admission. Doctor is read-only and offline: it does
not upload telemetry, make an automatic network diagnostic request, repair
packages, or reveal token/proxy credential values. See
[`doctor` diagnostics](cli/doctor-v0.md).

## Model and cache location

The VRhino binary installation and model storage are separate. On Linux the
release tree may remain under `~/.local/share/vrhino`, while model/cache data
defaults to `~/.vrhino`. On Windows use an explicit `--cache-root` to select
your intended model storage directory independently of the extracted ZIP.

```powershell
.\vrhino.exe --cache-root C:\VRhino-Models pull vrhino/ltx-video-v0.9.1:1.1.1
```

Set `VRHINO_HOME` before pulling to place model/cache data on a larger
filesystem:

```bash
export VRHINO_HOME=/mnt/large-disk/vrhino
vrhino pull vrhino/ltx-video-v0.9.1:1.1.1
```

An explicit `--cache-root PATH` overrides `VRHINO_HOME`; otherwise
`VRHINO_HOME` overrides the `~/.vrhino` default.

Public Hugging Face downloads work anonymously. VRhino prefers the official
Hugging Face endpoint. If that path has a qualifying availability failure,
VRhino may transparently retry through `https://hf-mirror.com`, a third-party
service. Selection is based on observed availability, not GeoIP or location,
and no `HF_ENDPOINT` configuration is required. The native HTTPS stack honors
the standard `HTTPS_PROXY` and `NO_PROXY` environment variables for both
transports.

If an authenticated official request is needed, set `HF_TOKEN`. VRhino may
send that credential to official Hugging Face, but never automatically
forwards it to the third-party mirror or a cross-host redirect. Credentials
are not printed or persisted. Model identity remains fixed by repository,
revision, artifact path, expected size, and SHA256; mirror bytes must pass the
same exact checks before entering the cache.

## Linux shell pull and run examples

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.1

vrhino run vrhino/ltx-video-v0.9.1:1.1.1 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

Another qualified model path is:

```bash
vrhino pull vrhino/wan2.1-t2v-1.3b:1.0.1

vrhino run vrhino/wan2.1-t2v-1.3b:1.0.1 \
  --prompt "a rhinoceros walking through a snowy forest" \
  --output wan-output.mp4
```

The historically qualified Linux Mochi path is below. Its frozen requirement
of at least 80 GiB available device memory excludes the 24 GiB Windows
qualification host; do not lower that admission or interpret exclusion as failure.

```bash
vrhino pull vrhino/mochi-1-preview:1.0.1

vrhino run vrhino/mochi-1-preview:1.0.1 \
  --prompt "a red panda runs through snow." \
  --output mochi-output.mp4
```

The initial LTX pull downloads about 24.77 GB; the Wan source acquisition is
about 16.36 GiB; and the Mochi source acquisition is about 37.28 GiB. Allow
additional disk space for source data, local conversion, and the installed
runnable package. Supported interrupted HTTP downloads can resume.

The release contains no model weights. Review the
[upstream LTX model terms](models/ltx-video-v0.9.1.md) before pulling or using
that model, the [Wan source and license notice](models/wan2.1-t2v-1.3b.md)
before pulling or using Wan, and the
[Mochi source and license notice](models/mochi-1-preview.md) before pulling or
using Mochi.
