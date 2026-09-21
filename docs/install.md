# Install VRhino

## Release status

**Linux: [v0.9.0-alpha](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.9.0-alpha)**.
**Windows: v0.8.0-alpha** (existing qualified rc9 package).
See the [v0.9 release and checksum record](release/v0.9.0-alpha.md) and the
[v0.8 Windows qualification record](release/v0.8.0-alpha.md).
Older release assets remain unchanged; v0.9 Windows binaries are not published.

## Requirements

- Linux v0.9 package: x86_64, glibc 2.35 or newer; CUDA 12.8 / SM80 build, tested on A800 80 GB
- Windows package: native Windows x64 CUDA; qualification baseline
  Windows 10 Pro 22H2 / build 19045, RTX 3090 24 GiB, NVIDIA driver 610.47
- compatible NVIDIA GPU and NVIDIA Driver
- sufficient GPU VRAM and disk space
- network access for the initial model pull

The historical v0.8 Linux release and Windows rc9 clean-host qualification passed. Wan,
LTX, MuseTalk, and LatentSync passed on the Windows baseline. Mochi remote
import, installation, doctor, and admission passed, while full inference was
not run on the 24 GiB host because its unchanged admission requires
85,899,345,920 bytes of available device memory. The baseline driver is not a
qualified minimum; universal Windows/GPU coverage and macOS are not claimed.

You do not install a CUDA Toolkit, standalone cuDNN, Python, PyTorch, Diffusers,
Transformers, Conda, system FFmpeg/x264, system curl, Visual Studio/Build Tools,
CMake, Ninja, MSYS2/MinGW or WSL for the Windows product. VRhino
bundles the required user-space runtime and media components. The host NVIDIA
Driver remains required.

## Linux one-command installation

```bash
curl -fsSL https://raw.githubusercontent.com/pixelrhino-ai/vrhino/main/install.sh | sh
```

The POSIX shell installer supports Linux x86_64 with glibc 2.35 or newer. It uses
curl, tar and standard Linux command-line tools, without sudo, Python, a model
download or GPU execution. The current script selects the published v0.9 alpha
explicitly (GitHub's `latest` release can still point to the older stable entry).

It verifies a SHA256 pinned in the script before extracting, checks the package's
`SHA256SUMS`, and tests `--version` before activating the installation. Downloads
use HTTPS and retry transient failures. Interrupted installs can be rerun.
The package is about 1.45 GB compressed / 2.1 GB extracted; allow at least 4 GB
free for staging, plus separate model storage.

Default locations:

- Package: `${XDG_DATA_HOME:-$HOME/.local/share}/vrhino-v0.9.0-alpha`
- Commands: `~/.local/bin/vrhino` and `~/.local/bin/vrhino-wan-family-convert`

The installer adds PATH entries to `.profile` and applicable Bash/Zsh profiles
without duplicating them. A child shell cannot change the current terminal's
PATH, so follow the printed `export PATH=...` instruction or open a new shell.
It does not overwrite unrelated executables or an existing manually installed
package. Repeating installation verifies and reuses its own intact package.
An install lock prevents two installers using the same prefix concurrently.

To inspect the script first or choose locations:

```bash
curl -fsSL https://raw.githubusercontent.com/pixelrhino-ai/vrhino/main/install.sh -o install.sh
sh install.sh --help
sh install.sh --prefix "$HOME/apps/vrhino-v0.9.0-alpha" --bin-dir "$HOME/.local/bin"
```

`--no-modify-path` leaves shell profiles unchanged. `--archive /path/to/archive`
uses an already downloaded archive with the same pinned SHA256; it does not
accept arbitrary builds. `VRHINO_INSTALL_DIR` and `VRHINO_BIN_DIR` are equivalent
to the corresponding location options when set in the installer environment.
When using a custom package location, use that location as `VRHINO_ROOT` in the
Wan2.2 quick start. Existing installations made by manual extraction should use
their existing commands or choose a fresh prefix rather than overwrite them.

The installer prints progress and exits nonzero on failure. It does not install
an NVIDIA driver or establish model/hardware qualification; run `vrhino doctor`
after installation. For manual installation, use the instructions below.

## Linux download

Download these two files from the
[v0.9.0-alpha release](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.9.0-alpha):

- [Linux CUDA archive](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.9.0-alpha/vrhino-linux-x86_64-cuda-v0.9.0-alpha-candidate.tar.gz)
- [SHA256 file](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.9.0-alpha/vrhino-linux-x86_64-cuda-v0.9.0-alpha-candidate.tar.gz.sha256)

The published filenames retain `candidate`; these are the released alpha assets.
The archive is 1,449,009,466 bytes. Exact hashes are in the
[release record](release/v0.9.0-alpha.md). Model weights are separate.

If both files are in `~/Downloads`, run:

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.9.0-alpha-candidate.tar.gz.sha256
mkdir -p "$HOME/.local/share"
# Extract into a fresh directory; do not overlay an older installation.
tar -xzf vrhino-linux-x86_64-cuda-v0.9.0-alpha-candidate.tar.gz -C "$HOME/.local/share"
export PATH="$HOME/.local/share/vrhino-v0.9.0-alpha/bin:$PATH"
```

Add the same `export PATH=...` line to your shell profile if desired.
Keep the entire `~/.local/share/vrhino-v0.9.0-alpha` directory intact: the launcher
resolves bundled libraries and the media encoder relative to it.

Verify from any directory:

```bash
vrhino --version   # v0.9.0-alpha
vrhino device
vrhino doctor
```

For Wan2.2, continue with the [quick start](models/wan2.2-quickstart.md)
([中文](models/wan2.2-quickstart.zh-CN.md)). It uses `evaluate` with a local
qualified package; the schema1 `pull` / `run` examples below remain separate.

No `sudo`, repository clone, environment activation, or manual
`LD_LIBRARY_PATH` configuration is required.

## Windows ZIP installation

Download these two files from the v0.8.0-alpha release:

- `vrhino-windows-x86_64-cuda-v0.8.0-alpha-d460de0-rc9.zip`
- `vrhino-windows-x86_64-cuda-v0.8.0-alpha-d460de0-rc9.zip.sha256`

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

The package reports `v0.8.0-alpha`. These examples use native
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

The [platform matrix](release/v0.8.0-alpha.md) records the exact qualification
scope for these five paths. No model weights or converted VRMs are in either
release archive.

If installation, pull, or run readiness is unclear, use `vrhino doctor` for a
privacy-safe local report, or `vrhino doctor MODEL` to include installed
package health and preset admission. Doctor is read-only and offline: it does
not upload telemetry, make an automatic network diagnostic request, repair
packages, or reveal token/proxy credential values. See
[`doctor` diagnostics](cli/doctor-v0.md).

## Model and cache location

The VRhino binary installation and model storage are separate. On Linux the
v0.9 release tree may remain under `~/.local/share/vrhino-v0.9.0-alpha`, while model/cache data
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
