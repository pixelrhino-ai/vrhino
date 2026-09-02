# Install VRhino

## Release status

The current downloadable release is **v0.5.0-alpha**. The Public source
contract for **v0.6.0-alpha** is frozen, but its release archive and checksum
have not been produced. The download instructions below therefore remain the
authoritative instructions for the published v0.5.0-alpha artifact; no
v0.6.0-alpha asset identity is implied by this document.

## Requirements

- Linux x86_64
- glibc 2.35 or newer
- compatible NVIDIA GPU and NVIDIA Driver
- sufficient GPU VRAM and disk space
- network access for the initial model pull

VRhino was qualified on Ubuntu 22.04. An exact minimum NVIDIA Driver version
has not yet been qualified.

You do not install a CUDA Toolkit, cuDNN, Python, PyTorch, Diffusers,
Transformers, Conda, system FFmpeg/x264, system curl, or build tools. VRhino
bundles the required user-space runtime and media components. The host NVIDIA
Driver remains required.

## Download and install

Download these two files from the
[v0.5.0-alpha release](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.5.0-alpha):

- `vrhino-linux-x86_64-cuda-v0.5.0-alpha.tar.gz`
- `vrhino-linux-x86_64-cuda-v0.5.0-alpha.tar.gz.sha256`

If both files are in `~/Downloads`, run:

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.5.0-alpha.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-v0.5.0-alpha.tar.gz -C "$HOME/.local/share"
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

## Native API in v0.6.0-alpha

The v0.6.0-alpha Native API server is built into the same primary `vrhino`
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

The exact successor package references intended for v0.6.0-alpha are:

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

If installation, pull, or run readiness is unclear, use `vrhino doctor` for a
privacy-safe local report, or `vrhino doctor MODEL` to include installed
package health and preset admission. Doctor is read-only and offline: it does
not upload telemetry, make an automatic network diagnostic request, repair
packages, or reveal token/proxy credential values. See
[`doctor` diagnostics](cli/doctor-v0.md).

## Model and cache location

The VRhino binary installation and model storage are separate. The extracted
release tree may remain under `~/.local/share/vrhino`, while models, downloads,
and cache data default to `~/.vrhino`.

Set `VRHINO_HOME` before pulling to place model/cache data on a larger
filesystem:

```bash
export VRHINO_HOME=/mnt/large-disk/vrhino
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
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

## Pull and run

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0

vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

Another qualified model path is:

```bash
vrhino pull vrhino/wan2.1-t2v-1.3b:1.0.0

vrhino run vrhino/wan2.1-t2v-1.3b:1.0.0 \
  --prompt "a rhinoceros walking through a snowy forest" \
  --output wan-output.mp4
```

The qualified Mochi path is:

```bash
vrhino pull vrhino/mochi-1-preview:1.0.0

vrhino run vrhino/mochi-1-preview:1.0.0 \
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
