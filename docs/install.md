# Install VRhino v0.2.1-alpha

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
[v0.2.1-alpha release](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.2.1-alpha):

- `vrhino-linux-x86_64-cuda-v0.2.1-alpha.tar.gz`
- `vrhino-linux-x86_64-cuda-v0.2.1-alpha.tar.gz.sha256`

If both files are in `~/Downloads`, run:

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.2.1-alpha.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-v0.2.1-alpha.tar.gz -C "$HOME/.local/share"
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
```

No `sudo`, repository clone, environment activation, or manual
`LD_LIBRARY_PATH` configuration is required.

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

The other qualified model path is:

```bash
vrhino pull vrhino/wan2.1-t2v-1.3b:1.0.0

vrhino run vrhino/wan2.1-t2v-1.3b:1.0.0 \
  --prompt "a rhinoceros walking through a snowy forest" \
  --output wan-output.mp4
```

The initial LTX pull downloads about 24.77 GB; the Wan source acquisition is
about 16.36 GiB. Allow additional disk space for source data, local conversion,
and the installed runnable package. Supported interrupted HTTP downloads can
resume.

The release contains no model weights. Review the
[upstream LTX model terms](models/ltx-video-v0.9.1.md) before pulling or using
that model, and the [Wan source and license notice](models/wan2.1-t2v-1.3b.md)
before pulling or using Wan.
