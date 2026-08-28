# VRhino Doctor v0

`vrhino doctor [MODEL]` produces a concise, privacy-safe local readiness report:

```text
vrhino doctor
vrhino doctor namespace/name:version
```

Without a model, it reports the VRhino version and build, OS and glibc, selected
GPU and NVIDIA Driver/CUDA facts, total and currently available VRAM, cache
filesystem/free space, sanitized network configuration state, and bundled media
helper readiness.

With an exact model identity, it also reports the fixed source and license
identity, installation state, expected and resolved artifact counts, package
component health, selected preset, and declared VRAM admission status.

Doctor is read-only, offline, non-repairing, and non-uploading. It does not run
inference, download data, contact Hugging Face or `hf-mirror.com`, mutate the
cache, or repair damaged packages.

## Privacy

- `HF_TOKEN`, `HTTPS_PROXY`, and `NO_PROXY` are shown only as `set` or
  `not set`.
- Token values, proxy URLs and credentials, prompts, videos, environment dumps,
  and arbitrary filesystem listings are not emitted.
- Default cache paths are abbreviated and custom roots are sanitized.

## Result and exit status

- `READY`: applicable deterministic checks pass; exit status `0`.
- `WARNING`: no deterministic blocker exists, but a non-blocking concern was
  found; exit status `0`.
- `FAILED`: a deterministic device, cache, media, package, or admission blocker
  exists; exit status `1`.
- Actual interruption exits with status `130`.

`READY` means the observed preflight environment appears ready. It is not a
guarantee that numerical inference will complete successfully.
