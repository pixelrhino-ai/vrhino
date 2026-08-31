# LatentSync 1.6 Lip-Sync Diffusion Source and License Notice

The qualified Public package identity is:

`vrhino/latentsync-1.6:1.0.0`

It is a bounded `lip_sync` product using workflow family
`lip_sync_diffusion_workflow_v1`. It accepts a source video and driving audio,
processes aligned faces at 512×512, and writes an MP4 with the driving audio.
The Public-safe face-analysis stack uses MediaPipe BlazeFace Short Range and
DWPose rather than LatentSync's optional InsightFace auxiliary weights.

## Distribution boundary

- Pixel Rhino does not distribute or host original model checkpoints.
- Pixel Rhino does not distribute converted `.vrm` model components.
- `vrhino pull` obtains the exact assets from official upstream sources,
  verifies fixed sizes and SHA256 identities, converts them locally, and
  transactionally installs the immutable package.
- Cached source, converted, and installed artifacts remain local to the user.
- The Public repository contains no qualification media, oracle tensors,
  checkpoints, converted components, or generated videos.

## Required upstream assets

The fixed 12-artifact source plan uses:

- LatentSync 1.6 Temporal Conditional UNet and OpenAI Whisper Tiny checkpoint
  from `ByteDance/LatentSync-1.6` at
  `c42c7e6c8e9c213626389fa7d9a3c444b8536353`;
- SD AutoencoderKL from `stabilityai/sd-vae-ft-mse` at
  `31f26fdeee1355a5c34592e401dd41e45d25a493`;
- MediaPipe BlazeFace Short Range from Google's versioned model URL;
- DWPose `dw-ll_ucoco_384.pth` from `yzd-v/DWPose` at
  `1a7144101628d69ee7a3768d1ee3a094070dc388`;
- the pinned DWPose configuration from `TMElyralab/MuseTalk`;
- pinned LatentSync UNet and scheduler configurations, fixed mask, and Whisper
  mel filters from `bytedance/LatentSync` at
  `a229c3948406bc2cf6eaf4873e662e70c6a04746`; and
- Whisper preprocessing metadata from `openai/whisper-tiny` at
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.

Exact filenames, URLs, byte sizes, and SHA256 values are retained in the
[Public source plan](../../registry/models/vrhino/latentsync-1.6/1.0.0/source-plan.json).

## Use

```bash
vrhino pull vrhino/latentsync-1.6:1.0.0
vrhino info vrhino/latentsync-1.6:1.0.0
vrhino doctor vrhino/latentsync-1.6:1.0.0
vrhino run vrhino/latentsync-1.6:1.0.0 \
  --video input.mp4 \
  --audio driving.wav \
  --output output.mp4
```

`--seed` is optional. The v1 package freezes 25 FPS workflow semantics,
16-frame temporal chunks, 20 DDIM steps, guidance 1.5, eta 0, alignment, mask,
and sampling policy. These are not user-facing tuning controls. Python,
PyTorch, Diffusers, Transformers-Python, MediaPipe, MMPose/MMCV, InsightFace,
and system FFmpeg are not production dependencies.

## Qualification scope

The package was qualified on an NVIDIA GeForce RTX 4090 D with the CUDA
backend. Observed peak internal allocation was approximately 19.9 GiB for the
qualification profile. This is an observation, not a minimum VRAM requirement.
No hard VRAM admission threshold is currently declared.

## Model licenses and responsible use

VRhino supports local acquisition and conversion of the listed upstream models
for a bounded lip-sync diffusion workflow using video and driving audio. Pixel
Rhino does not distribute model weights or converted VRMs.

LatentSync model use remains subject to the CreativeML Open RAIL++-M License
and its Attachment A use-based restrictions. Locally converted model
representations remain subject to those terms; conversion changes
representation, not licensing. Other components remain subject to their
respective MIT or Apache-2.0 terms.

The complete license texts, source mapping, attribution, and representation-
change notice are in the
[LatentSync third-party notices](../../licenses/models/latentsync-1.6/THIRD_PARTY_NOTICES.txt).
No upstream owner endorses VRhino. Users are responsible for lawful and
consented use of input video, faces, images, and audio.
