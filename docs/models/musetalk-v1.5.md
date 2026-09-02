# MuseTalk v1.5 Lip-Sync Source and License Notice

The qualified public package identity is:

`vrhino/musetalk-v1.5:1.0.1`

It is a bounded `lip_sync` product using workflow family
`lip_sync_workflow_v1`. It accepts a source video and driving audio and writes
an MP4 with the driving audio. The public stack uses MediaPipe BlazeFace,
DWPose, and MediaPipe Selfie Multiclass for face analysis and masking.

## Distribution boundary

- Pixel Rhino does not distribute or host original model checkpoints.
- Pixel Rhino does not distribute converted `.vrm` model components.
- `vrhino pull` obtains the exact model assets from their official upstream
  sources, verifies fixed sizes and SHA256 identities, converts them locally,
  and transactionally installs the immutable package.
- Cached source, converted, and installed artifacts remain local to the user.
- The release and public repository contain no MuseTalk test media,
  qualification media, oracle tensors, or generated videos.

## Required upstream models

The fixed source plan uses:

- MuseTalk v1.5 UNet from `TMElyralab/MuseTalk` at
  `3ef28bc5cff08c90ad8178a25f1b570cd800170f`;
- Whisper Tiny from `openai/whisper-tiny` at
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`;
- SD AutoencoderKL from `stabilityai/sd-vae-ft-mse` at
  `31f26fdeee1355a5c34592e401dd41e45d25a493`;
- MediaPipe BlazeFace Short Range from Google's versioned model URL;
- DWPose `dw-ll_ucoco_384.pth` from `yzd-v/DWPose` at
  `1a7144101628d69ee7a3768d1ee3a094070dc388`; and
- MediaPipe Selfie Multiclass 256x256 from Google's versioned model URL.

The exact filenames, URLs, byte sizes and SHA256 values are retained in the
[public source plan](../../registry/models/vrhino/musetalk-v1.5/1.0.1/source-plan.json).

## Use

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.1
vrhino info vrhino/musetalk-v1.5:1.0.1
vrhino doctor vrhino/musetalk-v1.5:1.0.1
vrhino run vrhino/musetalk-v1.5:1.0.1 \
  --video input.mp4 \
  --audio driving.wav \
  --output output.mp4
```

The Product contract requires `video` and `audio`, with exact 25 FPS video and
at least 40 ms of decodable audio. `seed` is optional with default `11001`;
`output` is optional with default `output.mp4`; duration is audio-derived.
Internal detector, geometry, audio-padding, alignment, masking, TTA, tensor,
and RNG controls are frozen by the workflow and are not Product parameters.
Python, PyTorch, TensorFlow, MediaPipe, Diffusers, MMPose/MMCV and system
FFmpeg are not production dependencies.

## Qualification scope

The package was qualified on an NVIDIA GeForce RTX 4090 D with the CUDA
backend. Metal implementations of generic primitives used by this stack are
present but were not numerically qualified for this package. One full-sample
qualification observed approximately 6.27 GiB peak internal device allocation;
this is an observation, not a minimum VRAM requirement.

## Model licenses and responsible use

VRhino supports local acquisition and conversion of the listed upstream
models. Pixel Rhino does not distribute model weights. Use remains subject to
each upstream model license, including MuseTalk's CreativeML OpenRAIL-M
use-based restrictions. A locally converted model remains subject to those
terms; conversion changes representation, not licensing.

The other model terms include MIT and Apache-2.0 licenses as itemized in the
[MuseTalk third-party notices](../../licenses/models/musetalk-v1.5/THIRD_PARTY_NOTICES.txt).
No upstream owner endorses VRhino. Users are responsible for lawful and
consented use of input video, faces, images, and audio.
