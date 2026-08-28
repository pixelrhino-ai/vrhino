# Wan2.1 T2V 1.3B Source and License Notice

The qualified VRhino Public Alpha model path is:

`vrhino/wan2.1-t2v-1.3b:1.0.0`

## Distribution boundary

- The VRhino binary archive contains no Wan model weights.
- Pixel Rhino does not distribute converted Wan `.vrm` weights in this Alpha.
- `vrhino pull` downloads original artifacts directly from the fixed official
  upstream source.
- Native conversion and runnable-package installation occur locally on the
  user's machine.
- Cached source, converted, and installed model artifacts remain local to the
  user.

## Exact upstream source

- Provider: Hugging Face
- Repository: `Wan-AI/Wan2.1-T2V-1.3B`
- Revision: `37ec512624d61f7aa208f7ea8140a131f93afc9a`
- License: Apache License 2.0
- License reference:
  <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/37ec512624d61f7aa208f7ea8140a131f93afc9a/LICENSE.txt>

## Qualified default profile

The current default profile is 832×480, 81 frames at 16 FPS, 50 sampling
steps, CFG 5, flow shift 5, seed 5701, and Flow UniPC order 2.

One qualification run observed a peak device allocation of approximately
13.06 GiB. This is an observed result on the qualification hardware, not a
minimum VRAM requirement or a guarantee for other hardware and workloads.

## Alpha scope

This release qualifies only the exact model package, upstream revision, and
default profile above on the current Linux x86_64 NVIDIA CUDA platform. It does
not claim support for Wan 14B, Wan2.2, every Wan checkpoint, every NVIDIA GPU,
or every Linux distribution. An exact minimum VRAM requirement has not been
qualified.

Review and comply with the upstream license before downloading, converting, or
using the model. The VRhino binary license grants no model, training-data,
input-content, output-content, or other upstream rights.
