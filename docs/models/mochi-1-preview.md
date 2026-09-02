# Mochi 1 Preview Source and License Notice

The qualified VRhino Public Alpha model path is:

`vrhino/mochi-1-preview:1.0.1`

## Distribution boundary

- The VRhino binary archive contains no Mochi model weights.
- Pixel Rhino does not distribute converted Mochi `.vrm` weights in this
  Alpha.
- `vrhino pull` obtains original artifacts from the fixed official upstream
  revision, with the documented availability fallback behavior.
- Native conversion and runnable-package installation occur locally on the
  user's machine.
- The installed model is executed by VRhino's Shared Native Runtime.

## Exact upstream source

- Provider: Hugging Face
- Repository: `genmo/mochi-1-preview`
- Revision: `14be5fcea23095ed330cb214647916a451e38b6e`
- License: Apache License 2.0, as declared by the model card at this fixed
  revision
- Model-card and license reference:
  <https://huggingface.co/genmo/mochi-1-preview/blob/14be5fcea23095ed330cb214647916a451e38b6e/README.md>

The fixed source plan contains 20 required artifacts totaling approximately
37.28 GiB. Actual network transfer can be lower when verified content is
already present in the local content-addressed cache. After verified
installation, VRhino reclaims source-only Mochi data while retaining artifacts
required by the runnable package.

## Qualified default profile

The successor package uses `vrhino.product.input-schema.v1`: `prompt` is
required; `seed` is optional with default `11001`; and `output` is optional
with default `output.mp4`. Its frozen Product profile is 848×480, 163 frames at
30 FPS, 64 sampling steps, and guidance 6.

The preset is admitted only when VRhino's planner sees at least 80 GiB of
available device memory. This is the current product support and admission
threshold for the qualified preset, not an empirically proven absolute minimum
VRAM requirement.

## Alpha scope

This release qualifies only the exact model package, upstream revision, and
default profile above on the current Linux x86_64 NVIDIA CUDA platform. It does
not claim support for every Mochi checkpoint, every NVIDIA GPU, or every Linux
distribution.

Review and comply with the upstream license before downloading, converting, or
using the model. The VRhino binary license grants no model, training-data,
input-content, output-content, or other upstream rights.
