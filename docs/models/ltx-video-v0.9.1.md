# LTX-Video v0.9.1 Source and License Notice

VRhino Public Alpha's first qualified model path is:

`vrhino/ltx-video-v0.9.1:1.1.1`

## Product contract

The successor package uses `vrhino.product.input-schema.v1`: `prompt` is
required; `seed` is optional with default `5703`; and `output` is optional with
default `output.mp4`. Its frozen profile is 704×480, 121 frames at 25 FPS,
40 sampling steps, and guidance 3. These frozen facts are not additional
request parameters.

## Distribution boundary

- The VRhino binary archive contains no LTX model weights.
- Pixel Rhino does not distribute converted LTX `.vrm` weights in this Alpha.
- `vrhino pull` downloads original artifacts directly from the fixed upstream
  source.
- Native conversion and runnable-package installation occur locally on the
  user's machine.
- Cached source, converted, and installed model artifacts remain local to the
  user.

## Exact upstream source

- Provider: Hugging Face
- Repository: `Lightricks/LTX-Video`
- Revision: `8984fa25007f376c1a299016d0957a37a2f797bb`
- Model artifact: `ltx-video-2b-v0.9.1.safetensors`
- Version-specific license artifact: `ltx-video-2b-v0.9.1.license.txt`
- License reference:
  <https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/ltx-video-2b-v0.9.1.license.txt>

## User responsibility

Review and comply with the upstream LTX v0.9.1 license before downloading,
converting, or using the model. The VRhino binary license grants no model,
training-data, input-content, output-content, or other upstream rights.

The source repository, immutable revision, license identifier, license
artifact, and provenance are also retained in the VRhino model metadata and
installed runnable package.
