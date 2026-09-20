# Wan2.2 alpha example

This example uses the already tested 832x480 / 81-frame / 40-step BF16 setup.
Start with a Linux CUDA host with sufficient resources; the measured host was
an A800 80 GB. The options include a 30 GiB weight cache and sampled resource
limits. Edit prompt/negative_prompt/seed in request.json as needed.

Weights are not included. Supply a package produced from the pinned qualified
source with the native Wan-family converter; arbitrary model revisions will be
rejected. Source pins and input requirements are in
share/vrhino/converters/wan2_2_t2v_a14b/source-contract.json. The converter is
available as bin/vrhino-wan-family-convert. It requires model source, official
semantic source, spec directory and a NEW output directory, in that order.
It streams conversion; do not convert again if you already have the qualified
package. A VRM alone is insufficient: retain the complete converter output,
including conditioning resources, tokenizer and declarations.

Copy this example directory to a writable location. From that copy:

```sh
mkdir -p assets
ln -s /absolute/path/to/converted-package assets/package
ln -s /absolute/path/to/qualified-model-source assets/source
/path/to/runtime/bin/vrhino preflight vrhino-model.json local-resources.json
/path/to/runtime/bin/vrhino evaluate vrhino-model.json local-resources.json request.json options.json ./output
```

The output directory must be new. Relative resource paths resolve against
local-resources.json, not the process working directory. Preflight verifies the
pinned artifact hashes; it does not copy the large model. Conversion metadata
must match this pinned example. Do not edit integrity hashes to accept mismatches.
See supervision.json for completion and evaluation.mp4 for the video.

Wan2.2 uses the explicit evaluate entry in this alpha; ordinary schema2 run is
pending. Numerical alignment and wider configuration coverage remain in progress.
Python is not required for the native execution path.
