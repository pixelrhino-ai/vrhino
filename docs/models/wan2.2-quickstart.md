# Wan2.2 native video quick start — v0.9 alpha

[中文](wan2.2-quickstart.zh-CN.md)

Install the [Linux v0.9 package](../install.md) first. This guide uses the shipped
native `evaluate` entry, including tokenizer, UMT5, denoising, VAE and MP4 output.
No Python inference environment is needed. Ordinary schema2 `run` is pending.

## 1. Prepare the model package

The tested setup is A800 80 GB, BF16, 832×480, 81 frames and 40 steps.
The example sets a 30 GiB weight-cache budget. Observed peak device usage was
about 44.8 GiB; this is not a guarantee that a smaller GPU will fit. Keep enough
host memory and disk for the source, conversion and outputs: `model.vrm` alone
is 114,816,482,368 bytes, and the complete package contains additional resources.

Use the qualified source revisions:

- Model files: [Wan-AI/Wan2.2-T2V-A14B on ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B),
  revision `3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7`.
- Official semantic source: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2),
  revision `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.

Obtain those revisions using the providers' download facilities and retain the
original files and directory layout. The converter checks the bundled source
contract before accepting weights. Core content also matches HF revision
`c8c270b13ee05bfa474194ac9fb07a5868a97cea`, but the whole directories are not
interchangeable: README/configuration metadata differ. Do not replace files or
edit integrity hashes to bypass checks. Review the upstream model terms.

If you already have the qualified converter output, reuse the **complete package**
and skip conversion. A standalone VRM is not enough: conditioning, tokenizer,
programs and provenance files are required. Otherwise, with absolute local paths:

```bash
VRHINO_ROOT="$HOME/.local/share/vrhino-v0.9.0-alpha"
MODEL_SOURCE=/absolute/path/to/Wan2.2-T2V-A14B
SEMANTIC_SOURCE=/absolute/path/to/Wan2.2-official-source
CONVERTED_PACKAGE=/absolute/path/to/new-converted-package

"$VRHINO_ROOT/bin/vrhino-wan-family-convert" \
  "$MODEL_SOURCE" "$SEMANTIC_SOURCE" \
  "$VRHINO_ROOT/share/vrhino/converters/wan2_2_t2v_a14b" \
  "$CONVERTED_PACKAGE"
```

The output directory must be new. Conversion streams the large model; it does not
run the official Python pipeline. The source contract shipped beside the spec
lists the exact required files. There is no published Wan2.2 `vrhino pull` alias
in this alpha; the local package example is the supported entry described here.

## 2. Copy and connect the example

Set `VRHINO_ROOT`, `MODEL_SOURCE` and `CONVERTED_PACKAGE` as above, including when
reusing an existing package. Use a fresh working directory:

```bash
cp -a "$VRHINO_ROOT/share/vrhino/examples/wan2.2" "$HOME/wan22-first-run"
cd "$HOME/wan22-first-run"
mkdir -p assets
ln -s "$CONVERTED_PACKAGE" assets/package
ln -s "$MODEL_SOURCE" assets/source
"$VRHINO_ROOT/bin/vrhino" preflight vrhino-model.json local-resources.json
```

Preflight validates identities and wiring without inference. Hash verification
reads the large artifacts and can take several minutes. Relative resource paths
are resolved against `local-resources.json`. Keep declarations and integrity pins
unchanged; a mismatched package should be corrected at its source.

## 3. Generate a video

Edit `request.json` to set `prompt`, `negative_prompt` and `seed`. For the first
run keep the shipped geometry and `options.json`, including its 30 GiB cache and
40-step schedule. Use an otherwise idle GPU; resource monitoring includes usage
from unrelated GPU processes. Then run:

```bash
"$VRHINO_ROOT/bin/vrhino" evaluate \
  vrhino-model.json local-resources.json request.json options.json ./output
```

`output` must not already exist. Successful completion produces
`output/evaluation.mp4`; inspect `output/supervision.json` for completion status.
A successful run is recorded as `EXECUTED_UNQUALIFIED`: execution completed, while
reference numerical qualification remains separate. It is not itself an error.
If interrupted, retain the logs and use a new output directory for another run.

See [tested settings and alpha notes](../release/v0.9-wan2.2-known-limitations.md)
and the [evaluation command reference](../product/bounded-evaluation.md).
