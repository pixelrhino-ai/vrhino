# Wan2.2 native video quick start — alpha

[中文](wan2.2-quickstart.zh-CN.md)

Wan2.2 uses the normal VRhino Product path. Python is not needed for conversion or inference.

## Pull

Set `VRHINO_HOME` first if the model cache should live on a large disk. The fixed Hugging Face revision is downloaded, verified, converted and installed automatically:

```bash
vrhino pull vrhino/wan2.2-t2v-a14b:1.0.0
```

The first pull is large. Source inputs are about 127 GB and the generated `model.vrm` is about 115 GB; conversion needs additional temporary space. The pull contract uses only `Wan-AI/Wan2.2-T2V-A14B` at Hugging Face revision `c8c270b13ee05bfa474194ac9fb07a5868a97cea`, plus checksum-pinned semantic files from the official Wan2.2 repository.

## Run

```bash
vrhino run vrhino/wan2.2-t2v-a14b:1.0.0 \
  --prompt "a red panda running through fresh snow" \
  --output wan22.mp4
```

The package supplies the tokenizer, UMT5 conditioning, dual-binding execution program, sampling program, BF16 policy, memory budget, 832×480 geometry, 81 frames and 40 steps. The tested configuration is an A800 80 GB. Actual memory depends on the GPU and other processes.

Wan2.2 runs as an alpha-unqualified package: its Product structure and representative native runs are validated, while generic BF16 reference qualification remains tracked separately as HOLD. This status is shown at run time and does not introduce a model-specific Runtime, Backend, CUDA path or precision exception.

See [tested settings and alpha notes](../release/v0.9-wan2.2-known-limitations.md) for qualification details.
