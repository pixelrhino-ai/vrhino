Describe the problem and resulting behavior.

Validation performed (host tests, synthetic fixtures, and CUDA qualification when applicable):

Target Public `main` from a topic branch and wait for the required `Linux host and offline tokenizer build` check before squash merging. See [CONTRIBUTING.md](https://github.com/pixelrhino-ai/vrhino/blob/main/CONTRIBUTING.md). Report unavailable local validation accurately.

For Runtime/backend/precision changes, explain the generic need and how bounded graph semantics and model-neutral policy are preserved. Keep private media, weights, evidence, and generated builds outside Git. Source presence does not establish released model support.

If promoting private research, record its Public base SHA and submit only the clean production diff. Published release tags and assets are immutable; fixes require a new version.
