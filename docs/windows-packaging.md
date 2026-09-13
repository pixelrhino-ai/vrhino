# Windows CUDA source-build candidate

`tools/package_windows_build.py` is a Python 3.10+ standard-library **build tool**.
Python is not part of the Windows product or its runtime requirements. The Linux
assembler remains separate and unchanged. This tool produces an unpublished ZIP
candidate; it does not establish clean-machine or real-model qualification.

The Windows layout has `vrhino.exe`, `vrhino-ffmpeg.exe` and the explicit 39-DLL
allowlist at its root. All 83 converter/spec resources are read as canonical Git
blobs from the requested commit into `share/vrhino/converters`. It also contains `VERSION`,
`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, consolidated `licenses/`, required
corresponding sources under `sources/`, `SOURCE-BUILD.json`, and `SHA256SUMS`.
No model weights, caches, videos, test executables or build logs are copied.

## Inputs

Build the native `vrhino` CMake target from the intended commit using Release,
MSVC x64, CUDA enabled, native tokenizers and product CLI enabled, and SM86.
Record the actual toolchain and build configuration in a build-info JSON file:

```json
{
  "source_commit": "<full git SHA>",
  "executable_sha256": "<built vrhino.exe SHA256>",
  "configuration": {
    "build_type": "Release",
    "cuda": true,
    "tokenizer": true,
    "product_cli": true,
    "gpu_architectures": ["86"],
    "cuda_runtime_linkage": "Shared"
  },
  "toolchain": {
    "msvc": "<exact version>",
    "msvc_toolset": "<exact version>",
    "cmake": "<exact version>",
    "ninja": "<exact version>",
    "rust": "<exact version>",
    "cuda_toolkit": "<exact version>",
    "nvcc": "<exact version>",
    "cudnn": "<exact version>"
  }
}
```

Supply a separately audited dependency directory. It must contain exactly the
40 runtime files (helper plus 39 DLLs), explicit license/source files, and
`dependency-manifest.json`. No implicit search of Toolkit, Visual Studio, Conda,
MSYS or PATH locations occurs. The manifest schema is:

```json
{
  "schema_version": 1,
  "runtime": [
    {
      "path": "<allowlisted DLL or vrhino-ffmpeg.exe>",
      "sha256": "<hash>",
      "version": "<component version>",
      "origin": "<public upstream URL, pinned package or build identity>",
      "license": "<upstream identifier or governing terms>",
      "license_files": ["licenses/<retained original text>"]
    }
  ],
  "resources": [
    {
      "path": "licenses/<file or sources/... instead>",
      "sha256": "<hash>",
      "origin": "<exact source identity>"
    }
  ],
  "dynamic_closure": {
    "nvrtc_required": true,
    "nvrtc_component_version": "12.8.61",
    "nvjitlink_included": false,
    "basis": "<bounded native load-trace evidence summary>"
  }
}
```

Each resource path must start with `licenses/` or `sources/`. The assembler
checks file presence, hashes, path safety, case uniqueness, license references,
and exact input membership. It rejects any unexpected dependency-root file.
Do not include private paths, credentials or private evidence in provenance.
Required toolchain fields and shared CUDA linkage are checked. Recognizable
absolute local paths, credential fields, token formats and URL credentials are
rejected without echoing their values. Review free-form provenance too: pattern
checks cannot establish that arbitrary text contains no confidential data.
The manifest is an explicit audited input, not an automatic legal determination.

The qualified NVRTC 12.8.61 DLL hashes are pinned in the assembler. Both compiler
and builtins must be local: cuDNN can otherwise reject a supported compiled plan
or resolve these DLLs from a developer installation. nvJitLink is not in this
closure. Changes to the closure require new evidence and review.

Keep original CUDA/NVRTC and cuDNN license materials, MSVC redistribution terms,
and notices for libarchive, curl, OpenSSL, compression libraries and static
tokenizer dependencies. Include exact FFmpeg/x264 source and native build
scripts, applicable libsoxr/libiconv/xz corresponding source and distributor
patches, MinGW notices and GCC runtime exceptions. Library package metadata alone is not a substitute
for actual license texts. Do not apply historical VRhino binary licenses to new
Apache-2.0 source builds.

## Assembly and validation

```powershell
python tools/package_windows_build.py `
  --build-dir <fresh-build> --dependency-root <audited-input> `
  --build-info <build-info.json> --source-commit <full-SHA> `
  --output <new-stage-directory> --zip <new-candidate.zip> --check

# Repeat without --check to assemble and ZIP the candidate.
python tools/test_package_windows_build.py -v
```

The source HEAD must match, and native sources must be clean. Native specs and
the root VERSION/LICENSE/NOTICE/THIRD_PARTY_NOTICES.md come from Git object bytes,
not mutable checkout bytes. The current 83-file spec closure is enforced; a
changed closure needs explicit packaging review. This does not normalize or
edit the checkout. With Windows `core.autocrlf=true`, registry files marked
`text=auto` can differ from their byte-pinned Git blobs. Native specs are marked
`-text`. The existing hygiene/public-contract validators are unchanged and still
run unconditionally before packaging tests in CI. A separate Git-byte snapshot
check diagnoses checkout differences; it does not replace that CI gate.

The tests include the original seven assembler cases, five input rejection
cases, and focused canonical-resource/provenance regressions. They need Git and
Python as build/test tools, but no NVIDIA DLL downloads. During assembly,
the staged product is started with an isolated PATH and its reported commit and
version are checked. PE32+ AMD64 direct and delay imports are parsed without
dumpbin or third-party Python libraries. Every import must resolve to the
allowlist or an explicit Windows/driver boundary; an installed unknown DLL does
not satisfy this check. This static check does not replace native dynamic-load
tracing. The assembler does not change execution policy or modify PE bytes.

Copying uses a private staging directory, then publishes that directory only
after validation. Existing outputs are never overwritten. ZIP members are
sorted and use fixed timestamps/modes; identical inputs and compression tooling
produce identical ZIP bytes. The adjacent `.zip.sha256` hashes the ZIP.
`SOURCE-BUILD.json` inventories payload files; `SHA256SUMS` additionally hashes
that metadata. As usual, SHA256SUMS does not hash itself.

Run local package probes with PATH containing only the candidate and Windows
system locations, clearing developer/runtime overrides. Test product startup,
device, doctor, cache operations, tokenizer, CUDA resource cleanup, actual cuDNN
compiled-plan execution, and bundled media. Keep validation-only executables
outside the released file inventory; when temporarily placing them alongside a
relocated test copy, audit loaded DLLs against that copy and exclude them from
the final ZIP. Exercise spaces/Unicode paths and an unrelated working directory.
Record full DLL load paths, including child processes; both NVRTC modules must
load locally. Windows system DLLs/API sets and proven driver modules belong to
the host, not the package.

Clean-machine validation and any new candidate's real Wan/LTX inference require
their own validation gates. Assembly does not authorize publishing, tagging,
committing or pushing.
