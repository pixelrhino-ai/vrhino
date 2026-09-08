# Source packaging

Source Git contains production code, build logic, specification files, license notices and the vendored tokenizer source closure. It contains no NVIDIA runtime libraries, FFmpeg executable, model weights or release archive. Existing v0.5/v0.6 release bytes and tags are immutable.

A source build produces `vrhino` and `vrhino-native`. `tools/package_source_build.py` assembles fresh VRhino executables with an explicitly supplied, audited dependency bundle. A convenient dependency input is the extracted official v0.6.0-alpha package linked in the README: its `lib/`, `media/`, `licenses/` and root dependency notices supply the existing runtime/legal closure. Those binary inputs stay outside source Git. Alternatively reconstruct that layout from the pinned upstream components and retained build/legal manifests. The script checks required paths, rejects host-driver bundles, copies dependency bytes unchanged, writes project source license notices and checksums, and adjusts RUNPATH only on VRhino-owned executables.

```sh
python3 tools/package_source_build.py --build-dir build \
  --dependency-root /path/to/extracted/vrhino \
  --output /path/to/new-package --check
python3 tools/package_source_build.py --build-dir build \
  --dependency-root /path/to/extracted/vrhino \
  --output /path/to/new-package
```

Python 3 here is optional distribution tooling from the standard library, never a Native build or inference dependency. `patchelf` is required for package assembly. The output directory must not exist. This command does not create a tag, upload an archive or publish a release. Each new binary package still needs relocation, dependency, media and model qualification before distribution. Dependency reuse alone does not establish new package qualification.

The media boundary is a separate FFmpeg/x264 process receiving raw frames. No system FFmpeg fallback is used. `release/media/build-media-lip-sync.sh` reconstructs the qualified media helper from FFmpeg 4.4.2, x264 revision `5db6aa6cab1b146e07b60cc1736a01f21da01154`, an explicit libsoxr development sysroot, an empty work directory and an output prefix. Its exact flags include GPL/x264 and exclude nonfree components. Source archive hashes, notices and corresponding-source requirements are under `licenses/media/` and `release/compliance-overlay/licenses/media/`; package distributors must retain those source/legal materials, including the libsoxr-related records. Upstream archives are available in the existing release’s `media/sources/` and from their identified upstreams. Build tools include C/C++ compilers, make, nasm, pkg-config and patchelf.

NVIDIA runtime terms are separate from the MIT cuDNN frontend source license. Preserve the audited CUDA 12.8/cuDNN 9.8 runtime objects without patching, stripping or replacing their metadata. `licenses/nvidia/` records the binary distribution boundary. The NVIDIA driver is a host prerequisite: never bundle `libcuda.so`. System library license/source obligations and static tokenizer attributions remain in the dependency notices. Current project source is Apache-2.0; older released binaries retain their original binary license. The new source grant does not relicense third-party components or model weights.
