# Windows converter large-file metadata overflow

Base: `e7e24ac2c231bf0dc29d94a39ccf52ba8570b47b`.

## Proven failure

The two failing calls are in VRhino-owned `native/src/product/converter.cpp`:
`FrozenTensorSource::FrozenTensorSource` and
`SafeTensorReader::SafeTensorReader`. Both declared `struct stat information`
and called unqualified `fstat(descriptor, &information)`.

The Windows adapter already supplied `fstat(int, struct _stat64*)`, forwarding
to `_fstat64`. However, UCRT's legacy `struct stat` is a different structure
whose `st_size` is a 32-bit `_off_t`, including on Windows x64. The exact-match
global `fstat(int, struct stat*)` overload was selected instead. UCRT implements
that overload using `_fstat64i32`. A 64-bit timestamp does not imply a 64-bit
file-size field. Declaring the adapter's `off_t` as `int64_t` does not change
UCRT's previously declared structure.

Read-only native diagnostics on the same open binary file demonstrated
`_fstat64i32` returning -1 with `errno=132` (`EOVERFLOW`), whereas `_fstat64`
returned 0 and the exact sizes below. Source SHA256 verification succeeded;
this failure is unrelated to source acquisition, JSON decoding, or inference.

| Product conversion | Source artifact | Exact bytes | Reader |
| --- | --- | ---: | --- |
| Wan 2.1 1.0.1 | `diffusion_pytorch_model.safetensors` | 5,676,070,424 | `FrozenTensorSource` |
| LTX Video 0.9.1 1.1.1 | `ltx-video-2b-v0.9.1.safetensors` | 5,716,863,844 | `SafeTensorReader` |

Wan's `import_wan_model` verifies source sizes/hashes and loads frozen tensor
maps before constructing the runtime and UMT5 `FrozenTensorSource` objects.
The first runtime source triggered the failure; the 11,361,920,418-byte UMT5
source used the same affected constructor. LTX's `import_ltx` verifies
its checkpoint before constructing `SafeTensorReader`, so it failed before
tensor-map validation and output writing. Both paths originate in VRhino, not
a vendored library.

## Fix and integer audit

The shared Windows adapter now names its status type `FileStatus`, explicitly
aliasing `_stat64`, with a compile-time assertion that `st_size` is eight bytes.
Both affected readers use that type. On POSIX the alias remains `struct stat`,
so the same system `fstat` and existing file/offset behavior are retained.
There are no architecture conditions or model-specific changes.

The surrounding size and offset audit found:

- The stat result is checked for failure/negative or insufficient length before
  conversion to `uint64_t`. Reader file sizes, tensor offsets/lengths, write
  results and format offsets are `uint64_t`; tensor dimensions are `int64_t`.
- Safetensors headers are capped at 64 MiB before conversion to `size_t`.
  Tensor ranges must fit the measured file, including canonical coverage.
  Frozen ranges use subtraction-based bounds checks and checked shape products.
  Relative reads reject out-of-range offsets and lengths before file access.
- Windows `off_t` and `ssize_t` are explicitly `int64_t`. Validated input
  offsets cannot exceed the nonnegative signed 64-bit file size. The positional
  adapter uses `_lseeki64`, rejects negative offsets, limits each CRT transfer
  to `INT_MAX`, and checks positive transfer results before converting them to
  `size_t`. Copy loops use bounded buffers, not a narrowed whole-file length.
- Streaming output sizes/offsets use checked 64-bit sums and alignment. On the
  supported x64 host `size_t` is 64-bit. Impossible signed seek offsets fail
  in the positional adapter; this patch does not expand format/admission limits.
- `pytorch_zip.cpp` already selects `_stat64` under Windows, with 64-bit file
  and ZIP64 offsets, subtraction-based file bounds and bounded metadata reads.
  It requires no change. Its small-member conversion to `size_t` is bounded
  by the caller's size limit and the x64 address space.
- `tflite.cpp` uses stream positions and explicitly caps its entire input at
  64 MiB before allocating. Fixed checkpoint converters use frozen 64-bit sizes
  or `std::filesystem::file_size`, and chunk/shape conversions do not narrow a
  whole large file to Windows `long` or `int`.
- Runtime VRM/safetensors mapping already uses `GetFileSizeEx` and checks the
  result against `size_t`. Stable verification and Windows cache admission use
  `FILE_STANDARD_INFO::EndOfFile.QuadPart`. Their POSIX `struct stat` calls are
  excluded by Windows compilation; none need this converter-only fix.

## Regression design

`vrhino-native-converter-tests` creates Windows sparse safetensors with payloads
of `2^31 + 17` and `2^32 + 17` bytes. Each fixture asserts physical allocation
below 1 MiB and exercises exact `SafeTensorReader::file_size()`, reads beyond
the boundary through both production readers, and streams a final BF16 tensor
into a small safetensors output with byte-for-byte sentinel verification.
It also rejects overflowing relative offsets, end-of-file overruns, invalid
frozen source ranges and overflowing shape products.

The same new test linked against the unchanged pre-fix converter library
failed at both boundaries with `PACKAGE_INVALID: truncated safetensors file`.
No multi-gigabyte physical fixture, model download, or inference is used by
this regression. Existing non-Windows test behavior is preserved; the Windows
test uses `_getpid` instead of importing POSIX `unistd.h`.

## Scope

Only converter host-file metadata portability and its regression are changed.
Runtime, architecture implementations, backend, CUDA kernels, precision policy,
model mappings, presets, admission budgets and output formats are unchanged.
The historical frozen rc4 distribution is preserved. Linux runtime execution
is not claimed; the POSIX comparison is structural.
