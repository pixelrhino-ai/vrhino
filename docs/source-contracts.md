# Source contracts and extensions

The v0.1 container header/layout is documented in `spec/vrm-v0.1.md`. Its historical baseline description predates executable quantization. Current loaders and verifiers in `native/src/loader.cpp` and `native/src/product/vrm_verification.cpp` define accepted metadata and fail closed on unsupported encoding or required capability. Tensor alignment, shape/byte bounds and payload checksums are checked before use.

The current tensor runtime also contains INT8 block dequantization and preconditioned INT8 support; consult `native/include/vrhino/quantization/`, `native/src/quantization/` and the VRM/INT8 synthetic tests for the executable contracts. A new quantization mode requires generic semantic support and independent qualification. It is not implied by simply adding metadata.

PrecisionPolicy is a shared execution contract. Storage, operation inputs/outputs, accumulation and persistent state can have distinct precision roles. Graph dtype validation and explicit Cast operations preserve those roles. No architecture-specific tolerance or fixed-block precision exception is introduced by this source publication.

Sampling declarations and Component graphs describe execution using supported generic semantics. Unsupported program/component/precision versions reject before execution. Product model manifests bind immutable artifacts and describe required inputs, components, presets and source acquisition. Model metadata refers to separately licensed weights; the source repository contains no production model tensors.
