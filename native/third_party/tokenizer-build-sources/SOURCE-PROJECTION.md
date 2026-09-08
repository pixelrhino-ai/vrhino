# Source-only distribution changes

VRhino omits the three prebuilt WebAssembly outputs shipped by wit-bindgen 0.57.1: `src/rt/libwit_bindgen_cabi.a`, `src/rt/wit_bindgen_cabi_realloc.o`, and `src/rt/wit_bindgen_cabi_wasip3.o`. The corresponding C/Rust sources and original upstream licenses remain unchanged. These outputs are used only by the upstream Wasm target branch; they are not required by VRhino’s supported Native host tokenizer build. VRhino does not provide a WebAssembly runtime or qualify this vendor tree as a ready-to-link Wasm SDK.

The per-file Cargo checksum record and the outer source inventory are updated to describe this exact source-only projection. The upstream package identity in Cargo.lock is unchanged. This is a deliberate removal of compiled artifacts, not a network or private-cache fallback. Linux offline closure and missing/corrupt-source rejection are requalified against the new inventory.
