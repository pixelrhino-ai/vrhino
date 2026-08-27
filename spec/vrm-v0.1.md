# VRM v0.1 Binary Specification

All integer fields are little-endian. Header is exactly 128 bytes (`<8sHHBBH16s16sQQQQQQQQ16s>`):

```text
magic[8] = "VRHINO\0\1"
format_major u16 = 0
format_minor u16 = 1
endianness u8 = 1 (little)
alignment_log2 u8 = 6 (64-byte tensor alignment)
flags u16
profile_id[16]
architecture_id[16]
metadata_offset/length u64
tensor_table_offset/length u64
graph_offset/length u64
data_offset u64
file_size u64
payload_blake2b_128[16]
```

Metadata、tensor table、graph section 均为 canonical UTF-8 JSON：sorted keys、compact separators、禁止 NaN。各 section 按 header 顺序排列；tensor data 从 `data_offset` 开始，单 tensor 相对 data section 64-byte 对齐。checksum 覆盖 header 后全部 payload，防止 metadata/table/graph/data 被静默修改。

未知 typed metadata 可以保留；未知 required feature、format major、graph/primitive version 必须拒绝。v0.1 不存 tokenizer payload，只存 external asset descriptor。量化字段存在但唯一可执行值是 `none`。

Metadata typed value 由 canonical JSON 映射：string、signed integer、finite float、bool、array、shape(integer array)、dtype/enum(string)，blob reference 使用 `{kind, offset, length, media_type}` object。模型 metadata 覆盖固定 source/revision、profile/architecture、default dtype、完整 denoiser config（hidden size/layers/heads/head dim/patch/RoPE/norm）、latent contract、sampling defaults、VAE config 及 tokenizer/text-encoder external asset descriptors。

v0.1 section 必须全部存在，未知 optional metadata key 可忽略并保留；header flag、required capability、implementation/schema version 或 tensor encoding 未知时必须在执行前失败。Backend 决定 device residency。
