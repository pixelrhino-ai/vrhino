# Stage 2A FACT / REPORT

**PASS：Linux 隔离候选的原始 source → Schema2 Runnable Package → 正常 Native 安装与运行已完成。** 公共 HTTPS Distribution 和完整 numerical qualification 仍为 HOLD。本轮未进入 Release，等待 CTO 审查。

批准的 public main 在开始与结束时均为 `e560faf43ef426dc0db9c40cf30cf90f4d78ac8e`。全部实现位于独立 detached worktree `$STAGE2A_WORKTREE`；未修改 public main。机器证据在该 worktree 的 `proof/stage2a/FACT.json` 和相关日志中，模型、媒体、私有 fixture、构建和证据目录均为本机资产，不作为 Git 源码提交。

## 1 结果和适用范围

| 检查 | 结果 |
| --- | --- |
| 原始 checkpoint 直接 Native 转换 | PASS；无旧 VRM 输入或 Schema1 中间产物 |
| 完整 Schema2 Runnable Package admission | PASS；715 denoiser / 297 decoder slots，完整 Program 与资源引用 |
| 空白模型缓存的正常 source-backed pull | PASS；仅原始 source CAS 预置，实际调用 Native converter |
| 第二个空白模型缓存的隔离 Registry pull | PASS；15 artifacts 全部经 Native HTTPS 下载和校验 |
| 最终候选完整正常 Product run | PASS；704×480、121 frames、25 FPS、40-step Typed FlowEuler |
| 冻结 reference 对照 | PASS；完整 MP4 和全部 121 个 RGB24 frames 逐字节一致 |
| 公开 Schema1 Product 回归 | PASS；同一 CLI 正常 pull/run，40 步和冻结 MP4 一致 |
| 公共 CTest | 86/87 PASS；59/59 host，27/28 CUDA；唯一失败为既有 F32 offset |
| 公共 HTTPS Distribution 资格 | HOLD；仅本机隔离 HTTPS Registry 证据 |
| 完整 numerical qualification / Release | HOLD / 未授权 |

## 2 能力边界和最小实现

合并基线中的 Schema2 工具从已有 Schema1 VRM 获取 binding 表，并通过 `MappedSource` 复制 VRM 内权重。该方式不能证明原始 checkpoint 的直接转换。原有 Product LTX converter 已提供 Safetensors reader、冻结 checkpoint 验证、1012 项 tensor mapping、descriptor 检查、Native streaming writer、CAS admission 和 immutable package publication，足以补齐本轮路径。

本轮把 source/path/mapping helper 从原有 LTX converter 复用到新的 Schema2 Product converter。它读取原始 Safetensors，以既有 mapping 构建绑定，生成既有 closed topology 和 programs.v2 声明，并使用原有 writer。所有 1012 个 tensor 的 source/destination dtype、shape 和全部 payload 字节均比较一致；名称关系使用原有冻结 mapping，未转换 dtype 或重新排列 tensor 内容。独立工具和正常 source-backed pull 分别执行真实转换，得到相同 runtime SHA256。

Package 返回前调用既有完整结构性 Product preflight；Native import 再要求生成 manifest 与固定候选 template 一致，逐 artifact 验证并入 CAS，最后原子发布 immutable manifest。取消、错误和 fresh output 保护保留；已有输出目录不会被清理。Package 包含全部声明、T5 shards、index、SentencePiece 和 license 的相对资源路径，无本机 source 路径作为产品运行依赖。

实现没有新的 Converter 框架、Sampling 算法或 Execution Engine。Converter 动态依赖只有系统 C/C++ 库，不含 CUDA、Python 或 PyTorch。Python 仅用于本机 fixture transport、观察和只读审计；生产 acquisition、转换、conditioning、sampling、decoder 与 MP4 encoding 均由 Native 执行。

## 3 Source 和 Package provenance

LTX checkpoint 和 license 固定为 Hugging Face `Lightricks/LTX-Video@8984fa25007f376c1a299016d0957a37a2f797bb`。T5 与 tokenizer 固定为 `PixArt-alpha/PixArt-XL-2-1024-MS@b89adadeccd9ead2adcb9fa2825d3fabec48d404`。六个 source artifacts 的大小和 SHA256 均由 Native 验证；完整记录在 [direct source contract](../../native/specs/ltx_schema2_v1/direct-source-contract.json)。未进行上游重复下载。

| 身份 | 字节数或 SHA256 |
| --- | --- |
| 原始 LTX checkpoint | 5,716,863,844 bytes；`a23200896c5eddf215c7cb9517820c5763a2b054eb62ba86cbce6b871a4577e3` |
| 原始 source artifacts 合计 | 24,766,961,087 bytes |
| 原有 tensor mapping | `b0ae4b597e617a287d62a66546a395f43e83631b9602b43c9726afec2a10f53b` |
| 直接 source contract | `17c5e0fb064d5edb88dfb80b78bd6fbde4f3fd366b14e1ece649042ff6846bd5` |
| 新 Schema2 VRM | 5,717,174,784 bytes；`066c9fd8da14932b80d895973293a996ee645166c05758d4948aa8f168e34b31` |
| Package manifest | `d8bddac9375ffc9ccf10bcfc3cd1122e285a22c90608c19e03149c0232c00acc` |
| 15 Package artifacts 合计 | 24,767,425,906 bytes |
| 冻结 40-step programs.v2 | `3e087be341d6a4210c4a06e1d28fe342c7ee34e0d3406432056c3e9a66993ee7` |
| 最终 Native CLI | `9b133f07a18c828ea669f56caca29cf5254cf4a8cb130840cbeae61850c49bef` |
| Native converter 工具 | `b3d40279ca3cd3268ac4a0ede0f235e29d3305bdac652bfa30ba52e178021d4d` |

候选 identity 为 `review/ltx-video-v0.9.1-schema2:0.2.0`，与公开 `vrhino/ltx-video-v0.9.1:1.1.1` 独立。声明保留 `alpha_unqualified` 和 `local_test_only`，没有公开 Registry entry。15 个 artifact 的完整 size/SHA256 列表在 [固定候选 manifest](../../native/specs/ltx_schema2_v1/vrhino-model.json)。VRM metadata 记录 source revision、checkpoint hash、mapping hash、source contract hash 和 converter version。

`proof/stage2a/native-identity.json`、`full-source-sha256.json`、`candidate.patch` 和 CMake cache 冻结实际 dirty candidate 的来源；嵌入的 Git HEAD 单独不能代表未提交实现的完整身份。

## 4 正常 Native pull 和隔离分发

首条路径使用 fresh cache：`models` 与 Product CAS 起始均为空，仅将本机已有的六个原始 source blobs 链接到 source CAS。正常 `vrhino pull review/ltx-video-v0.9.1-schema2:0.2.0` 通过二进制旁已安装 specs 自动发现计划，无 converter spec override。Native acquisition 验证复用 24,766,961,087 bytes，上游下载 0 bytes；输出明确记录 `Converter invoked: yes`、`Conversion performed: yes`、`Already installed: no`。总耗时约 347.42 秒。

第二条路径使用独立 loopback HTTPS Registry、私有 CA 和另一完全空白缓存，无任何 Product CAS 预置。正常 Native `pull` 获取 manifest 和全部 15 artifacts；server 与 CAS 观察分别确认 15 次 artifact transfer、24,767,425,906 bytes 和全部 size/SHA256。总耗时约 89.95 秒。正常 `info` 和重复 `pull` 通过；重复拉取为 installed-package，下载 0 bytes。

两条路径分别证明 source acquisition/conversion 与 Registry Package transport；不是通过把已安装 Package 当作直接转换输入。TLS 的验证范围是本机私有 CA，不能据此授予公共 HTTPS 分发资格。

## 5 最终候选完整 Product smoke

最终 run 使用第二条 Registry 路径安装的 Package 和上述 CLI，正常 `vrhino run`，没有 qualification prefix、步骤覆盖或 Legacy fallback。配置为 prompt `A red cube rests on a table in soft daylight.`、seed 5703、CFG 3、冻结 negative prompt、704×480、121 frames、25 FPS 和 40-step Typed FlowEuler。

Native SentencePiece 与双 shard T5 conditioning、40 次实际完成回调、Native decoder 和 MP4 输出全部完成。Schema2 记录一个 sampling stage entry 加 40 completion callbacks；Native 在 decode 前还检查 completed_steps、observer 次数和 program_complete。进程退出码 0，总耗时约 153.21 秒，峰值 device memory 13.17 GiB；GPU 在进程前后均为 0 MiB。

主机 GPU 为 RTX 4090 D，UUID `GPU-f44fb070-ef0c-a31c-d0e2-f58d9aa9c253`，driver 580.105.08。构建使用 GCC 11.4、CUDA 12.8.93、Release、SM89、Native tokenizers 与 Product CLI ON。encoder 为原有 VRhino FFmpeg，独立 FFmpeg 仅观察媒体。原始模型、Package 与 binary 身份分别绑定，不以历史 Stage 1D binary 作为本轮执行证据。

## 6 Schema1 对照和回归

新输出与保留的冻结 Schema1 reference 完整 MP4 字节一致：42,869 bytes，SHA256 `f47d33546e0c7369f71cc54030235df1d04e57167f854a916d73e3d4dba39534`。全部 121 个 704×480 RGB24 frames 逐字节相同，拼接 SHA256 为 `278c36111ee794173d54ec47476f885a8573a40370df05614ef449b9c00b2734`；25/1 FPS，时长 4.84 秒。

同一候选 CLI 另行执行公开 Schema1 `vrhino/ltx-video-v0.9.1:1.1.1` 正常 pull/run，复用既有 verified cache。pull 为 already-installed；run 完成全部 40 个 numbered completion events、Native conditioning/decoder 和完整 MP4，输出仍与冻结 reference 字节一致，GPU 回到 0 MiB。原 Schema1 VRM 保留 `267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d`；manifest 前后均为 `6c41babc7ae117e7815b48e18ae7538d8deb39b58c26812e19fbf49f3784aaa1`。

这些比较证明具体模型、binary 和执行配置下的完整媒体结果。没有把它扩大为所有 40 步中间 tensors、raw decoder float 或完整 LTX numerical qualification 的证明。

## 7 Changed files 和 Ownership

| Ownership | Changed files |
| --- | --- |
| Product Converter | `native/src/product/converter_ltx.cpp`、`converter_ltx_internal.h`、`converter_ltx_schema2.cpp`；`native/include/vrhino/product/converter.h`；`native/tools/ltx_schema2_convert.cpp` |
| Converter / Distribution data | `native/specs/ltx_schema2_v1/{direct-source-contract,source-plan,pull-plan,vrhino-model}.json` |
| Build integration | `native/CMakeLists.txt`；加入源文件、复用结构性 Product preflight，以及小型公共负向测试 |
| Tests | `native/tests/ltx_schema2_converter_tests.cpp`、`ltx_schema2_source_product_smoke.py`、`registry_fixture_tests.py` |
| Documentation | `docs/product/ltx-schema2.md`、本 FACT / REPORT |

Shared Runtime、Architecture、Sampling、Backend、CUDA、PrecisionPolicy 源码和模型专属语义增量均为 **零**。Wan Schema2 v1、原有 LTX Schema1 specs、公开 Registry、VERSION 和历史 Release 文件无改动。

## 8 Admission 负向验证和已知失败

公开 CTest 为 86/87 PASS：59/59 host，27/28 CUDA。Wan Schema2 admission/wiring、LTX programs.v2、resource ownership、lifecycle、异常和取消回归通过。Frozen F32 FlowEuler 测试逐位比较 3/40 步 schedule、timestep、delta、CFG 及独立 F32 recurrence，并通过 56 个负向案例。

Converter 有 13 个负向案例，覆盖取消、磁盘不足、已有输出保护、missing source、symlink/escape、spec drift、checkpoint size 和 mapping dtype/shape/transformation/inventory。另用大小完全相同但 SHA256 错误的 checkpoint 验证 Native 拒绝，未创建输出。真实 Package preparation 的 13 个负向案例全部拒绝，包括 frozen geometry/authority、UTF8、seed、program drift、混合 capability、missing shard 和禁止结构性 admission 授予 numerical qualification。

原有 source acquisition HTTPS fixture 与 Registry safety fixture 通过，覆盖默认 TLS/HTTP 策略、checksum/length/schema/space fail-closed、Range/no-Range、resume、取消、redirect、并发和 CAS reuse。Registry fixture 原先把已支持的 schema 2 当作未知 schema，现改为 99；保留 `PACKAGE_VERSION_UNSUPPORTED` 断言，没有修改 admission 代码或数值门槛。

唯一公共 CTest 失败是既有 `vrhino-neural-graph-layernorm-f32-offset-cuda-tests`。候选与保留 pristine baseline 在当前 GPU 上的完整 transcript 字节相同，SHA256 均为 `2e078b287e697a667f85007209c4b0ffac486ddae4ec5e7d9b37575239940861`；仍为 40 witnesses、20 FAIL。对应 Backend/F32 test 文件在 baseline 与批准 main 间相同，本轮也没有改动。

`BASELINE_F32_LAYERNORM=FAIL_UNCHANGED`

`GENERIC_BF16_NUMERICAL_STATUS=HOLD`

两次观察问题保留原始日志并纠正：并行 CPU admission 与 source fixture 的硬链接元数据变更冲突，原有 backing guard 正确拒绝，顺序重测 PASS；Schema1 观察器误用 Schema2 日志计数，依据原有 `Sampling n/40` 实现验证现有日志后 PASS，未重复模型运行。它们没有触发放宽资源或数值检查。

## 9 清理和公开 Distribution 证据边界

初始工作盘只有约 1.3 GiB 空间，按要求停止大型转换路线。用户随后授权清理；逐份验证 SHA256 后，将四份重复 CAS 存储通过硬链接去重，保留路径、内容、权限、reference 和证据，共释放 40,269,574,162 bytes，约 37.50 GiB。`proof/stage2a/cleanup.jsonl` 记录每个动作。只读模型内容未改变，没有上游大规模重复下载。权限切换造成的执行中断在环境恢复后继续完成；早期 HOLD 已被本轮完整证据替代。

公共 Distribution 仍缺少经过授权的实际公共 Registry entry、公共 HTTPS endpoint 上的最终固定 artifact/descriptor、独立客户端获取/安装证据和 Release Candidate 的正常审查/CI/安装资格。当前 native source plan 与隔离 Registry 证明技术安装路径，未证明公共服务的实际可达性或授予发布资格。没有修改公开模型 identity 或 `v0.9.4-alpha`。

## 10 CTO 决策边界

**值得进入 Linux Release Candidate 审查准备。** 原始 source 的 Native conversion、完整 Runnable Package、正式 Native installation orchestration 与完整 Product execution 已接通，不需要模型专属 Runtime/Backend/CUDA/Precision 妥协。

本轮不进入 Release：无 commit、push、PR、merge、公开 Registry 发布、历史 Release 修改、新 BF16 candidate 或模型计算优化。后续工作须由 CTO 审查决定；正常 CI、公共分发证据和 numerical qualification 的各自边界继续保留。
