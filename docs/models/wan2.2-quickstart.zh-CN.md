# Wan2.2 原生视频生成上手指南 — v0.9 alpha

[English](wan2.2-quickstart.md)

先按[安装说明](../install.md)安装 Linux v0.9 包。本指南使用随包提供的
`evaluate`，依次执行原生 tokenizer、UMT5、采样、VAE 和 MP4 输出，无需 Python
推理环境。普通 schema2 `run` 支持仍待完成。

## 1. 准备模型包

已测试配置：A800 80 GB、BF16、832×480、81 帧、40 步。示例设置 30 GiB 权重缓存，
实测峰值显存约 44.8 GiB；这不等于保证更小的显卡一定可运行。预留模型源文件、转换包
和输出所需的磁盘及主机内存：仅 `model.vrm` 就有 114,816,482,368 字节，完整包还有其他资源。

使用以下固定来源：

- 模型：[ModelScope 的 Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)，
  revision `3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7`。
- 官方语义源码：[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)，
  revision `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`。

通过来源站点的下载方式获取上述版本，保持原文件和目录结构。转换器按随包合同验证来源。
核心文件也匹配 HF revision `c8c270b13ee05bfa474194ac9fb07a5868a97cea`，但整个目录不能
直接互换：README/configuration 站点元数据有差异。不要替换这些文件或修改哈希来跳过检查。
使用前请阅读上游模型条款。

已有合格转换产物时直接复用**完整包**，无需重新转换。单独一个 VRM 不够，还需要 conditioning、
tokenizer、programs 和来源声明。需要首次转换时，设置本机绝对路径：

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

输出目录必须是新目录。转换采用流式处理，不运行官方 Python 推理。spec 目录中的
`source-contract.json` 列出精确输入要求。本 alpha 尚未发布 Wan2.2 的 `vrhino pull`
别名，使用下述本地模型包入口。

## 2. 复制并连接示例

按上例设置 `VRHINO_ROOT`、`MODEL_SOURCE` 和 `CONVERTED_PACKAGE`；复用既有包时也需要
这三个变量。工作目录应为新目录：

```bash
cp -a "$VRHINO_ROOT/share/vrhino/examples/wan2.2" "$HOME/wan22-first-run"
cd "$HOME/wan22-first-run"
mkdir -p assets
ln -s "$CONVERTED_PACKAGE" assets/package
ln -s "$MODEL_SOURCE" assets/source
"$VRHINO_ROOT/bin/vrhino" preflight vrhino-model.json local-resources.json
```

Preflight 检查身份和组件连接，不执行推理。读取大文件验证哈希可能需要几分钟。
相对资源路径基于 `local-resources.json` 所在目录解析。保持声明和完整性哈希不变；
若出现不匹配，应检查来源包。

## 3. 生成视频

编辑 `request.json` 中的 `prompt`、`negative_prompt` 和 `seed`。第一次保留示例分辨率、
帧数以及 `options.json` 的 30 GiB 缓存和 40 步配置。尽量使用空闲 GPU；资源监控也会统计
其他进程占用的显存。

```bash
"$VRHINO_ROOT/bin/vrhino" evaluate \
  vrhino-model.json local-resources.json request.json options.json ./output
```

`output` 必须尚不存在。完成后查看 `output/evaluation.mp4`，并用
`output/supervision.json` 确认完成状态。成功执行标记 `EXECUTED_UNQUALIFIED` 表示
本次执行已完成、reference 数值资格单独管理，本身不是报错。中断时保留日志，重试使用新的输出目录。

更多说明见[已测试配置与 alpha 说明](../release/v0.9-wan2.2-known-limitations.md)和
[evaluate 命令文档](../product/bounded-evaluation.md)。
