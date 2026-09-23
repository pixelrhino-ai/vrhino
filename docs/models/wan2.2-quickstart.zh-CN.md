# Wan2.2 原生视频生成上手指南 — Alpha

[English](wan2.2-quickstart.md)

Wan2.2 使用 VRhino 的普通 Product 路径，转换和推理都不需要 Python。

## 拉取

如果希望把模型缓存放到大容量磁盘，请先设置 `VRHINO_HOME`。下面的命令会自动下载、校验、转换并安装固定版本的来源：

```bash
vrhino pull vrhino/wan2.2-t2v-a14b:1.0.0
```

首次拉取的数据量较大：来源输入约 127 GB，生成的 `model.vrm` 约 115 GB，转换还需要额外临时空间。正式拉取仅使用 Hugging Face 的 `Wan-AI/Wan2.2-T2V-A14B` 固定 revision `c8c270b13ee05bfa474194ac9fb07a5868a97cea`，以及官方 Wan2.2 仓库中按校验和固定的语义文件。

## 运行

```bash
vrhino run vrhino/wan2.2-t2v-a14b:1.0.0 \
  --prompt "一只小熊猫在新雪中奔跑" \
  --output wan22.mp4
```

模型包已声明 tokenizer、UMT5 conditioning、双 binding Execution Program、Sampling Program、BF16 策略、内存预算、832×480、81 帧和 40 步默认值。已测试硬件为 A800 80 GB，实际显存仍会受到 GPU 和其他进程影响。

Wan2.2 以 Alpha 未完成数值资格的状态运行：Product 结构及代表性原生运行已经验证，通用 BF16 reference 资格仍单独保持 HOLD。运行时会提示这一状态；实现没有加入模型专属 Runtime、Backend、CUDA 路径或精度例外。

资格细节见[测试配置与 Alpha 说明](../release/v0.9-wan2.2-known-limitations.md)。
