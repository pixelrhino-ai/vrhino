<p align="center">
  <img src="assets/branding/banner.png" alt="VRhino" width="960">
</p>

# VRhino

**用共享原生运行时在本地运行 AI 视频模型，无需为每个模型维护 Python 环境。**

[English](README.md) | 简体中文

## 安装

Linux x86_64 一行安装 **v0.9.1-alpha**，无需 sudo：

```bash
curl -fsSL https://raw.githubusercontent.com/pixelrhino-ai/vrhino/main/install.sh | sh
```

脚本会下载、校验安装包并配置命令路径。当前终端按脚本末尾提示刷新 PATH，然后运行：

```bash
vrhino doctor
```

[手动下载与安装选项](docs/install.md) ·
[Windows v0.8 CUDA 包](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.8.0-alpha)

需要兼容的 NVIDIA GPU/驱动和足够的内存。模型权重单独下载；原生执行无需安装 Python 或 CUDA Toolkit。

## 可用模型

| 模型 | 用途 | 使用入口 |
|---|---|---|
| **Wan2.2 T2V A14B** | 文生视频 | `pull` / `run` · `vrhino/wan2.2-t2v-a14b:1.0.0` |
| Wan2.1 T2V 1.3B | 文生视频 | `pull` / `run` · `vrhino/wan2.1-t2v-1.3b:1.0.1` |
| LTX-Video 0.9.1 | 文生视频 | `pull` / `run` · `vrhino/ltx-video-v0.9.1:1.1.1` |
| Mochi 1 Preview | 文生视频 | `pull` / `run` · `vrhino/mochi-1-preview:1.0.1` |
| MuseTalk 1.5 | 口型同步 | `pull` / `run` · `vrhino/musetalk-v1.5:1.0.1` |
| LatentSync 1.6 | 口型同步 | `pull` / `run` · `vrhino/latentsync-1.6:1.0.1` |

各模型的硬件要求和已测试平台不同，详见[安装与模型说明](docs/install.md)。项目目前处于 Alpha 阶段。

## 生成视频

```bash
vrhino pull vrhino/wan2.2-t2v-a14b:1.0.0
vrhino run vrhino/wan2.2-t2v-a14b:1.0.0 \
  --prompt "a red panda running through fresh snow" --output video.mp4
```

首次 `pull` 会下载、校验并转换模型，安装到本地缓存。需要把模型放到更大的磁盘时，在下载前设置 `VRHINO_HOME`。

Wan2.2 与其他文生视频模型使用相同的 Product 命令。首次 `pull` 数据量较大，
因为 VRhino 会校验并转换固定版本的上游权重。存储、硬件和 Alpha 资格说明见
[Wan2.2 上手指南](docs/models/wan2.2-quickstart.zh-CN.md)。

### 口型同步

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.1
vrhino run vrhino/musetalk-v1.5:1.0.1 \
  --video input.mp4 --audio driving.wav --output lipsync.mp4
```

LatentSync 使用相同的视频/音频命令格式，替换为上表中的模型包 ID 即可。

## 更多文档

- [安装与缓存配置](docs/install.md)
- [架构](docs/architecture.md) · [源码构建](docs/source-build.md)
- [本地 Native API — v0.6.0-alpha 合同](docs/api/native-api-v1.md)（`vrhino serve`）
- [v0.9.1 发布与校验和](docs/release/v0.9.1-alpha.md) · [Windows v0.8 支持范围](docs/release/v0.8.0-alpha.md)
- [参与贡献](CONTRIBUTING.md)

项目自有源码采用 [Apache-2.0](LICENSE)。已分发二进制遵循其[二进制许可证](licenses/VRHINO-BINARY-LICENSE.txt)，
依赖和模型遵循各自条款。详见[第三方声明](THIRD_PARTY_NOTICES.md)及[模型许可说明](docs/install.md#linux-shell-pull-and-run-examples)。
