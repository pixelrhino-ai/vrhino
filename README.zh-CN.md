<p align="center">
  <img src="logo.png" alt="VRhino" width="160">
</p>

# VRhino

[English](README.md) | 简体中文

VRhino 是一个自包含的原生视频模型运行时与模型打包系统，目标是在本地运行
AI 视频模型时，不再维护每个模型各自的 Python 环境。

## VRhino 是什么？

VRhino 正在探索一种面向 AI 视频生成的、类似 GGUF + llama.cpp 的模型分发与
运行方式。它把受支持的模型检查点转换成 `.vrm`，再由共享原生运行时执行。

```text
模型 / Checkpoint
        ↓
VRhino 转换
        ↓
      .vrm
        ↓
共享原生运行时
        ↓
      后端
```

项目目前仍处于 Alpha 阶段，在模型覆盖、生态和成熟度上还不能与 llama.cpp
相提并论。

## 当前 Alpha

当前版本是 **v0.4.0-alpha**。

- 平台：Linux x86_64
- 后端：NVIDIA CUDA
- 已验证的公开模型路径：
  - `vrhino/ltx-video-v0.9.1:1.1.0`
  - `vrhino/wan2.1-t2v-1.3b:1.0.0`
  - `vrhino/mochi-1-preview:1.0.0`
  - `vrhino/musetalk-v1.5:1.0.0`（`lip_sync`）
- 已在 Ubuntu 22.04、glibc 2.35 环境验证
- 除兼容的 NVIDIA 驱动外，所需用户态运行库均随 VRhino 提供

你需要兼容的 NVIDIA GPU 和 NVIDIA 驱动，以及足够的显存和磁盘空间。你不
需要安装 Python、PyTorch、Diffusers、Conda、CUDA Toolkit、cuDNN 或系统
FFmpeg。

## 安装

从 [GitHub Releases](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.4.0-alpha)
下载
[发布包](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.4.0-alpha/vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz)
和对应的
[校验文件](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.4.0-alpha/vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz.sha256)。
假设两个文件都在 `~/Downloads`，运行：

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-v0.4.0-alpha.tar.gz -C "$HOME/.local/share"
printf '\nexport PATH="$HOME/.local/share/vrhino/bin:$PATH"\n' >> "$HOME/.profile"
export PATH="$HOME/.local/share/vrhino/bin:$PATH"
```

请保持解压后的 `vrhino` 目录完整，因为其中包含运行时、随包动态库和媒体编码
组件。安装后可在任意目录验证：

```bash
vrhino --version
vrhino device
vrhino doctor
```

安装不需要 `sudo`，也不需要克隆仓库。如果浏览器没有把文件保存到
`~/Downloads`，请参阅[安装说明](docs/install.md)。

## 快速开始

当前 Public Alpha 支持四个已验证的模型路径：

- `vrhino/ltx-video-v0.9.1:1.1.0`
- `vrhino/wan2.1-t2v-1.3b:1.0.0`
- `vrhino/mochi-1-preview:1.0.0`
- `vrhino/musetalk-v1.5:1.0.0`

拉取一个确定的模型包，例如：

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
```

然后生成视频：

```bash
vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

MuseTalk 使用类型化的视频与音频输入，而不是文本提示词：

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.0
vrhino run vrhino/musetalk-v1.5:1.0.0 \
  --video input.mp4 \
  --audio driving.wav \
  --output output.mp4
```

下一个候选版本还准备了以下 Public Mode-C 元数据：

- `vrhino/latentsync-1.6:1.0.0`（`lip_sync`）

LatentSync 使用相同的类型化视频/音频 CLI 形式。VRhino 从上游下载 12 个固定的
推理文件并在本地转换；Pixel Rhino 不分发模型权重或转换后的 VRM。当前
v0.4.0-alpha 二进制早于这份元数据，普通用户需等待后续经过明确验证的二进制
版本后才能使用。

首次拉取会从模型的固定上游版本下载原始文件，在本机完成转换，并把可运行模型
安装到 VRhino 本地缓存。LTX 源数据约为 24.77 GB，Wan 源数据约为 16.36 GiB，
Mochi 源数据约为 37.28 GiB，MuseTalk 源数据约为 4.01 GiB。VRhino 发布包本身
不包含模型权重或转换后的模型组件。

`vrhino pull` 默认优先使用 Hugging Face 官方端点；当官方端点不可用时，
VRhino 可能自动回退到第三方镜像完成公开模型下载。

如需生成便于提交给支持人员的隐私安全本地诊断报告，可运行 `vrhino doctor`
或 `vrhino doctor MODEL`。该命令不会上传遥测，也不会自动发起网络诊断请求。

模型与缓存数据默认保存在 `~/.vrhino`。如需使用更大的文件系统，可在拉取前设置
`VRHINO_HOME`，例如：

```bash
export VRHINO_HOME=/mnt/large-disk/vrhino
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
```

这不会改变 VRhino 二进制文件的安装目录。可运行包完成校验并成功安装后，pull
会清理不再需要的源文件数据，同时保留已安装数据和共享 CAS 数据。

## 工作方式

`vrhino pull` 下载固定的上游模型版本，校验并缓存源文件，在本机以原生转换器
生成 VRhino 模型格式，然后安装成不可变的本地包。`vrhino run` 使用共享原生
运行时执行该模型，并通过随包提供的媒体组件输出 MP4。

## 文档

- [安装与系统要求](docs/install.md)
- [模型命令](docs/cli/model-cli-v0.md)
- [`pull` 命令](docs/cli/pull-v0.md)
- [`run` 命令](docs/cli/run-v0.md)
- [`doctor` 诊断命令](docs/cli/doctor-v0.md)
- [LTX-Video v0.9.1 来源与许可证说明](docs/models/ltx-video-v0.9.1.md)
- [Wan2.1 T2V 1.3B 来源与许可证说明](docs/models/wan2.1-t2v-1.3b.md)
- [Mochi 1 Preview 来源与许可证说明](docs/models/mochi-1-preview.md)
- [MuseTalk v1.5 来源、使用与许可证说明](docs/models/musetalk-v1.5.md)
- [LatentSync 1.6 来源、使用与许可证说明](docs/models/latentsync-1.6.md)
- [VRM 格式规范](spec/vrm-v0.1.md)
- [可运行模型包规范](spec/model-package-v0.md)

## Alpha 限制

当前版本支持 Linux x86_64、NVIDIA CUDA 后端，以及上面列出的四个确定且已
验证的模型路径。它不声称支持所有 NVIDIA GPU、所有 Linux 发行版、所有模型
检查点或所有视频模型架构。Alpha 期间接口和兼容性可能发生变化。

详情见 [Alpha 限制](docs/alpha-limitations.md)。

## 许可证

VRhino 二进制是专有软件，按
[VRhino Alpha Binary License](licenses/VRHINO-BINARY-LICENSE.txt) 分发。
第三方组件继续适用各自的许可证，详见
[THIRD_PARTY_NOTICES.txt](licenses/THIRD_PARTY_NOTICES.txt)。

模型许可证彼此独立。VRhino 不授予模型权重、输入、输出或其他第三方内容的
任何权利。

对于 MuseTalk，Pixel Rhino 不分发模型权重；用户从固定的官方上游获取模型并
在本地转换。使用仍受各上游模型许可证约束，其中包括 MuseTalk 的 CreativeML
OpenRAIL-M 用途限制。用户须确保输入媒体的使用合法并已获得同意。本项目不暗示
任何上游权利人对 VRhino 的认可或背书。

已准备的 LatentSync Public 模型包同样遵循不分发权重的 Mode-C 边界。
LatentSync 模型的使用仍受 CreativeML Open RAIL++-M License 及其 Attachment A
用途限制约束。本次仓库变更仅准备元数据，并不表示支持 LatentSync 的二进制
版本已经发布。
