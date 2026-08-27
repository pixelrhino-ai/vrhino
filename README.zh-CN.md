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

当前公开版本是 **v0.1.0-alpha**。

- 平台：Linux x86_64
- 后端：NVIDIA CUDA
- 首个模型：`vrhino/ltx-video-v0.9.1:1.1.0`
- 已在 Ubuntu 22.04、glibc 2.35 环境验证
- 除兼容的 NVIDIA 驱动外，所需用户态运行库均随 VRhino 提供

你需要兼容的 NVIDIA GPU 和 NVIDIA 驱动，以及足够的显存和磁盘空间。你不
需要安装 Python、PyTorch、Diffusers、Conda、CUDA Toolkit、cuDNN 或系统
FFmpeg。

## 安装

从 [GitHub Releases](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.1.0-alpha)
下载
[发布包](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.1.0-alpha/vrhino-linux-x86_64-cuda-alpha.tar.gz)
和对应的
[校验文件](https://github.com/pixelrhino-ai/vrhino/releases/download/v0.1.0-alpha/vrhino-linux-x86_64-cuda-alpha.tar.gz.sha256)。
假设两个文件都在 `~/Downloads`，运行：

```bash
cd "$HOME/Downloads"
sha256sum -c vrhino-linux-x86_64-cuda-alpha.tar.gz.sha256
mkdir -p "$HOME/.local/share"
tar -xzf vrhino-linux-x86_64-cuda-alpha.tar.gz -C "$HOME/.local/share"
printf '\nexport PATH="$HOME/.local/share/vrhino/bin:$PATH"\n' >> "$HOME/.profile"
export PATH="$HOME/.local/share/vrhino/bin:$PATH"
```

请保持解压后的 `vrhino` 目录完整，因为其中包含运行时、随包动态库和媒体编码
组件。安装后可在任意目录验证：

```bash
vrhino --version
vrhino device
```

安装不需要 `sudo`，也不需要克隆仓库。如果浏览器没有把文件保存到
`~/Downloads`，请参阅[安装说明](docs/install.md)。

## 快速开始

先拉取当前支持的模型：

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.0
```

然后生成视频：

```bash
vrhino run vrhino/ltx-video-v0.9.1:1.1.0 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

首次拉取会从上游下载约 24.77 GB 的原始模型文件，在本机完成转换，并把可运行
模型安装到 VRhino 本地缓存。VRhino 发布包本身不包含模型权重。

## 工作方式

`vrhino pull` 下载固定的上游模型版本，校验并缓存源文件，在本机以原生转换器
生成 VRhino 模型格式，然后安装成不可变的本地包。`vrhino run` 使用共享原生
运行时执行该模型，并通过随包提供的媒体组件输出 MP4。

## 文档

- [安装与系统要求](docs/install.md)
- [模型命令](docs/cli/model-cli-v0.md)
- [`pull` 命令](docs/cli/pull-v0.md)
- [`run` 命令](docs/cli/run-v0.md)
- [LTX-Video v0.9.1 来源与许可证说明](docs/models/ltx-video-v0.9.1.md)
- [VRM 格式规范](spec/vrm-v0.1.md)
- [可运行模型包规范](spec/model-package-v0.md)

## Alpha 限制

当前版本支持 Linux x86_64、NVIDIA CUDA 后端和首个已验证的 LTX 模型路径。
它不声称支持所有 NVIDIA GPU、所有 Linux 发行版或所有视频模型架构。Alpha
期间接口和兼容性可能发生变化。

详情见 [Alpha 限制](docs/alpha-limitations.md)。

## 许可证

VRhino 二进制是专有软件，按
[VRhino Alpha Binary License](licenses/VRHINO-BINARY-LICENSE.txt) 分发。
第三方组件继续适用各自的许可证，详见
[THIRD_PARTY_NOTICES.txt](licenses/THIRD_PARTY_NOTICES.txt)。

模型许可证彼此独立。VRhino 不授予模型权重、输入、输出或其他第三方内容的
任何权利。
