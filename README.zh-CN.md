<p align="center">
  <img src="assets/branding/banner.png" alt="VRhino" width="960">
</p>

# VRhino

**面向本地 AI 视频模型的原生运行时。**

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

## 发布状态

当前源码版本为 **v0.8.0-alpha，尚未发布**，这是首个支持原生 Windows 的发布线。
最新已发布版本仍为
[v0.7.0-alpha](https://github.com/pixelrhino-ai/vrhino/releases/tag/v0.7.0-alpha)，
其 Linux x86_64 CUDA 范围、标签和发布资产保持不可变。

Windows 资格验证基线为 Windows 10 Pro 22H2 / build 19045、RTX 3090 24 GiB、
NVIDIA 驱动 610.47。冻结 rc6 已通过干净主机上的 Wan 和 LTX 完整验证，但它
保留 v0.7 构建身份，只作为历史证据，不是 v0.8 下载包。版本 PR 合并后必须
新建 rc7，并完成 Wan、LTX、MuseTalk、LatentSync 四条路径的资格验证后才能发布。

Mochi 保留历史 Linux 支持范围。其冻结准入要求至少 80 GiB 可用设备显存，
因此不属于这台 24 GiB Windows 主机的验证范围；这不代表模型测试失败。
不宣称覆盖所有 Windows/GPU，也不宣称支持 macOS。详见
[v0.8 支持矩阵和待完成门禁](docs/release/v0.8.0-alpha.md)。

最终用户需要兼容的 NVIDIA GPU/驱动、足够的内存和磁盘空间，以及获取未缓存
模型源文件所需的网络。不需要安装 CUDA Toolkit、独立 cuDNN、Visual Studio、
Build Tools、CMake、Ninja、Python、PyTorch、Diffusers、Transformers、Conda、
系统 FFmpeg、MSYS2/MinGW 或 WSL。开发者构建环境另有要求。

**v0.6.0-alpha** 加入机器可读
Product 合同，以及供桌面客户端、原生客户端、本地守护进程工具、CLI 辅助工具和
其他社区集成使用的本地 Native API。

v0.6.0-alpha 确定的五个 Public 模型路径为：

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

## 安装

现有 v0.7 Linux 下载方式及计划中的 Windows ZIP 布局见[安装说明](docs/install.md)。
**此次版本准备没有发布任何 v0.8 rc7 下载包。** 不得把 rc6 改名为 v0.8，
也不得把它追加到历史 v0.7 发布。

请保持完整包目录。Windows 包将 `vrhino.exe`、媒体助手和 39 个运行库 DLL
放在一起，不依赖系统 FFmpeg 或手动配置 CUDA 库路径。核对发布的 ZIP SHA256
后，从该包运行 `--version`、`device` 和 `doctor`。

## CLI 示例

v0.6.0-alpha 引入的模型包身份保持不变，各平台资格状态由 v0.8 支持矩阵单独记录：

- `vrhino/ltx-video-v0.9.1:1.1.1`
- `vrhino/wan2.1-t2v-1.3b:1.0.1`
- `vrhino/mochi-1-preview:1.0.1`
- `vrhino/musetalk-v1.5:1.0.1`
- `vrhino/latentsync-1.6:1.0.1`

以下示例使用 Linux shell 和已验证的安装包。Windows PowerShell 写法见
[安装说明](docs/install.md)。拉取一个确定的模型包，例如：

```bash
vrhino pull vrhino/ltx-video-v0.9.1:1.1.1
```

然后生成视频：

```bash
vrhino run vrhino/ltx-video-v0.9.1:1.1.1 \
  --prompt "a cat walking in snow" \
  --output output.mp4
```

MuseTalk 使用类型化的视频与音频输入，而不是文本提示词：

```bash
vrhino pull vrhino/musetalk-v1.5:1.0.1
vrhino run vrhino/musetalk-v1.5:1.0.1 \
  --video input.mp4 \
  --audio driving.wav \
  --output output.mp4
```

v0.6.0-alpha 包含以下 Public Mode-C 支持：

- `vrhino/latentsync-1.6:1.0.1`（`lip_sync`）

LatentSync 使用相同的类型化视频/音频 CLI 形式。VRhino 从上游下载 12 个固定的
推理文件并在本地转换；Pixel Rhino 不分发模型权重或转换后的 VRM。

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
vrhino pull vrhino/ltx-video-v0.9.1:1.1.1
```

这不会改变 VRhino 二进制文件的安装目录。可运行包完成校验并成功安装后，pull
会清理不再需要的源文件数据，同时保留已安装数据和共享 CAS 数据。

## 工作方式

`vrhino pull` 下载固定的上游模型版本，校验并缓存源文件，在本机以原生转换器
生成 VRhino 模型格式，然后安装成不可变的本地包。`vrhino run` 使用共享原生
运行时执行该模型，并通过随包提供的媒体组件输出 MP4。

successor 模型包通过
[ProductInputSchema v1](docs/product/product-input-schema-v1.md) 声明类型化输入、
参数、默认值和输出。客户端无需解析 CLI 文本即可查看同一合同：

```bash
vrhino info vrhino/ltx-video-v0.9.1:1.1.1 --json
```

主可执行文件还可启动 v0.6.0-alpha 引入的本地 Native API：

```bash
vrhino serve
# 等价的显式写法：
vrhino serve --host 127.0.0.1 --port 11435
```

服务端内置在主 `vrhino` 可执行文件中；没有独立服务端软件包，也不依赖 Python
或 Node 服务。Native API v1 alpha 是**本地 API**：默认监听
`127.0.0.1:11435`，不提供认证、TLS 或 CORS，也不应直接暴露到不受信任的
Internet。显式绑定非回环地址时会输出警告。完整七路由合同与本地绝对媒体路径
规则见 [Native API v1 合同](docs/api/native-api-v1.md)。

## 文档

- [安装与系统要求](docs/install.md)
- [v0.8 发布准备和资格验证范围](docs/release/v0.8.0-alpha.md)
- [模型命令](docs/cli/model-cli-v0.md)
- [ProductInputSchema v1](docs/product/product-input-schema-v1.md)
- [模型信息 JSON v1](docs/product/model-info-json-v1.md)
- [Native API v1 alpha](docs/api/native-api-v1.md)
- [v0.6.0-alpha 构建源可追溯边界](docs/release/build-source-provenance.md)
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

历史 Linux 资格与待完成的 v0.8 Windows 资格分开记录。发布前必须在精确 rc7
字节上通过四条 Windows 产品路径。Alpha 期间接口和兼容性可能变化；不宣称
覆盖所有 NVIDIA GPU、Windows/Linux 版本、模型检查点或架构，也不宣称支持 macOS。

MuseTalk 和 LatentSync 的技术执行、完整媒体校验、人工视觉合理性检查与客观
口型同步质量是不同结论。经校准的客观质量门禁仍独立延后；不宣称客观质量 PASS，
也不把该研究门禁的延后视为 Windows 专属技术阻塞。

详情见 [Alpha 限制](docs/alpha-limitations.md)。

## 从源码构建

Public main 包含生产源码、Shared Runtime、NeuralGraph、Backend、Converter 和原生 Product/API 实现。参见[源码构建与测试](docs/source-build.md)、[架构](docs/architecture.md)、[打包](docs/source-packaging.md)和[贡献指南](CONTRIBUTING.md)。源码存在不等于历史版本已支持；HunyuanVideo 和 CogVideoX canary 不加入 v0.6 已支持模型列表。

## 许可证

VRhino 项目自有源码采用 [Apache-2.0](LICENSE)。第三方组件保留各自许可证，参见[第三方源码声明](THIRD_PARTY_NOTICES.md)。现有已发布二进制继续适用其随包提供的 [Alpha Binary License](licenses/VRHINO-BINARY-LICENSE.txt)。

模型许可证彼此独立。VRhino 不授予模型权重、输入、输出或其他第三方内容的
任何权利。

对于 MuseTalk，Pixel Rhino 不分发模型权重；用户从固定的官方上游获取模型并
在本地转换。使用仍受各上游模型许可证约束，其中包括 MuseTalk 的 CreativeML
OpenRAIL-M 用途限制。用户须确保输入媒体的使用合法并已获得同意。本项目不暗示
任何上游权利人对 VRhino 的认可或背书。

LatentSync Public 模型包同样遵循不分发权重的 Mode-C 边界。
LatentSync 模型的使用仍受 CreativeML Open RAIL++-M License 及其 Attachment A
用途限制约束。
