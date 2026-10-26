<div align="center">
  <img src="src/logo/SparkTTS.jpg" alt="Spark-TTS Logo" width="180">

  <h1>Spark-TTS 工程优化版</h1>

  <p>
    <strong>面向中文长文本、实时语音克隆和 OpenAI API 兼容场景的生产化 Spark-TTS</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-10a37f" alt="OpenAI API">
    <img src="https://img.shields.io/badge/Acceleration-TensorRT--LLM-76b900" alt="TensorRT-LLM">
    <img src="https://img.shields.io/badge/Deploy-CPU%20%7C%20CUDA-blue" alt="CPU and CUDA">
    <img src="https://img.shields.io/badge/Code%20License-Apache%202.0-yellow" alt="Apache 2.0">
    <img src="https://img.shields.io/badge/Model%20License-CC%20BY--NC--SA%204.0-lightgrey" alt="CC BY-NC-SA 4.0">
  </p>
</div>

## 实测性能对比

测试环境为 NVIDIA RTX 4060 Ti 16 GB，使用 BF16，固定文本测试前预热一次。
RTF 越低越好；RTF 小于 `1.0` 表示生成速度快于实时播放。

| 测试场景 | 指标 | 优化前 | 优化后 | 改善 |
|---|---|---:|---:|---:|
| 固定文本单请求 | 平均延迟 | 11.93 s | 3.31 s | 3.61x |
| 固定文本单请求 | 平均 RTF | 0.997 | 0.273 | 3.65x |
| 固定文本，8 请求 | 4 并发总墙钟时间 | 42.941 s | 12.305 s | 3.49x |
| 固定文本，8 请求 | 4 并发聚合 RTF | 0.261 | **0.080** | 降低 69.3% |
| 9 段长文本 | 端到端耗时 | 69.6 s | 27.1 s | 2.57x |

单请求数据比较 PyTorch BF16 API 与 TensorRT-LLM BF16 API；并发数据比较
串行推理与 4 路 in-flight batching。两组的测试文本、生成音频长度和运行
后端并不完全相同，因此应分别解读，不能直接横向比较。优化后 4 并发可达到约 `12.51x`
实时音频吞吐。

### 并发基准明细

| 并发数 | 优化前聚合 RTF | 优化后聚合 RTF | 优化前墙钟 (s) | 优化后墙钟 (s) | 优化后吞吐 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.266 | 0.262 | 43.845 | 43.217 | 3.81x 实时 |
| 2 | 0.259 | 0.152 | 42.666 | 23.509 | 6.60x 实时 |
| 4 | 0.261 | **0.080** | 42.941 | **12.305** | **12.51x 实时** |

每组提交 8 个固定文本请求，响应格式为 16 kHz WAV。性能数据来自仓库内的
[单路并发基准](benchmarks/2026-10-23-spark-tts-api-rtf.md)、
[4 路批处理基准](benchmarks/2026-10-23-spark-tts-api-rtf-batch4.md) 和
[TensorRT-LLM 对比基准](runtime/triton_trtllm/sidecar/README.md)。
实际结果会受 GPU、模型精度、文本长度和批处理配置影响。

## 项目定位

本项目基于 [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)，
针对实际部署中常见的**推理速度、长文本稳定性、音色管理和接口集成**问题做了系统性优化。
在保留 Spark-TTS 零样本音色克隆能力的基础上，提供完整的 Web UI、REST API、
CPU/CUDA 部署方案以及可复现的基准测试。

## 项目亮点

- **新增可视化 Web 界面**：无需编写代码，打开浏览器即可选择音色、输入文本、
  生成并试听语音，同时支持声音管理和长文本模式。
- **长文本直接生成一段完整音频**：粘贴最多 4096 字符的长文本，系统会自动按
  中英文标点分段、并行合成，再按原文顺序拼接为一个完整音频文件，可直接播放或下载。
- **推理速度显著提升**：在 RTX 4060 Ti 16 GB 上，约 10 秒音频的生成耗时由
  11–13 秒降至通常不到 3 秒。
- **生产化接口与部署**：提供 OpenAI 兼容 API、音色管理接口、Docker 容器和
  CPU/CUDA 双后端，方便接入现有业务系统。

## 为什么选择这个版本

| 能力 | 说明 |
|---|---|
| **实时级 GPU 推理** | TensorRT-LLM BF16、4 路 in-flight batching、参考音频 Token 缓存和减少 GPU/CPU 往返 |
| **稳定生成长文本** | 最多 4096 字符，按中英文标点智能分段，保持英文单词边界并维持原文顺序 |
| **面向中文的文本规范化** | 自动将数字、日期、时间、百分比等转换为更适合播报的口语表达 |
| **完整音色管理** | 上传、录音、试听、重命名、替换参考音频和删除；支持用唯一名称或 Voice ID 调用 |
| **OpenAI 兼容接口** | 支持 `/v1/audio/speech`、`/v1/models`、标准 `voice` 参数和 6 种音频格式 |
| **可直接使用的前端** | 提供合成页、声音管理页、长文本模式和实时耗时指标 |
| **从本机到容器** | 支持 CPU、PyTorch CUDA、NVIDIA TensorRT-LLM 统一容器 |
| **可验证的工程质量** | 单元测试、RTF 基准、并发测试、长文本与异常静音检测 |

## 核心能力

### 1. 4 路批处理与低延迟推理

GPU 服务将原来的全局推理锁拆分为加载锁和推理并发控制，允许最多 4 个请求
进入 TensorRT-LLM，并由 Triton 的 in-flight batching 合并计算。参考音频的
BiCodec Token 会按文件和修改时间缓存，避免重复编码同一音色。

```text
单请求 / 批量请求 / 长文本
           │
           ▼
   FastAPI + OpenAI 兼容层
           │
           ▼
  最多 4 路并发推理槽
           │
           ▼
 TensorRT-LLM BF16 + Triton
           │
           ▼
 BiCodec 解码 → 静音清理 → 音频编码
```

### 2. 长文本不会粗暴截断

`POST /v1/audio/long_text` 会先清理控制字符、零宽字符和异常空白，再根据中英文
标点建立分段边界。默认目标长度为 `32–128` 字符，不会为了满足上限在无标点处
硬切；后续分段可并行推理，最终按原始顺序拼接。

仓库内的 9 段长文本测试中：

- 串行分段调用约 `69.6s`
- 长文本批量接口约 `27.1s`
- 端到端提速约 `2.57x`
- 合成音频约 `283.4s`，聚合 RTF 为 `0.096`
- 最长连续静音 `0.94s`，未发现超过 3 秒的异常静音

详见 [长文本稳定性报告](benchmarks/2026-10-23-tts-test1-stability.md)。

### 3. 中文播报更自然

推理前会自动处理中文口语场景中的常见表达，例如：

- `2026年` → `二零二六年`
- `18:30` → `十八点三十分`
- `35%` → `百分之三十五`
- 金额、电话、小数及其他常见数字表达

原始文本和口语化文本之间仍保留明确处理边界，方便排查合成效果。

### 4. 音色可持久化管理

声音保存在 `SPARK_TTS_VOICE_DIR`，每个音色包含参考 WAV 和
`metadata.json`。生成时可以直接使用显示名称；名称重复时会要求改用唯一
Voice ID，避免误选。

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/api/voices` | 列出全部音色 |
| `POST` | `/api/voices` | 上传参考音频并创建音色 |
| `GET` | `/api/voices/{voice}` | 查询音色信息 |
| `PATCH` | `/api/voices/{voice}` | 修改名称或参考文本 |
| `GET` | `/api/voices/{voice}/audio` | 获取参考音频 |
| `PUT` | `/api/voices/{voice}/audio` | 替换参考音频 |
| `DELETE` | `/api/voices/{voice}` | 删除音色 |

浏览器访问 `/voices` 即可使用完整管理页面。更多约定见
[声音管理文档](docs/voice-management.md)。

### 5. 标准 OpenAI 接口与多格式输出

基础接口与 OpenAI Audio Speech API 的调用方式一致，并将音色作为可复用资源：

```http
POST /v1/audio/speech
```

```json
{
  "model": "spark-tts",
  "input": "你好，欢迎使用 Spark-TTS。",
  "voice": "客服女声",
  "response_format": "mp3",
  "speed": 1.0
}
```

支持 `mp3`、`opus`、`aac`、`flac`、`wav` 和 `pcm`。响应头会返回
`duration`、`rtf`、`batch-size` 和 `X-Spark-TTS-Segments` 等运行指标。

## 快速开始

### 1. 安装环境

推荐 Python 3.12：

```bash
git clone <your-repository-url>
cd spark-tts

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

国内网络环境可以使用镜像：

```bash
pip install -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com
```

### 2. 下载模型

```bash
python dl-models.py pretrained_models/Spark-TTS-0.5B
```

模型目录最终应包含：

```text
pretrained_models/Spark-TTS-0.5B/
├── config.yaml
├── LLM/
├── BiCodec/
└── wav2vec2-large-xlsr-53/
```

### 3. 启动 CPU 服务

```bash
SPARK_TTS_MODEL_DIR=pretrained_models/Spark-TTS-0.5B \
SPARK_TTS_DEVICE=cpu \
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

打开以下页面：

- 语音合成：<http://127.0.0.1:8000/>
- 声音管理：<http://127.0.0.1:8000/voices>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/healthz>

### 4. 创建音色

```bash
curl -X POST http://127.0.0.1:8000/api/voices \
  -F "audio=@example/prompt_audio.wav" \
  -F "prompt_text=参考音频对应的准确文字" \
  -F "name=示例音色"
```

返回结果中的 `id` 或唯一显示名称都可以作为后续请求的 `voice` 参数。

### 5. 合成语音

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spark-tts",
    "input": "这是由 Spark-TTS 生成的示例语音。",
    "voice": "示例音色",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

## 部署方式

### CPU API 容器

根目录 `Dockerfile` 使用 CPU 版 PyTorch，并内置 `ffmpeg`：

```bash
docker build -f Dockerfile -t spark-tts-api:cpu .

docker run --rm \
  -p 8000:8000 \
  -v /绝对路径/Spark-TTS-0.5B:/models/Spark-TTS-0.5B:ro \
  -v spark-tts-voices:/data/voices \
  spark-tts-api:cpu
```

### NVIDIA TensorRT-LLM 容器

`Dockerfile.cuda` 将 FastAPI、TensorRT-LLM、Triton 和 BiCodec 组合为一个服务：

```bash
docker build -f Dockerfile.cuda -t spark-tts-api:bf16 .

docker run -d --name spark-tts-api \
  --gpus "device=0" \
  --shm-size 4g \
  -p 8168:8000 \
  -p 8169:8101 \
  -p 8170:8102 \
  -v /绝对路径/Spark-TTS-0.5B:/models/Spark-TTS-0.5B:ro \
  -v spark-tts-voices:/data/voices \
  -v /绝对路径/spark-tts-artifacts:/artifacts \
  spark-tts-api:bf16
```

首次启动会转换检查点并编译 TensorRT 引擎，通常需要数分钟；后续启动会复用
`/artifacts` 中的结果。更多说明见
[TensorRT-LLM 运行文档](runtime/triton_trtllm/sidecar/README.md)。

### Docker 镜像再分发说明

本仓库只提供 Dockerfile，不随源代码发布预构建镜像。根目录的 Apache 2.0
许可证不会替代镜像中所含基础镜像、系统软件、Python 包、运行工具或模型权重的
独立许可。自行构建并公开分发镜像时，需要注意：

- `Dockerfile` 通过 Debian 安装 FFmpeg。FFmpeg 可能包含 GPL 或 LGPL
  组件，需要保留 `/usr/share/doc/` 中的版权与许可文件，并按所使用构建的条款
  提供相应源码或书面要约。具体边界见
  [FFmpeg Legal](https://ffmpeg.org/legal.html)。
- `Dockerfile.cpu` 的 `funasr-cpu:26.9.18` 是预置基础镜像。公开分发前应确认
  该镜像及其中 FunASR 组件的来源、许可证和再分发权限，不能默认由本项目的
  Apache 2.0 许可证覆盖。
- `Dockerfile.cuda` 使用 `nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3`。
  NVIDIA NGC、Triton、TensorRT-LLM 和 CUDA 组件适用各自的 NVIDIA 软件条款；
  进入镜像的 NVIDIA 文件仍需保留其 BSD-3-Clause 声明。
- 镜像中安装的 Python 包继续受各自许可证约束。分发时应保留
  `*.dist-info` 中的许可文件；其中 `soxr` 使用 LGPL-2.1-or-later，
  `tqdm` 使用 MPL-2.0 与 MIT。
- 模型权重采用 CC BY-NC-SA 4.0，不属于代码许可证。运行命令通过只读卷挂载
  权重时，应把模型许可随交付物一并提供给使用者；如果把权重烘焙进镜像，镜像
  分发同样受非商业和相同方式共享条款约束。
- 对外发布镜像前，应把仓库中的
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 以及镜像内各组件的许可
  文件一并加入最终发布物。当前 Dockerfile 不会自动生成完整的第三方许可汇总。

面向商业用途的镜像或服务还需要另行取得模型权利方授权。以上说明用于提示
再分发义务，不构成法律意见。

### 原版 Gradio UI

仍保留上游 Web UI：

```bash
python webui.py \
  --model_dir pretrained_models/Spark-TTS-0.5B \
  --device 0 \
  --server_name 0.0.0.0 \
  --server_port 8720
```

## 主要 API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/healthz` | 服务、模型和运行状态 |
| `GET` | `/v1/models` | OpenAI 模型列表兼容 |
| `POST` | `/v1/audio/speech` | 单段语音合成 |
| `POST` | `/v1/audio/speech/batch` | 多段文本批量合成 |
| `POST` | `/v1/audio/long_text` | 自动分段的 4096 字符长文本合成 |
| `GET/POST` | `/api/voices` | 查询或创建音色 |
| `GET/PATCH/DELETE` | `/api/voices/{voice}` | 管理音色 |
| `GET/PUT` | `/api/voices/{voice}/audio` | 管理参考音频 |

批量接口示例：

```json
{
  "model": "spark-tts",
  "inputs": ["第一段文本。", "第二段文本。"],
  "voice": "示例音色",
  "response_format": "wav",
  "speed": 1.0
}
```

长文本接口示例：

```json
{
  "model": "spark-tts",
  "input": "需要合成的完整长文本",
  "voice": "示例音色",
  "response_format": "wav",
  "speed": 1.0,
  "min_segment_chars": 32,
  "max_segment_chars": 128
}
```

## 常用配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SPARK_TTS_MODEL_DIR` | `pretrained_models/Spark-TTS-0.5B` | 模型目录 |
| `SPARK_TTS_VOICE_DIR` | `data/voices` | 音色数据目录 |
| `SPARK_TTS_DEVICE` | `cpu` | `cpu`、`cuda:0` 等设备 |
| `SPARK_TTS_LLM_DTYPE` | `auto` | `auto`、`bfloat16`、`float16`、`float32` |
| `SPARK_TTS_PROMPT_CACHE_SIZE` | `16` | 缓存参考音频 Token 的数量 |
| `SPARK_TTS_PRELOAD` | `1` | 启动时是否预加载模型 |
| `SPARK_TTS_API_KEY` | 空 | 设置后启用 Bearer Token 鉴权 |
| `SPARK_TTS_MAX_TEXT_CHARS` | `2000` | 普通请求最大字符数 |
| `SPARK_TTS_MAX_LONG_TEXT_CHARS` | `4096` | 长文本最大字符数 |
| `SPARK_TTS_MAX_BATCH_ITEMS` | `128` | 单次批量请求最大条目数 |
| `SPARK_TTS_MAX_UPLOAD_MB` | `20` | 参考音频上传大小上限 |
| `SPARK_TTS_MAX_REFERENCE_SECONDS` | `30` | 参考音频最长时长 |
| `SPARK_TTS_TRT_PRECISION` | `bf16` | TensorRT-LLM 精度 |
| `SPARK_TTS_TRT_MAX_BATCH_SIZE` | `4` | TensorRT-LLM 最大批大小 |

## 测试与基准

运行单元测试：

```bash
pytest -q
```

执行端到端 RTF 基准：

```bash
python -m runtime.triton_trtllm.benchmark_api_rtf \
  --base-url http://127.0.0.1:8168 \
  --voice "示例音色" \
  --levels 1,2,4
```

测试覆盖 OpenAI API、音色管理、长文本分段、中文口语化、音频清理、
TensorRT-LLM 客户端、精度配置和推理性能。相关设计和结果位于：

- [GPU 并发推理优化方案](docs/tts-gpu-concurrency-optimization.md)
- [单路与并发 RTF 基准](benchmarks/2026-10-23-spark-tts-api-rtf.md)
- [4 路批处理基准](benchmarks/2026-10-23-spark-tts-api-rtf-batch4.md)
- [长文本稳定性报告](benchmarks/2026-10-23-tts-test1-stability.md)

## 项目来源与致谢

基础模型和原始实现来自
[SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)。论文：

> Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with
> Single-Stream Decoupled Speech Tokens

- [论文](https://arxiv.org/abs/2503.01710)
- [项目主页](https://sparkaudio.github.io/spark-tts/)
- [模型页面](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

本项目保留上游代码许可、模型架构和署名信息，并在工程化、推理性能和部署能力上
进行了扩展。代码与模型权重适用不同许可证，使用前请阅读下方
[许可证与使用边界](#许可证与使用边界)。

## 使用声明

本项目的音色克隆能力仅应用于合法、获得授权的场景。严禁用于未经许可的
声音模仿、身份冒充、欺诈、深度伪造或其他违法用途。使用者应自行确保数据、
声音权利和生成内容符合当地法律法规及平台政策。

## 许可证与使用边界

### 代码许可

本仓库代码基于上游 Spark-TTS 修改，并继续采用
[Apache License 2.0](LICENSE)。当前 `LICENSE` 文件与上游
`SparkAudio/Spark-TTS` 的 Apache 2.0 许可文本逐字节一致。对上游文件的修改
已在相应源文件头部标注，完整变更记录见 Git 历史。

### 模型权重许可

模型权重不包含在上述 Apache 2.0 授权中。根据上游模型卡，`Spark-TTS-0.5B`
权重已从 Apache 2.0 更新为
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)：

- 仅允许非商业用途。
- 使用或修改模型时必须保留适当署名。
- 修改作品和衍生作品必须继续采用 CC BY-NC-SA 4.0。

如果需要将模型权重用于商业场景，应另行取得模型权利方的授权。模型许可原文见
[Hugging Face 模型页面](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)。

### 第三方组件

`sparktts/modules/blocks/layers.py`、
`sparktts/modules/encoder_decoder/wave_generator.py` 和
`sparktts/modules/speaker/perceiver_encoder.py` 包含改编自 MIT 项目的代码。
对应版权和完整 MIT 许可文本见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

`runtime/triton_trtllm` 中部分 NVIDIA Triton/TensorRT-LLM 文件带有
BSD-3-Clause 声明，相关版权和许可头均已保留。BSD-3-Clause 与 Apache 2.0
兼容，不构成协议冲突。

Python 包、系统包、Docker 基础镜像和模型权重不属于上述代码许可范围。分发
源码或镜像时的具体注意事项见[Docker 镜像再分发说明](#docker-镜像再分发说明)。

## 交个朋友 / 支持项目

如果这个项目对你有帮助，欢迎请我喝杯咖啡；也欢迎加微信交个朋友，聊聊音频、产品和技术。
支持完全自愿，不影响任何使用权益。

<table>
  <tr>
    <td align="center">
      <img src="weixin.jpg" alt="作者个人微信二维码" width="260"><br>
      <sub>扫码加微信，交个朋友</sub>
    </td>
    <td align="center">
      <img src="donation.jpg" alt="项目捐赠二维码" width="260"><br>
      <sub>请我喝杯咖啡，支持项目持续维护</sub>
    </td>
  </tr>
</table>
