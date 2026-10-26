# Spark-TTS GPU 并发推理优化方案

## 目标

在同一张 GPU、同一模型实例、不进一步量化的情况下，让多个文本分段真正并发进入
TensorRT-LLM，而不是在 API 进程内排队。

目标场景：

- 长文本拆成约 10 个 64-128 字符的分段。
- 每段使用同一个音色和参考音频。
- 4 路并发合成，期望整组长文本耗时接近 `ceil(10 / 4)` 次单批推理时间。
- 保留原有单请求 `/v1/audio/speech` 接口。
- 新增批量接口，减少客户端调度和 HTTP 往返开销。

## 当前瓶颈

当前 `perf/gpu-fast-inference` 分支虽然可以同时接受多个 HTTP 请求，但 GPU 推理仍
被以下限制串行化：

1. `UnifiedTensorRTEngine.synthesize()` 使用全局 `RLock` 包住整个
   `model.inference()`。
2. `TritonLLMClient` 只提交 `batch size = 1` 的请求。
3. TensorRT-LLM 引擎构建参数为 `--max_batch_size 1`。
4. Triton `config.pbtxt` 的 `max_batch_size` 和 `preferred_batch_size` 都是 `1`，
   动态批处理等待时间为 `0`。
5. 音频 decoder 当前也随整个模型锁串行执行。

因此现有 2 路和 4 路测试的聚合 RTF 基本不变，增加并发只会增加排队延迟。

## 优化架构

```text
批量文本请求或并发单请求
          |
          v
最多 4 个推理槽 + Triton in-flight batching
          |
          v
TensorRT-LLM BF16 engine, max_batch_size=4
          |
          +--> 按请求提取 semantic/global token
          |
          v
BiCodec detokenize
          |
          v
按输入顺序返回音频
```

### 1. TensorRT-LLM 真批处理

- 保持 BF16，不引入 W8A16、FP8 或其它新量化。
- 构建独立引擎目录 `trt_engines_bf16_b4`。
- 引擎参数：
  - `max_batch_size=4`
  - `max_seq_len=4096`
  - `max_num_tokens=16384`
- Triton 配置：
  - `max_batch_size=4`
  - `preferred_batch_size=[4]`
  - `max_queue_delay_microseconds=2000`
  - paged KV cache 至少 `16384` tokens

### 2. API 并发控制

- 将模型加载锁与推理锁分离。
- 删除覆盖整个人工推理链路的全局锁。
- 使用大小为 `SPARK_TTS_TRT_MAX_BATCH_SIZE` 的 semaphore 限制同时进入
  TensorRT-LLM 的请求数。
- 由 Triton 的 `inflight_fused_batching` 合并同一时间窗口内的请求。
- 保留模型加载阶段的互斥，避免重复初始化。

### 3. 批量 API

新增：

```http
POST /v1/audio/speech/batch
```

请求示例：

```json
{
  "model": "spark-tts",
  "inputs": ["第一段文本。", "第二段文本。"],
  "voice": "low-female1",
  "response_format": "wav",
  "speed": 1.0
}
```

服务端行为：

1. 按输入顺序完成中文口语化转写。
2. 最多 4 路并发调用推理引擎。
3. Triton 将并发请求合并为 BF16 batch。
4. 按输入顺序拼接波形，只编码一次最终音频。

批量接口适合长文本一次性提交；原有单请求接口仍可由客户端并发调用。

### 4. 长文本接口

新增：

```http
POST /v1/audio/long_text
```

请求字段与 `/v1/audio/speech` 保持一致：

```json
{
  "model": "spark-tts",
  "input": "需要合成的完整长文本",
  "voice": "low-female1",
  "response_format": "wav",
  "speed": 1.0,
  "min_segment_chars": 32,
  "max_segment_chars": 128
}
```

服务端处理流程：

1. 单次输入最多 `4096` 个字符。
2. 先移除换行、制表符、控制字符、零宽字符和中文排版空格；英文单词之间
   只保留一个空格，避免把 `I need` 变成 `Ineed`。
3. 按中文或英文标点计算边界，默认以 `32-128` 字符为目标进行分段。
4. 不会为了满足长度上限而在无标点位置硬切；无法找到范围内边界时，会延伸到下一个标点。
5. 分段完成后逐段执行中文数字、日期、时间和百分比口语化。
6. 所有分段进入批量推理，最多 4 条同时提交给 TensorRT-LLM。
7. 按原始分段顺序拼接 PCM，只执行一次最终音频编码。
8. 返回格式与 `/v1/audio/speech` 相同，并增加以下响应头：
   - `duration`：最终音频时长，单位秒，已考虑语速。
   - `rtf`：本次推理耗时除以原始合成音频时长。
   - `batch-size`：本次实际使用的并发 batch 大小，最大为 4。
   - `X-Spark-TTS-Segments`：长文本实际分段数。

### 5. 后续可选优化

以下项目作为第二阶段，不纳入本次必须实现：

- 为 BiCodec decoder 增加变长 token padding 和真正的 batch detokenize。
- 将 LLM 生成和 audio decoder 拆成流水线，减少 tail latency。
- 评估 TensorRT-LLM KV cache/prefix reuse，复用相同参考音色的公共 prompt。
- 使用 Nsight Systems 或 torch profiler 拆分 prompt、generation、decoder 和
  HTTP/编码耗时。
- 多 GPU 时部署独立副本；单 GPU 上不要简单启动多个完整模型实例。

## 预期收益

假设 10 段总串行推理时间为 30 秒，每段约 3 秒：

- 4 路真正并发且每批开销接近单条时，理论下限约为
  `ceil(10 / 4) × 3 = 9 秒`。
- 6 秒目标需要 batch 内部效率或 prompt 复用进一步降低单批耗时，不能只靠
  “同时发 4 个 HTTP 请求”实现。
- 当 batch 成本随批量亚线性增长时，聚合 RTF 和吞吐量会明显改善。
- 单请求延迟在负载高时仍会上升，应同时报告 P50/P95，而不只看总时长。

## 验证方案

1. 编译 BF16 batch=4 TensorRT 引擎。
2. 验证单条请求输出与优化前兼容。
3. 使用固定输入分别测试 1、2、4 并发。
4. 对比聚合 RTF、总墙钟时间、吞吐和 P50/P95 延迟。
5. 用长文本批量接口测试 10 段顺序拼接。
6. 检查音频是否存在超过 3 秒的连续静音。
7. 确认失败请求不会导致返回音频乱序。

## 本地验证结果

验证环境：

- NVIDIA GeForce RTX 4060 Ti 16 GB
- TensorRT-LLM 0.17.0.post1
- BF16，`max_batch_size=4`
- 音色：`low-female1`
- 固定文本，每组 8 个请求，正式测试前预热 1 次

优化前后对比：

| 并发数 | 优化前聚合 RTF | 优化后聚合 RTF | 优化前墙钟 (s) | 优化后墙钟 (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.266 | 0.262 | 43.845 | 43.217 |
| 2 | 0.259 | 0.152 | 42.666 | 23.509 |
| 4 | 0.261 | 0.080 | 42.941 | 12.305 |

4 并发下：

- 8 个请求全部成功。
- 平均单请求延迟约 `6.04s`，P50 为 `6.14s`。
- 聚合 RTF 从约 `0.26` 降低到 `0.080`。
- 相对单路墙钟时间提速约 `3.5x`。
- 相同墙钟时间内生成的音频约为实时的 `12.5x`。

长文本批量接口验证：

- 输入为 `tests/tts-test1.txt`，按标点拆分为 9 段。
- 原串行分段调用约 `69.6s`。
- `long_text` 接口约 `27.1s`，端到端提速约 `2.57x`。
- 合成音频总时长约 `283.4s`，聚合 RTF 为 `0.096`。
- PCM 检测最长连续静音为 `0.94s`，未发现超过 3 秒的异常静音。

长文本批量接口没有达到与 4 个独立请求相同的 `3.5x`，说明 LLM 批处理已经生效，
但 BiCodec detokenize 和最终音频编码仍有进一步流水化空间。

## 回滚

- 将 `SPARK_TTS_TRT_MAX_BATCH_SIZE` 设为 `1`，并使用
  `trt_engines_bf16` 原引擎即可回滚到单请求路径。
- 不同 batch size 使用不同引擎目录，避免覆盖已验证的 batch=1 引擎。
- 原始单请求 API 行为保持不变。
