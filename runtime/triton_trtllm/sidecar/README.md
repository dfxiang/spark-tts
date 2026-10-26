# BF16 TensorRT-LLM sidecar

This service runs only the Spark-TTS LLM through TensorRT-LLM and Triton. The
existing BiCodec tokenizer and vocoder are loaded once by the FastAPI process,
which serves the same OpenAI-compatible routes and HTML voice-cloning page as
the original service.

## Start

The default model path is `D:\models\spark-tts`. Override it with `-ModelDir`
if necessary:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\runtime\triton_trtllm\sidecar\run.ps1 `
  -ModelDir "D:\models\spark-tts"
```

If the NGC registry is slow, the same image can be sourced through the
`nvcr.m.daocloud.io` mirror with
`-BaseImage nvcr.m.daocloud.io/nvidia/tritonserver:25.02-trtllm-python-py3`.

The first startup converts the LLM checkpoint and builds the TensorRT engine.
This can take several minutes and uses about 1 GB for the BF16 engine. Later
starts reuse `sidecar/artifacts`.

Follow startup progress with:

```powershell
docker logs -f spark-tts-trtllm-api
```

Wait until the OpenAI-compatible service is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8168/healthz
```

## Fixed benchmark

The benchmark sends the same target text to the existing API on port `8168` and
to the sidecar on port `8180`. It reuses the voice
`voice_b241260c3376` and its reference WAV from the existing
`spark-tts-voices` volume. This benchmark targets the direct Triton pipeline;
the production OpenAI endpoint is available on port `8168`.

```powershell
docker exec spark-tts-trtllm-bf16 python `
  /workspace/runtime/triton_trtllm/benchmark_sidecar.py `
  --baseline-url http://host.docker.internal:8168 `
  --sidecar-url http://127.0.0.1:8000 `
  --output-dir /artifacts/benchmark_results
```

Results are written to:

```text
runtime/triton_trtllm/sidecar/artifacts/benchmark_results/
```

The JSON contains every run plus mean and median latency/RTF. Both baseline and
sidecar waveforms are saved as 16 kHz WAV files for listening and quality
comparison.

## Measured result

Measured on the local RTX 4060 Ti 16 GB with the same reference voice and target
text, one warmup followed by three measured requests per backend:

| Backend | Mean latency | Mean RTF |
|---|---:|---:|
| Existing PyTorch BF16 API | 11.93 s | 0.997 |
| Unified OpenAI-compatible TensorRT-LLM API | 3.31 s | 0.273 |

Latency improved by `3.61x`; RTF improved by `3.65x`. The TensorRT service generated
slightly longer audio than the sampled PyTorch baseline, so latency speedup and
RTF speedup are reported separately.

The TensorRT output was transcribed by the local FunASR service and matched the
fixed target text. The unified service uses about 6.4 GB of the 16 GB GPU.
