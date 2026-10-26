"""Benchmark end-to-end Spark-TTS API RTF at multiple concurrency levels."""

from __future__ import annotations

import argparse
import io
import json
import math
import platform
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TEXT = (
    "人工智能正在改变我们的工作与生活方式，语音合成技术也在快速进步。"
    "今天我们用一段固定的中文测试文本，比较不同并发条件下的实时率和响应延迟。"
)


@dataclass(frozen=True)
class RequestMeasurement:
    index: int
    latency_seconds: float
    audio_seconds: float
    response_bytes: int

    @property
    def rtf(self) -> float:
        return self.latency_seconds / self.audio_seconds


@dataclass(frozen=True)
class LevelResult:
    concurrency: int
    wall_seconds: float
    requests: tuple[RequestMeasurement, ...]
    errors: tuple[str, ...]


class BenchmarkError(RuntimeError):
    """Raised when the benchmark endpoint returns an invalid response."""


def _request_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wav_duration_seconds(content: bytes) -> float:
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            frame_rate = audio.getframerate()
            frame_count = audio.getnframes()
    except (EOFError, wave.Error) as exc:
        raise BenchmarkError("response is not a valid WAV file") from exc
    if frame_rate <= 0 or frame_count <= 0:
        raise BenchmarkError("response WAV has no audio frames")
    return frame_count / frame_rate


def _post_speech(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[float, int]:
    request = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"HTTP {exc.code}: {detail[:1000] or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"request failed: {exc.reason}") from exc

    if "audio/wav" not in content_type:
        raise BenchmarkError(
            f"expected audio/wav, got {content_type or 'no content type'}"
        )
    return _wav_duration_seconds(content), len(content)


def _timed_request(
    *,
    index: int,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[RequestMeasurement | None, float, str | None]:
    started_at = time.perf_counter()
    try:
        audio_seconds, response_bytes = _post_speech(
            base_url=base_url,
            payload=payload,
            timeout=timeout,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        return None, elapsed, f"request {index}: {type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started_at
    return (
        RequestMeasurement(
            index=index,
            latency_seconds=elapsed,
            audio_seconds=audio_seconds,
            response_bytes=response_bytes,
        ),
        elapsed,
        None,
    )


def _run_level(
    *,
    base_url: str,
    payload: dict[str, Any],
    concurrency: int,
    request_count: int,
    timeout: float,
) -> LevelResult:
    started_at = time.perf_counter()
    measurements: list[RequestMeasurement] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _timed_request,
                index=index,
                base_url=base_url,
                payload=payload,
                timeout=timeout,
            )
            for index in range(1, request_count + 1)
        ]
        for future in as_completed(futures):
            measurement, _, error = future.result()
            if measurement is not None:
                measurements.append(measurement)
            if error is not None:
                errors.append(error)
    wall_seconds = time.perf_counter() - started_at
    measurements.sort(key=lambda item: item.index)
    return LevelResult(
        concurrency=concurrency,
        wall_seconds=wall_seconds,
        requests=tuple(measurements),
        errors=tuple(errors),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _check_service(base_url: str, timeout: float) -> dict[str, Any]:
    health = _request_json(f"{base_url}/healthz", timeout)
    if health.get("status") != "ok":
        raise BenchmarkError(f"service is not healthy: {health}")
    return health


def _gpu_description() -> str:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    return "; ".join(lines) if process.returncode == 0 and lines else "unavailable"


def _summary(level: LevelResult) -> dict[str, float | int]:
    successful = level.requests
    latencies = [item.latency_seconds for item in successful]
    request_rtfs = [item.rtf for item in successful]
    audio_seconds = sum(item.audio_seconds for item in successful)
    result: dict[str, float | int] = {
        "concurrency": level.concurrency,
        "requests": len(successful) + len(level.errors),
        "successes": len(successful),
        "errors": len(level.errors),
        "wall_seconds": level.wall_seconds,
        "audio_seconds": audio_seconds,
    }
    if successful:
        result.update(
            {
                "mean_latency_seconds": statistics.fmean(latencies),
                "p50_latency_seconds": statistics.median(latencies),
                "p95_latency_seconds": _percentile(latencies, 95),
                "mean_request_rtf": statistics.fmean(request_rtfs),
                "aggregate_rtf": level.wall_seconds / audio_seconds,
                "audio_seconds_per_wall_second": audio_seconds
                / level.wall_seconds,
            }
        )
    return result


def _format_report(
    *,
    generated_at: str,
    base_url: str,
    voice: str,
    model: str,
    text: str,
    health: dict[str, Any],
    levels: Sequence[LevelResult],
    warmups: int,
    title: str,
) -> str:
    summaries = [_summary(level) for level in levels]
    gpu = _gpu_description()
    successful_summaries = [item for item in summaries if item["successes"]]
    total_successes = sum(int(item["successes"]) for item in summaries)
    total_requests = sum(int(item["requests"]) for item in summaries)
    best_summary = (
        min(successful_summaries, key=lambda item: float(item["aggregate_rtf"]))
        if successful_summaries
        else None
    )
    serial_summary = next(
        (item for item in summaries if item["concurrency"] == 1),
        None,
    )
    lines = [
        f"# {title}",
        "",
        f"- 生成时间：`{generated_at}`",
        f"- 服务地址：`{base_url}`",
        f"- 模型：`{model}`",
        f"- 音色：`{voice}`",
        f"- 服务设备：`{health.get('device', 'unknown')}`",
        f"- LLM 精度：`{health.get('llm_dtype', 'unknown')}`",
        f"- GPU：`{gpu}`",
        f"- 客户端：`Python {platform.python_version()} / {platform.platform()}`",
        f"- 每组请求数：`{len(levels[0].requests) + len(levels[0].errors)}`"
        if levels
        else "- 每组请求数：`0`",
        f"- 预热请求数：`{warmups}`",
        "- 响应格式：`audio/wav`，16 kHz",
        "",
        "## 测试文本",
        "",
        f"> {text}",
        "",
        "## 结果概览",
        "",
    ]
    if best_summary is None:
        lines.append(f"- 共提交 `{total_requests}` 个请求，全部失败。")
    else:
        lines.extend(
            [
                f"- 共提交 `{total_requests}` 个请求，成功 `{total_successes}` 个。",
                (
                    f"- 最佳聚合 RTF 为 `{float(best_summary['aggregate_rtf']):.3f}`，"
                    f"出现在 `{best_summary['concurrency']}` 并发。"
                ),
            ]
        )
        if serial_summary and serial_summary["successes"]:
            serial_rtf = float(serial_summary["aggregate_rtf"])
            serial_p50 = float(serial_summary["p50_latency_seconds"])
            comparisons = []
            for item in successful_summaries:
                if item is serial_summary:
                    continue
                delta = (float(item["aggregate_rtf"]) / serial_rtf - 1.0) * 100
                direction = "上升" if delta > 0 else "下降"
                comparisons.append(
                    f"`{item['concurrency']}` 并发的聚合 RTF 同比{direction} "
                    f"`{abs(delta):.1f}%`，P50 延迟 "
                    f"`{float(item['p50_latency_seconds']):.2f}s`"
                )
            lines.append(
                f"- 串行 P50 延迟为 `{serial_p50:.2f}s`；"
                + "；".join(comparisons)
                + "。"
            )
            if int(best_summary["concurrency"]) == 1:
                lines.append(
                    "- 当前部署下，提高并发没有改善聚合吞吐，主要增加了请求排队延迟。"
                )
    lines.extend(
        [
            "",
        "## 指标定义",
        "",
        "- **延迟**：从发出 HTTP 请求到完整收到 WAV 音频的端到端时间。",
        "- **单请求 RTF**：`请求延迟 / 返回音频时长`，越小越好。",
        "- **聚合 RTF**：`整组墙钟时间 / 整组生成音频总时长`，包含并发排队影响。",
        "- **音频吞吐**：每秒墙钟时间生成的音频秒数；值等于 `1 / 聚合 RTF`。",
        "- RTF 小于 `1.0` 表示生成速度快于实时播放。",
        "",
        "## 汇总结果",
        "",
        "| 并发数 | 成功/请求 | 墙钟时间 (s) | 音频总时长 (s) "
        "| 平均延迟 (ms) | P50 延迟 (ms) | P95 延迟 (ms) "
        "| 聚合 RTF | 平均单请求 RTF | 音频吞吐 (x realtime) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        if item["successes"]:
            lines.append(
                f"| {item['concurrency']} | {item['successes']}/{item['requests']} "
                f"| {item['wall_seconds']:.3f} | {item['audio_seconds']:.3f} "
                f"| {float(item['mean_latency_seconds']) * 1000:.1f} "
                f"| {float(item['p50_latency_seconds']) * 1000:.1f} "
                f"| {float(item['p95_latency_seconds']) * 1000:.1f} "
                f"| {float(item['aggregate_rtf']):.3f} "
                f"| {float(item['mean_request_rtf']):.3f} "
                f"| {float(item['audio_seconds_per_wall_second']):.3f} |"
            )
        else:
            lines.append(
                f"| {item['concurrency']} | 0/{item['requests']} "
                f"| {item['wall_seconds']:.3f} | 0.000 | - | - | - | - | - | - |"
            )

    lines.extend(["", "## 请求明细", ""])
    for level in levels:
        attempts = len(level.requests) + len(level.errors)
        lines.extend(
            [
                f"### 并发 {level.concurrency}",
                "",
                f"发起 `{attempts}` 个请求，成功 `{len(level.requests)}` 个。",
                "",
                "| 请求 | 延迟 (s) | 音频时长 (s) | 单请求 RTF | 响应大小 (KiB) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for request in level.requests:
            lines.append(
                f"| {request.index} | {request.latency_seconds:.3f} "
                f"| {request.audio_seconds:.3f} | {request.rtf:.3f} "
                f"| {request.response_bytes / 1024:.1f} |"
            )
        if not level.requests:
            lines.append("| - | - | - | - | - |")
        if level.errors:
            lines.extend(["", "失败请求：", ""])
            lines.extend(f"- `{error}`" for error in level.errors)
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- 测试通过本机 API 完成，包含 HTTP、文本口语化、推理、WAV 编码和传输开销。",
            "- 并发请求共享同一个服务进程；聚合 RTF 比平均单请求 RTF 更能反映并发负载。",
            "- 首次请求可能受模型和 CUDA 图缓存影响，因此正式测量前执行了预热请求。",
            "- 生成音频长度会随模型输出变化，RTF 比单独比较延迟更适合评价实时性能。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from exc
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("levels must contain positive integers")
    return levels


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end Spark-TTS API RTF at 1, 2, 4 concurrency."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9616")
    parser.add_argument("--voice", default="low-female1")
    parser.add_argument("--model", default="spark-tts")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--levels", type=_parse_levels, default=(1, 2, 4))
    parser.add_argument(
        "--requests",
        type=int,
        default=8,
        help="Total measured requests per concurrency level.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--title", default="Spark-TTS API RTF Benchmark")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=script_dir / "benchmark_results" / "api_rtf.md",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.requests < max(args.levels):
        parser.error("--requests must be at least the largest concurrency level")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.cooldown < 0:
        parser.error("--cooldown cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "speed": 1.0,
    }
    health = _check_service(base_url, args.timeout)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    for index in range(1, args.warmups + 1):
        print(f"Warmup {index}/{args.warmups}", flush=True)
        _post_speech(base_url=base_url, payload=payload, timeout=args.timeout)

    levels: list[LevelResult] = []
    for concurrency in args.levels:
        print(
            f"Running concurrency={concurrency}, requests={args.requests}",
            flush=True,
        )
        result = _run_level(
            base_url=base_url,
            payload=payload,
            concurrency=concurrency,
            request_count=args.requests,
            timeout=args.timeout,
        )
        levels.append(result)
        summary = _summary(result)
        if summary["successes"]:
            print(
                f"  aggregate_rtf={float(summary['aggregate_rtf']):.3f} "
                f"mean_request_rtf={float(summary['mean_request_rtf']):.3f} "
                f"wall_seconds={summary['wall_seconds']:.3f}",
                flush=True,
            )
        else:
            print(f"  all requests failed: {result.errors}", flush=True)
        if args.cooldown:
            time.sleep(args.cooldown)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    report = _format_report(
        generated_at=generated_at,
        base_url=base_url,
        voice=args.voice,
        model=args.model,
        text=args.text,
        health=health,
        levels=levels,
        warmups=args.warmups,
        title=args.title,
    )
    args.output_md.write_text(report, encoding="utf-8")

    json_output = args.json_output or args.output_md.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    result_document = {
        "generated_at": generated_at,
        "config": {
            "base_url": base_url,
            "voice": args.voice,
            "model": args.model,
            "text": args.text,
            "levels": list(args.levels),
            "requests_per_level": args.requests,
            "warmups": args.warmups,
            "cooldown_seconds": args.cooldown,
            "timeout_seconds": args.timeout,
        },
        "health": health,
        "gpu": _gpu_description(),
        "levels": [
            {
                "summary": _summary(level),
                "requests": [asdict(request) for request in level.requests],
                "errors": list(level.errors),
            }
            for level in levels
        ],
    }
    json_output.write_text(
        json.dumps(result_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Markdown report: {args.output_md}", flush=True)
    print(f"Raw JSON: {json_output}", flush=True)


if __name__ == "__main__":
    main()
