"""Run long Chinese text through the TTS API and report silence anomalies."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PUNCTUATION = "，,、：:。！？!?；;…"
CLOSING_MARKS = "”’」』）》】〕〉"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "tts_stability.md"


@dataclass(frozen=True)
class SilenceRun:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class AudioAnalysis:
    duration_seconds: float
    sample_rate: int
    leading_silence: float
    trailing_silence: float
    longest_silence: float
    silence_runs: list[SilenceRun] = field(default_factory=list)


@dataclass
class SegmentResult:
    index: int
    text: str
    latency_seconds: float = 0.0
    response_bytes: int = 0
    audio: AudioAnalysis | None = None
    error: str | None = None

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def rtf(self) -> float | None:
        if self.audio is None or self.audio.duration_seconds <= 0:
            return None
        return self.latency_seconds / self.audio.duration_seconds


def normalize_text(text: str) -> str:
    """Remove whitespace and invisible Unicode formatting/control characters."""
    visible: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isspace():
            continue
        if category.startswith("C") or category.startswith("Z"):
            continue
        visible.append(character)
    return "".join(visible)


def _punctuation_boundaries(text: str) -> list[int]:
    boundaries: set[int] = {len(text)}
    index = 0
    while index < len(text):
        if text[index] not in PUNCTUATION:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in PUNCTUATION:
            end += 1
        while end < len(text) and text[end] in CLOSING_MARKS:
            end += 1
        boundaries.add(end)
        index = end
    return sorted(boundaries)


def split_text(
    text: str,
    max_chars: int = 128,
    min_chars: int = 64,
) -> list[str]:
    """Split only at punctuation into chunks constrained to min/max length."""
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    if max_chars < min_chars:
        raise ValueError("max_chars must be at least min_chars")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    boundaries = _punctuation_boundaries(normalized)
    target = (min_chars + max_chars) / 2
    states: dict[int, tuple[int, float, int | None]] = {0: (0, 0.0, None)}
    for end in boundaries:
        best: tuple[int, float, int | None] | None = None
        for start, (count, deviation, _) in states.items():
            length = end - start
            if length < min_chars or length > max_chars:
                continue
            candidate = (
                count + 1,
                deviation + abs(length - target),
                start,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            states[end] = best

    end = len(normalized)
    if end not in states:
        raise ValueError(
            f"cannot split into chunks of {min_chars}-{max_chars} characters "
            "using punctuation boundaries"
        )

    chunks: list[str] = []
    while end > 0:
        _, _, previous = states[end]
        if previous is None:
            raise ValueError("internal splitter error")
        chunks.append(normalized[previous:end])
        end = previous
    chunks.reverse()
    return chunks


def _pcm_peak_window(samples: array, start: int, end: int, channels: int) -> int:
    first = start * channels
    last = end * channels
    return max((abs(value) for value in samples[first:last]), default=0)


def analyze_wav(
    content: bytes,
    *,
    silence_threshold_db: float = -40.0,
    window_seconds: float = 0.02,
) -> AudioAnalysis:
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            pcm = audio.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError("response is not a valid WAV file") from exc

    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {sample_width * 8}-bit")
    if channels not in {1, 2}:
        raise ValueError(f"expected mono or stereo WAV, got {channels} channels")
    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV contains no audio frames")

    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()

    threshold = max(1, int(32768 * (10 ** (silence_threshold_db / 20.0))))
    frames_per_window = max(1, int(sample_rate * window_seconds))
    silent_runs: list[SilenceRun] = []
    silent_start: int | None = None

    for start in range(0, frame_count, frames_per_window):
        end = min(frame_count, start + frames_per_window)
        is_silent = _pcm_peak_window(samples, start, end, channels) <= threshold
        if is_silent and silent_start is None:
            silent_start = start
        elif not is_silent and silent_start is not None:
            silent_runs.append(
                SilenceRun(
                    start_seconds=silent_start / sample_rate,
                    end_seconds=start / sample_rate,
                )
            )
            silent_start = None
    if silent_start is not None:
        silent_runs.append(
            SilenceRun(
                start_seconds=silent_start / sample_rate,
                end_seconds=frame_count / sample_rate,
            )
        )

    duration = frame_count / sample_rate
    leading = (
        silent_runs[0].duration_seconds
        if silent_runs and silent_runs[0].start_seconds == 0
        else 0.0
    )
    trailing = (
        silent_runs[-1].duration_seconds
        if silent_runs and abs(silent_runs[-1].end_seconds - duration) < 1e-6
        else 0.0
    )
    return AudioAnalysis(
        duration_seconds=duration,
        sample_rate=sample_rate,
        leading_silence=leading,
        trailing_silence=trailing,
        longest_silence=max(
            (run.duration_seconds for run in silent_runs),
            default=0.0,
        ),
        silence_runs=silent_runs,
    )


def _speech_request(
    *,
    base_url: str,
    model: str,
    voice: str,
    text: str,
    speed: float,
    timeout: float,
) -> tuple[bytes, float]:
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": speed,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000] or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc
    return content, time.perf_counter() - started_at


def _check_health(base_url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}/healthz", timeout=timeout) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health.get("status") != "ok":
        raise RuntimeError(f"service is not healthy: {health}")
    return health


def _write_wav(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _format_report(
    *,
    generated_at: str,
    input_path: Path,
    base_url: str,
    model: str,
    voice: str,
    min_chars: int,
    max_chars: int,
    silence_seconds: float,
    silence_threshold_db: float,
    health: dict[str, Any],
    segments: Sequence[SegmentResult],
    total_wall_seconds: float,
) -> tuple[str, bool]:
    successful = [segment for segment in segments if segment.error is None]
    failed = [segment for segment in segments if segment.error is not None]
    total_audio = sum(
        segment.audio.duration_seconds
        for segment in successful
        if segment.audio is not None
    )
    aggregate_rtf = total_wall_seconds / total_audio if total_audio > 0 else None

    long_silences: list[tuple[SegmentResult, SilenceRun]] = []
    for segment in successful:
        if segment.audio is None:
            continue
        for run in segment.audio.silence_runs:
            if run.duration_seconds >= silence_seconds:
                long_silences.append((segment, run))

    boundary_silences: list[tuple[int, float, float, float]] = []
    for previous, current in zip(successful, successful[1:]):
        if previous.audio is None or current.audio is None:
            continue
        boundary = (
            previous.audio.trailing_silence + current.audio.leading_silence
        )
        if boundary >= silence_seconds:
            boundary_silences.append(
                (
                    current.index,
                    previous.audio.trailing_silence,
                    current.audio.leading_silence,
                    boundary,
                )
            )

    passed = not failed and not long_silences and not boundary_silences
    verdict = "PASS" if passed else "FAIL"
    lines = [
        "# Spark-TTS 长文本稳定性测试报告",
        "",
        f"- 结论：**{verdict}**",
        f"- 生成时间：`{generated_at}`",
        f"- 输入文件：`{input_path}`",
        f"- API：`{base_url}`",
        f"- 模型：`{model}`",
        f"- 音色：`{voice}`",
        f"- 服务设备：`{health.get('device', 'unknown')}`",
        f"- LLM 精度：`{health.get('llm_dtype', 'unknown')}`",
        f"- 每段字符数范围：`{min_chars}-{max_chars}`",
        f"- 静音阈值：`{silence_threshold_db:.1f} dBFS`",
        f"- 异常静音判定：连续静音 `>= {silence_seconds:.1f}s`",
        f"- 分段数：`{len(segments)}`",
        f"- 成功/失败：`{len(successful)}/{len(failed)}`",
        f"- 总墙钟时间：`{total_wall_seconds:.3f}s`",
        f"- 合成音频总时长：`{total_audio:.3f}s`",
        (
            f"- 聚合 RTF：`{aggregate_rtf:.3f}`"
            if aggregate_rtf is not None
            else "- 聚合 RTF：`N/A`"
        ),
        "",
        "## 判定",
        "",
    ]
    if passed:
        lines.append(
            f"未发现 API 错误，也未发现单段内部或相邻段拼接边界超过 "
            f"`{silence_seconds:.1f}s` 的静音。"
        )
    else:
        if failed:
            lines.append(f"- `{len(failed)}` 个分段请求或音频解析失败。")
        if long_silences:
            lines.append(f"- `{len(long_silences)}` 个单段内部静音区间超限。")
        if boundary_silences:
            lines.append(
                f"- `{len(boundary_silences)}` 个相邻段拼接边界静音超限。"
            )

    lines.extend(
        [
            "",
            "## 分段结果",
            "",
            "| 段 | 字符数 | 延迟 (s) | 音频时长 (s) | RTF "
            "| 首部静音 (s) | 尾部静音 (s) | 最长静音 (s) | 状态 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for segment in segments:
        if segment.error is not None or segment.audio is None:
            lines.append(
                f"| {segment.index} | {segment.character_count} "
                f"| {segment.latency_seconds:.3f} | - | - | - | - | - | 失败 |"
            )
            continue
        status = "通过"
        if segment.audio.longest_silence >= silence_seconds:
            status = "静音超限"
        lines.append(
            f"| {segment.index} | {segment.character_count} "
            f"| {segment.latency_seconds:.3f} "
            f"| {segment.audio.duration_seconds:.3f} "
            f"| {segment.rtf:.3f} "
            f"| {segment.audio.leading_silence:.3f} "
            f"| {segment.audio.trailing_silence:.3f} "
            f"| {segment.audio.longest_silence:.3f} | {status} |"
        )

    lines.extend(["", "## 异常静音", ""])
    if not long_silences and not boundary_silences and not failed:
        lines.append(f"未发现连续 `{silence_seconds:.1f}s` 以上的静音。")
    for segment, run in long_silences:
        lines.append(
            f"- 第 `{segment.index}` 段内部："
            f"`{run.start_seconds:.3f}s - {run.end_seconds:.3f}s`，"
            f"持续 `{run.duration_seconds:.3f}s`。"
        )
    for index, previous_tail, current_head, total in boundary_silences:
        lines.append(
            f"- 第 `{index - 1}` 段尾部 `{previous_tail:.3f}s` + "
            f"第 `{index}` 段首部 `{current_head:.3f}s` = "
            f"`{total:.3f}s`，若顺序拼接将超过阈值。"
        )
    for segment in failed:
        lines.append(f"- 第 `{segment.index}` 段失败：`{segment.error}`")

    lines.extend(["", "## 分段文本", ""])
    for segment in segments:
        lines.extend(
            [
                f"### 第 {segment.index} 段（{segment.character_count} 字符）",
                "",
                f"> {segment.text}",
                "",
            ]
        )

    lines.extend(
        [
            "## 方法说明",
            "",
            "- 文本只按标点边界拆分，并通过动态规划保证每段长度在指定范围内。",
            "- 所有 Unicode 空白、控制字符和格式字符在分段前移除。",
            "- API 按分段顺序串行调用，响应格式为 16-bit PCM WAV；测试程序直接分析 PCM。",
            "- 静音按 20 ms 窗口和峰值阈值检测，不生成合并音频或静音音频文件。",
            "- 拼接边界静音按“前段尾部静音 + 后段首部静音”计算，用于评估后续合并风险。",
            "",
        ]
    )
    return "\n".join(lines), passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split long text, call Spark-TTS serially, and detect long silence."
    )
    parser.add_argument("input", type=Path, nargs="?", default=Path("tests/tts-test1.txt"))
    parser.add_argument("--base-url", default="http://127.0.0.1:9616")
    parser.add_argument("--model", default="spark-tts")
    parser.add_argument("--voice", default="low-female1")
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=128)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--silence-seconds", type=float, default=3.0)
    parser.add_argument("--silence-threshold-db", type=float, default=-40.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Optional directory for saving per-segment WAV files.",
    )
    args = parser.parse_args()
    if args.min_chars < 1:
        parser.error("--min-chars must be positive")
    if args.max_chars < args.min_chars:
        parser.error("--max-chars must be at least --min-chars")
    if args.silence_seconds <= 0:
        parser.error("--silence-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    source = args.input.read_text(encoding="utf-8-sig")
    chunks = split_text(source, args.max_chars, args.min_chars)
    if not chunks:
        raise SystemExit("input contains no visible text")
    if any(
        len(chunk) < args.min_chars or len(chunk) > args.max_chars
        for chunk in chunks
    ):
        raise SystemExit("splitter produced a chunk outside the configured range")

    base_url = args.base_url.rstrip("/")
    health = _check_health(base_url, args.timeout)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    segments: list[SegmentResult] = []
    test_started = time.perf_counter()
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"[{index}/{len(chunks)}] {len(chunk)} chars",
            end="",
            flush=True,
        )
        segment = SegmentResult(index=index, text=chunk)
        try:
            content, latency = _speech_request(
                base_url=base_url,
                model=args.model,
                voice=args.voice,
                text=chunk,
                speed=args.speed,
                timeout=args.timeout,
            )
            segment.latency_seconds = latency
            segment.response_bytes = len(content)
            segment.audio = analyze_wav(
                content,
                silence_threshold_db=args.silence_threshold_db,
            )
            if args.audio_dir:
                _write_wav(args.audio_dir / f"segment-{index:02d}.wav", content)
            print(
                f" -> {latency:.3f}s, audio={segment.audio.duration_seconds:.3f}s, "
                f"RTF={segment.rtf:.3f}, max_silence="
                f"{segment.audio.longest_silence:.3f}s",
                flush=True,
            )
        except Exception as exc:
            segment.error = f"{type(exc).__name__}: {exc}"
            print(f" -> FAILED: {segment.error}", flush=True)
        segments.append(segment)

    total_wall = time.perf_counter() - test_started
    report, passed = _format_report(
        generated_at=generated_at,
        input_path=args.input,
        base_url=base_url,
        model=args.model,
        voice=args.voice,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        silence_seconds=args.silence_seconds,
        silence_threshold_db=args.silence_threshold_db,
        health=health,
        segments=segments,
        total_wall_seconds=total_wall,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
