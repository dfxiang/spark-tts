"""Compare the existing API with the TensorRT-LLM sidecar on fixed input."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.triton_trtllm.benchmark_protocol import (  # noqa: E402
    build_openai_speech_request,
    build_triton_infer_request,
)

DEFAULT_TEXT = (
    "Artificial intelligence is changing the way we work and live, and speech "
    "synthesis is advancing quickly. Today we measure a short passage to "
    "understand generation latency and the real time factor."
)
DEFAULT_VOICE_ID = "voice_b241260c3376"
DEFAULT_REFERENCE_AUDIO = Path(f"/data/voices/{DEFAULT_VOICE_ID}/reference.wav")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail[:1000]}") from exc


def _load_reference(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"reference audio must be 16 kHz, got {sample_rate}")
    if waveform.shape[1] != 1:
        raise ValueError("reference audio must be mono")
    if not waveform.size:
        raise ValueError("reference audio is empty")
    return np.ascontiguousarray(waveform[:, 0], dtype=np.float32)


def _request_baseline(
    *,
    base_url: str,
    text: str,
    voice_id: str,
    timeout: float,
) -> tuple[float, bytes]:
    payload = build_openai_speech_request(target_text=text, voice_id=voice_id)
    started_at = time.perf_counter()
    pcm = _post_json(f"{base_url}/v1/audio/speech", payload, timeout)
    return time.perf_counter() - started_at, pcm


def _request_sidecar(
    *,
    base_url: str,
    reference_wav: np.ndarray,
    reference_text: str,
    text: str,
    timeout: float,
) -> tuple[float, np.ndarray]:
    payload = build_triton_infer_request(
        waveform=reference_wav.tolist(),
        reference_text=reference_text,
        target_text=text,
    )
    started_at = time.perf_counter()
    response = json.loads(
        _post_json(f"{base_url}/v2/models/spark_tts/infer", payload, timeout)
    )
    elapsed = time.perf_counter() - started_at
    outputs = {item["name"]: item for item in response.get("outputs", [])}
    if "waveform" not in outputs:
        raise RuntimeError(f"sidecar response has no waveform: {response}")
    waveform = np.asarray(outputs["waveform"]["data"], dtype=np.float32).reshape(-1)
    if not waveform.size:
        raise RuntimeError("sidecar returned empty audio")
    return elapsed, waveform


def _summarize(runs: list[dict[str, float]]) -> dict[str, float]:
    latencies = [run["wall_seconds"] for run in runs]
    rtfs = [run["rtf"] for run in runs]
    return {
        "mean_wall_seconds": statistics.fmean(latencies),
        "median_wall_seconds": statistics.median(latencies),
        "mean_rtf": statistics.fmean(rtfs),
        "median_rtf": statistics.median(rtfs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the existing API against the TensorRT-LLM sidecar."
    )
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8168")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8180")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    reference_wav = _load_reference(args.reference_audio)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.warmup_runs):
        _request_baseline(
            base_url=args.baseline_url.rstrip("/"),
            text=args.text,
            voice_id=args.voice_id,
            timeout=args.timeout,
        )
        _request_sidecar(
            base_url=args.sidecar_url.rstrip("/"),
            reference_wav=reference_wav,
            reference_text=args.reference_text,
            text=args.text,
            timeout=args.timeout,
        )

    baseline_runs: list[dict[str, float]] = []
    sidecar_runs: list[dict[str, float]] = []
    for run_index in range(1, args.runs + 1):
        baseline_seconds, baseline_pcm = _request_baseline(
            base_url=args.baseline_url.rstrip("/"),
            text=args.text,
            voice_id=args.voice_id,
            timeout=args.timeout,
        )
        baseline_waveform = (
            np.frombuffer(baseline_pcm, dtype="<i2").astype(np.float32) / 32768.0
        )
        baseline_audio_seconds = baseline_waveform.size / 16000
        baseline_rtf = baseline_seconds / baseline_audio_seconds
        baseline_runs.append(
            {
                "wall_seconds": baseline_seconds,
                "audio_seconds": baseline_audio_seconds,
                "rtf": baseline_rtf,
            }
        )
        sf.write(
            args.output_dir / f"baseline_run{run_index}.wav",
            baseline_waveform,
            16000,
            subtype="PCM_16",
        )

        sidecar_seconds, sidecar_waveform = _request_sidecar(
            base_url=args.sidecar_url.rstrip("/"),
            reference_wav=reference_wav,
            reference_text=args.reference_text,
            text=args.text,
            timeout=args.timeout,
        )
        sidecar_audio_seconds = sidecar_waveform.size / 16000
        sidecar_rtf = sidecar_seconds / sidecar_audio_seconds
        sidecar_runs.append(
            {
                "wall_seconds": sidecar_seconds,
                "audio_seconds": sidecar_audio_seconds,
                "rtf": sidecar_rtf,
            }
        )
        sf.write(
            args.output_dir / f"sidecar_run{run_index}.wav",
            sidecar_waveform,
            16000,
            subtype="PCM_16",
        )
        print(
            json.dumps(
                {
                    "run": run_index,
                    "baseline": baseline_runs[-1],
                    "sidecar": sidecar_runs[-1],
                }
            ),
            flush=True,
        )

    baseline_summary = _summarize(baseline_runs)
    sidecar_summary = _summarize(sidecar_runs)
    result = {
        "config": {
            "baseline_url": args.baseline_url.rstrip("/"),
            "sidecar_url": args.sidecar_url.rstrip("/"),
            "voice_id": args.voice_id,
            "reference_audio": str(args.reference_audio),
            "reference_seconds": reference_wav.size / 16000,
            "reference_text": args.reference_text,
            "target_text": args.text,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
        },
        "baseline": {
            "runs": baseline_runs,
            "summary": baseline_summary,
        },
        "sidecar": {
            "runs": sidecar_runs,
            "summary": sidecar_summary,
        },
        "comparison": {
            "latency_speedup": (
                baseline_summary["mean_wall_seconds"]
                / sidecar_summary["mean_wall_seconds"]
            ),
            "rtf_improvement": (
                baseline_summary["mean_rtf"] / sidecar_summary["mean_rtf"]
            ),
        },
    }
    output_path = args.output_dir / "bf16_sidecar_benchmark.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Results written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
