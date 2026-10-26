"""OpenAI-compatible FastAPI adapter backed by the local Triton server."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import numpy as np
import soundfile as sf

from api.app import Settings, create_app
from runtime.triton_trtllm.benchmark_protocol import build_triton_infer_request

logger = logging.getLogger("spark_tts_triton_adapter")

JsonRequester = Callable[[str, dict[str, Any], float], dict[str, Any]]


class TritonEngine:
    """Drop-in inference engine for the existing FastAPI application."""

    def __init__(
        self,
        settings: Settings,
        *,
        request_json: JsonRequester | None = None,
    ) -> None:
        self.settings = settings
        self.triton_url = os.getenv(
            "SPARK_TTS_TRITON_URL", "http://127.0.0.1:8000"
        ).rstrip("/")
        self.timeout = float(os.getenv("SPARK_TTS_TRITON_TIMEOUT", "300"))
        self._request_json = request_json or self._http_json
        self._lock = RLock()
        self._ready = False

    @property
    def loaded(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return "cuda:0 via tensorrt-llm"

    @property
    def llm_dtype(self) -> str:
        return "torch.bfloat16"

    def preload(self) -> None:
        with urllib.request.urlopen(
            f"{self.triton_url}/v2/health/ready",
            timeout=10,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Triton is not ready: HTTP {response.status}")
        self._ready = True

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        prompt_text: str | None,
    ) -> np.ndarray:
        with self._lock:
            started_at = time.perf_counter()
            waveform, sample_rate = sf.read(
                reference_audio,
                dtype="float32",
                always_2d=True,
            )
            if sample_rate != 16000:
                raise ValueError(f"reference audio must be 16 kHz, got {sample_rate}")
            if waveform.shape[1] != 1 or not waveform.size:
                raise ValueError("reference audio must be non-empty mono audio")

            payload = build_triton_infer_request(
                waveform=waveform[:, 0].tolist(),
                reference_text=prompt_text or "",
                target_text=text,
            )
            response = self._request_json(
                "/v2/models/spark_tts/infer",
                payload,
                self.timeout,
            )

        outputs = {
            item["name"]: item
            for item in response.get("outputs", [])
        }
        if "waveform" not in outputs:
            raise RuntimeError(f"Triton response has no waveform: {response}")
        audio = np.asarray(outputs["waveform"]["data"], dtype=np.float32).reshape(-1)
        if not audio.size:
            raise RuntimeError("Triton returned empty audio")

        elapsed = time.perf_counter() - started_at
        audio_seconds = audio.size / 16000
        logger.info(
            "Synthesized %.2fs audio in %.2fs (RTF %.3f, text_chars=%d)",
            audio_seconds,
            elapsed,
            elapsed / audio_seconds,
            len(text),
        )
        return audio

    def _http_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.triton_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Triton returned HTTP {exc.code}: {detail[:2000]}"
            ) from exc


def _resolve_ffmpeg(configured: str) -> str | None:
    if os.path.sep in configured:
        return configured if Path(configured).is_file() else None
    executable = shutil.which(configured)
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def create_adapter_app():
    settings = Settings.from_env()
    ffmpeg = _resolve_ffmpeg(settings.ffmpeg_bin)
    settings = replace(
        settings,
        device="cuda:0",
        llm_dtype="bfloat16",
        ffmpeg_bin=ffmpeg or settings.ffmpeg_bin,
    )
    return create_app(settings, TritonEngine(settings))


app = create_adapter_app()
