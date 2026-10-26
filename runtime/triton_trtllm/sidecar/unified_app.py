"""Existing Spark-TTS API with only the LLM delegated to TensorRT-LLM."""

from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Any

import numpy as np
import torch

from api.app import Settings, create_app
from cli.SparkTTS import SparkTTS
from runtime.triton_trtllm.llm_client import TritonLLMClient

logger = logging.getLogger("spark_tts_trtllm_api")


class UnifiedTensorRTEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: SparkTTS | None = None
        self._load_lock = RLock()
        self.max_batch_size = max(
            1,
            int(os.getenv("SPARK_TTS_TRT_MAX_BATCH_SIZE", "4")),
        )
        self._inference_slots = BoundedSemaphore(self.max_batch_size)
        self._triton_url = os.getenv(
            "SPARK_TTS_TRTLLM_URL", "http://127.0.0.1:8000"
        ).rstrip("/")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return "cuda:0"

    @property
    def llm_dtype(self) -> str:
        return "torch.bfloat16"

    def preload(self) -> None:
        with self._load_lock:
            self._load()

    def _load(self) -> SparkTTS:
        if self._model is None:
            with urllib.request.urlopen(
                f"{self._triton_url}/v2/health/ready",
                timeout=10,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"TensorRT-LLM is not ready: HTTP {response.status}"
                    )
            self._model = SparkTTS(
                self.settings.model_dir,
                torch.device("cuda:0"),
                llm_dtype=torch.bfloat16,
                prompt_cache_size=self.settings.prompt_cache_size,
                llm_backend=TritonLLMClient(self._triton_url),
            )
            logger.info("Loaded FastAPI pipeline with TensorRT-LLM LLM backend")
        return self._model

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        prompt_text: str | None,
    ) -> np.ndarray:
        started_at = time.perf_counter()
        with self._load_lock:
            model = self._load()
        with self._inference_slots:
            waveform = model.inference(
                text=text,
                prompt_speech_path=reference_audio,
                prompt_text=prompt_text,
            )

        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not audio.size:
            raise RuntimeError("Spark-TTS returned empty audio")
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

    def synthesize_batch(
        self,
        texts: list[str],
        reference_audio: Path,
        prompt_text: str | None,
    ) -> list[np.ndarray]:
        """Run up to max_batch_size requests concurrently for Triton batching."""
        if not texts:
            return []
        with ThreadPoolExecutor(
            max_workers=min(self.max_batch_size, len(texts))
        ) as executor:
            futures = [
                executor.submit(
                    self.synthesize,
                    text,
                    reference_audio,
                    prompt_text,
                )
                for text in texts
            ]
            return [future.result() for future in futures]


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


def create_unified_app():
    settings = Settings.from_env()
    ffmpeg = _resolve_ffmpeg(settings.ffmpeg_bin)
    settings = replace(
        settings,
        device="cuda:0",
        llm_dtype="bfloat16",
        ffmpeg_bin=ffmpeg or settings.ffmpeg_bin,
    )
    return create_app(settings, UnifiedTensorRTEngine(settings))


app = create_unified_app()
