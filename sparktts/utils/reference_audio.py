"""Reference-audio preparation for stable voice-cloning prompts."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_REFERENCE_TAIL_SILENCE_SECONDS = 1.2
TAIL_SILENCE_THRESHOLD_DBFS = -80.0


def _as_samples(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim == 0:
        raise ValueError("audio must contain at least one sample")
    if samples.ndim > 2:
        raise ValueError("audio must be mono or use samples x channels layout")
    return samples


def _trailing_silence_seconds(
    audio: np.ndarray,
    sample_rate: int,
) -> float:
    amplitudes = np.abs(audio)
    if amplitudes.ndim == 2:
        amplitudes = amplitudes.max(axis=1)
    threshold = 10.0 ** (TAIL_SILENCE_THRESHOLD_DBFS / 20.0)
    audible = np.flatnonzero(amplitudes > threshold)
    if audible.size == 0:
        return amplitudes.size / sample_rate
    return (amplitudes.size - int(audible[-1]) - 1) / sample_rate


def pad_reference_tail(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_seconds: float = DEFAULT_REFERENCE_TAIL_SILENCE_SECONDS,
) -> np.ndarray:
    """Top up the reference audio's trailing silence to a stable boundary."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if target_seconds < 0:
        raise ValueError("target_seconds cannot be negative")

    samples = _as_samples(audio)
    current_seconds = _trailing_silence_seconds(samples, sample_rate)
    missing_samples = round((target_seconds - current_seconds) * sample_rate)
    if missing_samples <= 0:
        return np.asarray(samples).copy()

    padding = [(0, missing_samples)]
    padding.extend((0, 0) for _ in range(samples.ndim - 1))
    return np.pad(samples, padding, mode="constant")


@contextmanager
def reference_audio_for_prompt(
    source: Path,
    *,
    target_tail_seconds: float = DEFAULT_REFERENCE_TAIL_SILENCE_SECONDS,
) -> Iterator[Path]:
    """Yield the source or a temporary copy with sufficient trailing silence."""

    import soundfile as sf

    audio, sample_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    padded = pad_reference_tail(
        audio,
        sample_rate=sample_rate,
        target_seconds=target_tail_seconds,
    )
    if padded.shape == audio.shape:
        yield source
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="spark-tts-reference-",
            suffix=".wav",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        sf.write(
            temporary_path,
            padded,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        yield temporary_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
