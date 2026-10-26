"""Cleanup for long digital-silence spans in generated speech waveforms."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("spark_tts_api.audio_cleaning")

SILENCE_THRESHOLD_DBFS = -80.0
MIN_LONG_SILENCE_SECONDS = 1.0
ANALYSIS_FRAME_SECONDS = 0.02


def _silent_run_ranges(
    waveform: np.ndarray,
    sample_rate: int,
) -> list[tuple[int, int]]:
    """Return sample ranges occupied by complete near-silent analysis frames."""

    frame_samples = max(1, round(sample_rate * ANALYSIS_FRAME_SECONDS))
    frame_count = (waveform.size + frame_samples - 1) // frame_samples
    padded = np.pad(
        np.abs(waveform),
        (0, frame_count * frame_samples - waveform.size),
    )
    frame_peaks = padded.reshape(frame_count, frame_samples).max(axis=1)
    threshold = 10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0)
    silent_frames = frame_peaks <= threshold

    transitions = np.diff(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                silent_frames.astype(np.int8),
                np.zeros(1, dtype=np.int8),
            )
        )
    )
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    minimum_samples = round(sample_rate * MIN_LONG_SILENCE_SECONDS)

    ranges: list[tuple[int, int]] = []
    for frame_start, frame_end in zip(starts, ends):
        sample_start = int(frame_start) * frame_samples
        sample_end = min(waveform.size, int(frame_end) * frame_samples)
        if sample_end - sample_start >= minimum_samples:
            ranges.append((sample_start, sample_end))
    return ranges


def clean_generated_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Drop fully silent segments and long digital-silence spans.

    Natural pauses remain untouched. Only continuous near-silent spans of at
    least one second are removed, which targets decoder dropouts that otherwise
    become large blank regions after segment concatenation.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    audio = np.nan_to_num(
        np.asarray(waveform, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    if audio.size == 0:
        return audio

    threshold = 10.0 ** (SILENCE_THRESHOLD_DBFS / 20.0)
    if float(np.max(np.abs(audio))) <= threshold:
        logger.warning(
            "Dropped fully silent generated segment (%.3fs)",
            audio.size / sample_rate,
        )
        return np.empty(0, dtype=np.float32)

    silent_ranges = _silent_run_ranges(audio, sample_rate)
    if not silent_ranges:
        return audio

    kept: list[np.ndarray] = []
    cursor = 0
    for start, end in silent_ranges:
        kept.append(audio[cursor:start])
        cursor = end
    kept.append(audio[cursor:])
    cleaned = np.concatenate(kept)
    removed_seconds = sum(end - start for start, end in silent_ranges) / sample_rate
    logger.warning(
        "Removed %d long silent span(s) totaling %.3fs from generated audio",
        len(silent_ranges),
        removed_seconds,
    )
    return cleaned
