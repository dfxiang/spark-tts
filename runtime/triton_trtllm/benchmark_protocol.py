"""Request builders shared by the TensorRT-LLM benchmark runner."""

from __future__ import annotations

from collections.abc import Sequence


def build_openai_speech_request(
    *,
    target_text: str,
    voice_id: str,
) -> dict[str, object]:
    """Build the baseline request sent to the existing Spark-TTS API."""
    return {
        "model": "spark-tts",
        "input": target_text,
        "voice": voice_id,
        "response_format": "pcm",
        "speed": 1.0,
    }


def build_triton_infer_request(
    *,
    waveform: Sequence[float],
    reference_text: str,
    target_text: str,
) -> dict[str, list[dict[str, object]]]:
    """Build a Triton HTTP request for the combined Spark-TTS pipeline."""
    return {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": [1, len(waveform)],
                "datatype": "FP32",
                "data": list(waveform),
            },
            {
                "name": "reference_wav_len",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": [[len(waveform)]],
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text],
            },
        ]
    }
