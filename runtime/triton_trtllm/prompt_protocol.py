"""Prompt construction shared with the TensorRT-LLM Triton pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from sparktts.utils.token_parser import TASK_TOKEN_MAP


def _flatten_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "squeeze"):
        token_ids = token_ids.squeeze().tolist()
    if isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes)):
        return [int(token_id) for token_id in token_ids]
    return [int(token_ids)]


def build_prompt(
    *,
    text: str,
    reference_text: str | None,
    global_token_ids: Any,
    semantic_token_ids: Any,
) -> tuple[str, Any]:
    """Build the Spark-TTS prompt with the same empty-reference behavior as the API."""
    global_tokens = "".join(
        f"<|bicodec_global_{token_id}|>"
        for token_id in _flatten_ids(global_token_ids)
    )
    inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
    ]
    if reference_text:
        inputs.extend([reference_text, text])
    else:
        inputs.append(text)
    inputs.extend(
        [
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]
    )
    if reference_text:
        semantic_tokens = "".join(
            f"<|bicodec_semantic_{token_id}|>"
            for token_id in _flatten_ids(semantic_token_ids)
        )
        inputs.extend(
            [
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        )
    return "".join(inputs), global_token_ids
