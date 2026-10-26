"""TensorRT-LLM Triton client used by the existing Spark-TTS pipeline."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

import torch

JsonRequester = Callable[[str, dict[str, Any], float], dict[str, Any]]


class TritonLLMClient:
    """Expose the subset of the Hugging Face generate API used by SparkTTS."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        request_json: JsonRequester | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("SPARK_TTS_TRTLLM_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout or float(
            os.getenv("SPARK_TTS_TRTLLM_TIMEOUT", "300")
        )
        self._request_json = request_json or self._http_json

    def generate(self, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("TensorRT-LLM bridge supports batch size 1")

        prompt_ids = input_ids.detach().cpu().to(torch.int32).tolist()
        request_output_len = int(kwargs.get("max_new_tokens", 2000))
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, input_ids.shape[1]],
                    "datatype": "INT32",
                    "data": prompt_ids,
                },
                {
                    "name": "input_lengths",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[input_ids.shape[1]]],
                },
                {
                    "name": "request_output_len",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[request_output_len]],
                },
                {
                    "name": "streaming",
                    "shape": [1, 1],
                    "datatype": "BOOL",
                    "data": [[False]],
                },
                {
                    "name": "end_id",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[int(kwargs["eos_token_id"])]],
                },
                {
                    "name": "pad_id",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[int(kwargs["pad_token_id"])]],
                },
                {
                    "name": "runtime_top_k",
                    "shape": [1, 1],
                    "datatype": "INT32",
                    "data": [[int(kwargs.get("top_k", 1))]],
                },
                {
                    "name": "runtime_top_p",
                    "shape": [1, 1],
                    "datatype": "FP32",
                    "data": [[float(kwargs.get("top_p", 1.0))]],
                },
                {
                    "name": "temperature",
                    "shape": [1, 1],
                    "datatype": "FP32",
                    "data": [[float(kwargs.get("temperature", 0.8))]],
                },
            ]
        }
        response = self._request_json(
            "/v2/models/tensorrt_llm/infer",
            payload,
            self.timeout,
        )
        outputs = {item["name"]: item for item in response.get("outputs", [])}
        if "output_ids" not in outputs or "sequence_length" not in outputs:
            raise RuntimeError(f"TensorRT-LLM response is incomplete: {response}")

        sequence_length = int(
            torch.tensor(outputs["sequence_length"]["data"]).reshape(-1)[0]
        )
        generated_ids = (
            torch.tensor(outputs["output_ids"]["data"])
            .reshape(-1)[:sequence_length]
            .tolist()
        )
        generated = torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        return torch.cat((input_ids.to(torch.long), generated), dim=1)

    def _http_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
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
                f"TensorRT-LLM returned HTTP {exc.code}: {detail[:2000]}"
            ) from exc
