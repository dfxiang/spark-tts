from __future__ import annotations

import unittest

import torch

from runtime.triton_trtllm.llm_client import TritonLLMClient


class TritonLLMClientTest(unittest.TestCase):
    def test_generate_returns_prompt_and_generated_tokens(self) -> None:
        captured: dict[str, object] = {}

        def fake_request(
            path: str,
            payload: dict[str, object],
            timeout: float,
        ) -> dict[str, object]:
            captured.update(path=path, payload=payload, timeout=timeout)
            return {
                "outputs": [
                    {"name": "output_ids", "data": [[20, 21, 2]]},
                    {"name": "sequence_length", "data": [3]},
                ]
            }

        client = TritonLLMClient(
            "http://127.0.0.1:8000",
            request_json=fake_request,
        )
        generated = client.generate(
            input_ids=torch.tensor([[10, 11]]),
            max_new_tokens=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            eos_token_id=2,
            pad_token_id=0,
        )

        self.assertEqual(captured["path"], "/v2/models/tensorrt_llm/infer")
        inputs = {
            item["name"]: item
            for item in captured["payload"]["inputs"]  # type: ignore[index]
        }
        self.assertEqual(inputs["input_ids"]["data"], [[10, 11]])
        self.assertEqual(inputs["input_lengths"]["data"], [[2]])
        self.assertEqual(inputs["request_output_len"]["data"], [[3]])
        self.assertEqual(inputs["end_id"]["data"], [[2]])
        self.assertEqual(inputs["pad_id"]["data"], [[0]])
        self.assertEqual(inputs["runtime_top_k"]["data"], [[50]])
        self.assertAlmostEqual(inputs["runtime_top_p"]["data"][0][0], 0.95)
        self.assertAlmostEqual(inputs["temperature"]["data"][0][0], 0.8)
        self.assertTrue(torch.equal(generated, torch.tensor([[10, 11, 20, 21, 2]])))


if __name__ == "__main__":
    unittest.main()
