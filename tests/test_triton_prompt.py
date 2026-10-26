from __future__ import annotations

import unittest

from runtime.triton_trtllm.prompt_protocol import build_prompt


class TritonPromptTest(unittest.TestCase):
    def test_empty_reference_text_matches_the_no_prompt_baseline(self) -> None:
        global_ids = [1, 2]
        semantic_ids = [3, 4, 5]

        prompt, returned_global_ids = build_prompt(
            text="target",
            reference_text="",
            global_token_ids=global_ids,
            semantic_token_ids=semantic_ids,
        )

        self.assertEqual(
            prompt,
            "<|task_tts|><|start_content|>target<|end_content|>"
            "<|start_global_token|><|bicodec_global_1|>"
            "<|bicodec_global_2|><|end_global_token|>",
        )
        self.assertEqual(returned_global_ids, global_ids)
        self.assertNotIn("semantic", prompt)

    def test_reference_text_includes_semantic_context(self) -> None:
        prompt, _ = build_prompt(
            text="target",
            reference_text="reference",
            global_token_ids=[1],
            semantic_token_ids=[3, 4],
        )

        self.assertIn(
            "<|start_semantic_token|><|bicodec_semantic_3|>"
            "<|bicodec_semantic_4|>",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
