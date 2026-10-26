from __future__ import annotations

import unittest

from runtime.triton_trtllm.benchmark_protocol import (
    build_openai_speech_request,
    build_triton_infer_request,
)


class BenchmarkProtocolTest(unittest.TestCase):
    def test_both_services_receive_the_same_fixed_input(self) -> None:
        target_text = "A fixed benchmark sentence."
        waveform = [0.1, -0.2, 0.3]

        baseline = build_openai_speech_request(
            target_text=target_text,
            voice_id="voice_fixed",
        )
        sidecar = build_triton_infer_request(
            waveform=waveform,
            reference_text="",
            target_text=target_text,
        )

        inputs = {item["name"]: item for item in sidecar["inputs"]}
        self.assertEqual(baseline["input"], target_text)
        self.assertEqual(inputs["target_text"]["data"], [target_text])
        self.assertEqual(inputs["reference_text"]["data"], [""])
        self.assertEqual(inputs["reference_wav"]["data"], waveform)
        self.assertEqual(inputs["reference_wav"]["shape"], [1, 3])
        self.assertEqual(inputs["reference_wav"]["datatype"], "FP32")
        self.assertEqual(inputs["reference_wav_len"]["data"], [[3]])


if __name__ == "__main__":
    unittest.main()
