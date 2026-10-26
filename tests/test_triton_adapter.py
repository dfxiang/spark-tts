from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from api.app import Settings
from runtime.triton_trtllm.sidecar.adapter import TritonEngine


class TritonEngineTest(unittest.TestCase):
    def test_synthesize_sends_reference_and_target_text_to_triton(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.wav"
            sf.write(reference, np.zeros(1600, dtype=np.float32), 16000)
            captured: dict[str, object] = {}

            def fake_request(
                path: str,
                payload: dict[str, object],
                timeout: float,
            ) -> dict[str, object]:
                captured.update(path=path, payload=payload, timeout=timeout)
                return {
                    "outputs": [
                        {
                            "name": "waveform",
                            "data": [0.1, -0.2, 0.3],
                        }
                    ]
                }

            engine = TritonEngine(
                Settings(
                    model_dir=root / "model",
                    voice_dir=root / "voices",
                    device="cuda:0",
                ),
                request_json=fake_request,
            )

            waveform = engine.synthesize(
                text="target text",
                reference_audio=reference,
                prompt_text=None,
            )

        self.assertEqual(captured["path"], "/v2/models/spark_tts/infer")
        inputs = {
            item["name"]: item
            for item in captured["payload"]["inputs"]  # type: ignore[index]
        }
        self.assertEqual(inputs["reference_text"]["data"], [""])
        self.assertEqual(inputs["target_text"]["data"], ["target text"])
        self.assertEqual(inputs["reference_wav_len"]["data"], [[1600]])
        np.testing.assert_allclose(waveform, [0.1, -0.2, 0.3], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
