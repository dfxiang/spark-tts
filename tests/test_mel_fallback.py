from __future__ import annotations

import unittest

import torch

from sparktts.utils.mel import MelSpectrogram


class MelFallbackTest(unittest.TestCase):
    def test_matches_torchaudio_reference_values(self) -> None:
        mel = MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=640,
            hop_length=320,
            f_min=10,
            f_max=None,
            n_mels=128,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )
        waveform = torch.linspace(-1, 1, 8000).unsqueeze(0)

        with torch.inference_mode():
            output = mel(waveform)

        self.assertEqual(tuple(output.shape), (1, 128, 26))
        self.assertAlmostEqual(float(output.sum()), 103.2839584350586, places=3)
        self.assertAlmostEqual(float(output.mean()), 0.031034843996167183, places=7)
        self.assertAlmostEqual(float(output.max()), 6.751327037811279, places=5)


if __name__ == "__main__":
    unittest.main()
