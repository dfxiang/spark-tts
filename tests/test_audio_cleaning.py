from __future__ import annotations

import unittest

import numpy as np

from api.audio_cleaning import clean_generated_waveform


class GeneratedWaveformCleanupTest(unittest.TestCase):
    def test_removes_a_long_internal_digital_silence(self) -> None:
        speech_before = np.full(16000, 0.1, dtype=np.float32)
        long_silence = np.zeros(16000 * 3, dtype=np.float32)
        speech_after = np.full(16000, 0.08, dtype=np.float32)

        cleaned = clean_generated_waveform(
            np.concatenate((speech_before, long_silence, speech_after)),
            sample_rate=16000,
        )

        np.testing.assert_array_equal(
            cleaned,
            np.concatenate((speech_before, speech_after)),
        )

    def test_preserves_normal_short_pauses(self) -> None:
        waveform = np.concatenate(
            (
                np.full(8000, 0.1, dtype=np.float32),
                np.zeros(4000, dtype=np.float32),
                np.full(8000, 0.1, dtype=np.float32),
            )
        )

        cleaned = clean_generated_waveform(waveform, sample_rate=16000)

        np.testing.assert_array_equal(cleaned, waveform)

    def test_drops_a_fully_silent_segment(self) -> None:
        cleaned = clean_generated_waveform(
            np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
        )

        assert cleaned.size == 0


if __name__ == "__main__":
    unittest.main()
