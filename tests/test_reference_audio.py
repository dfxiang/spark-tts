from __future__ import annotations

import unittest

import numpy as np

from sparktts.utils.reference_audio import pad_reference_tail


class ReferenceAudioPaddingTest(unittest.TestCase):
    def test_tops_up_trailing_silence_to_target(self) -> None:
        audio = np.concatenate(
            (
                np.full(16000, 0.1, dtype=np.float32),
                np.zeros(3200, dtype=np.float32),
            )
        )

        padded = pad_reference_tail(
            audio,
            sample_rate=16000,
            target_seconds=1.2,
        )

        assert padded.size == audio.size + 16000
        np.testing.assert_array_equal(padded[: audio.size], audio)
        np.testing.assert_array_equal(padded[audio.size :], 0.0)

    def test_keeps_reference_with_enough_trailing_silence(self) -> None:
        audio = np.concatenate(
            (
                np.full(16000, 0.1, dtype=np.float32),
                np.zeros(24000, dtype=np.float32),
            )
        )

        padded = pad_reference_tail(
            audio,
            sample_rate=16000,
            target_seconds=1.2,
        )

        np.testing.assert_array_equal(padded, audio)

    def test_pads_each_channel_preserving_shape(self) -> None:
        audio = np.full((8000, 2), 0.1, dtype=np.float32)

        padded = pad_reference_tail(
            audio,
            sample_rate=16000,
            target_seconds=1.0,
        )

        assert padded.shape == (24000, 2)
        np.testing.assert_array_equal(padded[8000:], 0.0)


if __name__ == "__main__":
    unittest.main()
