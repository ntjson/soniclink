from __future__ import annotations

import unittest

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.dsp import goertzel_power, hann_window


class GoertzelTests(unittest.TestCase):
    def test_goertzel_power_distinguishes_1200_hz_from_2200_hz(self) -> None:
        tone_1200 = _tone(DEFAULT_CONFIG.f0_hz)
        tone_2200 = _tone(DEFAULT_CONFIG.f1_hz)
        window = hann_window(DEFAULT_CONFIG.samples_per_symbol)

        power_1200_at_1200 = goertzel_power(tone_1200, DEFAULT_CONFIG.f0_hz, DEFAULT_CONFIG.sample_rate_hz, window)
        power_1200_at_2200 = goertzel_power(tone_1200, DEFAULT_CONFIG.f1_hz, DEFAULT_CONFIG.sample_rate_hz, window)
        power_2200_at_2200 = goertzel_power(tone_2200, DEFAULT_CONFIG.f1_hz, DEFAULT_CONFIG.sample_rate_hz, window)
        power_2200_at_1200 = goertzel_power(tone_2200, DEFAULT_CONFIG.f0_hz, DEFAULT_CONFIG.sample_rate_hz, window)

        self.assertGreater(power_1200_at_1200, power_1200_at_2200 * 10.0)
        self.assertGreater(power_2200_at_2200, power_2200_at_1200 * 10.0)


def _tone(frequency_hz: float) -> np.ndarray:
    sample_offsets = np.arange(DEFAULT_CONFIG.samples_per_symbol, dtype=np.float64)
    return np.sin((2.0 * np.pi * frequency_hz * sample_offsets) / DEFAULT_CONFIG.sample_rate_hz)


if __name__ == "__main__":
    unittest.main()
