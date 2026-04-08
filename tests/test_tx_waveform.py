from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.tx import synthesize_transmission, write_wav


class TxWaveformTests(unittest.TestCase):
    def test_hello_transmission_has_expected_total_duration_and_sample_count(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)

        self.assertEqual(samples.shape, (246720,))
        self.assertAlmostEqual(samples.size / DEFAULT_CONFIG.sample_rate_hz, 5.14, places=12)

    def test_transmission_output_is_mono_and_finite(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)

        self.assertEqual(samples.ndim, 1)
        self.assertTrue(np.isfinite(samples).all())

    def test_peak_amplitude_does_not_exceed_full_scale(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)

        self.assertLessEqual(float(np.max(np.abs(samples))), 1.0)
        self.assertLessEqual(float(np.max(np.abs(samples))), DEFAULT_CONFIG.burst_amplitude + 1e-12)

    def test_dominant_energy_for_zero_symbol_is_near_1200_hz(self) -> None:
        samples = synthesize_transmission(np.array([0, 0, 0], dtype=np.uint8), DEFAULT_CONFIG)
        symbol = _extract_data_symbol(samples, data_symbol_index=1)

        self.assertEqual(_dominant_frequency(symbol), 1200.0)

    def test_dominant_energy_for_one_symbol_is_near_2200_hz(self) -> None:
        samples = synthesize_transmission(np.array([1, 1, 1], dtype=np.uint8), DEFAULT_CONFIG)
        symbol = _extract_data_symbol(samples, data_symbol_index=1)

        self.assertEqual(_dominant_frequency(symbol), 2200.0)

    def test_leading_and_trailing_silence_lengths_are_correct(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)

        self.assertTrue(np.all(samples[: DEFAULT_CONFIG.leading_silence_samples] == 0.0))
        self.assertTrue(np.all(samples[-DEFAULT_CONFIG.trailing_silence_samples :] == 0.0))

    def test_active_burst_has_fade_in_and_fade_out_behavior(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)
        active = samples[
            DEFAULT_CONFIG.leading_silence_samples : samples.size - DEFAULT_CONFIG.trailing_silence_samples
        ]

        self.assertEqual(float(active[0]), 0.0)
        self.assertEqual(float(active[-1]), 0.0)
        self.assertLess(
            float(np.mean(np.abs(active[: DEFAULT_CONFIG.fade_samples]))),
            float(np.mean(np.abs(active[DEFAULT_CONFIG.fade_samples : 2 * DEFAULT_CONFIG.fade_samples]))),
        )
        self.assertLess(
            float(np.mean(np.abs(active[-DEFAULT_CONFIG.fade_samples :]))),
            float(np.mean(np.abs(active[-2 * DEFAULT_CONFIG.fade_samples : -DEFAULT_CONFIG.fade_samples]))),
        )

    def test_write_wav_outputs_mono_pcm(self) -> None:
        samples = synthesize_transmission(_hello_frame_bits(), DEFAULT_CONFIG)

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello.wav"
            write_wav(wav_path, samples, DEFAULT_CONFIG.sample_rate_hz)

            with wave.open(str(wav_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), DEFAULT_CONFIG.sample_rate_hz)
                self.assertEqual(wav_file.getnframes(), samples.size)


def _hello_frame_bits() -> np.ndarray:
    return bytes_to_bits(build_frame(validate_text("HELLO")))


def _extract_data_symbol(samples: np.ndarray, data_symbol_index: int) -> np.ndarray:
    active_start = DEFAULT_CONFIG.leading_silence_samples
    symbol_start = active_start + ((DEFAULT_CONFIG.tx_prefix_bit_count + data_symbol_index) * DEFAULT_CONFIG.samples_per_symbol)
    symbol_stop = symbol_start + DEFAULT_CONFIG.samples_per_symbol
    return samples[symbol_start:symbol_stop]


def _dominant_frequency(symbol_samples: np.ndarray) -> float:
    spectrum = np.fft.rfft(symbol_samples)
    frequencies = np.fft.rfftfreq(symbol_samples.size, d=1.0 / DEFAULT_CONFIG.sample_rate_hz)
    return float(frequencies[int(np.argmax(np.abs(spectrum)))])


if __name__ == "__main__":
    unittest.main()
