from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.rx import decode_wav
from acoustic_modem.tx import synthesize_transmission, write_wav
from acoustic_modem.types import FailureCode


class SyntheticRoundtripTests(unittest.TestCase):
    def test_clean_synthetic_roundtrip_decodes_hello(self) -> None:
        result = _decode_samples(_transmission("HELLO"))

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")
        self.assertEqual(result.recovered_length, 5)
        self.assertTrue(result.crc_ok)
        self.assertEqual(result.samples_per_symbol, DEFAULT_CONFIG.samples_per_symbol)
        self.assertFalse(result.clipping_warning)
        self.assertTrue(result.sync_found)
        self.assertIsNotNone(result.best_candidate_score)

    def test_hello_decodes_with_random_extra_leading_silence(self) -> None:
        rng = np.random.default_rng(1234)
        base = _transmission("HELLO")

        for extra_samples in rng.integers(0, (2 * DEFAULT_CONFIG.sample_rate_hz) + 1, size=5):
            with self.subTest(extra_samples=int(extra_samples)):
                padded = np.concatenate((np.zeros(int(extra_samples), dtype=np.float64), base))
                result = _decode_samples(padded)
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, "HELLO")

    def test_hello_decodes_with_random_extra_trailing_silence(self) -> None:
        rng = np.random.default_rng(5678)
        base = _transmission("HELLO")

        for extra_samples in rng.integers(0, (2 * DEFAULT_CONFIG.sample_rate_hz) + 1, size=5):
            with self.subTest(extra_samples=int(extra_samples)):
                padded = np.concatenate((base, np.zeros(int(extra_samples), dtype=np.float64)))
                result = _decode_samples(padded)
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, "HELLO")

    def test_hello_decodes_when_burst_is_shifted_from_nominal_start(self) -> None:
        base = _transmission("HELLO")
        early_shift = DEFAULT_CONFIG.samples_per_symbol // 2
        shifted = np.concatenate((base[early_shift:], np.zeros(early_shift, dtype=np.float64)))

        result = _decode_samples(shifted)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")
        self.assertNotEqual(result.start_sample, DEFAULT_CONFIG.leading_silence_samples)

    def test_resampled_44100_hz_input_still_decodes(self) -> None:
        downsampled = resample_poly(_transmission("HELLO"), 147, 160)

        result = _decode_samples(downsampled, sample_rate=44_100)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_48k_to_44k1_to_48k_resampling_still_decodes(self) -> None:
        base = _transmission("HELLO")
        downsampled = resample_poly(base, 147, 160)
        restored = resample_poly(downsampled, 160, 147)

        result = _decode_samples(restored)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_stereo_input_is_converted_to_mono_and_still_decodes(self) -> None:
        transmission = _transmission("HELLO")
        stereo = np.column_stack((transmission, transmission))

        result = _decode_samples(stereo)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_additional_short_ascii_payloads_decode(self) -> None:
        for payload in ("A", "OK", "TEST"):
            with self.subTest(payload=payload):
                result = _decode_samples(_transmission(payload))
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, payload)

    def test_corrupted_clean_synthetic_wav_fails_closed(self) -> None:
        corrupted = _transmission("HELLO").copy()
        _corrupt_last_crc_symbol(corrupted)

        result = _decode_samples(corrupted)

        self.assertFalse(result.success)
        self.assertIsNone(result.decoded_text)
        self.assertEqual(result.failure_code, FailureCode.CRC_MISMATCH)
        self.assertFalse(result.crc_ok)


def _decode_samples(samples: np.ndarray, sample_rate: int = DEFAULT_CONFIG.sample_rate_hz):
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "roundtrip.wav"
        sample_array = np.asarray(samples, dtype=np.float64)
        if sample_array.ndim == 2:
            wavfile.write(wav_path, sample_rate, _to_pcm16(sample_array))
        else:
            write_wav(wav_path, sample_array, sample_rate)
        return decode_wav(wav_path, DEFAULT_CONFIG)


def _transmission(text: str) -> np.ndarray:
    frame_bits = bytes_to_bits(build_frame(validate_text(text)))
    return synthesize_transmission(frame_bits, DEFAULT_CONFIG)


def _to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return np.round(clipped * np.iinfo(np.int16).max).astype(np.int16)


def _corrupt_last_crc_symbol(samples: np.ndarray) -> None:
    frame_bits = bytes_to_bits(build_frame(validate_text("HELLO")))
    symbol_index = DEFAULT_CONFIG.tx_prefix_bit_count + frame_bits.size - 1
    start = DEFAULT_CONFIG.leading_silence_samples + (symbol_index * DEFAULT_CONFIG.samples_per_symbol)
    stop = start + DEFAULT_CONFIG.samples_per_symbol
    sample_offsets = np.arange(DEFAULT_CONFIG.samples_per_symbol, dtype=np.float64)
    replacement = DEFAULT_CONFIG.burst_amplitude * np.sin(
        (2.0 * np.pi * DEFAULT_CONFIG.f0_hz * sample_offsets) / DEFAULT_CONFIG.sample_rate_hz
    )
    samples[start:stop] = replacement


if __name__ == "__main__":
    unittest.main()
