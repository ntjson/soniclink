from __future__ import annotations

import tempfile
import unittest

import numpy as np
from scipy.signal import resample_poly

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.types import FailureCode
from tests.support import corrupt_last_crc_symbol, decode_samples, transmission


class SyntheticRoundtripTests(unittest.TestCase):
    def test_clean_synthetic_roundtrip_decodes_hello(self) -> None:
        result = decode_samples(transmission("HELLO"))

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
        base = transmission("HELLO")

        for extra_samples in rng.integers(0, (2 * DEFAULT_CONFIG.sample_rate_hz) + 1, size=5):
            with self.subTest(extra_samples=int(extra_samples)):
                padded = np.concatenate((np.zeros(int(extra_samples), dtype=np.float64), base))
                result = decode_samples(padded)
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, "HELLO")

    def test_hello_decodes_with_random_extra_trailing_silence(self) -> None:
        rng = np.random.default_rng(5678)
        base = transmission("HELLO")

        for extra_samples in rng.integers(0, (2 * DEFAULT_CONFIG.sample_rate_hz) + 1, size=5):
            with self.subTest(extra_samples=int(extra_samples)):
                padded = np.concatenate((base, np.zeros(int(extra_samples), dtype=np.float64)))
                result = decode_samples(padded)
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, "HELLO")

    def test_hello_decodes_when_burst_is_shifted_from_nominal_start(self) -> None:
        base = transmission("HELLO")
        early_shift = DEFAULT_CONFIG.samples_per_symbol // 2
        shifted = np.concatenate((base[early_shift:], np.zeros(early_shift, dtype=np.float64)))

        result = decode_samples(shifted)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")
        self.assertNotEqual(result.start_sample, DEFAULT_CONFIG.leading_silence_samples)

    def test_resampled_44100_hz_input_still_decodes(self) -> None:
        downsampled = resample_poly(transmission("HELLO"), 147, 160)

        result = decode_samples(downsampled, sample_rate=44_100)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_48k_to_44k1_to_48k_resampling_still_decodes(self) -> None:
        base = transmission("HELLO")
        downsampled = resample_poly(base, 147, 160)
        restored = resample_poly(downsampled, 160, 147)

        result = decode_samples(restored)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_stereo_input_is_converted_to_mono_and_still_decodes(self) -> None:
        hello = transmission("HELLO")
        stereo = np.column_stack((hello, hello))

        result = decode_samples(stereo)

        self.assertTrue(result.success)
        self.assertEqual(result.decoded_text, "HELLO")

    def test_additional_short_ascii_payloads_decode(self) -> None:
        for payload in ("A", "OK", "TEST"):
            with self.subTest(payload=payload):
                result = decode_samples(transmission(payload))
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, payload)

    def test_corrupted_clean_synthetic_wav_fails_closed(self) -> None:
        corrupted = transmission("HELLO").copy()
        corrupt_last_crc_symbol(corrupted)

        result = decode_samples(corrupted)

        self.assertFalse(result.success)
        self.assertIsNone(result.decoded_text)
        self.assertEqual(result.failure_code, FailureCode.CRC_MISMATCH)
        self.assertFalse(result.crc_ok)

if __name__ == "__main__":
    unittest.main()
