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
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello.wav"
            write_wav(wav_path, _hello_transmission(), DEFAULT_CONFIG.sample_rate_hz)

            result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertTrue(result.success)
            self.assertEqual(result.decoded_text, "HELLO")
            self.assertEqual(result.recovered_length, 5)
            self.assertTrue(result.crc_ok)
            self.assertEqual(result.samples_per_symbol, DEFAULT_CONFIG.samples_per_symbol)
            self.assertFalse(result.clipping_warning)

    def test_44100_hz_input_is_resampled_and_still_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello_44100.wav"
            downsampled = resample_poly(_hello_transmission(), 147, 160)
            write_wav(wav_path, downsampled, 44_100)

            result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertTrue(result.success)
            self.assertEqual(result.decoded_text, "HELLO")

    def test_stereo_input_is_converted_to_mono_and_still_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello_stereo.wav"
            transmission = _hello_transmission()
            stereo = np.column_stack((transmission, transmission))
            wavfile.write(wav_path, DEFAULT_CONFIG.sample_rate_hz, _to_pcm16(stereo))

            result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertTrue(result.success)
            self.assertEqual(result.decoded_text, "HELLO")

    def test_corrupted_clean_synthetic_wav_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello_corrupted.wav"
            corrupted = _hello_transmission().copy()
            _corrupt_last_crc_symbol(corrupted)
            write_wav(wav_path, corrupted, DEFAULT_CONFIG.sample_rate_hz)

            result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertFalse(result.success)
            self.assertIsNone(result.decoded_text)
            self.assertEqual(result.failure_code, FailureCode.CRC_MISMATCH)
            self.assertFalse(result.crc_ok)


def _hello_transmission() -> np.ndarray:
    frame_bits = bytes_to_bits(build_frame(validate_text("HELLO")))
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
