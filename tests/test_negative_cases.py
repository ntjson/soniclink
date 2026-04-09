from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.rx import decode_wav
from acoustic_modem.tx import write_wav
from acoustic_modem.types import FailureCode, SyncCandidate
from tests.support import transmission


class NegativeCaseTests(unittest.TestCase):
    def test_decoder_uses_backup_candidate_when_best_candidate_fails_frame_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hello.wav"
            write_wav(wav_path, transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz)

            bad_candidate = SyncCandidate(
                start_sample=DEFAULT_CONFIG.leading_silence_samples + (DEFAULT_CONFIG.samples_per_symbol // 2),
                samples_per_symbol=DEFAULT_CONFIG.samples_per_symbol,
                match_score=0.95,
                coarse_region_start=DEFAULT_CONFIG.leading_silence_samples,
                coarse_region_end=DEFAULT_CONFIG.leading_silence_samples + DEFAULT_CONFIG.samples_per_symbol,
            )
            good_candidate = SyncCandidate(
                start_sample=DEFAULT_CONFIG.leading_silence_samples,
                samples_per_symbol=DEFAULT_CONFIG.samples_per_symbol,
                match_score=0.90,
                coarse_region_start=DEFAULT_CONFIG.leading_silence_samples,
                coarse_region_end=DEFAULT_CONFIG.leading_silence_samples + DEFAULT_CONFIG.samples_per_symbol,
            )

            with patch("acoustic_modem.rx.find_sync_candidates", return_value=[bad_candidate, good_candidate]):
                result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertTrue(result.success)
            self.assertEqual(result.decoded_text, "HELLO")
            self.assertEqual(result.start_sample, good_candidate.start_sample)
            self.assertEqual(result.best_candidate_score, bad_candidate.match_score)

    def test_silence_only_recording_fails_with_sync_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "silence.wav"
            silence = np.zeros(DEFAULT_CONFIG.sample_rate_hz, dtype=np.float64)
            write_wav(wav_path, silence, DEFAULT_CONFIG.sample_rate_hz)

            result = decode_wav(wav_path, DEFAULT_CONFIG)

            self.assertFalse(result.success)
            self.assertEqual(result.failure_code, FailureCode.SYNC_NOT_FOUND)
            self.assertFalse(result.sync_found)

if __name__ == "__main__":
    unittest.main()
