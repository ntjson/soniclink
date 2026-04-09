from __future__ import annotations

import unittest

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.rx import preprocess_audio
from acoustic_modem.sync import find_leader_regions, find_sync_candidates, score_sync_candidate
from acoustic_modem.tx import synthesize_transmission


class SyncTests(unittest.TestCase):
    def test_leader_scan_finds_candidate_region_in_clean_synthetic_wav(self) -> None:
        processed = preprocess_audio(_transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz, DEFAULT_CONFIG)

        leader_regions = find_leader_regions(processed, DEFAULT_CONFIG)

        self.assertTrue(leader_regions)
        region_start, region_end = leader_regions[0]
        self.assertLessEqual(region_start, DEFAULT_CONFIG.leading_silence_samples + DEFAULT_CONFIG.sync_hop_samples)
        self.assertGreater(region_end, DEFAULT_CONFIG.leading_silence_samples)

    def test_fine_scoring_prefers_correct_start_offset_over_nearby_incorrect_offsets(self) -> None:
        processed = preprocess_audio(_transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz, DEFAULT_CONFIG)
        correct_score = score_sync_candidate(
            processed,
            DEFAULT_CONFIG.leading_silence_samples,
            DEFAULT_CONFIG.samples_per_symbol,
            DEFAULT_CONFIG,
        )
        shifted_score = score_sync_candidate(
            processed,
            DEFAULT_CONFIG.leading_silence_samples + DEFAULT_CONFIG.sync_hop_samples,
            DEFAULT_CONFIG.samples_per_symbol,
            DEFAULT_CONFIG,
        )

        self.assertGreater(correct_score, shifted_score)

    def test_fine_scoring_prefers_correct_samples_per_symbol_over_clearly_wrong_values(self) -> None:
        processed = preprocess_audio(_transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz, DEFAULT_CONFIG)
        correct_score = score_sync_candidate(
            processed,
            DEFAULT_CONFIG.leading_silence_samples,
            DEFAULT_CONFIG.samples_per_symbol,
            DEFAULT_CONFIG,
        )
        wrong_score = score_sync_candidate(
            processed,
            DEFAULT_CONFIG.leading_silence_samples,
            DEFAULT_CONFIG.sync_min_samples_per_symbol,
            DEFAULT_CONFIG,
        )

        self.assertGreater(correct_score, wrong_score)

    def test_accepted_candidates_are_sorted_best_first(self) -> None:
        padded = np.concatenate(
            (
                np.zeros(DEFAULT_CONFIG.sync_hop_samples * 2, dtype=np.float64),
                _transmission("HELLO"),
                np.zeros(DEFAULT_CONFIG.sync_hop_samples * 3, dtype=np.float64),
            )
        )
        processed = preprocess_audio(padded, DEFAULT_CONFIG.sample_rate_hz, DEFAULT_CONFIG)

        candidates = find_sync_candidates(processed, DEFAULT_CONFIG)

        self.assertTrue(candidates)
        self.assertEqual(
            [candidate.match_score for candidate in candidates],
            sorted((candidate.match_score for candidate in candidates), reverse=True),
        )

    def test_candidate_list_is_capped_at_five_backups(self) -> None:
        long_signal = np.concatenate(
            (
                np.zeros(DEFAULT_CONFIG.sync_hop_samples * 2, dtype=np.float64),
                _transmission("HELLO"),
                np.zeros(DEFAULT_CONFIG.sync_hop_samples * 20, dtype=np.float64),
            )
        )
        processed = preprocess_audio(long_signal, DEFAULT_CONFIG.sample_rate_hz, DEFAULT_CONFIG)

        candidates = find_sync_candidates(processed, DEFAULT_CONFIG)

        self.assertLessEqual(len(candidates), DEFAULT_CONFIG.sync_candidate_limit)


def _transmission(text: str) -> np.ndarray:
    frame_bits = bytes_to_bits(build_frame(validate_text(text)))
    return synthesize_transmission(frame_bits, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
