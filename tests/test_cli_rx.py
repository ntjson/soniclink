from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.tx import synthesize_transmission, write_wav


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliRxTests(unittest.TestCase):
    def test_rx_cli_decodes_known_good_wav_and_writes_json_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "hello.wav"
            json_out = Path(temp_dir) / "result.json"
            debug_dir = Path(temp_dir) / "debug"
            write_wav(input_path, _transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz)

            result = _run_cli(
                "rx",
                "--in",
                str(input_path),
                "--json-out",
                str(json_out),
                "--debug-dir",
                str(debug_dir),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "HELLO")
            self.assertTrue(json_out.exists())
            self.assertTrue((debug_dir / "decode_summary.json").exists())

            summary = json.loads(json_out.read_text(encoding="ascii"))
            self.assertTrue(summary["success"])
            self.assertEqual(summary["decoded_text"], "HELLO")
            self.assertEqual(summary["input_path"], str(input_path))

    def test_rx_cli_invalid_path_and_unsupported_input_exit_with_code_2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.wav"
            missing_result = _run_cli("rx", "--in", str(missing_path))
            self.assertEqual(missing_result.returncode, 2)
            self.assertIn("unsupported_format", missing_result.stderr)

            invalid_path = Path(temp_dir) / "not_wav.txt"
            invalid_path.write_text("not a wav file\n", encoding="ascii")
            invalid_result = _run_cli("rx", "--in", str(invalid_path))
            self.assertEqual(invalid_result.returncode, 2)
            self.assertIn("unsupported_format", invalid_result.stderr)

    def test_rx_cli_expect_success_and_mismatch_behaviors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "hello.wav"
            mismatch_json = Path(temp_dir) / "mismatch.json"
            write_wav(input_path, _transmission("HELLO"), DEFAULT_CONFIG.sample_rate_hz)

            success_result = _run_cli("rx", "--in", str(input_path), "--expect", "HELLO")
            self.assertEqual(success_result.returncode, 0)
            self.assertEqual(success_result.stdout.strip(), "HELLO")

            mismatch_result = _run_cli(
                "rx",
                "--in",
                str(input_path),
                "--expect",
                "WORLD",
                "--json-out",
                str(mismatch_json),
            )
            self.assertEqual(mismatch_result.returncode, 5)
            self.assertEqual(mismatch_result.stdout.strip(), "HELLO")
            self.assertIn("expectation_mismatch", mismatch_result.stderr)

            summary = json.loads(mismatch_json.read_text(encoding="ascii"))
            self.assertEqual(summary["expected_text"], "WORLD")
            self.assertFalse(summary["expectation_met"])

    def test_rx_cli_sync_failure_exits_with_code_3(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "silence.wav"
            write_wav(input_path, np.zeros(DEFAULT_CONFIG.sample_rate_hz, dtype=np.float64), DEFAULT_CONFIG.sample_rate_hz)

            result = _run_cli("rx", "--in", str(input_path))

            self.assertEqual(result.returncode, 3)
            self.assertIn("sync_not_found", result.stderr)

    def test_rx_cli_frame_validation_failure_exits_with_code_4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "corrupted.wav"
            corrupted = _transmission("HELLO")
            _corrupt_last_crc_symbol(corrupted)
            write_wav(input_path, corrupted, DEFAULT_CONFIG.sample_rate_hz)

            result = _run_cli("rx", "--in", str(input_path))

            self.assertEqual(result.returncode, 4)
            self.assertIn("crc_mismatch", result.stderr)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acoustic_modem.cli", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _transmission(text: str) -> np.ndarray:
    frame_bits = bytes_to_bits(build_frame(validate_text(text)))
    return synthesize_transmission(frame_bits, DEFAULT_CONFIG)


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
