from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTxTests(unittest.TestCase):
    def test_tx_cli_creates_hello_wav_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "hello.wav"
            result = _run_cli("tx", "--text", "HELLO", "--out", str(out_path))

            self.assertEqual(result.returncode, 0)
            self.assertTrue(out_path.exists())
            self.assertIn("length=5", result.stdout)
            self.assertIn("sample_rate=48000", result.stdout)
            self.assertIn("duration=5.14s", result.stdout)

    def test_tx_cli_output_wav_has_expected_format_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "hello.wav"
            result = _run_cli("tx", "--text", "HELLO", "--out", str(out_path))

            self.assertEqual(result.returncode, 0)
            with wave.open(str(out_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 48_000)
                self.assertEqual(wav_file.getnframes(), 246_720)
                self.assertAlmostEqual(wav_file.getnframes() / wav_file.getframerate(), 5.14, places=12)

    def test_tx_cli_invalid_text_exits_with_code_2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "invalid.wav"
            result = _run_cli("tx", "--text", "CAFÉ", "--out", str(out_path))

            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)
            self.assertFalse(out_path.exists())


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acoustic_modem.cli", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


if __name__ == "__main__":
    unittest.main()
