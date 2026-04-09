from __future__ import annotations

import unittest
from pathlib import Path

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.rx import decode_wav


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "phone"


class PhoneRoundtripTests(unittest.TestCase):
    def test_phone_fixtures_decode_to_hello(self) -> None:
        fixtures = sorted(FIXTURE_DIR.rglob("*.wav")) if FIXTURE_DIR.exists() else []
        if not fixtures:
            self.skipTest("phone fixtures not present")

        for fixture_path in fixtures:
            with self.subTest(fixture=fixture_path.name):
                result = decode_wav(fixture_path, DEFAULT_CONFIG)
                self.assertTrue(result.success)
                self.assertEqual(result.decoded_text, "HELLO")
                self.assertTrue(result.crc_ok)


if __name__ == "__main__":
    unittest.main()
