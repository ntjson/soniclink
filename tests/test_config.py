import unittest

from acoustic_modem.config import DEFAULT_CONFIG


class ConfigTests(unittest.TestCase):
    def test_default_sample_count_constants_match_sdd(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.samples_per_symbol, 1920)
        self.assertEqual(DEFAULT_CONFIG.fade_samples, 240)
        self.assertEqual(DEFAULT_CONFIG.leading_silence_samples, 12000)
        self.assertEqual(DEFAULT_CONFIG.trailing_silence_samples, 12000)
        self.assertEqual(DEFAULT_CONFIG.silence_samples, 12000)


if __name__ == "__main__":
    unittest.main()
