import unittest

from acoustic_modem.framing import build_frame, crc16_ccitt_false, validate_text


class CrcTests(unittest.TestCase):
    def test_crc16_ccitt_false_standard_vector(self) -> None:
        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_crc16_ccitt_false_empty_input_uses_config_init(self) -> None:
        self.assertEqual(crc16_ccitt_false(b""), 0xFFFF)

    def test_hello_frame_crc_matches_golden_value(self) -> None:
        payload = validate_text("HELLO")
        frame = build_frame(payload)

        self.assertEqual(frame, bytes.fromhex("05 48 45 4C 4C 4F 15 CB"))
        self.assertEqual(crc16_ccitt_false(frame[:-2]), 0x15CB)


if __name__ == "__main__":
    unittest.main()
