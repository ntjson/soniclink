import unittest

import numpy as np

from acoustic_modem.framing import bits_to_bytes, build_frame, bytes_to_bits, parse_frame, validate_text
from acoustic_modem.types import FailureCode, FramingError


class FramingTests(unittest.TestCase):
    def test_validate_text_accepts_hello(self) -> None:
        self.assertEqual(validate_text("HELLO"), b"HELLO")

    def test_build_frame_for_hello_matches_golden_bytes(self) -> None:
        frame = build_frame(validate_text("HELLO"))
        self.assertEqual(frame, bytes.fromhex("05 48 45 4C 4C 4F 15 CB"))

    def test_frame_bits_for_hello_are_msb_first(self) -> None:
        frame = build_frame(validate_text("HELLO"))
        bits = bytes_to_bits(frame)

        self.assertEqual(
            "".join(str(int(bit)) for bit in bits),
            "0000010101001000010001010100110001001100010011110001010111001011",
        )

    def test_bits_roundtrip_is_lossless_and_msb_first(self) -> None:
        data = bytes.fromhex("80 01 A5")
        bits = bytes_to_bits(data)

        self.assertEqual(
            bits.tolist(),
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
            ],
        )
        self.assertEqual(bits_to_bytes(bits), data)

    def test_parse_frame_roundtrip_returns_payload_and_crc(self) -> None:
        frame = build_frame(validate_text("HELLO"))
        parsed = parse_frame(frame)

        self.assertEqual(parsed.length, 5)
        self.assertEqual(parsed.payload, b"HELLO")
        self.assertEqual(parsed.crc, 0x15CB)

    def test_validate_text_rejects_empty_input(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            validate_text("")

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_LENGTH)

    def test_validate_text_rejects_non_printable_ascii(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            validate_text("HELLO\n")

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_INPUT)

    def test_validate_text_rejects_non_ascii(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            validate_text("CAFÉ")

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_INPUT)

    def test_build_frame_rejects_non_printable_payload_bytes(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            build_frame(b"HELLO\x00")

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_INPUT)

    def test_bits_to_bytes_rejects_non_binary_values(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            bits_to_bytes(np.array([0, 1, 2, 0, 1, 0, 1, 0], dtype=np.int8))

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_INPUT)

    def test_bits_to_bytes_rejects_non_byte_aligned_input(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            bits_to_bytes(np.array([1, 0, 1], dtype=np.uint8))

        self.assertEqual(exc_info.exception.code, FailureCode.INVALID_INPUT)

    def test_parse_frame_rejects_crc_mismatch(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            parse_frame(bytes.fromhex("05 48 45 4C 4C 4F 15 CA"))

        self.assertEqual(exc_info.exception.code, FailureCode.CRC_MISMATCH)

    def test_parse_frame_rejects_non_ascii_payload(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            parse_frame(bytes.fromhex("01 1F 2D C1"))

        self.assertEqual(exc_info.exception.code, FailureCode.NON_ASCII_PAYLOAD)

    def test_parse_frame_rejects_truncated_payload(self) -> None:
        with self.assertRaises(FramingError) as exc_info:
            parse_frame(bytes.fromhex("05 48 45"))

        self.assertEqual(exc_info.exception.code, FailureCode.TRUNCATED_FRAME)


if __name__ == "__main__":
    unittest.main()
