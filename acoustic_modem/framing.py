from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from acoustic_modem.config import DEFAULT_CONFIG, ModemConfig
from acoustic_modem.types import FailureCode, FrameFields, FramingError


def validate_text(text: str, cfg: ModemConfig = DEFAULT_CONFIG) -> bytes:
    if not isinstance(text, str):
        raise FramingError(FailureCode.INVALID_INPUT, "text must be a string")

    try:
        payload = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise FramingError(
            FailureCode.INVALID_INPUT,
            "text must contain only printable ASCII characters",
        ) from exc

    return _validate_payload_bytes(
        payload,
        cfg=cfg,
        printable_error_code=FailureCode.INVALID_INPUT,
        printable_error_message="text must contain only printable ASCII characters",
    )


def crc16_ccitt_false(data: bytes | bytearray | memoryview, cfg: ModemConfig = DEFAULT_CONFIG) -> int:
    raw = _coerce_bytes(data, "data must be bytes-like")
    crc = cfg.crc_init

    for byte in raw:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ cfg.crc_poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc ^ cfg.crc_xorout


def build_frame(payload: bytes | bytearray | memoryview, cfg: ModemConfig = DEFAULT_CONFIG) -> bytes:
    validated_payload = _validate_payload_bytes(
        payload,
        cfg=cfg,
        printable_error_code=FailureCode.INVALID_INPUT,
        printable_error_message="payload must contain only printable ASCII bytes",
    )
    body = bytes([len(validated_payload)]) + validated_payload
    crc = crc16_ccitt_false(body, cfg)
    return body + crc.to_bytes(cfg.crc_bytes, byteorder="big")


def parse_frame(frame: bytes | bytearray | memoryview, cfg: ModemConfig = DEFAULT_CONFIG) -> FrameFields:
    raw = _coerce_bytes(frame, "frame must be bytes-like")

    if len(raw) < cfg.min_frame_bytes:
        raise FramingError(
            FailureCode.TRUNCATED_FRAME,
            "frame is shorter than the minimum valid length",
        )

    length = raw[0]
    if not cfg.min_payload_length <= length <= cfg.max_payload_length:
        raise FramingError(
            FailureCode.INVALID_LENGTH,
            f"frame length field must be {cfg.min_payload_length}..{cfg.max_payload_length}",
        )

    expected_length = cfg.length_field_bytes + length + cfg.crc_bytes
    if len(raw) < expected_length:
        raise FramingError(
            FailureCode.TRUNCATED_FRAME,
            "frame ended before the declared payload and CRC bytes",
        )
    if len(raw) != expected_length:
        raise FramingError(
            FailureCode.INVALID_LENGTH,
            "frame length does not match the declared payload length",
        )

    payload_end = cfg.length_field_bytes + length
    payload = raw[cfg.length_field_bytes:payload_end]
    validated_payload = _validate_payload_bytes(
        payload,
        cfg=cfg,
        printable_error_code=FailureCode.NON_ASCII_PAYLOAD,
        printable_error_message="payload must contain only printable ASCII bytes",
    )

    received_crc = int.from_bytes(raw[payload_end:expected_length], byteorder="big")
    expected_crc = crc16_ccitt_false(raw[:payload_end], cfg)
    if received_crc != expected_crc:
        raise FramingError(
            FailureCode.CRC_MISMATCH,
            f"crc mismatch: expected 0x{expected_crc:04X}, got 0x{received_crc:04X}",
        )

    return FrameFields(length=length, payload=validated_payload, crc=received_crc)


def bytes_to_bits(data: bytes | bytearray | memoryview) -> np.ndarray:
    raw = _coerce_bytes(data, "data must be bytes-like")
    byte_array = np.frombuffer(raw, dtype=np.uint8)
    return np.unpackbits(byte_array, bitorder="big")


def bits_to_bytes(bits: ArrayLike) -> bytes:
    bit_array = np.asarray(bits)
    if bit_array.ndim != 1:
        raise FramingError(FailureCode.INVALID_INPUT, "bits must be a one-dimensional array")
    if bit_array.size % 8 != 0:
        raise FramingError(FailureCode.INVALID_INPUT, "bit array length must be divisible by 8")
    if not np.isin(bit_array, (0, 1)).all():
        raise FramingError(FailureCode.INVALID_INPUT, "bit array must contain only 0 or 1 values")

    packed = np.packbits(bit_array.astype(np.uint8, copy=False), bitorder="big")
    return packed.tobytes()


def _coerce_bytes(value: bytes | bytearray | memoryview, type_error_message: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise FramingError(FailureCode.INVALID_INPUT, type_error_message)
    return bytes(value)


def _validate_payload_bytes(
    payload: bytes | bytearray | memoryview,
    cfg: ModemConfig,
    printable_error_code: FailureCode,
    printable_error_message: str,
) -> bytes:
    raw = _coerce_bytes(payload, "payload must be bytes-like")

    if not cfg.min_payload_length <= len(raw) <= cfg.max_payload_length:
        raise FramingError(
            FailureCode.INVALID_LENGTH,
            f"payload length must be {cfg.min_payload_length}..{cfg.max_payload_length} bytes",
        )

    if any(byte < cfg.printable_ascii_min or byte > cfg.printable_ascii_max for byte in raw):
        raise FramingError(printable_error_code, printable_error_message)

    return raw
