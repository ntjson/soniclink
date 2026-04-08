from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_FORMAT = "unsupported_format"
    SYNC_NOT_FOUND = "sync_not_found"
    INVALID_LENGTH = "invalid_length"
    CRC_MISMATCH = "crc_mismatch"
    NON_ASCII_PAYLOAD = "non_ascii_payload"
    TRUNCATED_FRAME = "truncated_frame"


@dataclass(frozen=True, slots=True)
class FrameFields:
    length: int
    payload: bytes
    crc: int


class FramingError(ValueError):
    __slots__ = ("code",)

    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code
