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


@dataclass(frozen=True, slots=True)
class DecodeResult:
    decoded_text: str | None
    failure_code: FailureCode | None
    recovered_length: int | None
    crc_ok: bool
    weak_symbol_count: int
    samples_per_symbol: int
    clipping_warning: bool
    sync_found: bool
    start_sample: int | None
    best_candidate_score: float | None = None

    @property
    def success(self) -> bool:
        return self.failure_code is None and self.decoded_text is not None and self.crc_ok


@dataclass(frozen=True, slots=True)
class SyncCandidate:
    start_sample: int
    samples_per_symbol: int
    match_score: float
    coarse_region_start: int
    coarse_region_end: int


class FramingError(ValueError):
    __slots__ = ("code",)

    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code
