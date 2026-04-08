from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModemConfig:
    printable_ascii_min: int = 0x20
    printable_ascii_max: int = 0x7E
    min_payload_length: int = 1
    max_payload_length: int = 16
    length_field_bytes: int = 1
    crc_bytes: int = 2
    crc_poly: int = 0x1021
    crc_init: int = 0xFFFF
    crc_xorout: int = 0x0000
    crc_refin: bool = False
    crc_refout: bool = False
    leader_bits: str = "111111111111"
    preamble_bits: str = "010101010101010101010101"
    sync_bits: str = "1110010110010110"

    @property
    def sync_word(self) -> int:
        return int(self.sync_bits, 2)

    @property
    def frame_overhead_bytes(self) -> int:
        return self.length_field_bytes + self.crc_bytes

    @property
    def min_frame_bytes(self) -> int:
        return self.frame_overhead_bytes + self.min_payload_length

    @property
    def max_frame_bytes(self) -> int:
        return self.frame_overhead_bytes + self.max_payload_length


DEFAULT_CONFIG = ModemConfig()
