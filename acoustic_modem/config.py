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
    f0_hz: float = 1200.0
    f1_hz: float = 2200.0
    sample_rate_hz: int = 48_000
    symbol_duration_ms: float = 40.0
    leading_silence_ms: float = 250.0
    trailing_silence_ms: float = 250.0
    burst_amplitude: float = 0.7
    fade_duration_ms: float = 5.0
    bandpass_low_hz: float = 800.0
    bandpass_high_hz: float = 2600.0
    preprocess_peak_target: float = 0.95
    clipping_abs_threshold: float = 0.98
    clipping_fraction_threshold: float = 0.01
    weak_symbol_confidence_threshold: float = 0.15

    @property
    def sync_word(self) -> int:
        return int(self.sync_bits, 2)

    @property
    def tx_prefix_bits(self) -> str:
        return f"{self.leader_bits}{self.preamble_bits}{self.sync_bits}"

    @property
    def tx_prefix_bit_count(self) -> int:
        return len(self.tx_prefix_bits)

    @property
    def frame_overhead_bytes(self) -> int:
        return self.length_field_bytes + self.crc_bytes

    @property
    def min_frame_bytes(self) -> int:
        return self.frame_overhead_bytes + self.min_payload_length

    @property
    def max_frame_bytes(self) -> int:
        return self.frame_overhead_bytes + self.max_payload_length

    @property
    def samples_per_symbol(self) -> int:
        return int(round(self.sample_rate_hz * (self.symbol_duration_ms / 1_000.0)))

    @property
    def fade_samples(self) -> int:
        return int(round(self.sample_rate_hz * (self.fade_duration_ms / 1_000.0)))

    @property
    def leading_silence_samples(self) -> int:
        return int(round(self.sample_rate_hz * (self.leading_silence_ms / 1_000.0)))

    @property
    def trailing_silence_samples(self) -> int:
        return int(round(self.sample_rate_hz * (self.trailing_silence_ms / 1_000.0)))

    @property
    def silence_samples(self) -> int:
        return self.leading_silence_samples


DEFAULT_CONFIG = ModemConfig()
