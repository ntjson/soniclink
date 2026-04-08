from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG, ModemConfig


def synthesize_transmission(frame_bits: np.ndarray, cfg: ModemConfig = DEFAULT_CONFIG) -> np.ndarray:
    payload_bits = _coerce_bits(frame_bits)
    tx_bits = np.concatenate((_bits_from_string(cfg.tx_prefix_bits), payload_bits))

    active_samples = _synthesize_active_burst(tx_bits, cfg)
    transmission = np.concatenate(
        (
            np.zeros(cfg.leading_silence_samples, dtype=np.float64),
            active_samples,
            np.zeros(cfg.trailing_silence_samples, dtype=np.float64),
        )
    )
    return transmission


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    sample_array = np.asarray(samples, dtype=np.float64)

    if sample_array.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not np.isfinite(sample_array).all():
        raise ValueError("samples must be finite")

    peak = float(np.max(np.abs(sample_array), initial=0.0))
    if peak > 1.0 + 1e-12:
        raise ValueError("samples must be within [-1.0, 1.0]")

    pcm = np.round(np.clip(sample_array, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _coerce_bits(bits: np.ndarray) -> np.ndarray:
    bit_array = np.asarray(bits)
    if bit_array.ndim != 1:
        raise ValueError("frame_bits must be a one-dimensional array")
    if not np.isin(bit_array, (0, 1)).all():
        raise ValueError("frame_bits must contain only 0 or 1 values")
    return bit_array.astype(np.uint8, copy=False)


def _bits_from_string(bit_string: str) -> np.ndarray:
    return np.fromiter((1 if bit == "1" else 0 for bit in bit_string), dtype=np.uint8)


def _synthesize_active_burst(bits: np.ndarray, cfg: ModemConfig) -> np.ndarray:
    samples_per_symbol = cfg.samples_per_symbol
    sample_offsets = np.arange(samples_per_symbol, dtype=np.float64)
    angular_scale = (2.0 * np.pi) / cfg.sample_rate_hz

    active = np.empty(bits.size * samples_per_symbol, dtype=np.float64)
    phase = 0.0

    for index, bit in enumerate(bits):
        frequency = cfg.f1_hz if int(bit) == 1 else cfg.f0_hz
        start = index * samples_per_symbol
        stop = start + samples_per_symbol
        active[start:stop] = np.sin(phase + (angular_scale * frequency * sample_offsets))
        phase = (phase + (angular_scale * frequency * samples_per_symbol)) % (2.0 * np.pi)

    active *= cfg.burst_amplitude
    _apply_burst_envelope(active, cfg.fade_samples)
    return active


def _apply_burst_envelope(active: np.ndarray, fade_samples: int) -> None:
    if active.size == 0 or fade_samples <= 0:
        return

    fade_count = min(fade_samples, active.size // 2)
    if fade_count == 0:
        return

    envelope = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_count, dtype=np.float64))
    active[:fade_count] *= envelope
    active[-fade_count:] *= envelope[::-1]
