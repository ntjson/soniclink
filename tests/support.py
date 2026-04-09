from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from acoustic_modem.config import DEFAULT_CONFIG
from acoustic_modem.framing import build_frame, bytes_to_bits, validate_text
from acoustic_modem.rx import decode_wav
from acoustic_modem.tx import synthesize_transmission, write_wav
from acoustic_modem.types import DecodeResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acoustic_modem.cli", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def transmission(text: str) -> np.ndarray:
    frame_bits = bytes_to_bits(build_frame(validate_text(text)))
    return synthesize_transmission(frame_bits, DEFAULT_CONFIG)


def corrupt_last_crc_symbol(samples: np.ndarray, text: str = "HELLO") -> None:
    frame_bits = bytes_to_bits(build_frame(validate_text(text)))
    symbol_index = DEFAULT_CONFIG.tx_prefix_bit_count + frame_bits.size - 1
    start = DEFAULT_CONFIG.leading_silence_samples + (symbol_index * DEFAULT_CONFIG.samples_per_symbol)
    stop = start + DEFAULT_CONFIG.samples_per_symbol
    sample_offsets = np.arange(DEFAULT_CONFIG.samples_per_symbol, dtype=np.float64)
    replacement = DEFAULT_CONFIG.burst_amplitude * np.sin(
        (2.0 * np.pi * DEFAULT_CONFIG.f0_hz * sample_offsets) / DEFAULT_CONFIG.sample_rate_hz
    )
    samples[start:stop] = replacement


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return np.round(clipped * np.iinfo(np.int16).max).astype(np.int16)


def decode_samples(samples: np.ndarray, sample_rate: int = DEFAULT_CONFIG.sample_rate_hz) -> DecodeResult:
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "roundtrip.wav"
        sample_array = np.asarray(samples, dtype=np.float64)
        if sample_array.ndim == 2:
            wavfile.write(wav_path, sample_rate, to_pcm16(sample_array))
        else:
            write_wav(wav_path, sample_array, sample_rate)
        return decode_wav(wav_path, DEFAULT_CONFIG)
