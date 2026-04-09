from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

from acoustic_modem.config import DEFAULT_CONFIG, ModemConfig


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, samples = wavfile.read(Path(path))
    return samples, int(sample_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    sample_array = np.asarray(samples)
    if sample_array.ndim == 1:
        return sample_array
    if sample_array.ndim == 2:
        return np.mean(sample_array, axis=1)
    raise ValueError("samples must be mono or stereo")


def pcm_to_float(samples: np.ndarray) -> np.ndarray:
    sample_array = np.asarray(samples)

    if np.issubdtype(sample_array.dtype, np.floating):
        return np.clip(sample_array.astype(np.float64, copy=False), -1.0, 1.0)
    if np.issubdtype(sample_array.dtype, np.signedinteger):
        info = np.iinfo(sample_array.dtype)
        scale = float(max(abs(info.min), info.max))
        return sample_array.astype(np.float64) / scale
    if np.issubdtype(sample_array.dtype, np.unsignedinteger):
        info = np.iinfo(sample_array.dtype)
        midpoint = (info.max + 1) / 2.0
        return (sample_array.astype(np.float64) - midpoint) / midpoint

    raise ValueError("unsupported WAV sample dtype")


def resample_audio(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return np.asarray(samples, dtype=np.float64)

    rate_gcd = gcd(sample_rate, target_rate)
    up = target_rate // rate_gcd
    down = sample_rate // rate_gcd
    return resample_poly(np.asarray(samples, dtype=np.float64), up, down)


def bandpass_filter(samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.size == 0:
        return sample_array.copy()

    sos_array = np.asarray(
        butter(6, (low_hz, high_hz), btype="bandpass", fs=sample_rate, output="sos"),
        dtype=np.float64,
    )
    min_length_for_filtfilt = (3 * (2 * sos_array.shape[0] + 1)) + 1
    if sample_array.size < min_length_for_filtfilt:
        return sample_array.copy()
    return sosfiltfilt(sos_array, sample_array)


def peak_normalize(samples: np.ndarray, target_peak: float) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(sample_array), initial=0.0))
    if peak == 0.0:
        return sample_array.copy()
    return sample_array * (target_peak / peak)


def detect_likely_clipping(
    samples: np.ndarray,
    abs_threshold: float,
    fraction_threshold: float,
) -> bool:
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.size == 0:
        return False
    clipped_fraction = float(np.mean(np.abs(sample_array) >= abs_threshold))
    return clipped_fraction > fraction_threshold


def hann_window(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    return np.hanning(length).astype(np.float64)


def goertzel_power(
    samples: np.ndarray,
    target_freq_hz: float,
    sample_rate_hz: int,
    window: np.ndarray | None = None,
) -> float:
    sample_array = np.asarray(samples, dtype=np.float64)
    if window is not None:
        window_array = np.asarray(window, dtype=np.float64)
        if window_array.shape != sample_array.shape:
            raise ValueError("window must match the sample shape")
        sample_array = sample_array * window_array

    omega = 2.0 * np.pi * target_freq_hz / sample_rate_hz
    coeff = 2.0 * np.cos(omega)
    state_prev = 0.0
    state_prev2 = 0.0

    for sample in sample_array:
        state = float(sample) + (coeff * state_prev) - state_prev2
        state_prev2 = state_prev
        state_prev = state

    power = (state_prev2**2) + (state_prev**2) - (coeff * state_prev * state_prev2)
    return float(max(power, 0.0))


def tone_energies(
    samples: np.ndarray,
    cfg: ModemConfig = DEFAULT_CONFIG,
    window: np.ndarray | None = None,
) -> tuple[float, float]:
    energy_0 = goertzel_power(samples, cfg.f0_hz, cfg.sample_rate_hz, window=window)
    energy_1 = goertzel_power(samples, cfg.f1_hz, cfg.sample_rate_hz, window=window)
    return energy_0, energy_1
