from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG, ModemConfig
from acoustic_modem.dsp import (
    bandpass_filter,
    detect_likely_clipping,
    hann_window,
    load_wav,
    pcm_to_float,
    peak_normalize,
    resample_audio,
    to_mono,
    tone_energies,
)
from acoustic_modem.framing import bits_to_bytes, parse_frame
from acoustic_modem.types import DecodeResult, FailureCode, FramingError


def preprocess_audio(samples: np.ndarray, sample_rate: int, cfg: ModemConfig = DEFAULT_CONFIG) -> np.ndarray:
    mono_float = to_mono(pcm_to_float(samples))
    resampled = resample_audio(mono_float, sample_rate, cfg.sample_rate_hz)
    filtered = bandpass_filter(resampled, cfg.sample_rate_hz, cfg.bandpass_low_hz, cfg.bandpass_high_hz)
    return peak_normalize(filtered, cfg.preprocess_peak_target)


def decode_wav(path: Path, cfg: ModemConfig = DEFAULT_CONFIG) -> DecodeResult:
    try:
        raw_samples, sample_rate = load_wav(path)
    except (OSError, ValueError):
        return DecodeResult(
            decoded_text=None,
            failure_code=FailureCode.UNSUPPORTED_FORMAT,
            recovered_length=None,
            crc_ok=False,
            weak_symbol_count=0,
            samples_per_symbol=cfg.samples_per_symbol,
            clipping_warning=False,
            sync_found=False,
            start_sample=None,
        )

    mono_float = to_mono(pcm_to_float(raw_samples))
    clipping_warning = detect_likely_clipping(
        mono_float,
        abs_threshold=cfg.clipping_abs_threshold,
        fraction_threshold=cfg.clipping_fraction_threshold,
    )
    processed = preprocess_audio(raw_samples, sample_rate, cfg)
    return _decode_clean_synthetic(processed, clipping_warning, cfg)


def _decode_clean_synthetic(
    processed_samples: np.ndarray,
    clipping_warning: bool,
    cfg: ModemConfig,
) -> DecodeResult:
    start_sample = cfg.leading_silence_samples
    samples_per_symbol = cfg.samples_per_symbol
    prefix_samples = cfg.tx_prefix_bit_count * samples_per_symbol
    data_start = start_sample + prefix_samples
    data_end = processed_samples.size - cfg.trailing_silence_samples

    if data_end <= data_start:
        return _failure_result(
            failure_code=FailureCode.TRUNCATED_FRAME,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=start_sample,
            cfg=cfg,
        )

    data_region = processed_samples[data_start:data_end]
    if data_region.size % samples_per_symbol != 0:
        return _failure_result(
            failure_code=FailureCode.TRUNCATED_FRAME,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=start_sample,
            cfg=cfg,
        )

    symbol_count = data_region.size // samples_per_symbol
    if symbol_count == 0:
        return _failure_result(
            failure_code=FailureCode.TRUNCATED_FRAME,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=start_sample,
            cfg=cfg,
        )

    window = hann_window(samples_per_symbol)
    bits = np.empty(symbol_count, dtype=np.uint8)
    weak_symbol_count = 0

    for index in range(symbol_count):
        symbol = data_region[index * samples_per_symbol : (index + 1) * samples_per_symbol]
        energy_0, energy_1 = tone_energies(symbol, cfg, window=window)
        bits[index] = 1 if energy_1 > energy_0 else 0
        confidence = abs(energy_1 - energy_0) / (energy_0 + energy_1 + 1e-12)
        if confidence < cfg.weak_symbol_confidence_threshold:
            weak_symbol_count += 1

    frame_bytes = bits_to_bytes(bits)
    recovered_length = frame_bytes[0] if frame_bytes else None

    try:
        parsed = parse_frame(frame_bytes, cfg)
    except FramingError as exc:
        return _failure_result(
            failure_code=exc.code,
            recovered_length=recovered_length,
            weak_symbol_count=weak_symbol_count,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=start_sample,
            cfg=cfg,
        )

    return DecodeResult(
        decoded_text=parsed.payload.decode("ascii"),
        failure_code=None,
        recovered_length=parsed.length,
        crc_ok=True,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=samples_per_symbol,
        clipping_warning=clipping_warning,
        sync_found=True,
        start_sample=start_sample,
    )


def _failure_result(
    failure_code: FailureCode,
    recovered_length: int | None,
    weak_symbol_count: int,
    clipping_warning: bool,
    sync_found: bool,
    start_sample: int | None,
    cfg: ModemConfig,
) -> DecodeResult:
    return DecodeResult(
        decoded_text=None,
        failure_code=failure_code,
        recovered_length=recovered_length,
        crc_ok=False,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=cfg.samples_per_symbol,
        clipping_warning=clipping_warning,
        sync_found=sync_found,
        start_sample=start_sample,
    )
