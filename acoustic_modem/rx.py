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
from acoustic_modem.sync import find_sync_candidates
from acoustic_modem.types import DecodeResult, FailureCode, FramingError, SyncCandidate


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
            best_candidate_score=None,
        )

    mono_float = to_mono(pcm_to_float(raw_samples))
    clipping_warning = detect_likely_clipping(
        mono_float,
        abs_threshold=cfg.clipping_abs_threshold,
        fraction_threshold=cfg.clipping_fraction_threshold,
    )
    processed = preprocess_audio(raw_samples, sample_rate, cfg)
    candidates = find_sync_candidates(processed, cfg)
    if not candidates:
        return _failure_result(
            failure_code=FailureCode.SYNC_NOT_FOUND,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=clipping_warning,
            sync_found=False,
            start_sample=None,
            best_candidate_score=None,
            samples_per_symbol=cfg.samples_per_symbol,
        )

    best_candidate_score = candidates[0].match_score
    last_failure: DecodeResult | None = None
    for candidate in candidates:
        result = demodulate_from_sync(
            processed,
            candidate,
            cfg=cfg,
            clipping_warning=clipping_warning,
            best_candidate_score=best_candidate_score,
        )
        if result.success:
            return result
        last_failure = result

    return last_failure or _failure_result(
        failure_code=FailureCode.SYNC_NOT_FOUND,
        recovered_length=None,
        weak_symbol_count=0,
        clipping_warning=clipping_warning,
        sync_found=False,
        start_sample=None,
        best_candidate_score=best_candidate_score,
        samples_per_symbol=cfg.samples_per_symbol,
    )


def demodulate_from_sync(
    samples: np.ndarray,
    sync: SyncCandidate,
    cfg: ModemConfig = DEFAULT_CONFIG,
    clipping_warning: bool = False,
    best_candidate_score: float | None = None,
) -> DecodeResult:
    data_start = sync.start_sample + (cfg.tx_prefix_bit_count * sync.samples_per_symbol)
    length_bits_result = _demodulate_bits(samples, data_start, 8, sync.samples_per_symbol, cfg)
    if isinstance(length_bits_result, DecodeResult):
        return _copy_result(length_bits_result, sync, clipping_warning, best_candidate_score)

    length_bits, weak_length_bits = length_bits_result
    length_byte = bits_to_bytes(length_bits)[0]
    if not cfg.min_payload_length <= length_byte <= cfg.max_payload_length:
        return _failure_result(
            failure_code=FailureCode.INVALID_LENGTH,
            recovered_length=length_byte,
            weak_symbol_count=weak_length_bits,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=sync.start_sample,
            best_candidate_score=best_candidate_score,
            samples_per_symbol=sync.samples_per_symbol,
        )

    total_frame_bits = 8 + (8 * length_byte) + 16
    frame_bits_result = _demodulate_bits(samples, data_start, total_frame_bits, sync.samples_per_symbol, cfg)
    if isinstance(frame_bits_result, DecodeResult):
        return _copy_result(
            frame_bits_result,
            sync,
            clipping_warning,
            best_candidate_score,
            recovered_length=length_byte,
        )

    frame_bits, weak_symbol_count = frame_bits_result
    frame_bytes = bits_to_bytes(frame_bits)

    try:
        parsed = parse_frame(frame_bytes, cfg)
    except FramingError as exc:
        return _failure_result(
            failure_code=exc.code,
            recovered_length=length_byte,
            weak_symbol_count=weak_symbol_count,
            clipping_warning=clipping_warning,
            sync_found=True,
            start_sample=sync.start_sample,
            best_candidate_score=best_candidate_score,
            samples_per_symbol=sync.samples_per_symbol,
        )

    return DecodeResult(
        decoded_text=parsed.payload.decode("ascii"),
        failure_code=None,
        recovered_length=parsed.length,
        crc_ok=True,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=sync.samples_per_symbol,
        clipping_warning=clipping_warning,
        sync_found=True,
        start_sample=sync.start_sample,
        best_candidate_score=best_candidate_score,
    )


def _decode_clean_synthetic(
    processed_samples: np.ndarray,
    clipping_warning: bool,
    cfg: ModemConfig,
) -> DecodeResult:
    candidate = SyncCandidate(
        start_sample=cfg.leading_silence_samples,
        samples_per_symbol=cfg.samples_per_symbol,
        match_score=1.0,
        coarse_region_start=cfg.leading_silence_samples,
        coarse_region_end=cfg.leading_silence_samples + (cfg.tx_prefix_bit_count * cfg.samples_per_symbol),
    )
    return demodulate_from_sync(
        processed_samples,
        candidate,
        cfg=cfg,
        clipping_warning=clipping_warning,
        best_candidate_score=candidate.match_score,
    )


def _demodulate_bits(
    samples: np.ndarray,
    data_start: int,
    bit_count: int,
    samples_per_symbol: int,
    cfg: ModemConfig,
) -> tuple[np.ndarray, int] | DecodeResult:
    if data_start < 0:
        return _failure_result(
            failure_code=FailureCode.TRUNCATED_FRAME,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=False,
            sync_found=True,
            start_sample=data_start,
            best_candidate_score=None,
            samples_per_symbol=samples_per_symbol,
        )

    total_data_samples = bit_count * samples_per_symbol
    data_end = data_start + total_data_samples
    if data_end > len(samples):
        return _failure_result(
            failure_code=FailureCode.TRUNCATED_FRAME,
            recovered_length=None,
            weak_symbol_count=0,
            clipping_warning=False,
            sync_found=True,
            start_sample=data_start,
            best_candidate_score=None,
            samples_per_symbol=samples_per_symbol,
        )

    window = hann_window(samples_per_symbol)
    bits = np.empty(bit_count, dtype=np.uint8)
    weak_symbol_count = 0
    for index in range(bit_count):
        symbol_start = data_start + (index * samples_per_symbol)
        symbol_end = symbol_start + samples_per_symbol
        symbol = samples[symbol_start:symbol_end]
        energy_0, energy_1 = tone_energies(symbol, cfg, window=window)
        bits[index] = 1 if energy_1 > energy_0 else 0
        confidence = abs(energy_1 - energy_0) / (energy_0 + energy_1 + 1e-12)
        if confidence < cfg.weak_symbol_confidence_threshold:
            weak_symbol_count += 1

    return bits, weak_symbol_count


def _copy_result(
    result: DecodeResult,
    sync: SyncCandidate,
    clipping_warning: bool,
    best_candidate_score: float | None,
    recovered_length: int | None = None,
) -> DecodeResult:
    return DecodeResult(
        decoded_text=result.decoded_text,
        failure_code=result.failure_code,
        recovered_length=recovered_length if recovered_length is not None else result.recovered_length,
        crc_ok=result.crc_ok,
        weak_symbol_count=result.weak_symbol_count,
        samples_per_symbol=sync.samples_per_symbol,
        clipping_warning=clipping_warning,
        sync_found=True,
        start_sample=sync.start_sample,
        best_candidate_score=best_candidate_score,
    )


def _failure_result(
    failure_code: FailureCode,
    recovered_length: int | None,
    weak_symbol_count: int,
    clipping_warning: bool,
    sync_found: bool,
    start_sample: int | None,
    best_candidate_score: float | None,
    samples_per_symbol: int,
) -> DecodeResult:
    return DecodeResult(
        decoded_text=None,
        failure_code=failure_code,
        recovered_length=recovered_length,
        crc_ok=False,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=samples_per_symbol,
        clipping_warning=clipping_warning,
        sync_found=sync_found,
        start_sample=start_sample,
        best_candidate_score=best_candidate_score,
    )
