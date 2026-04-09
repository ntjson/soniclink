from __future__ import annotations

from dataclasses import dataclass
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
from acoustic_modem.reporting import (
    decode_result_summary,
    failure_reason,
    is_frame_validation_failure,
    is_invalid_input_failure,
    write_debug_artifacts,
    write_summary_json,
)
from acoustic_modem.sync import find_sync_candidates
from acoustic_modem.types import DecodeResult, FailureCode, FramingError, SyncCandidate


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class _PreparedAudio:
    processed: np.ndarray
    clipping_warning: bool


@dataclass(frozen=True, slots=True)
class _DecodeContext:
    clipping_warning: bool
    sync_found: bool
    start_sample: int | None
    best_candidate_score: float | None
    samples_per_symbol: int


@dataclass(frozen=True, slots=True)
class _DemodulatedBits:
    bits: np.ndarray
    weak_symbol_count: int


@dataclass(frozen=True, slots=True)
class _DemodulationFailure:
    failure_code: FailureCode


def preprocess_audio(samples: np.ndarray, sample_rate: int, cfg: ModemConfig = DEFAULT_CONFIG) -> np.ndarray:
    mono_float = to_mono(pcm_to_float(samples))
    return _preprocess_mono_float(mono_float, sample_rate, cfg)


def decode_wav(path: Path, cfg: ModemConfig = DEFAULT_CONFIG) -> DecodeResult:
    try:
        raw_samples, sample_rate = load_wav(path)
    except (OSError, ValueError):
        return _failure_result(
            failure_code=FailureCode.UNSUPPORTED_FORMAT,
            recovered_length=None,
            weak_symbol_count=0,
            context=_default_context(cfg, clipping_warning=False, best_candidate_score=None),
        )

    prepared = _prepare_audio(raw_samples, sample_rate, cfg)
    candidates = find_sync_candidates(prepared.processed, cfg)
    if not candidates:
        return _failure_result(
            failure_code=FailureCode.SYNC_NOT_FOUND,
            recovered_length=None,
            weak_symbol_count=0,
            context=_default_context(cfg, clipping_warning=prepared.clipping_warning, best_candidate_score=None),
        )

    best_candidate_score = candidates[0].match_score
    last_failure: DecodeResult | None = None
    for candidate in candidates:
        result = demodulate_from_sync(
            prepared.processed,
            candidate,
            cfg=cfg,
            clipping_warning=prepared.clipping_warning,
            best_candidate_score=best_candidate_score,
        )
        if result.success:
            return result
        last_failure = result

    return last_failure or _failure_result(
        failure_code=FailureCode.SYNC_NOT_FOUND,
        recovered_length=None,
        weak_symbol_count=0,
        context=_default_context(
            cfg,
            clipping_warning=prepared.clipping_warning,
            best_candidate_score=best_candidate_score,
        ),
    )


def demodulate_from_sync(
    samples: np.ndarray,
    sync: SyncCandidate,
    cfg: ModemConfig = DEFAULT_CONFIG,
    clipping_warning: bool = False,
    best_candidate_score: float | None = None,
) -> DecodeResult:
    context = _sync_context(sync, clipping_warning, best_candidate_score)
    data_start = sync.start_sample + (cfg.tx_prefix_bit_count * sync.samples_per_symbol)

    length_bits_result = _demodulate_bits(samples, data_start, 8, sync.samples_per_symbol, cfg)
    if isinstance(length_bits_result, _DemodulationFailure):
        return _failure_result(
            failure_code=length_bits_result.failure_code,
            recovered_length=None,
            weak_symbol_count=0,
            context=context,
        )

    length_byte = bits_to_bytes(length_bits_result.bits)[0]
    if not cfg.min_payload_length <= length_byte <= cfg.max_payload_length:
        return _failure_result(
            failure_code=FailureCode.INVALID_LENGTH,
            recovered_length=length_byte,
            weak_symbol_count=length_bits_result.weak_symbol_count,
            context=context,
        )

    total_frame_bits = 8 + (8 * length_byte) + 16
    frame_bits_result = _demodulate_bits(samples, data_start, total_frame_bits, sync.samples_per_symbol, cfg)
    if isinstance(frame_bits_result, _DemodulationFailure):
        return _failure_result(
            failure_code=frame_bits_result.failure_code,
            recovered_length=length_byte,
            weak_symbol_count=0,
            context=context,
        )

    frame_bytes = bits_to_bytes(frame_bits_result.bits)

    try:
        parsed = parse_frame(frame_bytes, cfg)
    except FramingError as exc:
        return _failure_result(
            failure_code=exc.code,
            recovered_length=length_byte,
            weak_symbol_count=frame_bits_result.weak_symbol_count,
            context=context,
        )

    return _success_result(
        decoded_text=parsed.payload.decode("ascii"),
        recovered_length=parsed.length,
        weak_symbol_count=frame_bits_result.weak_symbol_count,
        context=context,
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
) -> _DemodulatedBits | _DemodulationFailure:
    if data_start < 0:
        return _DemodulationFailure(FailureCode.TRUNCATED_FRAME)

    total_data_samples = bit_count * samples_per_symbol
    data_end = data_start + total_data_samples
    if data_end > len(samples):
        return _DemodulationFailure(FailureCode.TRUNCATED_FRAME)

    window = hann_window(samples_per_symbol)
    bits = np.empty(bit_count, dtype=np.uint8)
    weak_symbol_count = 0
    for index in range(bit_count):
        symbol_start = data_start + (index * samples_per_symbol)
        symbol_end = symbol_start + samples_per_symbol
        symbol = samples[symbol_start:symbol_end]
        energy_0, energy_1 = tone_energies(symbol, cfg, window=window)
        bits[index] = 1 if energy_1 > energy_0 else 0
        confidence = abs(energy_1 - energy_0) / (energy_0 + energy_1 + _EPSILON)
        if confidence < cfg.weak_symbol_confidence_threshold:
            weak_symbol_count += 1

    return _DemodulatedBits(bits=bits, weak_symbol_count=weak_symbol_count)


def _prepare_audio(samples: np.ndarray, sample_rate: int, cfg: ModemConfig) -> _PreparedAudio:
    mono_float = to_mono(pcm_to_float(samples))
    clipping_warning = detect_likely_clipping(
        mono_float,
        abs_threshold=cfg.clipping_abs_threshold,
        fraction_threshold=cfg.clipping_fraction_threshold,
    )
    processed = _preprocess_mono_float(mono_float, sample_rate, cfg)
    return _PreparedAudio(processed=processed, clipping_warning=clipping_warning)


def _preprocess_mono_float(mono_float: np.ndarray, sample_rate: int, cfg: ModemConfig) -> np.ndarray:
    resampled = resample_audio(mono_float, sample_rate, cfg.sample_rate_hz)
    filtered = bandpass_filter(resampled, cfg.sample_rate_hz, cfg.bandpass_low_hz, cfg.bandpass_high_hz)
    return peak_normalize(filtered, cfg.preprocess_peak_target)


def _default_context(
    cfg: ModemConfig,
    *,
    clipping_warning: bool,
    best_candidate_score: float | None,
) -> _DecodeContext:
    return _DecodeContext(
        clipping_warning=clipping_warning,
        sync_found=False,
        start_sample=None,
        best_candidate_score=best_candidate_score,
        samples_per_symbol=cfg.samples_per_symbol,
    )


def _sync_context(
    sync: SyncCandidate,
    clipping_warning: bool,
    best_candidate_score: float | None,
) -> _DecodeContext:
    return _DecodeContext(
        clipping_warning=clipping_warning,
        sync_found=True,
        start_sample=sync.start_sample,
        best_candidate_score=best_candidate_score,
        samples_per_symbol=sync.samples_per_symbol,
    )


def _success_result(
    decoded_text: str,
    recovered_length: int,
    weak_symbol_count: int,
    context: _DecodeContext,
) -> DecodeResult:
    return DecodeResult(
        decoded_text=decoded_text,
        failure_code=None,
        recovered_length=recovered_length,
        crc_ok=True,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=context.samples_per_symbol,
        clipping_warning=context.clipping_warning,
        sync_found=context.sync_found,
        start_sample=context.start_sample,
        best_candidate_score=context.best_candidate_score,
    )


def _failure_result(
    failure_code: FailureCode,
    recovered_length: int | None,
    weak_symbol_count: int,
    context: _DecodeContext,
) -> DecodeResult:
    return DecodeResult(
        decoded_text=None,
        failure_code=failure_code,
        recovered_length=recovered_length,
        crc_ok=False,
        weak_symbol_count=weak_symbol_count,
        samples_per_symbol=context.samples_per_symbol,
        clipping_warning=context.clipping_warning,
        sync_found=context.sync_found,
        start_sample=context.start_sample,
        best_candidate_score=context.best_candidate_score,
    )
