from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_modem.config import DEFAULT_CONFIG, ModemConfig
from acoustic_modem.dsp import hann_window
from acoustic_modem.types import SyncCandidate


_FINE_OFFSET_STEP_SAMPLES = 60
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class _LeaderRegion:
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class _ToneProjectionCache:
    window: np.ndarray
    f0_cos: np.ndarray
    f0_sin: np.ndarray
    f1_cos: np.ndarray
    f1_sin: np.ndarray


@dataclass(frozen=True, slots=True)
class _KnownPrefixPattern:
    bits: np.ndarray
    expected_signs: np.ndarray


def find_leader_regions(samples: np.ndarray, cfg: ModemConfig = DEFAULT_CONFIG) -> list[tuple[int, int]]:
    sample_array = np.asarray(samples, dtype=np.float64)
    window_size = cfg.samples_per_symbol
    hop_size = cfg.sync_hop_samples

    if sample_array.size < window_size:
        return []

    window = hann_window(window_size)
    cache = _build_projection_cache(window_size, cfg, window)
    window_starts = np.arange(0, sample_array.size - window_size + 1, hop_size, dtype=np.int64)

    total_energies = np.empty(window_starts.size, dtype=np.float64)
    scores = np.empty(window_starts.size, dtype=np.float64)
    for index, start in enumerate(window_starts):
        scores[index], total_energies[index] = _symbol_score(sample_array[start : start + window_size], cache)

    noise_floor = _estimate_noise_floor(total_energies, cfg)
    qualifying = (scores > cfg.sync_leader_score_threshold) & (
        total_energies > (cfg.sync_leader_energy_threshold_multiplier * noise_floor)
    )

    regions: list[tuple[int, int]] = []
    region_start_index: int | None = None
    for index, is_qualified in enumerate(qualifying):
        if is_qualified and region_start_index is None:
            region_start_index = index
        elif not is_qualified and region_start_index is not None:
            region = _build_region(window_starts, region_start_index, index - 1, window_size)
            if (region.end_sample - region.start_sample) >= cfg.sync_leader_min_duration_samples:
                regions.append((region.start_sample, region.end_sample))
            region_start_index = None

    if region_start_index is not None:
        region = _build_region(window_starts, region_start_index, len(qualifying) - 1, window_size)
        if (region.end_sample - region.start_sample) >= cfg.sync_leader_min_duration_samples:
            regions.append((region.start_sample, region.end_sample))

    return regions


def score_sync_candidate(
    samples: np.ndarray,
    start_sample: int,
    samples_per_symbol: int,
    cfg: ModemConfig = DEFAULT_CONFIG,
) -> float:
    sample_array = np.asarray(samples, dtype=np.float64)
    pattern = _known_prefix_pattern(cfg)
    cache = _build_projection_cache(samples_per_symbol, cfg)
    return _score_sync_candidate(sample_array, start_sample, samples_per_symbol, pattern, cache)


def find_sync_candidates(samples: np.ndarray, cfg: ModemConfig = DEFAULT_CONFIG) -> list[SyncCandidate]:
    sample_array = np.asarray(samples, dtype=np.float64)
    accepted: dict[tuple[int, int], SyncCandidate] = {}
    pattern = _known_prefix_pattern(cfg)
    samples_per_symbol_values = list(
        range(
            cfg.sync_min_samples_per_symbol,
            cfg.sync_max_samples_per_symbol + 1,
            cfg.sync_samples_per_symbol_step,
        )
    )
    projection_caches = {
        samples_per_symbol: _build_projection_cache(samples_per_symbol, cfg)
        for samples_per_symbol in samples_per_symbol_values
    }

    for region_start, region_end in find_leader_regions(sample_array, cfg):
        for samples_per_symbol in samples_per_symbol_values:
            cache = projection_caches[samples_per_symbol]
            for offset in range(-cfg.samples_per_symbol, cfg.samples_per_symbol + 1, _FINE_OFFSET_STEP_SAMPLES):
                candidate_start = region_start + offset
                match_score = _score_sync_candidate(sample_array, candidate_start, samples_per_symbol, pattern, cache)
                if match_score < cfg.sync_match_threshold:
                    continue

                candidate = SyncCandidate(
                    start_sample=candidate_start,
                    samples_per_symbol=samples_per_symbol,
                    match_score=match_score,
                    coarse_region_start=region_start,
                    coarse_region_end=region_end,
                )
                key = (candidate_start, samples_per_symbol)
                current = accepted.get(key)
                if current is None or candidate.match_score > current.match_score:
                    accepted[key] = candidate

    sorted_candidates = sorted(accepted.values(), key=lambda candidate: candidate.match_score, reverse=True)
    return sorted_candidates[: cfg.sync_candidate_limit]


def _known_prefix_pattern(cfg: ModemConfig) -> _KnownPrefixPattern:
    bits = np.fromiter((1 if bit == "1" else 0 for bit in cfg.tx_prefix_bits), dtype=np.uint8)
    return _KnownPrefixPattern(bits=bits, expected_signs=np.where(bits == 1, 1.0, -1.0))


def _estimate_noise_floor(total_energies: np.ndarray, cfg: ModemConfig) -> float:
    baseline_cap = float(np.percentile(total_energies, cfg.sync_noise_floor_percentile))
    baseline = total_energies[total_energies <= baseline_cap]
    if baseline.size == 0:
        baseline = total_energies
    return float(np.percentile(baseline, cfg.sync_noise_floor_percentile))


def _build_region(window_starts: np.ndarray, start_index: int, end_index: int, window_size: int) -> _LeaderRegion:
    start_sample = int(window_starts[start_index])
    end_sample = int(window_starts[end_index] + window_size)
    return _LeaderRegion(start_sample=start_sample, end_sample=end_sample)


def _build_projection_cache(
    samples_per_symbol: int,
    cfg: ModemConfig,
    window: np.ndarray | None = None,
) -> _ToneProjectionCache:
    sample_offsets = np.arange(samples_per_symbol, dtype=np.float64)
    window_array = window if window is not None else hann_window(samples_per_symbol)
    omega_0 = (2.0 * np.pi * cfg.f0_hz) / cfg.sample_rate_hz
    omega_1 = (2.0 * np.pi * cfg.f1_hz) / cfg.sample_rate_hz

    return _ToneProjectionCache(
        window=window_array,
        f0_cos=window_array * np.cos(omega_0 * sample_offsets),
        f0_sin=window_array * np.sin(omega_0 * sample_offsets),
        f1_cos=window_array * np.cos(omega_1 * sample_offsets),
        f1_sin=window_array * np.sin(omega_1 * sample_offsets),
    )


def _score_sync_candidate(
    samples: np.ndarray,
    start_sample: int,
    samples_per_symbol: int,
    pattern: _KnownPrefixPattern,
    cache: _ToneProjectionCache,
) -> float:
    if start_sample < 0 or samples_per_symbol <= 0:
        return float("-inf")

    total_known_samples = pattern.bits.size * samples_per_symbol
    end_sample = start_sample + total_known_samples
    if end_sample > samples.size:
        return float("-inf")

    symbol_scores = np.empty(pattern.bits.size, dtype=np.float64)
    for index in range(pattern.bits.size):
        symbol_start = start_sample + (index * samples_per_symbol)
        symbol_end = symbol_start + samples_per_symbol
        symbol_scores[index], _ = _symbol_score(samples[symbol_start:symbol_end], cache)

    return float(np.mean(pattern.expected_signs * symbol_scores))


def _symbol_score(samples: np.ndarray, cache: _ToneProjectionCache) -> tuple[float, float]:
    symbol = np.asarray(samples, dtype=np.float64)
    if symbol.shape != cache.window.shape:
        raise ValueError("symbol length does not match the projection cache")

    energy_0 = _projection_energy(symbol, cache.f0_cos, cache.f0_sin)
    energy_1 = _projection_energy(symbol, cache.f1_cos, cache.f1_sin)
    total_energy = energy_0 + energy_1
    score = (energy_1 - energy_0) / (total_energy + _EPSILON)
    return score, total_energy


def _projection_energy(symbol: np.ndarray, basis_cos: np.ndarray, basis_sin: np.ndarray) -> float:
    in_phase = float(np.dot(symbol, basis_cos))
    quadrature = float(np.dot(symbol, basis_sin))
    return (in_phase * in_phase) + (quadrature * quadrature)
