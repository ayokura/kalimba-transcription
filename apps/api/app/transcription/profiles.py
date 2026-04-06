from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .constants import (
    ACTIVE_RANGE_START_CLUSTER_MAX_DURATION,
    ATTACK_REFINED_ONSET_MAX_INTERVAL,
    ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX,
    ONSET_ATTACK_MIN_BROADBAND_GAIN,
    ONSET_ATTACK_MIN_HIGH_BAND_FLUX,
    ONSET_ATTACK_MODERATE_GAIN,
    ONSET_ATTACK_MODERATE_GAIN_MIN_FLUX,
    ONSET_ENERGY_WINDOW_SECONDS,
    SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY,
)
from .models import OnsetAttackProfile, OnsetWaveformStats


LEADING_GAP_START_MARGIN = 0.05
GAP_ONSET_MIN_BROADBAND_GAIN = 0.95
GAP_ONSET_MAX_KURTOSIS = 2.5
GAP_ONSET_KURTOSIS_OVERRIDE_MIN_GAIN = 100.0
GAP_ONSET_MAX_POST_CREST = 0.0  # disabled; set to e.g. 3.8 to enable
GAP_ONSET_MIN_POST_SUSTAIN_RATIO = 0.4
PRE_PERFORMANCE_GAP_REJECT_MAX_POST_AUTOCORR = 0.45
PRE_PERFORMANCE_GAP_REJECT_MIN_DIFF_CENTROID = 1000.0
ONSET_WAVEFORM_STATS_MIN_FFT = 512
ONSET_WAVEFORM_STATS_MAX_FFT = 4096
GAP_ONSET_REJECT_MAX_BROADBAND_GAIN = 2.0
GAP_ONSET_REJECT_MAX_HIGH_BAND_FLUX = 0.5
_KURTOSIS_WINDOW_SECONDS = 0.02


def _lookup_onset_attack_profile(
    onset_profiles: dict[float, OnsetAttackProfile] | None,
    onset_time: float,
) -> OnsetAttackProfile | None:
    if onset_profiles is None:
        return None
    return onset_profiles.get(round(onset_time, 4))


def precompute_onset_waveform_stats(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[float, OnsetWaveformStats]:
    """Pre-compute simple waveform/spectral diagnostics for each onset."""
    stats: dict[float, OnsetWaveformStats] = {}
    window_samples = int(sample_rate * _KURTOSIS_WINDOW_SECONDS)
    sustain_onset_samples = int(sample_rate * 0.04)
    sustain_check_offset = int(sample_rate * 0.10)
    sustain_check_samples = int(sample_rate * 0.04)
    min_lag = int(sample_rate / 2000)
    max_lag = int(sample_rate / 150)
    for onset_time in onset_times:
        onset_sample = max(int(onset_time * sample_rate), 0)
        pre_seg = np.array(audio[max(0, onset_sample - window_samples):onset_sample], copy=True)
        seg = np.array(audio[onset_sample:min(len(audio), onset_sample + window_samples)], copy=True)
        diff_centroid = 0.0
        if len(pre_seg) >= 256 and len(seg) >= 256:
            diff_centroid = _positive_diff_spectral_centroid(pre_seg, seg, sample_rate)
        onset_chunk = audio[onset_sample:onset_sample + sustain_onset_samples]
        sustain_start = onset_sample + sustain_check_offset
        sustain_chunk = audio[sustain_start:sustain_start + sustain_check_samples]
        if len(onset_chunk) >= sustain_onset_samples and len(sustain_chunk) >= sustain_check_samples:
            onset_rms = float(np.sqrt(np.mean(onset_chunk ** 2)))
            sustain_rms = float(np.sqrt(np.mean(sustain_chunk ** 2)))
            post_sustain_ratio = sustain_rms / onset_rms if onset_rms > 1e-10 else 0.0
        else:
            post_sustain_ratio = 1.0
        stats[round(onset_time, 4)] = OnsetWaveformStats(
            kurtosis=_waveform_kurtosis(seg),
            crest=_waveform_crest_factor(seg),
            post_autocorr_20ms=_normalized_autocorrelation(seg, min_lag, max_lag),
            diff_centroid=diff_centroid,
            post_sustain_ratio=post_sustain_ratio,
        )
    return stats


def estimate_pre_performance_start(active_ranges: list[tuple[float, float]]) -> float | None:
    if not active_ranges:
        return None

    first_start, first_end = active_ranges[0]
    if first_end - first_start >= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION:
        return first_start

    for range_start, range_end in active_ranges[1:]:
        if range_end - range_start >= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION:
            return range_start

    return first_start


def _build_analysis_window_chunks(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    if end_sample - start_sample < 512:
        return None

    attack_end = min(start_sample + window_samples, end_sample)
    pre_start = max(0, start_sample - window_samples)
    pre_chunk = audio[pre_start:start_sample]
    attack_chunk = audio[start_sample:attack_end]
    sustain_start = max(start_sample, end_sample - window_samples)
    sustain_chunk = audio[sustain_start:end_sample]
    if len(pre_chunk) < 512 or len(attack_chunk) < 512 or len(sustain_chunk) < 512:
        return None
    return pre_chunk, attack_chunk, sustain_chunk


def _chunk_spectrum(chunk: np.ndarray, sample_rate: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return frequencies, spectrum


def _broadband_chunk_energy(chunk: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(chunk, dtype=np.float64))))


def _positive_spectral_flux(
    reference_spectrum: np.ndarray,
    target_spectrum: np.ndarray,
    frequencies: np.ndarray,
    *,
    min_frequency: float = 0.0,
) -> float:
    if min_frequency > 0.0:
        mask = frequencies >= min_frequency
        if not np.any(mask):
            return 0.0
        reference = reference_spectrum[mask]
        target = target_spectrum[mask]
    else:
        reference = reference_spectrum
        target = target_spectrum
    positive_delta = np.maximum(target - reference, 0.0)
    return float(np.sum(positive_delta) / (np.sum(reference) + 1e-6))


def _waveform_kurtosis(signal: np.ndarray) -> float:
    """Excess kurtosis (Fisher). Normal distribution = 0, periodic ≈ -1.5."""
    if len(signal) < 4:
        return 0.0
    var = float(np.var(signal))
    if var < 1e-20:
        return 0.0
    mean = float(np.mean(signal))
    return float(np.mean((signal - mean) ** 4) / (var * var)) - 3.0


def _waveform_crest_factor(signal: np.ndarray) -> float:
    """Peak / RMS. Sine wave ≈ 1.41, impulsive spike >> 1."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-20:
        return 0.0
    return float(np.max(np.abs(signal))) / rms


def _normalized_autocorrelation(
    signal: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> float:
    centered = np.asarray(signal, dtype=np.float64)
    if centered.size == 0:
        return 0.0
    centered = centered - float(np.mean(centered))
    energy = float(np.sum(centered * centered))
    if energy < 1e-12:
        return 0.0
    max_valid_lag = min(max_lag, len(centered))
    if min_lag >= max_valid_lag:
        return 0.0

    full_corr = np.correlate(centered, centered, mode="full")
    zero_lag_index = len(centered) - 1
    lag_corr = full_corr[zero_lag_index + min_lag:zero_lag_index + max_valid_lag]
    if lag_corr.size == 0:
        return 0.0
    return float(np.max(lag_corr / energy))


def _waveform_stats_n_fft(pre_signal: np.ndarray, post_signal: np.ndarray) -> int:
    segment_len = max(len(pre_signal), len(post_signal))
    bounded = max(segment_len, ONSET_WAVEFORM_STATS_MIN_FFT)
    return min(ONSET_WAVEFORM_STATS_MAX_FFT, 1 << math.ceil(math.log2(bounded)))


@lru_cache(maxsize=8)
def _rfft_frequency_bins(sample_rate: int, n_fft: int) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, 1 / sample_rate)


def _positive_diff_spectral_centroid(
    pre_signal: np.ndarray,
    post_signal: np.ndarray,
    sample_rate: int,
) -> float:
    if len(pre_signal) < 256 or len(post_signal) < 256:
        return 0.0
    n_fft = _waveform_stats_n_fft(pre_signal, post_signal)
    pre_spec = np.abs(np.fft.rfft(pre_signal, n=n_fft)) ** 2
    post_spec = np.abs(np.fft.rfft(post_signal, n=n_fft)) ** 2
    diff_spec = np.maximum(post_spec - pre_spec, 0.0)
    diff_energy = float(np.sum(diff_spec))
    if diff_energy < 1e-10:
        return 0.0
    freqs = _rfft_frequency_bins(sample_rate, n_fft)
    return float(np.sum(freqs * diff_spec)) / diff_energy


def compute_onset_attack_profile(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    *,
    window_seconds: float = ONSET_ENERGY_WINDOW_SECONDS,
) -> OnsetAttackProfile | None:
    window_samples = max(int(sample_rate * window_seconds), 512)
    onset_sample = max(int(onset_time * sample_rate), 0)
    pre_start = max(0, onset_sample - window_samples)
    pre_chunk = audio[pre_start:onset_sample]
    attack_end = min(onset_sample + window_samples, len(audio))
    attack_chunk = audio[onset_sample:attack_end]
    if len(pre_chunk) < 512 or len(attack_chunk) < 512:
        return None

    n_fft = max(4096, 1 << int(np.ceil(np.log2(max(len(pre_chunk), len(attack_chunk))))))
    frequencies, pre_spectrum = _chunk_spectrum(pre_chunk, sample_rate, n_fft)
    _, attack_spectrum = _chunk_spectrum(attack_chunk, sample_rate, n_fft)

    pre_energy = _broadband_chunk_energy(pre_chunk)
    attack_energy = _broadband_chunk_energy(attack_chunk)
    broadband_gain = (attack_energy + 1e-6) / (pre_energy + 1e-6)
    broadband_flux = _positive_spectral_flux(pre_spectrum, attack_spectrum, frequencies)
    high_band_flux = _positive_spectral_flux(
        pre_spectrum,
        attack_spectrum,
        frequencies,
        min_frequency=SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY,
    )
    is_valid = high_band_flux >= ONSET_ATTACK_MIN_HIGH_BAND_FLUX or (
        broadband_gain >= ONSET_ATTACK_MIN_BROADBAND_GAIN
        and high_band_flux >= ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX
    ) or (
        broadband_gain >= ONSET_ATTACK_MODERATE_GAIN
        and high_band_flux >= ONSET_ATTACK_MODERATE_GAIN_MIN_FLUX
    )

    return OnsetAttackProfile(
        onset_time=onset_time,
        broadband_onset_gain=broadband_gain,
        high_band_spectral_flux=high_band_flux,
        broadband_spectral_flux=broadband_flux,
        is_valid_attack=is_valid,
    )


def precompute_onset_attack_profiles(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[float, OnsetAttackProfile]:
    profiles: dict[float, OnsetAttackProfile] = {}
    for onset_time in onset_times:
        profile = compute_onset_attack_profile(audio, sample_rate, onset_time)
        if profile is not None:
            profiles[round(onset_time, 4)] = profile
    return profiles


def refine_onset_times_by_attack_profile(
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
) -> list[float]:
    if not onset_times:
        return onset_times

    refined: list[float] = []
    for onset_time in onset_times:
        if not refined:
            refined.append(onset_time)
            continue

        previous_time = refined[-1]
        if onset_time - previous_time >= ATTACK_REFINED_ONSET_MAX_INTERVAL:
            refined.append(onset_time)
            continue

        previous_profile = onset_profiles.get(round(previous_time, 4))
        current_profile = onset_profiles.get(round(onset_time, 4))
        if current_profile is None:
            continue

        should_replace_previous = previous_profile is None
        if not should_replace_previous and not previous_profile.is_valid_attack and current_profile.is_valid_attack:
            should_replace_previous = True

        if should_replace_previous:
            refined[-1] = onset_time
            continue

        refined.append(onset_time)

    return refined


def filter_gap_onsets_by_attack(
    onset_times: list[float],
    active_ranges: list[tuple[float, float]],
    onset_profiles: dict[float, OnsetAttackProfile],
    waveform_stats: dict[float, OnsetWaveformStats] | None = None,
) -> list[float]:
    """Return onset_times with obvious-noise gap onsets removed."""
    if not active_ranges:
        return onset_times

    pre_performance_start = estimate_pre_performance_start(active_ranges)

    def _in_active_range(time: float) -> bool:
        for range_start, range_end in active_ranges:
            if range_start - 0.05 <= time <= range_end + 0.05:
                return True
        return False

    filtered: list[float] = []
    for time in onset_times:
        if _in_active_range(time):
            filtered.append(time)
            continue
        profile = onset_profiles.get(round(time, 4))
        if profile is None:
            filtered.append(time)
            continue
        if waveform_stats is not None and pre_performance_start is not None and time < pre_performance_start:
            stats = waveform_stats.get(round(time, 4))
            if (
                stats is not None
                and stats.post_autocorr_20ms < PRE_PERFORMANCE_GAP_REJECT_MAX_POST_AUTOCORR
                and stats.diff_centroid > PRE_PERFORMANCE_GAP_REJECT_MIN_DIFF_CENTROID
            ):
                continue
        if (
            profile.broadband_onset_gain < GAP_ONSET_REJECT_MAX_BROADBAND_GAIN
            and profile.high_band_spectral_flux < GAP_ONSET_REJECT_MAX_HIGH_BAND_FLUX
        ):
            continue
        filtered.append(time)
    return filtered
