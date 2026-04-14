from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .audio import cached_hanning, cached_rfftfreq
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


def _stack_audio_segments(
    audio: np.ndarray,
    starts: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``audio[starts[i]:starts[i]+length]`` as zero-padded rows.

    Returns ``(matrix, lengths)`` where ``matrix`` has shape
    ``(n_starts, length)`` with each row containing the slice (clipped to
    audio bounds and zero-padded on the right) and ``lengths`` records the
    number of valid leading samples per row.
    """
    audio_len = len(audio)
    n = len(starts)
    matrix = np.zeros((n, length), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)
    if length <= 0 or n == 0:
        return matrix, lengths
    for idx, start in enumerate(starts):
        s = int(max(0, start))
        e = int(min(audio_len, start + length))
        if e > s:
            chunk = audio[s:e]
            matrix[idx, : chunk.size] = chunk
            lengths[idx] = chunk.size
    return matrix, lengths


def _stack_onset_windows(
    audio: np.ndarray,
    onset_samples: np.ndarray,
    *,
    pre_samples: int,
    post_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pre/post zero-padded matrices anchored at each onset sample."""
    pre_matrix, pre_lengths = _stack_audio_segments(
        audio, onset_samples - pre_samples, pre_samples
    )
    post_matrix, post_lengths = _stack_audio_segments(
        audio, onset_samples, post_samples
    )
    return pre_matrix, post_matrix, pre_lengths, post_lengths


def _row_wise_kurtosis(rows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Excess kurtosis (Fisher) per row over the leading ``lengths[i]`` samples."""
    n_rows = rows.shape[0]
    out = np.zeros(n_rows, dtype=np.float64)
    valid = lengths >= 4
    if not np.any(valid):
        return out
    counts = lengths.astype(np.float64)
    sums = rows.sum(axis=1)
    means = np.zeros(n_rows, dtype=np.float64)
    means[valid] = sums[valid] / counts[valid]
    centered = rows - means[:, None]
    # Mask padded tail samples to zero so they do not contribute to moments.
    mask = np.arange(rows.shape[1])[None, :] < lengths[:, None]
    centered = centered * mask
    sq = centered * centered
    var = np.zeros(n_rows, dtype=np.float64)
    var[valid] = sq[valid].sum(axis=1) / counts[valid]
    safe = valid & (var > 1e-20)
    if not np.any(safe):
        return out
    fourth = np.zeros(n_rows, dtype=np.float64)
    fourth[safe] = (sq[safe] * sq[safe]).sum(axis=1) / counts[safe]
    out[safe] = fourth[safe] / (var[safe] * var[safe]) - 3.0
    return out


def _row_wise_crest(rows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Peak / RMS per row over the leading ``lengths[i]`` samples."""
    n_rows = rows.shape[0]
    out = np.zeros(n_rows, dtype=np.float64)
    valid = lengths > 0
    if not np.any(valid):
        return out
    counts = lengths.astype(np.float64)
    sq_sum = (rows * rows).sum(axis=1)
    rms = np.zeros(n_rows, dtype=np.float64)
    rms[valid] = np.sqrt(sq_sum[valid] / counts[valid])
    safe = valid & (rms > 1e-20)
    if not np.any(safe):
        return out
    peaks = np.abs(rows).max(axis=1)
    out[safe] = peaks[safe] / rms[safe]
    return out


def _row_wise_autocorrelation(
    rows: np.ndarray,
    lengths: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> np.ndarray:
    """Normalized autocorrelation peak in ``[min_lag, max_lag)`` per row.

    Mirrors :func:`_normalized_autocorrelation` but processes all rows in a
    single batched FFT.
    """
    n_rows, width = rows.shape
    out = np.zeros(n_rows, dtype=np.float64)
    if width == 0 or n_rows == 0:
        return out
    valid = lengths > 0
    if not np.any(valid):
        return out
    counts = lengths.astype(np.float64)
    sums = rows.sum(axis=1)
    means = np.zeros(n_rows, dtype=np.float64)
    means[valid] = sums[valid] / counts[valid]
    mask = np.arange(width)[None, :] < lengths[:, None]
    centered = (rows - means[:, None]) * mask
    energy = (centered * centered).sum(axis=1)
    safe = valid & (energy > 1e-12)
    if not np.any(safe):
        return out
    # Per-row max valid lag is min(max_lag, lengths[i]); skip rows where
    # min_lag is already past the limit.
    max_valid_lag = np.minimum(max_lag, lengths.astype(np.int64))
    safe &= max_valid_lag > min_lag
    if not np.any(safe):
        return out
    # Batched FFT autocorrelation: corr = IFFT(|FFT(x)|^2).
    n_fft = 1 << int(math.ceil(math.log2(2 * width - 1))) if width > 1 else 2
    spec = np.fft.rfft(centered[safe], n=n_fft, axis=1)
    power = (spec.conj() * spec).real
    corr = np.fft.irfft(power, n=n_fft, axis=1)
    # Lag k corresponds to index k in the IFFT output (for k < width).
    upper = int(np.max(max_valid_lag[safe]))
    if upper <= min_lag:
        return out
    lag_slice = corr[:, min_lag:upper]
    # Mask out lag bins that exceed each row's individual max_valid_lag.
    lag_indices = np.arange(min_lag, upper)
    safe_indices = np.where(safe)[0]
    row_max = max_valid_lag[safe_indices][:, None]
    valid_lag_mask = lag_indices[None, :] < row_max
    masked = np.where(valid_lag_mask, lag_slice, -np.inf)
    peaks = masked.max(axis=1)
    out[safe_indices] = peaks / energy[safe_indices]
    return out


def _row_wise_diff_spectral_centroid(
    pre_rows: np.ndarray,
    post_rows: np.ndarray,
    pre_lengths: np.ndarray,
    post_lengths: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Vectorised :func:`_positive_diff_spectral_centroid` over batched onsets."""
    n_rows = pre_rows.shape[0]
    out = np.zeros(n_rows, dtype=np.float64)
    valid = (pre_lengths >= 256) & (post_lengths >= 256)
    if not np.any(valid):
        return out
    segment_len = max(int(pre_rows.shape[1]), int(post_rows.shape[1]))
    bounded = max(segment_len, ONSET_WAVEFORM_STATS_MIN_FFT)
    n_fft = min(ONSET_WAVEFORM_STATS_MAX_FFT, 1 << math.ceil(math.log2(bounded)))
    pre_spec = np.abs(np.fft.rfft(pre_rows[valid], n=n_fft, axis=1)) ** 2
    post_spec = np.abs(np.fft.rfft(post_rows[valid], n=n_fft, axis=1)) ** 2
    diff_spec = np.maximum(post_spec - pre_spec, 0.0)
    diff_energy = diff_spec.sum(axis=1)
    freqs = _rfft_frequency_bins(sample_rate, n_fft)
    weighted = (freqs[None, :] * diff_spec).sum(axis=1)
    safe = diff_energy >= 1e-10
    centroid = np.zeros(diff_spec.shape[0], dtype=np.float64)
    centroid[safe] = weighted[safe] / diff_energy[safe]
    out[valid] = centroid
    return out


def precompute_onset_waveform_stats(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[float, OnsetWaveformStats]:
    """Pre-compute simple waveform/spectral diagnostics for each onset."""
    stats: dict[float, OnsetWaveformStats] = {}
    if not onset_times:
        return stats
    window_samples = int(sample_rate * _KURTOSIS_WINDOW_SECONDS)
    sustain_onset_samples = int(sample_rate * 0.04)
    sustain_check_offset = int(sample_rate * 0.10)
    sustain_check_samples = int(sample_rate * 0.04)
    min_lag = int(sample_rate / 2000)
    max_lag = int(sample_rate / 150)

    onset_samples = np.maximum(
        np.asarray(onset_times, dtype=np.float64) * sample_rate, 0
    ).astype(np.int64)
    audio_f64 = np.asarray(audio, dtype=np.float64)

    pre_seg, seg, pre_len, seg_len = _stack_onset_windows(
        audio_f64,
        onset_samples,
        pre_samples=window_samples,
        post_samples=window_samples,
    )
    diff_centroids = _row_wise_diff_spectral_centroid(
        pre_seg, seg, pre_len, seg_len, sample_rate
    )
    kurtoses = _row_wise_kurtosis(seg, seg_len)
    crests = _row_wise_crest(seg, seg_len)
    autocorrs = _row_wise_autocorrelation(seg, seg_len, min_lag, max_lag)

    onset_chunks, onset_chunk_lens = _stack_audio_segments(
        audio_f64, onset_samples, sustain_onset_samples
    )
    sustain_chunks, sustain_chunk_lens = _stack_audio_segments(
        audio_f64, onset_samples + sustain_check_offset, sustain_check_samples
    )
    full_pair = (onset_chunk_lens >= sustain_onset_samples) & (
        sustain_chunk_lens >= sustain_check_samples
    )
    onset_rms = np.sqrt((onset_chunks * onset_chunks).sum(axis=1) / sustain_onset_samples)
    sustain_rms = np.sqrt(
        (sustain_chunks * sustain_chunks).sum(axis=1) / sustain_check_samples
    )
    post_sustain_ratio = np.ones(len(onset_times), dtype=np.float64)
    safe = full_pair & (onset_rms > 1e-10)
    post_sustain_ratio[full_pair & ~safe] = 0.0
    post_sustain_ratio[safe] = sustain_rms[safe] / onset_rms[safe]

    for idx, onset_time in enumerate(onset_times):
        stats[round(onset_time, 4)] = OnsetWaveformStats(
            kurtosis=float(kurtoses[idx]),
            crest=float(crests[idx]),
            post_autocorr_20ms=float(autocorrs[idx]),
            diff_centroid=float(diff_centroids[idx]),
            post_sustain_ratio=float(post_sustain_ratio[idx]),
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
    window = cached_hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window, n=n_fft))
    frequencies = cached_rfftfreq(n_fft, sample_rate)
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
    return cached_rfftfreq(n_fft, sample_rate)


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
    if not onset_times:
        return profiles

    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    audio_f64 = np.asarray(audio, dtype=np.float64)
    onset_samples = np.maximum(
        np.asarray(onset_times, dtype=np.float64) * sample_rate, 0
    ).astype(np.int64)

    pre_rows, attack_rows, pre_lens, attack_lens = _stack_onset_windows(
        audio_f64,
        onset_samples,
        pre_samples=window_samples,
        post_samples=window_samples,
    )
    valid = (pre_lens >= 512) & (attack_lens >= 512)
    if not np.any(valid):
        return profiles

    # The single-onset version Hanning-windows each chunk at its own length,
    # so partial-window rows must be handled individually to keep numerics
    # bit-identical at audio boundaries. Full-length rows go through the
    # batched FFT path.
    full_mask = (pre_lens == window_samples) & (attack_lens == window_samples) & valid
    partial_mask = valid & ~full_mask

    if np.any(full_mask):
        n_fft = max(4096, 1 << int(math.ceil(math.log2(window_samples))))
        window = cached_hanning(window_samples)
        pre_full = pre_rows[full_mask]
        attack_full = attack_rows[full_mask]
        pre_spec = np.abs(np.fft.rfft(pre_full * window[None, :], n=n_fft, axis=1))
        attack_spec = np.abs(
            np.fft.rfft(attack_full * window[None, :], n=n_fft, axis=1)
        )

        pre_energy = (pre_full * pre_full).mean(axis=1)
        attack_energy = (attack_full * attack_full).mean(axis=1)
        broadband_gain = (attack_energy + 1e-6) / (pre_energy + 1e-6)

        frequencies = cached_rfftfreq(n_fft, sample_rate)
        pre_sum = pre_spec.sum(axis=1)
        delta = np.maximum(attack_spec - pre_spec, 0.0)
        broadband_flux = delta.sum(axis=1) / (pre_sum + 1e-6)

        high_band_mask = frequencies >= SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY
        if np.any(high_band_mask):
            pre_high_sum = pre_spec[:, high_band_mask].sum(axis=1)
            high_band_flux = delta[:, high_band_mask].sum(axis=1) / (pre_high_sum + 1e-6)
        else:
            high_band_flux = np.zeros_like(broadband_flux)

        is_valid_attack = (
            (high_band_flux >= ONSET_ATTACK_MIN_HIGH_BAND_FLUX)
            | (
                (broadband_gain >= ONSET_ATTACK_MIN_BROADBAND_GAIN)
                & (high_band_flux >= ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX)
            )
            | (
                (broadband_gain >= ONSET_ATTACK_MODERATE_GAIN)
                & (high_band_flux >= ONSET_ATTACK_MODERATE_GAIN_MIN_FLUX)
            )
        )

        for row_idx, onset_idx in enumerate(np.where(full_mask)[0]):
            onset_time = onset_times[onset_idx]
            profiles[round(onset_time, 4)] = OnsetAttackProfile(
                onset_time=onset_time,
                broadband_onset_gain=float(broadband_gain[row_idx]),
                high_band_spectral_flux=float(high_band_flux[row_idx]),
                broadband_spectral_flux=float(broadband_flux[row_idx]),
                is_valid_attack=bool(is_valid_attack[row_idx]),
            )

    # Partial-window rows fall back to the single-onset implementation.
    if np.any(partial_mask):
        for onset_idx in np.where(partial_mask)[0]:
            onset_time = onset_times[onset_idx]
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
