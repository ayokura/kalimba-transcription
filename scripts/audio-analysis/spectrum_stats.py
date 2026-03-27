#!/usr/bin/env python3
"""
Compute spectral statistics at a specific onset time.
Used by /audio-spectrum skill.

Usage: uv run python spectrum_stats.py <audio_file> <onset_time> [window_ms]
"""
import sys
from pathlib import Path

import numpy as np
import librosa


def compute_spectral_stats(audio_path: str, onset_time: float, window_ms: float = 15) -> dict:
    """Compute spectral statistics at onset time."""
    audio, sr = librosa.load(audio_path, sr=None)

    onset_sample = int(sr * onset_time)
    window_samples = int(sr * window_ms / 1000)

    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]

    if len(segment) < 256:
        return {"error": "Segment too short"}

    n_fft = 2048
    spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    total_energy = np.sum(spectrum ** 2)
    if total_energy == 0:
        return {"error": "No energy in segment"}

    norm_spectrum = spectrum ** 2 / total_energy

    # Centroid
    centroid = np.sum(freqs * norm_spectrum) / np.sum(norm_spectrum)

    # BW90 (90% energy bandwidth)
    cumsum = np.cumsum(norm_spectrum)
    idx_05 = np.searchsorted(cumsum, 0.05)
    idx_95 = np.searchsorted(cumsum, 0.95)
    bandwidth_90 = freqs[idx_95] - freqs[idx_05]

    # Spread
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * norm_spectrum))

    # HF ratios
    hf_mask = freqs >= 2000
    hf_ratio = np.sum(spectrum[hf_mask] ** 2) / total_energy

    vhf_mask = freqs >= 8000
    vhf_ratio = np.sum(spectrum[vhf_mask] ** 2) / total_energy

    # Peak frequency
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    # Classification
    if bandwidth_90 > 6000 or centroid > 3000:
        classification = "NOISE"
    elif bandwidth_90 < 2000 and centroid < 1000:
        classification = "KALIMBA"
    else:
        classification = "UNCLEAR"

    return {
        "onset_time": onset_time,
        "window_ms": window_ms,
        "centroid_hz": round(centroid, 1),
        "bandwidth_90_hz": round(bandwidth_90, 1),
        "spread_hz": round(spread, 1),
        "hf_ratio_pct": round(hf_ratio * 100, 2),
        "vhf_ratio_pct": round(vhf_ratio * 100, 2),
        "peak_freq_hz": round(peak_freq, 1),
        "classification": classification,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: spectrum_stats.py <audio_file> <onset_time> [window_ms]")
        sys.exit(1)

    audio_path = sys.argv[1]
    onset_time = float(sys.argv[2])
    window_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 15

    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    result = compute_spectral_stats(audio_path, onset_time, window_ms)

    print("Spectral Statistics")
    print("=" * 40)
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Interpretation
    print()
    print("Interpretation")
    print("-" * 40)
    if result.get("classification") == "NOISE":
        print("  -> Likely NOISE (broadband, high centroid)")
    elif result.get("classification") == "KALIMBA":
        print("  -> Likely KALIMBA note (narrow band, low centroid)")
    else:
        print("  -> UNCLEAR - manual review recommended")


if __name__ == "__main__":
    main()
