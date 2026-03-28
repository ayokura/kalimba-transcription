#!/usr/bin/env python3
"""
Onset feature separation analysis.

Compare groups of onset samples across many waveform/spectral features
and report which features best separate the groups (Cohen's d, overlap analysis).

Usage:
  # JSON config (multi-group, comprehensive)
  uv run python scripts/audio-analysis/onset_separation_analysis.py --config samples.json

  # Quick CLI (two-group comparison on a single audio file)
  uv run python scripts/audio-analysis/onset_separation_analysis.py \\
    --audio path/to/audio.wav --real 1.87,3.15,5.06 --compare 4.16,16.97

  # Using fixture names (auto-resolved)
  uv run python scripts/audio-analysis/onset_separation_analysis.py \\
    --audio bwv147-restart-prefix-01 --real 1.87,3.15,5.06 --compare 4.16

JSON config format:
  {
    "groups": {
      "real": [
        {"audio": "bwv147-restart-prefix-01", "onset": 1.87, "label": "E5"},
        {"audio": "/full/path/to/audio.wav", "onset": 3.15, "label": "C5"}
      ],
      "noise_trailing": [
        {"audio": "a4-d4-f4-triad-repeat-01", "onset": 16.97, "label": "triad_noise"}
      ]
    },
    "reference_group": "real"
  }
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Fixture resolution
# ---------------------------------------------------------------------------
_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "apps/api/tests/fixtures/manual-captures"


def resolve_audio_path(name_or_path: str) -> str:
    """Resolve fixture name to full audio path, or return path as-is."""
    p = Path(name_or_path)
    if p.exists():
        return str(p)
    # Try as fixture name (with or without kalimba-17-c- prefix)
    for prefix in ["", "kalimba-17-c-"]:
        candidate = _FIXTURES_DIR / f"{prefix}{name_or_path}" / "audio.wav"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot resolve audio: {name_or_path}")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _bandpass(audio: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    n = len(audio)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    fft[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(fft, n=n)


def _normalized_autocorrelation(signal: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, int]:
    n = len(signal)
    if n < max_lag + 1:
        return 0.0, 0
    signal = signal - np.mean(signal)
    energy = float(np.sum(signal ** 2))
    if energy < 1e-20:
        return 0.0, 0
    best_corr, best_lag = 0.0, 0
    for lag in range(min_lag, min(max_lag, n)):
        corr = float(np.sum(signal[: n - lag] * signal[lag:])) / energy
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_corr, best_lag


def compute_onset_features(
    audio: np.ndarray,
    sr: int,
    onset_time: float,
) -> dict[str, float]:
    """Compute comprehensive waveform and spectral features at an onset time."""
    onset_sample = int(sr * onset_time)
    eps = 1e-10
    results: dict[str, float] = {}

    # --- RMS / broadband gain at multiple windows ---
    for win_ms in [20, 80]:
        samples = int(sr * win_ms / 1000)
        pre_seg = audio[max(0, onset_sample - samples) : onset_sample]
        post_seg = audio[onset_sample : min(len(audio), onset_sample + samples)]
        pre_rms = float(np.sqrt(np.mean(pre_seg**2))) if len(pre_seg) > 0 else 0.0
        post_rms = float(np.sqrt(np.mean(post_seg**2))) if len(post_seg) > 0 else 0.0
        results[f"rms_ratio_{win_ms}ms"] = post_rms / (pre_rms + eps)
        results[f"broadband_gain_{win_ms}ms"] = (
            (float(np.sum(post_seg**2)) + eps) / (float(np.sum(pre_seg**2)) + eps)
            if len(pre_seg) > 0
            else 0.0
        )

    # --- Spectral structure (20ms windows, FFT) ---
    win20 = int(sr * 0.02)
    pre20 = audio[max(0, onset_sample - win20) : onset_sample]
    post20 = audio[onset_sample : min(len(audio), onset_sample + win20)]
    n_fft = 4096

    if len(pre20) >= 256 and len(post20) >= 256:
        pre_spec = np.abs(np.fft.rfft(pre20, n=n_fft)) ** 2
        post_spec = np.abs(np.fft.rfft(post20, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        diff_spec = np.maximum(post_spec - pre_spec, 0)

        # Spectral flatness of new energy
        diff_nz = diff_spec[diff_spec > eps]
        results["diff_flatness"] = (
            float(np.exp(np.mean(np.log(diff_nz + eps))) / (np.mean(diff_nz) + eps))
            if len(diff_nz) > 0
            else 1.0
        )

        # Spectral crest of new energy
        results["diff_crest"] = (
            float(np.max(diff_spec)) / (float(np.mean(diff_spec)) + eps)
            if float(np.max(diff_spec)) > eps
            else 0.0
        )

        # Fraction of bins with energy increase > 1.5x
        results["gain_positive_frac"] = float(np.mean(post_spec > pre_spec * 1.5))

        # Band-specific flatness
        for bname, (lo, hi) in [("himid", (2000, 5000)), ("high", (5000, 10000)), ("vhigh", (10000, 20000))]:
            mask = (freqs >= lo) & (freqs < hi)
            band_nz = diff_spec[mask][diff_spec[mask] > eps]
            results[f"flatness_{bname}"] = (
                float(np.exp(np.mean(np.log(band_nz + eps))) / (np.mean(band_nz) + eps))
                if len(band_nz) > 5
                else 1.0
            )

        # Diff centroid
        diff_energy = float(np.sum(diff_spec))
        results["diff_centroid"] = float(np.sum(freqs * diff_spec)) / diff_energy if diff_energy > eps else 0.0

    # --- Harmonicity (autocorrelation) ---
    min_lag = int(sr / 2000)
    max_lag = int(sr / 150)
    post_ac, _ = _normalized_autocorrelation(post20, min_lag, max_lag)
    pre_ac, _ = _normalized_autocorrelation(pre20, min_lag, max_lag)
    results["post_autocorr_20ms"] = post_ac
    results["autocorr_change_20ms"] = post_ac - pre_ac

    # Long (50ms) autocorrelation
    win50 = int(sr * 0.05)
    long_post = audio[onset_sample : min(len(audio), onset_sample + win50)]
    long_ac, _ = _normalized_autocorrelation(long_post, min_lag, max_lag)
    long_pre = audio[max(0, onset_sample - win50) : onset_sample]
    long_pre_ac, _ = _normalized_autocorrelation(long_pre, min_lag, max_lag)
    results["long_autocorr"] = long_ac
    results["long_autocorr_change"] = long_ac - long_pre_ac

    # --- Band gains (5ms) ---
    BANDS = [
        ("sub_bass", 20, 200), ("bass", 200, 500), ("low_mid", 500, 1000),
        ("mid", 1000, 2000), ("hi_mid", 2000, 4000), ("presence", 4000, 8000),
        ("brilliance", 8000, 16000),
    ]
    t5 = int(sr * 5 / 1000)
    band_gains = []
    for bname, lo, hi in BANDS:
        filtered = _bandpass(audio, sr, lo, hi)
        pre5 = onset_sample - t5
        post5 = onset_sample + t5
        if pre5 >= 0 and post5 < len(filtered):
            pre_e = float(np.sqrt(np.mean(filtered[pre5:onset_sample] ** 2)))
            post_e = float(np.sqrt(np.mean(filtered[onset_sample:post5] ** 2)))
            g = post_e / (pre_e + eps)
            results[f"gain_{bname}"] = g
            band_gains.append(g)

    if band_gains:
        bg = np.array(band_gains)
        results["band_gain_cv"] = float(np.std(bg) / (np.mean(bg) + eps))
        results["band_gain_mean"] = float(np.mean(bg))

    # --- Crest factor / kurtosis ---
    for wms in [3, 5, 10]:
        s = int(sr * wms / 1000)
        seg = audio[onset_sample : min(len(audio), onset_sample + s)]
        if len(seg) > 10:
            results[f"crest_{wms}ms"] = float(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg**2)) + eps))

    if len(pre20) > 10 and len(post20) > 10:
        pre_crest = float(np.max(np.abs(pre20)) / (np.sqrt(np.mean(pre20**2)) + eps))
        post_crest = float(np.max(np.abs(post20)) / (np.sqrt(np.mean(post20**2)) + eps))
        results["post_crest_20ms"] = post_crest
        results["crest_change"] = post_crest - pre_crest

    if len(post20) > 10:
        var = float(np.var(post20))
        if var > 1e-20:
            mean = float(np.mean(post20))
            results["kurtosis_20ms"] = float(np.mean((post20 - mean) ** 4) / (var * var)) - 3.0
        else:
            results["kurtosis_20ms"] = 0.0

    # Max derivative 5ms
    post5ms = audio[onset_sample : min(len(audio), onset_sample + t5)]
    if len(post5ms) > 1:
        deriv = np.abs(np.diff(post5ms))
        results["max_deriv_5ms"] = float(np.max(deriv))
        results["mean_deriv_5ms"] = float(np.mean(deriv))

    return results


# ---------------------------------------------------------------------------
# Separation analysis
# ---------------------------------------------------------------------------

def cohens_d(real_vals: list[float], compare_vals: list[float]) -> float:
    """Cohen's d: standardized mean difference."""
    if not real_vals or not compare_vals:
        return 0.0
    r_std = float(np.std(real_vals))
    c_std = float(np.std(compare_vals))
    pool_std = float(np.sqrt((r_std**2 + c_std**2) / 2)) + 1e-10
    return abs(float(np.mean(real_vals)) - float(np.mean(compare_vals))) / pool_std


def overlap_status(real_vals: list[float], compare_vals: list[float]) -> str:
    """Check if groups are cleanly separated."""
    if min(real_vals) > max(compare_vals):
        return "CLEAN(R>C)"
    elif max(real_vals) < min(compare_vals):
        return "CLEAN(R<C)"
    else:
        return "overlap"


def print_separation_report(
    ref_data: list[dict[str, float]],
    compare_data: list[dict[str, float]],
    ref_name: str = "real",
    compare_name: str = "compare",
) -> None:
    """Print separation scores for all features, sorted by Cohen's d."""
    all_metrics = sorted({k for d in ref_data for k in d if not k.startswith("_")})

    scored = []
    for metric in all_metrics:
        ref_vals = [d[metric] for d in ref_data if metric in d]
        cmp_vals = [d[metric] for d in compare_data if metric in d]
        if not ref_vals or not cmp_vals:
            continue
        d = cohens_d(ref_vals, cmp_vals)
        direction = "R>C" if float(np.mean(ref_vals)) > float(np.mean(cmp_vals)) else "R<C"
        clean = overlap_status(ref_vals, cmp_vals)
        scored.append((d, metric, direction, clean, min(ref_vals), max(ref_vals), min(cmp_vals), max(cmp_vals)))

    scored.sort(reverse=True)

    print(f"\n{'=' * 120}")
    print(f"  {ref_name} ({len(ref_data)} samples) vs {compare_name} ({len(compare_data)} samples)")
    print(f"{'=' * 120}")
    print(f"  {'metric':<28} {'sep':>6} {'dir':>5} {'clean':>12}   {'R_range':>30}  {'C_range':>30}")
    print(f"  {'-' * 110}")

    for d_val, metric, direction, clean, r_min, r_max, c_min, c_max in scored:
        marker = "***" if d_val > 2.0 else "** " if d_val > 1.5 else "*  " if d_val > 1.0 else "   "
        r_range = f"[{r_min:>10.4f}, {r_max:>10.4f}]"
        c_range = f"[{c_min:>10.4f}, {c_max:>10.4f}]"
        print(f"  {metric:<28} {d_val:>6.2f} {direction:>5} {clean:>12} {marker} {r_range}  {c_range}")


def print_raw_values(
    groups: dict[str, list[dict]],
    top_n: int = 10,
) -> None:
    """Print raw feature values for top metrics across all groups."""
    # Collect all data and find top metrics
    all_data = []
    for name, data in groups.items():
        for d in data:
            all_data.append(d)

    all_metrics = sorted({k for d in all_data for k in d if not k.startswith("_")})

    # Find top metrics (highest separation in any pair)
    ref_name = next(iter(groups))
    ref_data = groups[ref_name]
    top_metrics_scored: dict[str, float] = {}
    for cmp_name, cmp_data in groups.items():
        if cmp_name == ref_name:
            continue
        for metric in all_metrics:
            ref_vals = [d[metric] for d in ref_data if metric in d]
            cmp_vals = [d[metric] for d in cmp_data if metric in d]
            if ref_vals and cmp_vals:
                d = cohens_d(ref_vals, cmp_vals)
                top_metrics_scored[metric] = max(top_metrics_scored.get(metric, 0), d)

    top_metrics = sorted(top_metrics_scored, key=lambda m: top_metrics_scored[m], reverse=True)[:top_n]

    print(f"\n{'=' * 120}")
    print(f"RAW VALUES — top {top_n} metrics")
    print(f"{'=' * 120}")
    print(f"  {'label':<30}", end="")
    for m in top_metrics:
        print(f"{m:>18}", end="")
    print()
    print(f"  {'-' * (30 + 18 * len(top_metrics))}")

    for group_name, data in groups.items():
        for d in data:
            label = f"{group_name}:{d.get('_label', '?')}"
            print(f"  {label:<30}", end="")
            for m in top_metrics:
                v = d.get(m, 0)
                print(f"{v:>18.4f}", end="")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Onset feature separation analysis")
    parser.add_argument("--config", help="JSON config file with sample groups")
    parser.add_argument("--audio", help="Audio file or fixture name (for quick CLI mode)")
    parser.add_argument("--real", help="Comma-separated onset times for reference group")
    parser.add_argument("--compare", help="Comma-separated onset times for comparison group")
    parser.add_argument("--top", type=int, default=10, help="Number of top metrics to show in raw values")
    parser.add_argument("--sr", type=int, default=None, help="Sample rate (default: native)")
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {}

    if args.config:
        # JSON config mode
        config = load_config(args.config)
        ref_group = config.get("reference_group", "real")
        audio_cache: dict[str, tuple[np.ndarray, int]] = {}

        for group_name, samples in config["groups"].items():
            groups[group_name] = []
            for sample in samples:
                audio_path = resolve_audio_path(sample["audio"])
                if audio_path not in audio_cache:
                    audio_cache[audio_path] = librosa.load(audio_path, sr=args.sr)
                audio, sr = audio_cache[audio_path]
                features = compute_onset_features(audio, sr, sample["onset"])
                features["_label"] = sample.get("label", f"{sample['onset']:.3f}")
                groups[group_name].append(features)

    elif args.audio and args.real and args.compare:
        # Quick CLI mode
        audio_path = resolve_audio_path(args.audio)
        audio, sr = librosa.load(audio_path, sr=args.sr)

        groups["real"] = []
        for t in args.real.split(","):
            t_float = float(t.strip())
            features = compute_onset_features(audio, sr, t_float)
            features["_label"] = f"{t_float:.3f}"
            groups["real"].append(features)

        groups["compare"] = []
        for t in args.compare.split(","):
            t_float = float(t.strip())
            features = compute_onset_features(audio, sr, t_float)
            features["_label"] = f"{t_float:.3f}"
            groups["compare"].append(features)

        ref_group = "real"
    else:
        parser.error("Provide either --config or (--audio + --real + --compare)")
        return

    # Print summary
    print("Onset Feature Separation Analysis")
    print("=" * 60)
    for name, data in groups.items():
        labels = [d.get("_label", "?") for d in data]
        marker = " (reference)" if name == ref_group else ""
        print(f"  {name}{marker}: {len(data)} samples ({', '.join(labels)})")

    # Separation reports
    ref_data = groups[ref_group]
    for cmp_name, cmp_data in groups.items():
        if cmp_name == ref_group:
            continue
        print_separation_report(ref_data, cmp_data, ref_name=ref_group, compare_name=cmp_name)

    # Raw values
    print_raw_values(groups, top_n=args.top)


if __name__ == "__main__":
    main()
