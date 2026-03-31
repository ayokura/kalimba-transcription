"""Track per-note peak frequency and energy over time.

Shows the actual peak frequency in a note's band at each time step,
with cents deviation from the tuning reference.  Useful for detecting
tuning drift, verifying note presence, and diagnosing FFT resolution
issues.

Usage:
    uv run python scripts/audio-analysis/note_peak_track.py <audio> <start> <duration> \\
        --notes D4,B4,G4 [--step 0.05] [--n-fft 8192] [--band-cents 100] [--min-energy 1.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

NOTE_FREQUENCIES = {
    "C3": 130.81, "D3": 146.83, "E3": 164.81, "F3": 174.61, "G3": 196.00, "A3": 220.00, "B3": 246.94,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.26, "F5": 698.46, "G5": 783.99, "A5": 880.00, "B5": 987.77,
    "C6": 1046.50, "D6": 1174.66, "E6": 1318.51,
}

FIXTURE_ROOT = Path(__file__).resolve().parent.parent.parent / "apps" / "api" / "tests" / "fixtures" / "manual-captures"


def resolve_audio_path(audio_arg: str) -> Path:
    p = Path(audio_arg)
    if p.exists():
        return p
    for prefix in [f"kalimba-17-c-{audio_arg}", audio_arg]:
        fixture_dir = FIXTURE_ROOT / prefix
        if fixture_dir.is_dir():
            return fixture_dir / "audio.wav"
    for d in FIXTURE_ROOT.iterdir():
        if d.is_dir() and audio_arg in d.name:
            return d / "audio.wav"
    print(f"Error: Cannot resolve audio path: {audio_arg}", file=sys.stderr)
    sys.exit(1)


def find_peak_in_band(
    audio: np.ndarray,
    sr: int,
    center_time: float,
    ref_freq: float,
    band_cents: float = 100.0,
    window_sec: float = 0.03,
    n_fft_min: int = 8192,
) -> tuple[float | None, float]:
    """Find the peak frequency and energy within ±band_cents of ref_freq."""
    center = int(center_time * sr)
    half = int(window_sec * sr / 2)
    chunk = audio[max(0, center - half):center + half]
    if len(chunk) < 256:
        return None, 0.0

    n_fft = max(n_fft_min, 1 << int(np.ceil(np.log2(len(chunk)))))
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    low = ref_freq * 2 ** (-band_cents / 1200)
    high = ref_freq * 2 ** (band_cents / 1200)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None, 0.0

    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]
    peak_idx = int(np.argmax(band_spectrum))
    return float(band_freqs[peak_idx]), float(band_spectrum[peak_idx])


def main():
    parser = argparse.ArgumentParser(description="Track per-note peak frequency over time")
    parser.add_argument("audio", help="Audio file path or fixture name")
    parser.add_argument("start", type=float, help="Start time in seconds")
    parser.add_argument("duration", type=float, help="Duration in seconds")
    parser.add_argument("--notes", default="D4", help="Comma-separated note names (default: D4)")
    parser.add_argument("--step", type=float, default=0.05, help="Time step in seconds (default: 0.05)")
    parser.add_argument("--n-fft", type=int, default=8192, help="Minimum FFT size (default: 8192)")
    parser.add_argument("--band-cents", type=float, default=100, help="Search band ±cents (default: 100)")
    parser.add_argument("--min-energy", type=float, default=1.0, help="Minimum energy to display (default: 1.0)")
    parser.add_argument("--window", type=float, default=0.03, help="Analysis window in seconds (default: 0.03)")
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio)
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    notes = [(name.strip(), NOTE_FREQUENCIES[name.strip()]) for name in args.notes.split(",")]
    times = np.arange(args.start, args.start + args.duration + args.step / 2, args.step)

    # Header
    header_parts = [f"{'Time':>9s}"]
    for name, _ in notes:
        header_parts.append(f"{'peak_Hz':>8s} {'cents':>6s} {'energy':>7s}")
    print("  ".join(header_parts))

    # Separator showing reference frequencies
    ref_parts = [f"{'':>9s}"]
    for name, freq in notes:
        ref_parts.append(f"  {name}={freq:.1f}Hz{'':>11s}")
    print("  ".join(ref_parts))

    for t in times:
        if t < 0 or t > len(audio) / sr:
            continue
        parts = [f"{t:9.3f}s"]
        for name, ref_freq in notes:
            peak_freq, energy = find_peak_in_band(
                audio, sr, t, ref_freq,
                band_cents=args.band_cents,
                window_sec=args.window,
                n_fft_min=args.n_fft,
            )
            if peak_freq is not None and energy >= args.min_energy:
                cents = 1200 * np.log2(peak_freq / ref_freq)
                in_40 = "✓" if abs(cents) <= 40 else "✗"
                parts.append(f"{peak_freq:8.1f} {cents:+5.0f}¢{in_40} {energy:7.1f}")
            else:
                parts.append(f"{'---':>8s} {'':>6s} {'---':>7s}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
