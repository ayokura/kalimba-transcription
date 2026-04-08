"""Trace per-note energy over time for a given audio file.

Usage:
    uv run python scripts/audio-analysis/energy_trace.py <audio> <start> <duration> [--notes G4,G5,E4] [--step 0.05] [--band 15]

Arguments:
    audio       Audio file path or fixture name
    start       Start time in seconds
    duration    Duration in seconds

Options:
    --notes     Comma-separated note names to track (default: C4,E4,G4,A4,C5,E5,G5,C6)
    --step      Time step in seconds (default: 0.05)
    --band      Frequency band half-width in Hz (default: 15)
    --n_fft     FFT size (default: 4096)
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

NOTE_FREQUENCIES = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61,
    "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23,
    "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.26, "F5": 698.46,
    "F#5": 739.99, "G5": 783.99, "G#5": 830.61, "A5": 880.00, "A#5": 932.33, "B5": 987.77,
    "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E6": 1318.51, "F6": 1396.91,
}

FIXTURE_ROOT = Path(__file__).resolve().parent.parent.parent / "apps" / "api" / "tests" / "fixtures" / "manual-captures"


def resolve_audio_path(audio_arg: str) -> Path:
    p = Path(audio_arg)
    if p.exists():
        return p
    # Try fixture name expansion (with and without kalimba-17-c prefix)
    for prefix in [f"kalimba-17-c-{audio_arg}", audio_arg]:
        fixture_dir = FIXTURE_ROOT / prefix
        if fixture_dir.is_dir():
            return fixture_dir / "audio.wav"
    # Try glob match
    for d in FIXTURE_ROOT.iterdir():
        if d.is_dir() and audio_arg in d.name:
            return d / "audio.wav"
    print(f"Error: Cannot resolve audio path: {audio_arg}", file=sys.stderr)
    sys.exit(1)


def energy_in_band(spectrum: np.ndarray, freqs: np.ndarray, center_freq: float, band_hz: float) -> float:
    mask = (freqs >= center_freq - band_hz) & (freqs <= center_freq + band_hz)
    return float(np.sum(spectrum[mask] ** 2))


def main():
    parser = argparse.ArgumentParser(description="Trace per-note energy over time")
    parser.add_argument("audio", help="Audio file path or fixture name")
    parser.add_argument("start", type=float, help="Start time in seconds")
    parser.add_argument("duration", type=float, help="Duration in seconds")
    parser.add_argument("--notes", default="C4,E4,G4,A4,C5,E5,G5,C6",
                        help="Comma-separated note names (default: C4,E4,G4,A4,C5,E5,G5,C6)")
    parser.add_argument("--step", type=float, default=0.05, help="Time step in seconds (default: 0.05)")
    parser.add_argument("--band", type=float, default=15.0, help="Frequency band half-width in Hz (default: 15)")
    parser.add_argument("--n_fft", type=int, default=4096, help="FFT size (default: 4096)")
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio)
    note_names = [n.strip() for n in args.notes.split(",")]
    note_freqs = {}
    for name in note_names:
        if name not in NOTE_FREQUENCIES:
            print(f"Error: Unknown note name: {name}", file=sys.stderr)
            sys.exit(1)
        note_freqs[name] = NOTE_FREQUENCIES[name]

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    total_duration = len(y) / sr

    start_t = args.start
    end_t = min(args.start + args.duration, total_duration)

    # Print header
    header = f"{'Time':>8s}"
    for name in note_names:
        header += f"  {name:>10s}"
    # Add ratio columns for first two notes
    if len(note_names) >= 2:
        header += f"  {note_names[0]}/{note_names[1]:>10s}"
    print(header)

    t = start_t
    while t < end_t:
        start_sample = int(t * sr)
        seg = y[start_sample:start_sample + args.n_fft]
        if len(seg) < args.n_fft:
            seg = np.pad(seg, (0, args.n_fft - len(seg)))
        spectrum = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(args.n_fft, 1.0 / sr)

        row = f"{t:8.3f}s"
        energies = {}
        for name in note_names:
            e = energy_in_band(spectrum, freqs, note_freqs[name], args.band)
            energies[name] = e
            row += f"  {e:10.1f}"

        if len(note_names) >= 2:
            denom = max(energies[note_names[1]], 1.0)
            ratio = energies[note_names[0]] / denom
            row += f"  {ratio:10.4f}"

        print(row)
        t += args.step


if __name__ == "__main__":
    main()
