#!/usr/bin/env python3
"""Measure per-tine partial ratios from a solo-note recording.

Usage:
    python scripts/audio-analysis/measure_partials.py <fixture_name_or_wav> [--max-partials N]

Runs the recognizer to identify segments, then for each confident solo-note
segment extracts spectral peaks from the sustain portion and reports the
measured partial frequency ratios vs the integer harmonic assumption.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import get_window

# Allow imports from apps/api
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "api"))

from app.tunings import get_default_tunings
from app.transcription.segments import detect_segments
from app.transcription.peaks import segment_peaks


FIXTURE_BASE = Path(__file__).resolve().parents[2] / "apps" / "api" / "tests" / "fixtures" / "manual-captures"


def resolve_audio(name_or_path: str) -> tuple[Path, str]:
    """Return (wav_path, tuning_id) for a fixture name or WAV path."""
    p = Path(name_or_path)
    if p.exists() and p.suffix == ".wav":
        return p, "kalimba-17-c"
    for d in FIXTURE_BASE.iterdir():
        if not d.is_dir():
            continue
        if name_or_path in d.name:
            wav = d / "audio.wav"
            req = d / "request.json"
            tuning_id = "kalimba-17-c"
            if req.exists():
                r = json.loads(req.read_text())
                tuning_id = r.get("tuning", {}).get("id", tuning_id)
            return wav, tuning_id
    raise FileNotFoundError(f"Cannot find fixture or WAV: {name_or_path}")


def load_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as mono float32."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(wf.getsampwidth(), np.int16)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        audio /= max(np.max(np.abs(audio)), 1e-9)
    return audio, sr


def find_spectral_peaks(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    fundamental_hz: float,
    max_partials: int = 8,
) -> list[dict]:
    """Extract spectral peaks from the sustain portion of a segment.

    Returns a list of detected partials with their frequency, ratio to
    fundamental, and deviation from the nearest integer multiple.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = audio[start_sample:end_sample]

    # Use the latter 60% as "sustain" to avoid attack transient
    sustain_start = int(len(segment) * 0.4)
    sustain = segment[sustain_start:]

    if len(sustain) < 1024:
        return []

    # Zero-pad for better frequency resolution
    n_fft = max(2 ** int(np.ceil(np.log2(len(sustain)))), 8192)
    window = get_window("hann", len(sustain))
    spectrum = np.abs(np.fft.rfft(sustain * window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Frequency resolution
    freq_res = sr / n_fft

    partials = []
    for partial_idx in range(1, max_partials + 1):
        # Search window: ±30% around expected position (generous for inharmonic partials)
        expected = fundamental_hz * partial_idx
        if partial_idx == 1:
            search_low = fundamental_hz * 0.9
            search_high = fundamental_hz * 1.1
        else:
            # For higher partials, search between midpoints to adjacent integer multiples
            search_low = fundamental_hz * (partial_idx - 0.5)
            search_high = fundamental_hz * (partial_idx + 0.5)

        mask = (freqs >= search_low) & (freqs <= search_high)
        if not np.any(mask):
            continue

        masked_spectrum = spectrum.copy()
        masked_spectrum[~mask] = 0.0
        peak_idx = np.argmax(masked_spectrum)
        peak_energy = float(masked_spectrum[peak_idx])

        if peak_energy < 1.0:  # noise floor
            continue

        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_idx < len(spectrum) - 1:
            alpha = float(spectrum[peak_idx - 1])
            beta = float(spectrum[peak_idx])
            gamma = float(spectrum[peak_idx + 1])
            if 2 * beta - alpha - gamma > 0:
                delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                peak_freq = float(freqs[peak_idx]) + delta * freq_res
            else:
                peak_freq = float(freqs[peak_idx])
        else:
            peak_freq = float(freqs[peak_idx])

        ratio = peak_freq / fundamental_hz
        nearest_int = round(ratio)
        deviation_cents = 1200.0 * np.log2(ratio / nearest_int) if nearest_int > 0 and ratio > 0 else 0.0

        partials.append({
            "partial": partial_idx,
            "expected_hz": round(expected, 1),
            "measured_hz": round(peak_freq, 1),
            "ratio": round(ratio, 4),
            "nearest_integer": nearest_int,
            "deviation_cents": round(deviation_cents, 1),
            "energy": round(peak_energy, 1),
        })

    return partials


def main():
    parser = argparse.ArgumentParser(description="Measure per-tine partial ratios")
    parser.add_argument("fixture", help="Fixture name or WAV path")
    parser.add_argument("--max-partials", type=int, default=6, help="Max partials to search")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    wav_path, tuning_id = resolve_audio(args.fixture)
    audio, sr = load_audio(wav_path)
    all_tunings = get_default_tunings()
    tuning = next((t for t in all_tunings if t.id == tuning_id), all_tunings[0])

    # Run segmenter + peaks to get solo notes
    segments, _est_start, _debug = detect_segments(audio, sr)

    print(f"Fixture: {wav_path.parent.name}")
    print(f"Tuning: {tuning_id} ({len(tuning.notes)} notes)")
    print(f"Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    print(f"Segments detected: {len(segments)}")
    print()

    all_partials = {}

    for seg in segments:
        start = seg.start_time
        end = seg.end_time
        duration = end - start

        if duration < 0.12:
            continue

        result = segment_peaks(audio, sr, start, end, tuning)
        candidates = result[0]
        primary_hyp = result[2]  # primary NoteHypothesis

        if not candidates or primary_hyp is None:
            continue

        note_name = primary_hyp.candidate.note_name
        frequency = primary_hyp.candidate.frequency

        # Skip if we already have this note with a longer duration (better data)
        if note_name in all_partials and all_partials[note_name]["duration"] >= duration:
            continue

        partials = find_spectral_peaks(audio, sr, start, end, frequency, args.max_partials)
        if partials:
            n_notes = len(candidates)
            all_partials[note_name] = {
                "frequency": frequency,
                "duration": round(duration, 3),
                "n_notes": n_notes,
                "partials": partials,
            }

    if args.json:
        print(json.dumps(all_partials, indent=2))
        return

    # Print table
    print(f"{'Note':<5} {'f0 (Hz)':>8} | ", end="")
    for i in range(1, args.max_partials + 1):
        print(f"  P{i} ratio (Δcents)", end="")
    print()
    print("-" * (16 + args.max_partials * 19))

    for note_name in ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
                       "C5", "D5", "E5", "F5", "G5", "A5", "B5",
                       "C6", "D6", "E6"]:
        if note_name not in all_partials:
            print(f"{note_name:<5} {'—':>8} |")
            continue

        data = all_partials[note_name]
        print(f"{note_name:<5} {data['frequency']:>8.1f} | ", end="")
        partials_by_idx = {p["partial"]: p for p in data["partials"]}
        for i in range(1, args.max_partials + 1):
            if i in partials_by_idx:
                p = partials_by_idx[i]
                dev = p["deviation_cents"]
                marker = "  " if abs(dev) < 20 else " *" if abs(dev) < 50 else " !"
                print(f"  {p['ratio']:6.3f} ({dev:+5.1f}){marker}", end="")
            else:
                print(f"  {'—':>6} {'':>8}", end="")
        print()

    # Summary
    print()
    print("Legend: no marker = <20 cents, * = 20-50 cents, ! = >50 cents deviation from integer")
    print()

    # Compute average deviation per partial
    print("Average |deviation| from integer by partial:")
    for i in range(1, args.max_partials + 1):
        devs = []
        for data in all_partials.values():
            for p in data["partials"]:
                if p["partial"] == i:
                    devs.append(abs(p["deviation_cents"]))
        if devs:
            print(f"  P{i}: mean={np.mean(devs):5.1f} cents, max={np.max(devs):5.1f} cents (n={len(devs)})")


if __name__ == "__main__":
    main()
