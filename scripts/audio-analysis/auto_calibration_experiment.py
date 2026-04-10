#!/usr/bin/env python3
"""Auto-calibration experiment: extract per-tine partials from performance solo segments.

Compares auto-extracted partial ratios with ground truth from repeat fixtures
to evaluate the feasibility of calibration-free partial table construction.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import get_window

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "api"))

from app.tunings import get_default_tunings
from app.transcription.segments import detect_segments
from app.transcription.peaks import segment_peaks

FIXTURE_BASE = Path(__file__).resolve().parents[2] / "apps" / "api" / "tests" / "fixtures" / "manual-captures"

# Ground truth from repeat fixtures (17-C, mic close)
GROUND_TRUTH_P2 = {
    "C4": 1.994,  # c4-repeat-01
    "D4": 2.006,  # d4-repeat-01
    "D5": 1.508,  # d5-repeat-01
}
GROUND_TRUTH_P3 = {
    "C4": 2.908,  # c4-repeat-01
    "D5": 2.648,  # d5-repeat-01
}


def load_audio(wav_path: Path) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(wf.getsampwidth(), np.int16)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        audio /= max(np.max(np.abs(audio)), 1e-9)
    return audio, sr


def find_partial_ratio(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    fundamental_hz: float,
    partial_idx: int,
    sustain_start_ratio: float = 0.4,
) -> float | None:
    """Extract a single partial ratio from the sustain portion of a segment."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = audio[start_sample:end_sample]

    sustain_start = int(len(segment) * sustain_start_ratio)
    sustain = segment[sustain_start:]

    if len(sustain) < 1024:
        return None

    n_fft = max(2 ** int(np.ceil(np.log2(len(sustain)))), 8192)
    window = get_window("hann", len(sustain))
    spectrum = np.abs(np.fft.rfft(sustain * window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freq_res = sr / n_fft

    # Search window
    if partial_idx == 1:
        search_low = fundamental_hz * 0.9
        search_high = fundamental_hz * 1.1
    else:
        search_low = fundamental_hz * (partial_idx - 0.5)
        search_high = fundamental_hz * (partial_idx + 0.5)

    mask = (freqs >= search_low) & (freqs <= search_high)
    if not np.any(mask):
        return None

    masked_spectrum = spectrum.copy()
    masked_spectrum[~mask] = 0.0
    peak_idx = np.argmax(masked_spectrum)
    peak_energy = float(masked_spectrum[peak_idx])

    if peak_energy < 1.0:
        return None

    # Parabolic interpolation
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

    return peak_freq / fundamental_hz


def main():
    wav_path = FIXTURE_BASE / "kalimba-17-c-bwv147-sequence-163-01" / "audio.wav"
    audio, sr = load_audio(wav_path)
    tuning = get_default_tunings()[0]

    segments, _, _ = detect_segments(audio, sr)
    print(f"17-C BWV147: {len(segments)} segments, {len(audio)/sr:.1f}s @ {sr}Hz")
    print()

    # Collect all segment data with quality metrics
    segment_data: list[dict] = []
    prev_end = 0.0

    for seg in segments:
        start = seg.start_time
        end = seg.end_time
        duration = end - start
        gap_from_prev = start - prev_end

        if duration < 0.10:
            prev_end = end
            continue

        result = segment_peaks(audio, sr, start, end, tuning, debug=True)
        candidates, debug, primary_hyp, _trace = result

        if not candidates or primary_hyp is None:
            prev_end = end
            continue

        note_name = primary_hyp.candidate.note_name
        frequency = primary_hyp.candidate.frequency
        n_notes = len(candidates)

        # Get onset_gain from debug
        onset_gain = None
        if debug and "primaryOnsetGain" in debug:
            onset_gain = debug["primaryOnsetGain"]

        p2 = find_partial_ratio(audio, sr, start, end, frequency, 2)
        p3 = find_partial_ratio(audio, sr, start, end, frequency, 3)

        segment_data.append({
            "note": note_name,
            "frequency": frequency,
            "start": round(start, 3),
            "duration": round(duration, 3),
            "gap": round(gap_from_prev, 3),
            "n_notes": n_notes,
            "onset_gain": onset_gain,
            "p2": round(p2, 4) if p2 else None,
            "p3": round(p3, 4) if p3 else None,
        })

        prev_end = end

    # ── Experiment: try different filter configurations ──

    filters = [
        ("no filter (all primaries)", lambda d: d["p2"] is not None),
        ("solo only (n_notes=1)", lambda d: d["p2"] is not None and d["n_notes"] == 1),
        ("solo + dur≥0.2s", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["duration"] >= 0.2),
        ("solo + dur≥0.3s", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["duration"] >= 0.3),
        ("solo + gap≥0.1s", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["gap"] >= 0.1),
        ("solo + dur≥0.2 + gap≥0.05", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["duration"] >= 0.2 and d["gap"] >= 0.05),
        ("solo + dur≥0.2 + gap≥0.1", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["duration"] >= 0.2 and d["gap"] >= 0.1),
        ("solo + dur≥0.3 + gap≥0.1", lambda d: d["p2"] is not None and d["n_notes"] == 1 and d["duration"] >= 0.3 and d["gap"] >= 0.1),
    ]

    for filter_name, filter_fn in filters:
        filtered = [d for d in segment_data if filter_fn(d)]

        # Aggregate: for each note, take the measurement with longest duration
        best_per_note: dict[str, dict] = {}
        for d in filtered:
            n = d["note"]
            if n not in best_per_note or d["duration"] > best_per_note[n]["duration"]:
                best_per_note[n] = d

        # Compare with ground truth
        gt_errors = []
        for gt_note, gt_p2 in GROUND_TRUTH_P2.items():
            if gt_note in best_per_note and best_per_note[gt_note]["p2"]:
                err = abs(best_per_note[gt_note]["p2"] - gt_p2)
                err_cents = abs(1200.0 * np.log2(best_per_note[gt_note]["p2"] / gt_p2)) if gt_p2 > 0 else 0
                gt_errors.append((gt_note, best_per_note[gt_note]["p2"], gt_p2, err_cents))

        print(f"━━ {filter_name} ━━")
        print(f"   Segments: {len(filtered)}, Notes covered: {len(best_per_note)}/17")

        if gt_errors:
            print(f"   Ground truth comparison (P2):")
            for note, measured, truth, err_c in gt_errors:
                status = "✓" if err_c < 20 else "△" if err_c < 50 else "✗"
                print(f"     {note}: measured={measured:.4f} truth={truth:.4f} Δ={err_c:.1f}c {status}")
            avg_err = np.mean([e[3] for e in gt_errors])
            print(f"   Avg P2 error: {avg_err:.1f} cents")
        else:
            print(f"   (no ground truth notes in filtered set)")
        print()

    # ── Detailed per-note table for best filter ──

    print("━━ Detail: solo + dur≥0.2 + gap≥0.05 — all notes ━━")
    best_filter = [d for d in segment_data
                   if d["p2"] is not None and d["n_notes"] == 1
                   and d["duration"] >= 0.2 and d["gap"] >= 0.05]

    best_per_note = {}
    all_per_note: dict[str, list[dict]] = {}
    for d in best_filter:
        n = d["note"]
        all_per_note.setdefault(n, []).append(d)
        if n not in best_per_note or d["duration"] > best_per_note[n]["duration"]:
            best_per_note[n] = d

    note_order = ["C4", "D4", "E4", "F4", "G4", "A4", "B4",
                  "C5", "D5", "E5", "F5", "G5", "A5", "B5",
                  "C6", "D6", "E6"]

    print(f"{'Note':<5} {'P2':>7} {'P3':>7} {'dur':>5} {'gap':>5} {'#seg':>4}  GT-P2")
    print("-" * 60)
    for note in note_order:
        if note in best_per_note:
            d = best_per_note[note]
            n_segs = len(all_per_note.get(note, []))
            gt = GROUND_TRUTH_P2.get(note)
            gt_str = ""
            if gt:
                err = abs(1200 * np.log2(d["p2"] / gt)) if d["p2"] and gt else 0
                gt_str = f"{gt:.3f} (Δ{err:.0f}c)"
            p3_str = f"{d['p3']:.3f}" if d["p3"] else "—"
            print(f"{note:<5} {d['p2']:>7.3f} {p3_str:>7} {d['duration']:>5.2f} {d['gap']:>5.2f} {n_segs:>4}  {gt_str}")
        else:
            print(f"{note:<5} {'—':>7}")

    # ── Show consistency across multiple segments for same note ──
    print()
    print("━━ Multi-segment consistency (notes with ≥2 segments) ━━")
    for note in note_order:
        segs = all_per_note.get(note, [])
        if len(segs) >= 2:
            p2s = [s["p2"] for s in segs if s["p2"]]
            if len(p2s) >= 2:
                std_cents = 1200 * np.std([np.log2(p) for p in p2s]) / np.log(2) if min(p2s) > 0 else 0
                print(f"  {note}: n={len(p2s)} P2 values={[round(p,3) for p in p2s]} std={np.std(p2s):.4f} ({std_cents:.1f}c)")


if __name__ == "__main__":
    main()
