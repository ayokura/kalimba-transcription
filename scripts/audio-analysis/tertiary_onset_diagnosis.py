#!/usr/bin/env python3
"""
Diagnose tertiary onset false positives across failing fixtures.

Collects tertiary decisions (accepted) from debug output, classifies them as
TRUE_POSITIVE or FALSE_POSITIVE by comparing with expected.json, and computes
candidate discriminator metrics (onset_gain, attack_peakiness, relative_onset_gain).

Usage:
  uv run python scripts/audio-analysis/tertiary_onset_diagnosis.py
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

TESTS_DIR = Path(__file__).resolve().parent.parent.parent / "apps" / "api" / "tests"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

from manual_capture_helpers import build_evaluation_audio_bytes, load_fixture, normalized_assertions
from fastapi.testclient import TestClient
from app.main import app
from app.transcription import onset_energy_gain, peak_energy_near
from app.tunings import note_name_to_frequency

FIXTURE_ROOT = TESTS_DIR / "fixtures" / "manual-captures"

FAILING_FIXTURES = [
    "kalimba-17-c-a4-d4-f4-triad-repeat-02",
    "kalimba-17-c-b4-d5-double-notes-01",
    "kalimba-17-c-bwv147-lower-context-roll-01",
    "kalimba-17-c-bwv147-lower-mixed-roll-01",
    "kalimba-17-c-c4-c5-octave-dyad-01",
    "kalimba-17-c-d4-repeat-01",
    "kalimba-17-c-e4-g4-b4-d5-four-note-repeat-01",
    "kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-02",
    "kalimba-17-c-e6-to-c4-sequence-17-01",
    "kalimba-17-c-e6-to-c4-sequence-51-01",
    "kalimba-17-c-triple-glissando-ascending-01",
]

TARGET_FIXTURE = "kalimba-17-c-bwv147-sequence-163-01"


def compute_attack_peakiness(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    target_frequency: float,
) -> float:
    """Compute early(0-20ms) / late(40-80ms) narrowband energy ratio at target_frequency."""
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    if end_sample - start_sample < 512:
        return 0.0

    early_end = min(start_sample + int(sample_rate * 0.02), end_sample)
    late_s = min(start_sample + int(sample_rate * 0.04), end_sample)
    late_e = min(start_sample + int(sample_rate * 0.08), end_sample)

    early_chunk = audio[start_sample:early_end]
    late_chunk = audio[late_s:late_e]

    if len(early_chunk) < 256 or len(late_chunk) < 256:
        return 0.0

    def _energy(chunk: np.ndarray) -> float:
        n_fft = max(4096, 1 << int(np.ceil(np.log2(len(chunk)))))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        return peak_energy_near(frequencies, spectrum, target_frequency)

    early_energy = _energy(early_chunk)
    late_energy = _energy(late_chunk)
    return (early_energy + 1e-6) / (late_energy + 1e-6)


def run_fixture(fixture_name: str) -> tuple[dict, dict, np.ndarray, int]:
    """Run transcription on a fixture and return (payload, expected, audio, sr)."""
    fixture_dir = FIXTURE_ROOT / fixture_name
    client = TestClient(app)
    request_payload, expected = load_fixture(fixture_dir)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim > 1:
        audio = audio[:, 0]

    return payload, expected, audio, sr


def analyze_fixture(fixture_name: str, is_target: bool = False):
    """Analyze tertiary decisions for a single fixture."""
    payload, expected, audio, sr = run_fixture(fixture_name)
    assertions = normalized_assertions(expected)
    debug = payload.get("debug", {})
    segment_candidates = debug.get("segmentCandidates", [])

    results = []

    for seg in segment_candidates:
        trail = seg.get("secondaryDecisionTrail", [])
        start_t = seg.get("startTime", 0)
        end_t = seg.get("endTime", start_t + 0.5)
        selected = seg.get("selectedNotes", [])

        if len(selected) < 3:
            continue

        primary_note = selected[0]
        primary_og = seg.get("primaryOnsetGain")

        # Use note_name_to_frequency for frequency lookup
        try:
            primary_freq = note_name_to_frequency(primary_note)
        except Exception:
            primary_freq = None

        if primary_og is None and primary_freq:
            primary_og = onset_energy_gain(audio, sr, start_t, end_t, primary_freq)

        # Find tertiary notes: identify which trail entries were accepted as 3rd+ notes
        # The trail processes candidates in order. Count accepted entries to find tertiaries.
        accepted_notes_in_order = []
        for entry in trail:
            if entry.get("accepted", False):
                accepted_notes_in_order.append(entry)

        # accepted_notes_in_order[0] = secondary (2nd), [1] = tertiary (3rd), etc.
        for accept_idx, entry in enumerate(accepted_notes_in_order):
            # accept_idx=0 is secondary, accept_idx>=1 is tertiary
            if accept_idx < 1:
                continue  # Skip secondary

            note_name = entry["noteName"]
            trail_og = entry.get("onsetGain")

            try:
                note_freq = note_name_to_frequency(note_name)
            except Exception:
                continue

            og = trail_og if trail_og is not None else onset_energy_gain(audio, sr, start_t, end_t, note_freq)
            peakiness = compute_attack_peakiness(audio, sr, start_t, end_t, note_freq)
            rel_og = og / primary_og if primary_og and primary_og > 0 else None

            # Classify: is this note in any expected note set?
            is_in_any_expected_set = False
            for req_set in assertions.get("requiredEventNoteSetOccurrences", {}):
                if note_name in req_set.split("+"):
                    is_in_any_expected_set = True
                    break

            # Also check ordered expectations
            if not is_in_any_expected_set:
                for ordered_set in assertions.get("expectedEventNoteSetsOrdered", []):
                    if note_name in ordered_set.split("+"):
                        is_in_any_expected_set = True
                        break

            label = "TRUE_POSITIVE" if is_in_any_expected_set else "FALSE_POSITIVE"

            results.append({
                "fixture": fixture_name,
                "time": start_t,
                "primary": primary_note,
                "tertiary_note": note_name,
                "note_index": accept_idx + 2,  # 1-indexed: 3 = tertiary
                "event_notes": "+".join(selected),
                "label": label,
                "onset_gain": og,
                "attack_peakiness": peakiness,
                "relative_onset_gain": rel_og,
                "primary_onset_gain": primary_og,
                "score": entry.get("score"),
                "fundamental_ratio": entry.get("fundamentalRatio"),
                "is_target": is_target,
            })

    return results


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt((var_a * (len(group_a) - 1) + var_b * (len(group_b) - 1)) / (len(group_a) + len(group_b) - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return float((mean_a - mean_b) / pooled_std)


def overlap_ratio(group_a: list[float], group_b: list[float]) -> float:
    if not group_a or not group_b:
        return 0.0
    min_a, max_a = min(group_a), max(group_a)
    min_b, max_b = min(group_b), max(group_b)
    overlap_start = max(min_a, min_b)
    overlap_end = min(max_a, max_b)
    if overlap_start >= overlap_end:
        return 0.0
    total_range = max(max_a, max_b) - min(min_a, min_b)
    if total_range < 1e-10:
        return 1.0
    return (overlap_end - overlap_start) / total_range


def main():
    all_results = []

    print("=== Analyzing Failing Fixtures ===")
    for fixture_name in FAILING_FIXTURES:
        print(f"  Processing {fixture_name.replace('kalimba-17-c-', '')}...", end=" ", flush=True)
        try:
            results = analyze_fixture(fixture_name, is_target=False)
            all_results.extend(results)
            fp_count = sum(1 for r in results if r["label"] == "FALSE_POSITIVE")
            tp_count = sum(1 for r in results if r["label"] == "TRUE_POSITIVE")
            print(f"TP={tp_count}, FP={fp_count}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n=== Analyzing Target Fixture ===")
    print(f"  Processing {TARGET_FIXTURE.replace('kalimba-17-c-', '')}...", end=" ", flush=True)
    try:
        results = analyze_fixture(TARGET_FIXTURE, is_target=True)
        all_results.extend(results)
        fp_count = sum(1 for r in results if r["label"] == "FALSE_POSITIVE")
        tp_count = sum(1 for r in results if r["label"] == "TRUE_POSITIVE")
        print(f"TP={tp_count}, FP={fp_count}")
    except Exception as e:
        print(f"ERROR: {e}")

    true_pos = [r for r in all_results if r["label"] == "TRUE_POSITIVE"]
    false_pos = [r for r in all_results if r["label"] == "FALSE_POSITIVE"]

    print(f"\n=== Summary ===")
    print(f"Total TRUE_POSITIVE tertiary: {len(true_pos)}")
    print(f"Total FALSE_POSITIVE tertiary: {len(false_pos)}")

    print(f"\n=== FALSE_POSITIVE Samples ===")
    for r in false_pos:
        rel_str = f"{r['relative_onset_gain']:.3f}" if r['relative_onset_gain'] is not None else "N/A"
        print(f"  {r['fixture'].replace('kalimba-17-c-', '')} @{r['time']:.2f}s: "
              f"primary={r['primary']}, tertiary={r['tertiary_note']}(#{r['note_index']}), "
              f"OG={r['onset_gain']:.2f}, peak={r['attack_peakiness']:.3f}, "
              f"relOG={rel_str}, score={r['score']:.1f}, FR={r['fundamental_ratio']:.3f}")

    print(f"\n=== TRUE_POSITIVE Samples ===")
    for r in true_pos[:40]:
        rel_str = f"{r['relative_onset_gain']:.3f}" if r['relative_onset_gain'] is not None else "N/A"
        print(f"  {r['fixture'].replace('kalimba-17-c-', '')} @{r['time']:.2f}s: "
              f"primary={r['primary']}, tertiary={r['tertiary_note']}(#{r['note_index']}), "
              f"OG={r['onset_gain']:.2f}, peak={r['attack_peakiness']:.3f}, "
              f"relOG={rel_str}, score={r['score']:.1f}, FR={r['fundamental_ratio']:.3f}")

    if len(true_pos) > 40:
        print(f"  ... and {len(true_pos) - 40} more")

    # Metric separation analysis
    print(f"\n=== Metric Separation Analysis ===")
    metrics = [
        ("onset_gain", "onset_gain"),
        ("attack_peakiness", "attack_peakiness"),
        ("relative_onset_gain", "relative_onset_gain"),
        ("score", "score"),
        ("fundamental_ratio", "fundamental_ratio"),
    ]

    for name, key in metrics:
        tp_vals = [r[key] for r in true_pos if r[key] is not None]
        fp_vals = [r[key] for r in false_pos if r[key] is not None]

        if not tp_vals or not fp_vals:
            print(f"\n  {name}: insufficient data (TP={len(tp_vals)}, FP={len(fp_vals)})")
            continue

        d = cohens_d(tp_vals, fp_vals)
        overlap = overlap_ratio(tp_vals, fp_vals)

        print(f"\n  {name}:")
        print(f"    TRUE_POSITIVE:  n={len(tp_vals)}, mean={np.mean(tp_vals):.3f}, "
              f"median={np.median(tp_vals):.3f}, min={min(tp_vals):.3f}, max={max(tp_vals):.3f}")
        print(f"    FALSE_POSITIVE: n={len(fp_vals)}, mean={np.mean(fp_vals):.3f}, "
              f"median={np.median(fp_vals):.3f}, min={min(fp_vals):.3f}, max={max(fp_vals):.3f}")
        print(f"    Cohen's d: {d:.3f}")
        print(f"    Overlap ratio: {overlap:.3f}")

        # Find best separation threshold
        all_vals = sorted(set(tp_vals + fp_vals))
        best_threshold = None
        best_accuracy = 0
        best_tp_recall = 0
        best_fp_reject = 0
        for threshold in all_vals:
            tp_correct = sum(1 for v in tp_vals if v >= threshold)
            fp_correct = sum(1 for v in fp_vals if v < threshold)
            accuracy = (tp_correct + fp_correct) / (len(tp_vals) + len(fp_vals))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_tp_recall = tp_correct / len(tp_vals)
                best_fp_reject = fp_correct / len(fp_vals)

        if best_threshold is not None:
            print(f"    Best threshold: {best_threshold:.3f} "
                  f"(accuracy={best_accuracy:.1%}, TP recall={best_tp_recall:.1%}, "
                  f"FP rejection={best_fp_reject:.1%})")


if __name__ == "__main__":
    main()
