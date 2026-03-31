"""Diagnose bounded candidate-pool coverage and dry combo scoring for chord-like misses.

Usage:
    uv run python scripts/audio-analysis/chord_pool_diagnosis.py <fixture-name> [--events 112,136]

This script:
1. Aligns expected score_structure events to detected segmentCandidates using the same
   DP ordered matching as score_alignment_diagnosis.py.
2. For target multi-note mismatches, inspects whether expected notes are present in a
   bounded union of:
   - initial ranking
   - residual after suppressing primary
   - residual after suppressing primary + first residual winner
3. Optionally ranks physically playable note combinations on that bounded pool with a
   few simple scoring families.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TESTS_DIR = REPO_ROOT / "apps" / "api" / "tests"

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

from fastapi.testclient import TestClient
from manual_capture_helpers import build_evaluation_audio_bytes, load_fixture

from app.main import app
from app.transcription.peaks import (
    harmonic_relation_multiple,
    is_physically_playable_chord,
    rank_tuning_candidates,
    suppress_harmonics,
)
from app.tunings import get_default_tunings

from score_alignment_diagnosis import build_time_converter, match_line, parse_content, resolve_fixture


POOL_LIMITS = (8, 10, 12, 14, 16, 20)
FORMULA_NAMES = ("sum", "support_alias", "sum_pairpen", "support_pairpen")


def load_alignment_overrides(fixture_dir: Path) -> dict[int, set[str]]:
    overrides_path = fixture_dir / "alignment_overrides.json"
    if not overrides_path.exists():
        return {}
    data = json.loads(overrides_path.read_text(encoding="utf-8"))
    return {item["eventIndex"]: set(item["expectedNotes"]) for item in data.get("overrides", [])}


def parse_event_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    items = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return sorted(set(items))


def score_sum(combo) -> float:
    return sum(item.score for item in combo)


def score_support_alias(combo) -> float:
    return sum(
        (item.fundamental_energy + 0.5 * item.overtone_energy) * (0.25 + 0.75 * item.fundamental_ratio)
        - 0.7 * item.subharmonic_alias_energy
        - item.octave_alias_penalty
        for item in combo
    )


def pair_penalty(combo) -> float:
    penalty = 0.0
    for left, right in itertools.combinations(combo, 2):
        multiple = harmonic_relation_multiple(left.candidate, right.candidate)
        if multiple is None:
            continue
        if multiple == 2.0:
            penalty += 0.35 * min(left.score, right.score)
        else:
            penalty += 0.8 * min(left.score, right.score)
    return penalty


def score_sum_pairpen(combo) -> float:
    return score_sum(combo) - pair_penalty(combo)


def score_support_pairpen(combo) -> float:
    return score_support_alias(combo) - pair_penalty(combo)


SCORERS = {
    "sum": score_sum,
    "support_alias": score_support_alias,
    "sum_pairpen": score_sum_pairpen,
    "support_pairpen": score_support_pairpen,
}


def collect_matched_events(fixture_dir: Path) -> tuple[dict[int, dict], np.ndarray, int, dict]:
    request_payload, expected = load_fixture(fixture_dir)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    client = TestClient(app)
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    payload = response.json()
    segments = payload["debug"]["segmentCandidates"]

    score_data = json.loads((fixture_dir / "score_structure.json").read_text(encoding="utf-8"))
    overrides = load_alignment_overrides(fixture_dir)
    orig_to_eval = build_time_converter(expected)

    matched: dict[int, dict] = {}
    for index, line in enumerate(score_data["lines"]):
        start = orig_to_eval(line.get("verifiedStartSec") or line["estimatedStartSec"])
        if index + 1 < len(score_data["lines"]):
            next_line = score_data["lines"][index + 1]
            end = orig_to_eval(next_line.get("verifiedStartSec") or next_line["estimatedStartSec"])
        else:
            end = 999.0
        detected = [
            {
                "notes": set(segment.get("selectedNotes", [])),
                "startTime": segment["startTime"],
                "endTime": segment["endTime"],
                "raw": segment,
            }
            for segment in segments
            if start <= segment["startTime"] < end and segment.get("selectedNotes")
        ]
        expected_events = parse_content(line["content"])
        for offset, event in enumerate(expected_events):
            event_num = line["eventRange"][0] + offset
            event["num"] = event_num
            if event_num in overrides:
                event["notes"] = overrides[event_num]
        results, _ = match_line(expected_events, detected)
        for expected_event, detected_segment in results:
            if detected_segment is None:
                continue
            matched[expected_event["num"]] = {
                "expectedNotes": sorted(expected_event["notes"]),
                "detectedNotes": sorted(detected_segment["notes"]),
                "startTime": detected_segment["startTime"],
                "endTime": detected_segment["endTime"],
                "segment": detected_segment["raw"],
            }

    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return matched, audio, sample_rate, request_payload["tuning"]


def choose_target_events(matched: dict[int, dict], explicit_events: list[int] | None) -> list[int]:
    if explicit_events is not None:
        return [event_num for event_num in explicit_events if event_num in matched]
    targets = []
    for event_num, item in sorted(matched.items()):
        expected = set(item["expectedNotes"])
        detected = set(item["detectedNotes"])
        if len(expected) < 2:
            continue
        if expected != detected:
            targets.append(event_num)
    return targets


def pool_by_limit(ranked, residual1, residual2) -> dict[int, set[str]]:
    results: dict[int, set[str]] = {}
    for limit in POOL_LIMITS:
        pool = set(h.candidate.note_name for h in ranked[:limit])
        pool |= set(h.candidate.note_name for h in residual1[:limit])
        pool |= set(h.candidate.note_name for h in residual2[:limit])
        results[limit] = pool
    return results


def best_entries_for_limit(ranked, residual1, residual2, limit: int) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for source_name, hypotheses in (
        ("initial", ranked[:limit]),
        ("residual1", residual1[:limit]),
        ("residual2", residual2[:limit]),
    ):
        for hypothesis in hypotheses:
            note_name = hypothesis.candidate.note_name
            current = best.get(note_name)
            if current is None or hypothesis.score > current["hypothesis"].score:
                best[note_name] = {"hypothesis": hypothesis, "source": source_name}
    return best


def rank_combinations(best_entries: dict[str, dict], target_notes: set[str], tuning, max_polyphony: int) -> dict[str, dict]:
    pool = list(best_entries.values())
    combos = []
    for size in range(2, min(max_polyphony, len(pool)) + 1):
        for subset in itertools.combinations(pool, size):
            hypotheses = [item["hypothesis"] for item in subset]
            keys = [item.candidate.key for item in hypotheses]
            if not is_physically_playable_chord(keys):
                continue
            combos.append(hypotheses)

    results = {}
    for name, scorer in SCORERS.items():
        ordered = sorted(combos, key=scorer, reverse=True)
        top = [
            {
                "rank": index,
                "notes": [item.candidate.note_name for item in combo],
                "score": round(scorer(combo), 2),
            }
            for index, combo in enumerate(ordered[:5], start=1)
        ]
        target_rank = None
        for index, combo in enumerate(ordered, start=1):
            if {item.candidate.note_name for item in combo} == target_notes:
                target_rank = index
                break
        results[name] = {"targetRank": target_rank, "top": top}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose chord candidate pools and simple combo scoring")
    parser.add_argument("fixture", help="Fixture name or path")
    parser.add_argument("--events", help="Comma-separated expected event numbers to inspect")
    parser.add_argument("--combo-limit", type=int, default=20, help="Bounded pool limit used for combo ranking")
    parser.add_argument("--max-polyphony", type=int, default=4, help="Maximum combo size to enumerate")
    parser.add_argument("--skip-combos", action="store_true", help="Only report pool completeness")
    args = parser.parse_args()

    fixture_dir = resolve_fixture(args.fixture)
    matched, audio, sample_rate, tuning_payload = collect_matched_events(fixture_dir)
    tuning = next(item for item in get_default_tunings() if item.id == tuning_payload["id"])
    event_numbers = choose_target_events(matched, parse_event_list(args.events))
    if not event_numbers:
        print("No matching target events found.")
        return

    fft_size = 8192
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_size)

    for event_num in event_numbers:
        item = matched[event_num]
        start = int(item["startTime"] * sample_rate)
        end = int(item["endTime"] * sample_rate)
        chunk = audio[start:end]
        spectrum = np.abs(librosa.stft(chunk, n_fft=fft_size, hop_length=max(1, len(chunk)), center=False)).mean(axis=1)

        ranked = rank_tuning_candidates(spectrum, frequencies, tuning, debug=False)
        primary = ranked[0]
        residual1_spec = suppress_harmonics(spectrum, frequencies, primary.candidate.frequency)
        residual1 = rank_tuning_candidates(residual1_spec, frequencies, tuning, debug=False)
        secondary = residual1[0]
        residual2_spec = suppress_harmonics(residual1_spec, frequencies, secondary.candidate.frequency)
        residual2 = rank_tuning_candidates(residual2_spec, frequencies, tuning, debug=False)

        pools = pool_by_limit(ranked, residual1, residual2)

        print(f"=== E{event_num}")
        print(f"expected={item['expectedNotes']} detected={item['detectedNotes']}")
        print("pool completeness:")
        for limit in POOL_LIMITS:
            missing = [note for note in item["expectedNotes"] if note not in pools[limit]]
            print(f"  top{limit}: missing={missing}")

        if args.skip_combos:
            print()
            continue

        best_entries = best_entries_for_limit(ranked, residual1, residual2, args.combo_limit)
        target_notes = set(item["expectedNotes"])
        target_present = target_notes <= set(best_entries.keys())
        print(f"combo pool: limit=top{args.combo_limit}, unique_notes={len(best_entries)}, target_present={target_present}")

        playable_counts = {2: 0, 3: 0, 4: 0}
        note_lookup = {note.note_name: note for note in tuning.notes}
        for size in range(2, min(args.max_polyphony, len(best_entries)) + 1):
            for subset in itertools.combinations(best_entries.keys(), size):
                if is_physically_playable_chord([note_lookup[name].key for name in subset]):
                    playable_counts[size] += 1
        print(f"playable combos: {playable_counts}")

        combo_results = rank_combinations(best_entries, target_notes, tuning, args.max_polyphony)
        for name in FORMULA_NAMES:
            result = combo_results[name]
            print(f"  {name}: target_rank={result['targetRank']}")
            for top_item in result["top"]:
                print(f"    {top_item['rank']}. {top_item['notes']} score={top_item['score']}")
        print()


if __name__ == "__main__":
    main()
