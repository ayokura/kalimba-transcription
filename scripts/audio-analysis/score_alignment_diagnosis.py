"""Compare expected events from score_structure with recognizer output using ordered matching.

Usage:
    uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture-name> [--verbose]

Arguments:
    fixture-name    Fixture name (e.g., bwv147-sequence-163-01)
    --verbose       Show exact matches too (default: failures only)
"""
import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TESTS_DIR = REPO_ROOT / "apps" / "api" / "tests"
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "manual-captures"

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

from manual_capture_helpers import build_evaluation_audio_bytes, load_fixture
from fastapi.testclient import TestClient
from app.main import app


def resolve_fixture(name: str) -> Path:
    """Resolve fixture name to directory path."""
    direct = FIXTURE_ROOT / name
    if direct.is_dir():
        return direct
    prefixed = FIXTURE_ROOT / f"kalimba-17-c-{name}"
    if prefixed.is_dir():
        return prefixed
    for d in FIXTURE_ROOT.iterdir():
        if d.is_dir() and name in d.name:
            return d
    print(f"Error: Cannot resolve fixture: {name}", file=sys.stderr)
    sys.exit(1)


def parse_content(content: str) -> list[dict]:
    """Parse score_structure content string into list of event dicts."""
    events = []
    for item in content.split(" / "):
        item = item.strip()
        notes = set(re.findall(r'[A-G]#?[0-9]', item))
        events.append({"content": item, "notes": notes})
    return events


def build_time_converter(expected: dict):
    """Build a function to convert original audio time to evaluation audio time.

    When evaluationWindows is present, the evaluation audio is the concatenation
    of those windows only, so the mapping is: each window maps to a contiguous
    block starting where the previous window ended in cropped time.

    When only ignoredRanges is present, the evaluation audio is the original
    minus the ignored ranges.
    """
    windows = expected.get("evaluationWindows", [])
    if windows:
        # evaluationWindows: audio = concat of windows in order
        sorted_wins = sorted(windows, key=lambda w: w["startSec"])
        cropped_offset = 0.0
        mapping = []  # (orig_start, orig_end, cropped_start)
        for w in sorted_wins:
            mapping.append((w["startSec"], w["endSec"], cropped_offset))
            cropped_offset += w["endSec"] - w["startSec"]

        def convert(t: float) -> float:
            for orig_start, orig_end, crop_start in mapping:
                if orig_start <= t < orig_end:
                    return crop_start + (t - orig_start)
            # Outside all windows: clamp to nearest boundary
            if mapping and t >= mapping[-1][1]:
                last = mapping[-1]
                return last[2] + (last[1] - last[0])
            return 0.0

        return convert

    ignored = []
    for r in expected.get("ignoredRanges", []):
        ignored.append((r["startSec"], r["endSec"]))
    ignored.sort()

    def convert(t: float) -> float:
        offset = 0.0
        for start, end in ignored:
            if t > end:
                offset += end - start
            elif t > start:
                return start - offset
        return t - offset

    return convert


def match_line(expected_events, detected_segs):
    """Ordered greedy matching: for each expected event, find best overlapping segment."""
    used_det = set()
    results = []

    for exp in expected_events:
        best_di = None
        best_score = -1

        for di, det in enumerate(detected_segs):
            if di in used_det:
                continue
            if exp["notes"] == det["notes"]:
                score = 2
            elif exp["notes"] & det["notes"]:
                score = 1
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_di = di

            furthest = max(used_det) if used_det else -1
            if di > furthest + len(expected_events) + 3:
                break

        if best_di is not None and best_score > 0:
            used_det.add(best_di)
            results.append((exp, detected_segs[best_di]))
        else:
            results.append((exp, None))

    unmatched_det = [detected_segs[di] for di in range(len(detected_segs)) if di not in used_det]
    return results, unmatched_det


def main():
    parser = argparse.ArgumentParser(description="Score structure alignment diagnosis")
    parser.add_argument("fixture", help="Fixture name or path")
    parser.add_argument("--verbose", action="store_true", help="Show exact matches too")
    args = parser.parse_args()

    fixture_dir = resolve_fixture(args.fixture)
    score_path = fixture_dir / "score_structure.json"
    if not score_path.exists():
        print(f"Error: {fixture_dir.name} has no score_structure.json", file=sys.stderr)
        sys.exit(1)

    client = TestClient(app)
    request_payload, expected = load_fixture(fixture_dir)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    if response.status_code != 200:
        print(f"Error: transcription request failed with status {response.status_code}", file=sys.stderr)
        print(response.text[:500], file=sys.stderr)
        sys.exit(1)
    payload = response.json()
    debug = payload.get("debug")
    if not isinstance(debug, dict) or "segmentCandidates" not in debug:
        print(f"Error: response missing debug.segmentCandidates (keys: {list(payload.keys())})", file=sys.stderr)
        sys.exit(1)
    segments = debug["segmentCandidates"]
    score_data = json.loads(score_path.read_text())
    lines = score_data["lines"]

    orig_to_eval = build_time_converter(expected)

    # Classify segments by line time boundaries
    line_bounds = []
    for i, line in enumerate(lines):
        start = orig_to_eval(line.get("verifiedStartSec") or line["estimatedStartSec"])
        if i + 1 < len(lines):
            end = orig_to_eval(lines[i + 1].get("verifiedStartSec") or lines[i + 1]["estimatedStartSec"])
        else:
            end = 999.0
        line_bounds.append((line["id"], start, end))

    seg_by_line = {}
    for name, start, end in line_bounds:
        seg_by_line[name] = [
            {"notes": set(s.get("selectedNotes", [])), "time": s["startTime"],
             "trail": s.get("secondaryDecisionTrail", []), "duration": s.get("durationSec", 0)}
            for s in segments if start <= s["startTime"] < end
        ]

    total_exact = 0
    total_subset = 0
    total_partial = 0
    total_miss = 0
    total_events = 0

    for line in lines:
        line_id = line["id"]
        expected_events = parse_content(line["content"])
        for i, evt in enumerate(expected_events):
            evt["num"] = line["eventRange"][0] + i

        detected = seg_by_line.get(line_id, [])
        results, unmatched = match_line(expected_events, detected)

        n = len(expected_events)
        exact = sum(1 for e, s in results if s and s["notes"] == e["notes"])
        subset = sum(1 for e, s in results if s and s["notes"] < e["notes"] and s["notes"])
        partial = sum(1 for e, s in results if s and (e["notes"] & s["notes"]) and s["notes"] != e["notes"] and not s["notes"] < e["notes"])
        miss = sum(1 for _, s in results if s is None)

        total_exact += exact
        total_subset += subset
        total_partial += partial
        total_miss += miss
        total_events += n

        pct = 100 * exact / n if n else 0
        pct2 = 100 * (exact + subset) / n if n else 0

        print(f"\n=== {line_id} ({n} exp, {len(detected)} det) exact={exact} subset={subset} partial={partial} miss={miss} ({pct:.0f}% exact, {pct2:.0f}% +sub) ===")

        for exp, seg in results:
            if seg is None:
                print(f"  ∅ E{exp['num']:3d} {exp['content']:30s} → NO MATCH")
                continue

            det_notes = seg["notes"]
            exp_notes = exp["notes"]

            if exp_notes == det_notes:
                if args.verbose:
                    print(f"  ✓ E{exp['num']:3d} {exp['content']}")
                continue

            missing = exp_notes - det_notes
            extra = det_notes - exp_notes

            # Look up rejection reasons for missing notes
            trail_parts = []
            for m in sorted(missing):
                for entry in seg["trail"]:
                    if entry["noteName"] == m and not entry["accepted"]:
                        trail_parts.append(f"{m}→{','.join(entry['reasons'][:2])}")
                        break
            trail_str = " " + " ".join(f"[{t}]" for t in trail_parts) if trail_parts else ""

            miss_str = f" miss={'+'.join(sorted(missing))}" if missing else ""
            extra_str = f" extra={'+'.join(sorted(extra))}" if extra else ""

            if not missing and extra:
                sym = "⊃"
            elif missing and not extra:
                sym = "⊂"
            elif missing and extra and (exp_notes & det_notes):
                sym = "△"
            else:
                sym = "✗"

            print(f"  {sym} E{exp['num']:3d} {exp['content']:30s} → {'+'.join(sorted(det_notes)):15s}{miss_str}{extra_str}{trail_str}")

        if unmatched:
            print(f"  +{len(unmatched)} extra segments")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_events} expected events, {len(segments)} segments")
    print(f"  Exact match:   {total_exact:3d}/{total_events} ({100*total_exact/total_events:.0f}%)")
    print(f"  Subset match:  {total_subset:3d}/{total_events} ({100*total_subset/total_events:.0f}%)")
    print(f"  Partial match: {total_partial:3d}/{total_events} ({100*total_partial/total_events:.0f}%)")
    print(f"  No match:      {total_miss:3d}/{total_events} ({100*total_miss/total_events:.0f}%)")


if __name__ == "__main__":
    main()
