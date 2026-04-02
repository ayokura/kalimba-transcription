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
    """Optimal ordered matching using dynamic programming.

    Finds the assignment of expected events to detected segments that
    maximizes total match score, subject to:
    - Each detected segment used at most once
    - Matched segment indices are strictly increasing (monotonic in time)
    - Expected events can be skipped (NO MATCH, score 0)

    This avoids the greedy pitfall where an early partial match consumes
    a segment that would have been an exact match for a later event.
    """
    n_exp = len(expected_events)
    n_det = len(detected_segs)

    def _score(exp, det):
        if exp["notes"] == det["notes"]:
            return 2
        if exp["notes"] & det["notes"]:
            return 1
        return 0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(ei, min_di):
        """Best total score for matching expected[ei:] using detected[min_di:]."""
        if ei >= n_exp:
            return 0
        # Option: skip this expected event
        best = dp(ei + 1, min_di)
        # Option: match to some detected[di]
        for di in range(min_di, n_det):
            s = _score(expected_events[ei], detected_segs[di])
            if s > 0:
                total = s + dp(ei + 1, di + 1)
                if total > best:
                    best = total
        return best

    def reconstruct(ei, min_di):
        if ei >= n_exp:
            return []
        skip_score = dp(ei + 1, min_di)
        best_total = skip_score
        best_di = None
        for di in range(min_di, n_det):
            s = _score(expected_events[ei], detected_segs[di])
            if s > 0:
                total = s + dp(ei + 1, di + 1)
                if total > best_total:
                    best_total = total
                    best_di = di
        if best_di is not None:
            return [(ei, best_di)] + reconstruct(ei + 1, best_di + 1)
        return [(ei, None)] + reconstruct(ei + 1, min_di)

    assignment = reconstruct(0, 0)

    results = []
    used_det = set()
    for ei, di in assignment:
        if di is not None:
            results.append((expected_events[ei], detected_segs[di]))
            used_det.add(di)
        else:
            results.append((expected_events[ei], None))

    unmatched_det = [detected_segs[di] for di in range(n_det) if di not in used_det]
    return results, unmatched_det


def main():
    parser = argparse.ArgumentParser(description="Score structure alignment diagnosis")
    parser.add_argument("fixture", help="Fixture name or path")
    parser.add_argument("--verbose", action="store_true", help="Show exact matches too")
    parser.add_argument("--line", type=str, default=None, help="Show only this line (e.g., R6)")
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

    # Load alignment overrides (patches for events where recording differs from score)
    overrides_path = fixture_dir / "alignment_overrides.json"
    event_overrides: dict[int, set[str]] = {}
    if overrides_path.exists():
        overrides_data = json.loads(overrides_path.read_text())
        for ov in overrides_data.get("overrides", []):
            event_overrides[ov["eventIndex"]] = set(ov["expectedNotes"])

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
             "trail": s.get("secondaryDecisionTrail", []), "duration": s.get("durationSec", 0),
             "source": s.get("segmentSource", []), "mergeReason": s.get("mergeReason", ""),
             "mergedFrom": s.get("mergedFrom"),
             "droppedBy": s.get("droppedBy", "")}
            for s in segments if start <= s["startTime"] < end and s.get("selectedNotes")
        ]

    total_exact = 0
    total_subset = 0
    total_partial = 0
    total_miss = 0
    total_events = 0

    filter_line = args.line.upper() if args.line else None

    for line in lines:
        line_id = line["id"]
        if filter_line and line_id != filter_line:
            continue
        expected_events = parse_content(line["content"])
        for i, evt in enumerate(expected_events):
            evt["num"] = line["eventRange"][0] + i
            if evt["num"] in event_overrides:
                evt["notes"] = event_overrides[evt["num"]]
                evt["overridden"] = True

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

        _blank_time = " " * 8
        for exp, seg in results:
            if seg is None:
                print(f"  ∅ {_blank_time} E{exp['num']:3d} {exp['content']:30s} → NO MATCH")
                continue

            det_notes = seg["notes"]
            exp_notes = exp["notes"]
            time_str = f"{seg['time']:7.2f}s"

            if exp_notes == det_notes:
                if args.verbose:
                    if exp.get("overridden"):
                        ov_notes = "+".join(sorted(exp_notes))
                        print(f"  ✓ {time_str} E{exp['num']:3d} {exp['content']:30s} (override: {ov_notes})")
                    else:
                        print(f"  ✓ {time_str} E{exp['num']:3d} {exp['content']}")
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

            src_str = f" [{','.join(seg['source'])}]" if seg.get('source') else ""
            print(f"  {sym} {time_str} E{exp['num']:3d} {exp['content']:30s}→ {'+'.join(sorted(det_notes)):15s}{miss_str}{extra_str}{trail_str}{src_str}")

        if unmatched:
            print(f"  +{len(unmatched)} extra segments:")
            for u in unmatched:
                notes_str = '+'.join(sorted(u['notes'])) if u['notes'] else '(empty)'
                src_str = f" [{','.join(u['source'])}]" if u.get('source') else ""
                merge_str = f" ({u['mergeReason']})" if u.get('mergeReason') else ""
                print(f"    t={u['time']:.3f}s {notes_str}{src_str}{merge_str}")

    active_segment_count = sum(1 for s in segments if s.get("selectedNotes"))
    dropped_segment_count = len(segments) - active_segment_count
    print(f"\n{'='*60}")
    line_note = f" (filtered: {filter_line})" if filter_line else ""
    print(f"SUMMARY{line_note}: {total_events} expected events, {active_segment_count} active segments ({dropped_segment_count} dropped)")
    if total_events:
        print(f"  Exact match:   {total_exact:3d}/{total_events} ({100*total_exact/total_events:.0f}%)")
        print(f"  Subset match:  {total_subset:3d}/{total_events} ({100*total_subset/total_events:.0f}%)")
        print(f"  Partial match: {total_partial:3d}/{total_events} ({100*total_partial/total_events:.0f}%)")
        print(f"  No match:      {total_miss:3d}/{total_events} ({100*total_miss/total_events:.0f}%)")


if __name__ == "__main__":
    main()
