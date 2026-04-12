"""Compare expected events with recognizer output using ordered matching.

Usage:
    uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture-name> [--verbose] [--mode events|segments] [--line LINE_ID]

Arguments:
    fixture-name    Fixture name (e.g., bwv147-sequence-163-01, c4-repeat-01)
    --verbose       Show exact matches too (default: failures only)
    --mode          Data source: 'events' (default, post-processed final output)
                    or 'segments' (raw segmentCandidates before event post-processing)
    --line          Show only the specified line id (e.g., R6). All lines by default.

Expected event source priority:
    1. score_structure.json — multi-line score (e.g. BWV147 sequence-163).
       Used when present for per-line matching.
    2. request.json:expectedPerformance.events — Web UI's clickable-kalimba-ui
       ordered event list. Set on every captured fixture. Synthesized into
       a single line `ALL` so the script also works on simple fixtures
       (e.g. c4-repeat-01, mixed-sequence-01).
    3. expected.json:assertions.expectedEventNoteSetsOrdered — last-resort
       fallback. Only set on promoted (completed) fixtures; derived from
       recognizer output, so it requires the recognizer to already match
       the truth.

    The script logs which source was used via stderr
    `[synthetic-line] using <source> (N events)` when falling back.

Environment:
    SCORE_ALIGNMENT_NO_CACHE=1  Disable the on-disk transcription cache
                                (apps/api/tests/.cache/score_alignment/). When set,
                                transcription requests always re-run the full pipeline.

Cache invalidation:
    The cache key includes a fingerprint of all .py files under
    apps/api/app/transcription/, so editing recognizer code automatically
    invalidates stale entries on the next run. Reverting an edit restores
    the original cache hit. Audio bytes and request data are also part of
    the key. Old entries accumulate over time and can be cleaned manually:
        rm -rf apps/api/tests/.cache/score_alignment/
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TESTS_DIR = REPO_ROOT / "apps" / "api" / "tests"
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "manual-captures"
CACHE_DIR = TESTS_DIR / ".cache" / "score_alignment"
RECOGNIZER_PKG_DIR = REPO_ROOT / "apps" / "api" / "app" / "transcription"

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

from manual_capture_helpers import build_evaluation_audio_bytes, load_fixture
from fastapi.testclient import TestClient
from app.main import app


@lru_cache(maxsize=1)
def _recognizer_code_fingerprint() -> str:
    """Hash all .py files under apps/api/app/transcription/.

    Included in the cache key so that any recognizer code edit automatically
    invalidates stale cache entries on the next run. Reverting the edit
    restores the original cache hit. Computed once per script invocation.
    """
    h = hashlib.sha256()
    for path in sorted(RECOGNIZER_PKG_DIR.rglob("*.py")):
        h.update(path.relative_to(RECOGNIZER_PKG_DIR).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _cache_key(audio_bytes: bytes, request_data: dict[str, str]) -> str:
    """Compute cache key from audio bytes + request data + recognizer fingerprint."""
    h = hashlib.sha256()
    h.update(audio_bytes)
    h.update(
        json.dumps(sorted(request_data.items()), ensure_ascii=False).encode("utf-8")
    )
    h.update(_recognizer_code_fingerprint().encode("utf-8"))
    return h.hexdigest()


_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    """Return True iff the env var is set to an explicit truthy value.

    Accepts "1", "true", "yes", "on" (case-insensitive). Anything else,
    including "0", "false", and the empty string, is treated as False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _cached_transcribe(
    client: TestClient, audio_bytes: bytes, request_data: dict[str, str]
) -> dict:
    """Run transcription with disk cache of the JSON response.

    Set ``SCORE_ALIGNMENT_NO_CACHE=1`` (or ``true``/``yes``/``on``) to bypass
    the cache entirely. Other values, including ``0`` and ``false``, leave
    the cache enabled.
    """
    cache_disabled = _env_flag("SCORE_ALIGNMENT_NO_CACHE")
    key = _cache_key(audio_bytes, request_data)
    key_prefix = key[:12]
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_disabled and cache_file.exists():
        invalid_reason: str | None = None
        payload: object = None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid_reason = str(exc)
        else:
            if not isinstance(payload, dict):
                invalid_reason = (
                    f"expected JSON object, got {type(payload).__name__}"
                )
        if invalid_reason is not None:
            print(
                f"[cache invalid] {key_prefix}: {invalid_reason}; treating as cache miss",
                file=sys.stderr,
            )
            try:
                cache_file.unlink()
            except OSError:
                pass
        else:
            print(f"[cache hit] {key_prefix}", file=sys.stderr)
            return payload

    print(f"[cache miss] {key_prefix}", file=sys.stderr)
    response = client.post(
        "/api/transcriptions",
        data=request_data,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    if response.status_code != 200:
        print(
            f"Error: transcription request failed with status {response.status_code}",
            file=sys.stderr,
        )
        print(response.text[:500], file=sys.stderr)
        sys.exit(1)
    raw_text = response.text
    payload = json.loads(raw_text)
    if not cache_disabled:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to a uniquely-named sibling temp file (mkstemp
            # guarantees no collision between concurrent invocations) and rename
            # into place so an interrupted run cannot leave a truncated cache
            # entry behind.
            fd, tmp_path_str = tempfile.mkstemp(
                dir=CACHE_DIR, prefix=f"{key}.", suffix=".tmp"
            )
            tmp_path = Path(tmp_path_str)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(raw_text)
                os.replace(tmp_path, cache_file)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        except OSError as exc:
            print(
                f"[cache write skipped] {key_prefix}: {exc}",
                file=sys.stderr,
            )
    return payload


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
    parser.add_argument("--mode", choices=["events", "segments"], default="events",
                        help="Data source: 'events' (post-processed, default) or 'segments' (raw segmentCandidates)")
    parser.add_argument("--line", type=str, default=None, help="Show only this line (e.g., R6)")
    args = parser.parse_args()

    fixture_dir = resolve_fixture(args.fixture)
    score_path = fixture_dir / "score_structure.json"

    client = TestClient(app)
    request_payload, expected = load_fixture(fixture_dir)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    request_data = {"tuning": json.dumps(request_payload["tuning"]), "debug": "true"}
    payload = _cached_transcribe(client, audio_bytes, request_data)
    debug = payload.get("debug")
    if not isinstance(debug, dict) or "segmentCandidates" not in debug:
        print(f"Error: response missing debug.segmentCandidates (keys: {list(payload.keys())})", file=sys.stderr)
        sys.exit(1)
    segments = debug["segmentCandidates"]
    merged_events = debug.get("mergedEvents", [])
    use_events_mode = args.mode == "events"

    if score_path.exists():
        score_data = json.loads(score_path.read_text())
        lines = score_data["lines"]
    else:
        # Fallback: build a synthetic single-line score from a simpler
        # fixture metadata source. This allows the diagnosis script to
        # work on simple fixtures (e.g. c4-repeat-01) that have no
        # multi-line score_structure.json. The synthetic line spans the
        # entire audio and contains all expected events in order.
        #
        # Source priority:
        #   1. request.json:expectedPerformance.events
        #      — the Web UI's clickable-kalimba-ui produces this for every
        #        captured fixture, so it is the most reliable source and the
        #        canonical "what was supposed to be played"
        #   2. expected.json:assertions.expectedEventNoteSetsOrdered
        #      — only set on promoted (completed) fixtures; derived from
        #        recognizer output by `event_note_sets()`, so it requires
        #        the recognizer to already match the truth
        synthetic_events: list[str] | None = None
        source_label: str = ""
        ep = request_payload.get("expectedPerformance") if isinstance(request_payload, dict) else None
        if isinstance(ep, dict):
            ep_events = ep.get("events")
            if isinstance(ep_events, list) and ep_events:
                tokens: list[str] = []
                for ev in ep_events:
                    keys = ev.get("keys") if isinstance(ev, dict) else None
                    if not isinstance(keys, list) or not keys:
                        continue
                    note_set = sorted(
                        {k["noteName"] for k in keys if isinstance(k, dict) and k.get("noteName")},
                        key=lambda n: (int(n[1:]), n[0]),  # octave then pitch class
                    )
                    if note_set:
                        tokens.append("+".join(note_set))
                if tokens:
                    synthetic_events = tokens
                    source_label = "request.json:expectedPerformance"
        if synthetic_events is None:
            ordered = expected.get("assertions", {}).get("expectedEventNoteSetsOrdered")
            if isinstance(ordered, list) and ordered:
                synthetic_events = list(ordered)
                source_label = "expected.json:expectedEventNoteSetsOrdered"
        if synthetic_events is None:
            print(
                f"Error: {fixture_dir.name} has no score_structure.json and no usable "
                f"fallback (request.json:expectedPerformance / expected.json:expectedEventNoteSetsOrdered)",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"[synthetic-line] using {source_label} ({len(synthetic_events)} events)",
            file=sys.stderr,
        )
        synthetic_content = " / ".join(synthetic_events)
        lines = [
            {
                "id": "ALL",
                "eventRange": [1, len(synthetic_events)],
                "content": synthetic_content,
                "estimatedStartSec": 0.0,
            }
        ]

    # Load alignment overrides (patches for events where recording differs from score)
    overrides_path = fixture_dir / "alignment_overrides.json"
    event_overrides: dict[int, set[str]] = {}
    if overrides_path.exists():
        overrides_data = json.loads(overrides_path.read_text())
        for ov in overrides_data.get("overrides", []):
            event_overrides[ov["eventIndex"]] = set(ov["expectedNotes"])

    orig_to_eval = build_time_converter(expected)

    # Build per-segment trail lookup for rejection reason cross-referencing.
    # Each entry carries the segment's selectedNotes so we can distinguish
    # "note was selected at segment level but dropped by post-processing"
    # from "note was rejected at segment level".
    seg_info_by_time: list[tuple[float, list[dict], set[str]]] = [
        (s["startTime"], s.get("secondaryDecisionTrail", []),
         set(s.get("selectedNotes", [])))
        for s in segments if s.get("selectedNotes")
    ]

    def find_trail_for_time(t: float) -> list[dict]:
        """Find the segment trail closest in time to *t*."""
        best_trail: list[dict] = []
        best_dist = float("inf")
        for seg_t, trail, _selected in seg_info_by_time:
            d = abs(seg_t - t)
            if d < best_dist:
                best_dist = d
                best_trail = trail
        return best_trail if best_dist < 0.1 else []

    def _was_selected_in_nearby_segment(t: float, note_name: str) -> bool:
        """Check if note_name was in selectedNotes of any segment near t."""
        for seg_t, _trail, selected in seg_info_by_time:
            if abs(seg_t - t) < 0.2 and note_name in selected:
                return True
        return False

    # Post-processing trace lookup: which pass dropped a note near a given time?
    pp_trace: list[dict] = payload.get("debug", {}).get("postProcessingTrace", [])

    def _find_drop_pass(t: float, note_name: str) -> str | None:
        """Find the post-processing pass that removed a note near time t."""
        for entry in pp_trace:
            for removed in entry.get("removed", []):
                if abs(removed["startTime"] - t) < 0.2 and note_name in removed.get("notes", []):
                    return entry["pass"]
        return None

    # Classify segments by line time boundaries
    line_bounds = []
    for i, line in enumerate(lines):
        start = orig_to_eval(line.get("verifiedStartSec") or line["estimatedStartSec"])
        if i + 1 < len(lines):
            end = orig_to_eval(lines[i + 1].get("verifiedStartSec") or lines[i + 1]["estimatedStartSec"])
        else:
            end = 999.0
        line_bounds.append((line["id"], start, end))

    if use_events_mode:
        # Build detected list from final post-processed events
        seg_by_line = {}
        for name, start, end in line_bounds:
            seg_by_line[name] = [
                {"notes": set(e["notes"]), "time": e["startTime"],
                 "trail": find_trail_for_time(e["startTime"]),
                 "duration": round(e["endTime"] - e["startTime"], 6),
                 "source": [], "mergeReason": "", "mergedFrom": None,
                 "droppedBy": ""}
                for e in merged_events if start <= e["startTime"] < end
            ]
    else:
        # Build detected list from raw segment candidates (pre-post-processing)
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
    total_extras = 0
    total_cosmetic_extras = 0
    total_events = 0

    # An "extra" is "cosmetic" when it's a single-note detection from a
    # very short window (< SHORT_SEGMENT_SECONDARY_GUARD_DURATION = 30 ms).
    # These are recognizer artefacts of the short-segment secondary guard
    # (commit 1f3bda4): a 6-16 ms segment whose spectral content is too
    # narrow for the FFT to resolve secondaries, so the guard preserves
    # only the primary as a tentative singleton.  Most of these primaries
    # are spectral artefacts of nearby real attacks rather than actual
    # played notes.  The metric tracks them separately so suppression
    # progress can be measured without confusing them with structural
    # extras (e.g., E115 spectral leakage that survives both A.2 and A.3).
    COSMETIC_EXTRA_MAX_DURATION = 0.030

    def _is_cosmetic_extra(seg: dict) -> bool:
        notes = seg.get("notes") or set()
        if len(notes) != 1:
            return False
        return seg.get("duration", float("inf")) < COSMETIC_EXTRA_MAX_DURATION

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
        total_extras += len(unmatched)
        total_cosmetic_extras += sum(1 for u in unmatched if _is_cosmetic_extra(u))
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

            # Look up rejection reasons for missing notes.
            # Priority: (1) postProcessingTrace drop pass, (2) secondaryDecisionTrail.
            # If a note was selected at the segment level but dropped by a
            # post-processing merge/suppress, show the pass name instead of
            # the misleading segment-level trail reason.
            trail_parts = []
            for m in sorted(missing):
                drop_pass = _find_drop_pass(seg["time"], m) if use_events_mode else None
                if drop_pass:
                    trail_parts.append(f"{m}→dropped-by:{drop_pass}")
                elif use_events_mode and _was_selected_in_nearby_segment(seg["time"], m):
                    trail_parts.append(f"{m}→dropped-by-post-processing")
                else:
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
                cosmetic_tag = " (cosmetic)" if _is_cosmetic_extra(u) else ""
                print(f"    t={u['time']:.3f}s {notes_str}{src_str}{merge_str}{cosmetic_tag}")

    if use_events_mode:
        det_count = len(merged_events)
        det_label = "final events"
    else:
        det_count = sum(1 for s in segments if s.get("selectedNotes"))
        det_label = f"active segments ({len(segments) - det_count} dropped)"
    print(f"\n{'='*60}")
    mode_note = f" [mode={args.mode}]"
    line_note = f" (filtered: {filter_line})" if filter_line else ""
    print(f"SUMMARY{mode_note}{line_note}: {total_events} expected events, {det_count} {det_label}")
    if total_events:
        print(f"  Exact match:   {total_exact:3d}/{total_events} ({100*total_exact/total_events:.0f}%)")
        print(f"  Subset match:  {total_subset:3d}/{total_events} ({100*total_subset/total_events:.0f}%)")
        print(f"  Partial match: {total_partial:3d}/{total_events} ({100*total_partial/total_events:.0f}%)")
        print(f"  No match:      {total_miss:3d}/{total_events} ({100*total_miss/total_events:.0f}%)")
        real_extras = total_extras - total_cosmetic_extras
        print(f"  Extra segments: {total_extras} ({real_extras} real + {total_cosmetic_extras} cosmetic <30ms guard artefacts)")


if __name__ == "__main__":
    main()
