"""Diagnose secondary note rejection patterns for bwv147-163 fixture."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "apps" / "api" / "tests"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

from manual_capture_helpers import build_evaluation_audio_bytes, load_fixture
from fastapi.testclient import TestClient
from app.main import app

FIXTURE_DIR = TESTS_DIR / "fixtures" / "manual-captures" / "kalimba-17-c-bwv147-sequence-163-01"


def main():
    client = TestClient(app)
    request_payload, expected = load_fixture(FIXTURE_DIR)
    audio_bytes = build_evaluation_audio_bytes(FIXTURE_DIR, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()

    events = payload["events"]
    debug = payload.get("debug", {})
    segment_candidates = debug.get("segmentCandidates", [])

    print(f"=== bwv147-163 Diagnostic ===")
    print(f"Total events: {len(events)}")
    print(f"Total segments analyzed: {len(segment_candidates)}")

    # Count multi-note vs single-note events
    multi_note = [e for e in events if len(e.get("notes", [])) > 1]
    single_note = [e for e in events if len(e.get("notes", [])) == 1]
    print(f"Multi-note events: {len(multi_note)}")
    print(f"Single-note events: {len(single_note)}")

    # Analyze secondary decision trails
    reason_counter = Counter()
    rejection_details = defaultdict(list)

    for seg in segment_candidates:
        trail = seg.get("secondaryDecisionTrail", [])
        start_t = seg.get("startTime", 0)
        selected = seg.get("selectedNotes", [])
        primary = selected[0] if selected else "?"
        source = seg.get("segmentSource", [])

        for entry in trail:
            if not entry.get("accepted", False):
                reasons = entry.get("reasons", [])
                note = entry.get("noteName", "?")
                score = entry.get("score", 0)
                fr = entry.get("fundamentalRatio", 0)
                og = entry.get("onsetGain")
                key = tuple(sorted(reasons))
                reason_counter[key] += 1
                rejection_details[key].append({
                    "time": start_t,
                    "primary": primary,
                    "rejected": note,
                    "score": score,
                    "fundamentalRatio": fr,
                    "onsetGain": og,
                    "scoreRatio": score / max(seg.get("secondaryDecisionTrail", [{}])[0].get("score", 1) if trail else 1, 0.001),
                    "primaryScore": next((c.get("score", 0) for c in seg.get("rankedCandidates", []) if c.get("noteName") == primary), 0),
                    "source": source,
                })

    print(f"\n=== Rejection Reason Summary ===")
    for reasons, count in reason_counter.most_common():
        print(f"  {' + '.join(reasons)}: {count}")

    # Show details for the most relevant rejection patterns
    for pattern in [
        ("recent-carryover-candidate",),
        ("score-below-threshold",),
        ("harmonic-related-to-selected",),
        ("fundamental-ratio-too-low",),
        ("score-below-threshold", "recent-carryover-candidate"),
        ("fundamental-ratio-too-low", "harmonic-related-to-selected"),
        ("fundamental-ratio-too-low", "recent-carryover-candidate"),
        ("score-below-threshold", "fundamental-ratio-too-low"),
        ("score-below-threshold", "harmonic-related-to-selected"),
    ]:
        key = tuple(sorted(pattern))
        details = rejection_details.get(key, [])
        if not details:
            continue
        print(f"\n=== {' + '.join(key)} ({len(details)} cases) ===")
        for d in details[:10]:
            og_str = f", onsetGain={d['onsetGain']:.1f}" if d['onsetGain'] is not None else ""
            src_str = f" [{','.join(d['source'])}]" if d.get('source') else ""
            print(f"  @{d['time']:.1f}s: primary={d['primary']}, rejected={d['rejected']}, "
                  f"score={d['score']:.0f}, FR={d['fundamentalRatio']:.3f}{og_str}{src_str}")

    # Show all segments where we got only 1 note but might expect more
    print(f"\n=== Single-note segments with strong rejected candidates ===")
    for seg in segment_candidates:
        selected = seg.get("selectedNotes", [])
        if len(selected) != 1:
            continue
        trail = seg.get("secondaryDecisionTrail", [])
        start_t = seg.get("startTime", 0)
        for entry in trail:
            if entry.get("accepted"):
                continue
            score = entry.get("score", 0)
            fr = entry.get("fundamentalRatio", 0)
            # Show cases where secondary had decent score or fundamental ratio
            primary_score = 0
            for c in seg.get("rankedCandidates", []):
                if c.get("noteName") == selected[0]:
                    primary_score = c.get("score", 0)
                    break
            if primary_score > 0 and score > primary_score * 0.05 and fr > 0.1:
                reasons = entry.get("reasons", [])
                og = entry.get("onsetGain")
                og_str = f", onsetGain={og:.1f}" if og is not None else ""
                source = seg.get("segmentSource", [])
                src_str = f" [{','.join(source)}]" if source else ""
                print(f"  @{start_t:.1f}s: primary={selected[0]}(score={primary_score:.0f}), "
                      f"rejected={entry['noteName']}(score={score:.0f}, FR={fr:.3f}{og_str}), "
                      f"reasons={reasons}{src_str}")


if __name__ == "__main__":
    main()
