"""D1 (4th term): mechanism taxonomy of every FN in the non-saturated set.

Classifies each ground-truth false negative by which pipeline stage lost
it, walking the debug payload the way #196 / #197 investigations did by
hand — but for the whole non-saturated set at once. The taxonomy drives
D2 target selection (largest bucket wins, not the author's favorite) and
the E2 recording-kit specs (which FN regimes to record against).

Buckets (first match in pipeline order):
  no-broadband-onset   no detected onset within tol of the GT time
                       (segments.py never saw it -> onset-detection regime)
  no-slot              onset exists but no candidate slot / event covers
                       the time (segment formation dropped it)
  slot-dropped         a covering candidateSlot exists but was dropped
                       (dropReason recorded; the per-tine oracle's regime)
  not-in-candidates    a surviving slot/event covers it but the GT note is
                       not among its candidates (peak detection/scoring)
  candidate-not-chosen the GT note IS a candidate of a covering slot/event
                       but not in the 1-best output (#178 candidate-tier
                       recoverable band)
  time-mismatch        the note appears in the 1-best output but outside
                       tol (timing regime, cf. #201 semantics)

Outputs:
  docs/research/fn-taxonomy-summary.json  — per-recording bucket counts
                                            (aggregate stats only, repo-safe)
  data/gt_drafts/fn-taxonomy-detail.json  — per-FN rows incl. note/time
                                            (local-only: note sequences of
                                            rights-unreviewed recordings)

Usage: uv run python scripts/audio-analysis/research/fn_taxonomy.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import note_f1_benchmark as nfb  # noqa: E402
from pertine_dualrun import gt_tx_ids  # noqa: E402

SUMMARY_OUT = REPO / "docs" / "research" / "fn-taxonomy-summary.json"
DETAIL_OUT = REPO / "data" / "gt_drafts" / "fn-taxonomy-detail.json"
ONSET_TOL = 0.08
SLOT_PAD = 0.05


def classify(fn: dict, payload: dict, tol: float) -> tuple[str, dict]:
    t, note = fn["time"], fn["note"]
    debug = payload.get("debug") or {}
    extra: dict = {}

    onsets = list(debug.get("gapValidatedOnsetTimes") or debug.get("onsetTimes") or [])
    if onsets and min(abs(o - t) for o in onsets) > ONSET_TOL:
        return "no-broadband-onset", extra

    # 1-best events covering the time
    events = payload.get("events") or []
    covering_events = [
        ev for ev in events
        if ev["startTimeSec"] - SLOT_PAD <= t <= ev["startTimeSec"] + ev.get("durationSec", 0) + SLOT_PAD
        or abs(ev["startTimeSec"] - t) <= tol
    ]
    # candidate slots covering the time (includes dropped ones)
    slots = payload.get("candidateSlots") or []
    covering_slots = [
        s for s in slots
        if s["startTime"] - SLOT_PAD <= t <= s["endTime"] + SLOT_PAD
        or abs(s["startTime"] - t) <= tol
    ]

    def slot_notes(s):
        out = set()
        for c in s.get("candidates") or []:
            out.add(f"{c['pitchClass']}{c['octave']}")
        p = s.get("primaryNote")
        if p:
            out.add(f"{p['pitchClass']}{p['octave']}")
        return out

    def event_notes(ev):
        return {f"{n['pitchClass']}{n['octave']}" for n in ev.get("notes") or []}

    # in 1-best somewhere else (timing regime)?
    for ev in events:
        if note in event_notes(ev) and abs(ev["startTimeSec"] - t) > tol:
            if not covering_events or note not in set().union(*(event_notes(e) for e in covering_events)):
                nearest = min((abs(ev["startTimeSec"] - t), ev["startTimeSec"]) for ev in events
                              if note in event_notes(ev))
                if nearest[0] <= 0.5:
                    extra["nearestSameNoteSec"] = round(nearest[1], 3)
                    return "time-mismatch", extra

    if not covering_events and not covering_slots:
        return "no-slot", extra

    live_slots = [s for s in covering_slots if not s.get("dropReason")]
    dropped = [s for s in covering_slots if s.get("dropReason")]

    if covering_events or live_slots:
        cand_union = set()
        for ev in covering_events:
            cand_union |= event_notes(ev)
        for s in live_slots:
            cand_union |= slot_notes(s)
        if note in cand_union:
            return "candidate-not-chosen", extra
        if dropped and any(note in slot_notes(s) for s in dropped):
            extra["dropReasons"] = sorted({s["dropReason"] for s in dropped if note in slot_notes(s)})
            return "slot-dropped", extra
        return "not-in-candidates", extra

    # only dropped slots cover it
    reasons = sorted({s["dropReason"] for s in dropped})
    extra["dropReasons"] = reasons
    if any(note in slot_notes(s) for s in dropped):
        return "slot-dropped", extra
    extra["noteAbsentFromDroppedSlot"] = True
    return "slot-dropped", extra


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    baseline = json.loads((REPO / "apps/api/tests/fixtures/free-performance-corpus/"
                           "benchmark_baseline.json").read_text())
    recs = baseline.get("recordings", {})

    summary, details = {}, {}
    for tx in gt_tx_ids():
        entry = recs.get(tx)
        if entry is not None and float(entry.get("minF1", 0.0)) >= 0.9999:
            continue  # saturated (current-benchmark classification)
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        payload = nfb.transcribe_payload(client, tx, debug=True)
        m = nfb.match_pairs(truth, nfb.collect_one_best(payload))
        buckets = Counter()
        rows = []
        for fn in m["falseNegatives"]:
            bucket, extra = classify(fn, payload, fn.get("tol", ONSET_TOL))
            buckets[bucket] += 1
            rows.append({"time": round(fn["time"], 3), "note": fn["note"],
                         "bucket": bucket, **extra})
        summary[tx[:8]] = {"fnTotal": len(rows), "buckets": dict(buckets),
                           "f1": round(m["f1"], 3)}
        details[tx[:8]] = rows
        print(f"{tx[:8]} FN={len(rows):3d} {dict(buckets)}")

    total = Counter()
    for s in summary.values():
        total.update(s["buckets"])
    print("\nTOTAL:", dict(total.most_common()))

    SUMMARY_OUT.write_text(json.dumps({
        "generator": "fn_taxonomy.py (D1, 4th term)",
        "classification": "current benchmark (baseline minF1>=1.0 = saturated)",
        "buckets": dict(total.most_common()),
        "perRecording": summary,
    }, ensure_ascii=False, indent=1) + "\n")
    DETAIL_OUT.write_text(json.dumps(details, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {SUMMARY_OUT.relative_to(REPO)} (aggregate) and "
          f"{DETAIL_OUT.relative_to(REPO)} (local-only detail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
