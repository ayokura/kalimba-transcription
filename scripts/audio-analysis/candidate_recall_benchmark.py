#!/usr/bin/env python3
"""Candidate top-K recall + confidence-ranking benchmark (#178 Phase 2).

The note-level F1 benchmark (note_f1_benchmark.py) measures only the recognizer's
*primary* output. This tool measures the multi-candidate output: when the primary
note set misses a ground-truth note, is that note recoverable from the surfaced
candidate pool (event ``alternateGroupings`` + ``candidateSlots``) — i.e. can the
user fix it with one tap instead of typing it? And how much *noise* does the
candidate layer add (candidates matching no real note), and is ``confidence`` a
usable ranking signal that separates real candidates from noise?

It reports three views:

1. Primary recall (recall@1) — identical to the F1 benchmark's recall.
2. Surfaced-candidate recovery — primary ∪ alternateGroupings ∪ candidateSlots:
   candidate-augmented recall, FN-recovery rate, candidate noise rate, and the
   confidence separation between matched (real) and unmatched (noise) candidates.
3. Ranked top-K recall (diagnostic, from debug.segmentCandidates.rankedCandidates)
   — for each ground-truth note, the rank of the correct note among the covering
   segment's scored hypotheses. Approximate (segment/onset granularity is fuzzy);
   labelled as a diagnostic, not a contract.

Note: with the current 6-recording corpus the primary recall is ~1.0, so the
*recovery* numbers are near-trivial — the immediate signal is candidate noise +
confidence separation. The recall@K instrument becomes discriminating as the
corpus gains harder free-performance recordings (#18) where the primary misses.

Usage:
  uv run python scripts/audio-analysis/candidate_recall_benchmark.py
  uv run python scripts/audio-analysis/candidate_recall_benchmark.py <tx-id> ...
  uv run python scripts/audio-analysis/candidate_recall_benchmark.py --json
  uv run python scripts/audio-analysis/candidate_recall_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import recognizer_fingerprint  # noqa: E402
from apps.api.app.main import app  # noqa: E402
from note_f1_benchmark import (  # noqa: E402
    CAPTURES_DIR,
    DATA_DIR,
    discover_tx_ids,
    load_ground_truth,
)

# Window for associating a surfaced candidate / ranked segment with a ground-truth
# onset. Wider than the F1 match tolerance because candidates carry the event /
# segment start time, and strum/gliss notes spread over ~0.1-0.2s.
CANDIDATE_TIME_WINDOW_SEC = 0.18
RANKED_TOP_K = (1, 2, 3, 5)


def _note_name(note: dict) -> str:
    return f"{note['pitchClass']}{note['octave']}"


def transcribe(client: TestClient, tx_id: str) -> dict:
    tx_dir = DATA_DIR / tx_id
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "true",
            "dryRun": "true",
            "force": "true",
        },
        files={"file": ("audio.wav", (tx_dir / "audio.wav").read_bytes(), "audio/wav")},
    )
    response.raise_for_status()
    return response.json()


def collect_primary(payload: dict) -> list[dict]:
    pairs: list[dict] = []
    for event in payload["events"]:
        for note in event["notes"]:
            pairs.append({"time": float(event["startTimeSec"]), "note": _note_name(note)})
    return pairs


def collect_candidates(payload: dict) -> list[dict]:
    """Surfaced candidate notes the review UI exposes for one-tap recovery.

    Each candidate is (time, note, confidence, source). Notes already in the
    primary output at that event are not re-emitted here — those are recall@1.
    """
    candidates: list[dict] = []
    for event in payload["events"]:
        time = float(event["startTimeSec"])
        primary_here = {_note_name(n) for n in event["notes"]}
        for ag in event.get("alternateGroupings") or []:
            conf = float(ag.get("confidence", 0.0))
            reason = ag.get("reason", "")
            notes: list[dict] = []
            if ag.get("alternateNote"):
                notes.append(ag["alternateNote"])
            for n in ag.get("combinedNotes") or []:
                notes.append(n)
            for group in ag.get("splitInto") or []:
                notes.extend(group)
            for n in notes:
                name = _note_name(n)
                if name in primary_here:
                    continue
                candidates.append({"time": time, "note": name, "confidence": conf, "source": f"ag:{reason}"})
    for slot in payload.get("candidateSlots") or []:
        time = float(slot["startTime"])
        conf = float(slot.get("confidence", 0.0))
        reason = slot.get("dropReason", "")
        slot_notes = [slot["primaryNote"], *(slot.get("candidates") or [])]
        for n in slot_notes:
            candidates.append({"time": time, "note": _note_name(n), "confidence": conf, "source": f"slot:{reason}"})
    return candidates


def _matched(pairs: list[dict], gt: dict, used: list[bool] | None = None, window: float | None = None) -> int:
    """Index of the nearest unused pair with the same note within tolerance, or -1."""
    tol = window if window is not None else gt["tol"]
    best_index, best_dt = -1, None
    for i, pred in enumerate(pairs):
        if (used is not None and used[i]) or pred["note"] != gt["note"]:
            continue
        dt = abs(pred["time"] - gt["time"])
        if dt > tol:
            continue
        if best_dt is None or dt < best_dt:
            best_index, best_dt = i, dt
    return best_index


def ranked_topk_recall(payload: dict, truth: list[dict]) -> dict:
    """Diagnostic: rank of each GT note among the covering segment(s)' scored
    rankedCandidates. Segments overlapping [t-window, t+window] are unioned and
    deduped by note name keeping the best score."""
    segments = (payload.get("debug") or {}).get("segmentCandidates") or []
    hits = {k: 0 for k in RANKED_TOP_K}
    hits["any"] = 0
    ranks: list[int] = []
    for gt in truth:
        t = gt["time"]
        pool: dict[str, float] = {}
        for seg in segments:
            start = seg.get("startTime")
            end = seg.get("endTime", start)
            if start is None:
                continue
            if start - CANDIDATE_TIME_WINDOW_SEC <= t <= (end if end is not None else start) + CANDIDATE_TIME_WINDOW_SEC:
                for cand in seg.get("rankedCandidates") or []:
                    name = cand.get("noteName")
                    score = float(cand.get("score", 0.0))
                    if name is not None and score > pool.get(name, float("-inf")):
                        pool[name] = score
        ordered = sorted(pool.items(), key=lambda kv: -kv[1])
        rank = next((i + 1 for i, (name, _) in enumerate(ordered) if name == gt["note"]), None)
        if rank is not None:
            ranks.append(rank)
            hits["any"] += 1
            for k in RANKED_TOP_K:
                if rank <= k:
                    hits[k] += 1
    n = len(truth)
    return {
        "recallAtK": {str(k): (hits[k] / n if n else 1.0) for k in RANKED_TOP_K},
        "recallAny": hits["any"] / n if n else 1.0,
        "meanRankWhenFound": (sum(ranks) / len(ranks)) if ranks else None,
    }


def evaluate(payload: dict, truth: list[dict]) -> dict:
    primary = collect_primary(payload)
    candidates = collect_candidates(payload)

    used_primary = [False] * len(primary)
    primary_tp = 0
    fn_after_primary: list[dict] = []
    for gt in sorted(truth, key=lambda p: p["time"]):
        idx = _matched(primary, gt, used_primary)
        if idx >= 0:
            used_primary[idx] = True
            primary_tp += 1
        else:
            fn_after_primary.append(gt)

    # Of the primary FNs, how many are recoverable from the surfaced candidates?
    used_cand = [False] * len(candidates)
    recovered: list[dict] = []
    unrecoverable: list[dict] = []
    for gt in fn_after_primary:
        idx = _matched(candidates, gt, used_cand, window=CANDIDATE_TIME_WINDOW_SEC)
        if idx >= 0:
            used_cand[idx] = True
            recovered.append({**gt, "confidence": candidates[idx]["confidence"], "source": candidates[idx]["source"]})
        else:
            unrecoverable.append(gt)

    # Candidate noise: a candidate matches no GT note within the window.
    matched_conf: list[float] = []
    noise_conf: list[float] = []
    for cand in candidates:
        gt_like = {"time": cand["time"], "note": cand["note"], "tol": CANDIDATE_TIME_WINDOW_SEC}
        is_real = _matched(truth, gt_like, window=CANDIDATE_TIME_WINDOW_SEC) >= 0
        (matched_conf if is_real else noise_conf).append(cand["confidence"])

    n = len(truth)
    primary_recall = primary_tp / n if n else 1.0
    augmented_recall = (primary_tp + len(recovered)) / n if n else 1.0
    fn_recovery_rate = (len(recovered) / len(fn_after_primary)) if fn_after_primary else None
    cand_total = len(candidates)
    cand_real = len(matched_conf)
    return {
        "truthNotes": n,
        "primaryRecall": primary_recall,
        "augmentedRecall": augmented_recall,
        "falseNegatives": len(fn_after_primary),
        "recovered": len(recovered),
        "unrecoverable": len(unrecoverable),
        "fnRecoveryRate": fn_recovery_rate,
        "candidatePoolSize": cand_total,
        "candidateReal": cand_real,
        "candidateNoise": cand_total - cand_real,
        "candidateNoiseRate": ((cand_total - cand_real) / cand_total) if cand_total else 0.0,
        "meanConfReal": (sum(matched_conf) / len(matched_conf)) if matched_conf else None,
        "meanConfNoise": (sum(noise_conf) / len(noise_conf)) if noise_conf else None,
        "rankedDiagnostic": ranked_topk_recall(payload, truth),
        "_recovered": recovered,
        "_unrecoverable": unrecoverable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate top-K recall + confidence benchmark (#178 Phase 2)")
    parser.add_argument("tx_ids", nargs="*", help="transaction IDs (default: all with ground_truth.json)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="list recovered / unrecoverable notes")
    args = parser.parse_args()

    tx_ids = args.tx_ids or discover_tx_ids()
    if not tx_ids:
        print("No ground_truth.json found under", CAPTURES_DIR, file=sys.stderr)
        return 1

    client = TestClient(app)
    results = []
    agg = {"truth": 0, "primaryTp": 0, "recovered": 0, "fn": 0, "candTotal": 0, "candReal": 0}
    conf_real: list[float] = []
    conf_noise: list[float] = []
    for tx_id in tx_ids:
        truth = load_ground_truth(CAPTURES_DIR / tx_id / "ground_truth.json")
        outcome = evaluate(transcribe(client, tx_id), truth)
        outcome["txId"] = tx_id
        results.append(outcome)
        agg["truth"] += outcome["truthNotes"]
        agg["primaryTp"] += round(outcome["primaryRecall"] * outcome["truthNotes"])
        agg["recovered"] += outcome["recovered"]
        agg["fn"] += outcome["falseNegatives"]
        agg["candTotal"] += outcome["candidatePoolSize"]
        agg["candReal"] += outcome["candidateReal"]
        if outcome["meanConfReal"] is not None:
            conf_real.extend([outcome["meanConfReal"]] * outcome["candidateReal"])
        if outcome["meanConfNoise"] is not None:
            conf_noise.extend([outcome["meanConfNoise"]] * outcome["candidateNoise"])

    micro_primary = agg["primaryTp"] / agg["truth"] if agg["truth"] else 1.0
    micro_aug = (agg["primaryTp"] + agg["recovered"]) / agg["truth"] if agg["truth"] else 1.0
    summary = {
        "recognizerFingerprint": recognizer_fingerprint()[:16],
        "recordings": len(results),
        "microPrimaryRecall": micro_primary,
        "microAugmentedRecall": micro_aug,
        "totalFalseNegatives": agg["fn"],
        "totalRecovered": agg["recovered"],
        "candidatePoolSize": agg["candTotal"],
        "candidateNoiseRate": ((agg["candTotal"] - agg["candReal"]) / agg["candTotal"]) if agg["candTotal"] else 0.0,
        "meanConfReal": (sum(conf_real) / len(conf_real)) if conf_real else None,
        "meanConfNoise": (sum(conf_noise) / len(conf_noise)) if conf_noise else None,
    }

    if args.json:
        for r in results:
            r.pop("_recovered", None)
            r.pop("_unrecoverable", None)
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return 0

    print(f"{'tx':38} {'GT':>4} {'R@1':>6} {'R+cand':>7} {'FN':>3} {'rec':>3} {'cand':>5} {'noise%':>7} {'rk@3':>6}")
    for r in results:
        rd = r["rankedDiagnostic"]
        print(
            f"{r['txId'][:36]:38} {r['truthNotes']:>4} {r['primaryRecall']:6.3f} {r['augmentedRecall']:7.3f}"
            f" {r['falseNegatives']:>3} {r['recovered']:>3} {r['candidatePoolSize']:>5}"
            f" {r['candidateNoiseRate']*100:6.1f}% {rd['recallAtK']['3']:6.3f}"
        )
        if args.verbose:
            for rec in r["_recovered"]:
                print(f"    RECOVERED {rec['time']:8.3f}s {rec['note']} (conf={rec['confidence']:.2f} via {rec['source']})")
            for fn in r["_unrecoverable"]:
                print(f"    UNRECOVERABLE {fn['time']:8.3f}s {fn['note']}")
    cr = f"{summary['meanConfReal']:.3f}" if summary["meanConfReal"] is not None else "n/a"
    cn = f"{summary['meanConfNoise']:.3f}" if summary["meanConfNoise"] is not None else "n/a"
    print(
        f"\nmicro recall@1={summary['microPrimaryRecall']:.3f} +candidates={summary['microAugmentedRecall']:.3f}"
        f"  FN={summary['totalFalseNegatives']} recovered={summary['totalRecovered']}"
        f"\ncandidate pool={summary['candidatePoolSize']} noise={summary['candidateNoiseRate']*100:.1f}%"
        f"  mean confidence real={cr} noise={cn}"
        f"\n({summary['recordings']} recordings, recognizer {summary['recognizerFingerprint']})"
    )
    if summary["microPrimaryRecall"] >= 0.999:
        print(
            "\nNote: primary recall is saturated on this corpus. Every surfaced candidate is\n"
            "therefore an *alternative to an already-correct primary* (a played note would be\n"
            "in the primary), so noise=100% is expected, not a defect -- and these candidates\n"
            "carry low confidence (~0.27), which is the desirable property (the UI can\n"
            "de-emphasize them). 'mean confidence real' needs primary misses to compute.\n"
            "The actionable signal now is the ranked top-K diagnostic (rk@3 < 1.0 marks GT\n"
            "notes that raw segment scoring under-ranks but downstream rescue recovers --\n"
            "Phase 3 calibration targets). recall@K / recovery become informative once harder\n"
            "free-performance recordings (#18) where the primary misses enter the GT corpus."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
