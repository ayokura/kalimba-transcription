#!/usr/bin/env python3
"""Note-level + candidate-aware benchmark for free-performance recordings.

Evaluates the currently-loaded recognizer against human-verified ground truth
(`ground_truth.json`, AGENTS.md schema) instead of exact event-sequence match.
This is the free-performance counterpart of the fixture regression suite:
fixtures assert "the transcription equals the score", this benchmark measures
"how close the transcription is to what was physically played" via layered
metrics:

1. one-best onset-only note precision / recall / F1 (the historical output);
2. candidate recall@K from primary output + surfaced alternates + dropped slots;
3. coarse correction burden for how expensive user repair would be;
4. hard-miss rate for notes absent from both one-best and surfaced candidates;
5. initial confidence-calibration aggregates.

The extra layers are intentionally benchmark-only: the production response
schema is not widened here.  The script requests debug output so the benchmark
can also summarize diagnostic ``rankedCandidates`` without making them public.

Ground truth discovery (default):
  apps/api/tests/fixtures/free-performance-corpus/<tx-id>/ground_truth.json
  apps/api/tests/fixtures/transaction-captures/<tx-id>/ground_truth.json
Repo-managed corpus items carry their own audio.wav + request.json. Local
transaction-captures fall back to data/transactions/<tx-id>/ for audio + tuning.

Usage:
  uv run python scripts/audio-analysis/note_f1_benchmark.py            # all GT
  uv run python scripts/audio-analysis/note_f1_benchmark.py <tx-id> ...
  uv run python scripts/audio-analysis/note_f1_benchmark.py --json
  uv run python scripts/audio-analysis/note_f1_benchmark.py --verbose  # FP/FN 明細

Matching: each ground-truth (timeSec, note) pair is matched one-to-one to the
nearest predicted (startTimeSec, note) pair with the same note name within
toleranceSec (per-onset override supported). Candidate-layer matching uses a
wider event/segment window because alternates and dropped slots are attached to
event/segment starts rather than human onset annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import (  # noqa: E402
    kalimba_dsp_fingerprint,
    recognizer_fingerprint,
)
from apps.api.app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"
FREE_PERFORMANCE_CORPUS_DIR = (
    REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "free-performance-corpus"
)

DEFAULT_TOLERANCE_SEC = 0.05
CANDIDATE_TIME_WINDOW_SEC = 0.18
CANDIDATE_RECALL_K = (1, 3, 5)
LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)


def _safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _note_name(note: dict) -> str:
    return f"{note['pitchClass']}{note['octave']}"


def load_ground_truth(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    default_tol = float(doc.get("toleranceSec", DEFAULT_TOLERANCE_SEC))
    pairs: list[dict] = []
    for onset in doc.get("onsets", []):
        tol = float(onset.get("toleranceSec", default_tol))
        for note in onset["notes"]:
            pairs.append({"time": float(onset["timeSec"]), "note": note, "tol": tol})
    return pairs


def corpus_dir_for(tx_id: str) -> Path | None:
    """Repo-managed corpus dir for tx_id, if present."""
    path = FREE_PERFORMANCE_CORPUS_DIR / tx_id
    if (path / "audio.wav").is_file() and (path / "request.json").is_file():
        return path
    return None


def transaction_dir_for(tx_id: str) -> Path:
    """Directory containing audio.wav/request.json for tx_id.

    Prefer repo-managed free-performance corpus for reproducibility; fall back
    to local data/transactions for dev-only transaction-capture ground truth.
    """
    return corpus_dir_for(tx_id) or (DATA_DIR / tx_id)


def ground_truth_path_for(tx_id: str) -> Path:
    corpus_dir = corpus_dir_for(tx_id)
    if corpus_dir is not None and (corpus_dir / "ground_truth.json").is_file():
        return corpus_dir / "ground_truth.json"
    return CAPTURES_DIR / tx_id / "ground_truth.json"


def review_status_path_for(tx_id: str) -> Path | None:
    corpus_dir = corpus_dir_for(tx_id)
    candidates = []
    if corpus_dir is not None:
        candidates.append(corpus_dir / "review_status.json")
    candidates.append(DATA_DIR / tx_id / "review_status.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def transcribe_payload(client: TestClient, tx_id: str, *, debug: bool = True) -> dict:
    tx_dir = transaction_dir_for(tx_id)
    audio_bytes = (tx_dir / "audio.wav").read_bytes()
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "true" if debug else "false",
            "dryRun": "true",
            "force": "true",
        },
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    return response.json()


def collect_one_best(payload: dict) -> list[dict]:
    pairs: list[dict] = []
    for event_index, event in enumerate(payload["events"]):
        for note in event["notes"]:
            pairs.append(
                {
                    "time": float(event["startTimeSec"]),
                    "note": _note_name(note),
                    "eventIndex": event_index,
                    "eventId": event.get("id"),
                }
            )
    return pairs


def transcribe(client: TestClient, tx_id: str) -> list[dict]:
    """Backward-compatible one-best helper used by older analysis scripts."""
    return collect_one_best(transcribe_payload(client, tx_id, debug=False))


def match_pairs(truth: list[dict], predicted: list[dict]) -> dict:
    used = [False] * len(predicted)
    matched: list[tuple[dict, dict]] = []
    false_negatives: list[dict] = []
    for gt in sorted(truth, key=lambda p: p["time"]):
        best_index = -1
        best_dt = None
        for i, pred in enumerate(predicted):
            if used[i] or pred["note"] != gt["note"]:
                continue
            dt = abs(pred["time"] - gt["time"])
            if dt > gt["tol"]:
                continue
            if best_dt is None or dt < best_dt:
                best_index, best_dt = i, dt
        if best_index >= 0:
            used[best_index] = True
            matched.append((gt, predicted[best_index]))
        else:
            false_negatives.append(gt)
    false_positives = [pred for i, pred in enumerate(predicted) if not used[i]]
    tp = len(matched)
    precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "truthNotes": len(truth),
        "predictedNotes": len(predicted),
        "tp": tp,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _nearest_match_index(
    pairs: list[dict],
    gt: dict,
    *,
    used: list[bool] | None = None,
    window: float | None = None,
    max_rank: int | None = None,
) -> int:
    """Index of nearest unused same-note pair within the requested tolerance."""
    tol = window if window is not None else gt["tol"]
    best_index = -1
    best_dt = None
    for i, pair in enumerate(pairs):
        if used is not None and used[i]:
            continue
        if max_rank is not None and pair.get("rank") is not None and int(pair["rank"]) > max_rank:
            continue
        if pair["note"] != gt["note"]:
            continue
        dt = abs(float(pair["time"]) - float(gt["time"]))
        if dt > tol:
            continue
        if best_dt is None or dt < best_dt:
            best_index, best_dt = i, dt
    return best_index


def _candidate_entry(
    *,
    time: float,
    note: str,
    source: str,
    source_field: str,
    rank: int,
    confidence: float | None = None,
    reason: str | None = None,
    end_time: float | None = None,
    score: float | None = None,
) -> dict:
    entry = {
        "time": float(time),
        "note": note,
        "source": source,
        "sourceField": source_field,
        "rank": int(rank),
    }
    if end_time is not None:
        entry["endTime"] = float(end_time)
    if confidence is not None:
        entry["confidence"] = round(float(confidence), 6)
    if reason:
        entry["reason"] = reason
    if score is not None:
        entry["score"] = round(float(score), 6)
    return entry


def collect_surface_candidates(payload: dict) -> list[dict]:
    """Collect review-visible candidate notes.

    Sources:
    - event ``alternateGroupings`` (soft alternates, split/merge alternatives);
    - top notes from ``candidateSlots`` (dropped segment candidates).

    Primary notes are deliberately excluded here because they are rank-1 in the
    one-best layer. Surfaced candidates start at rank 2 so recall@1 remains the
    historical one-best recall, while recall@3/@5 measures one-tap recoverability.
    """
    candidates: list[dict] = []
    for event_index, event in enumerate(payload.get("events", [])):
        time = float(event["startTimeSec"])
        primary_here = {_note_name(n) for n in event.get("notes", [])}
        rank = 2
        for ag in event.get("alternateGroupings") or []:
            confidence = _safe_float(ag.get("confidence"), 0.0)
            reason = ag.get("reason", "")
            notes_with_source: list[tuple[dict, str]] = []
            if ag.get("alternateNote"):
                notes_with_source.append((ag["alternateNote"], "softAlternate"))
            for n in ag.get("combinedNotes") or []:
                notes_with_source.append((n, "alternateGroupingCombined"))
            for group in ag.get("splitInto") or []:
                for n in group:
                    notes_with_source.append((n, "alternateGroupingSplit"))

            for note, source in notes_with_source:
                name = _note_name(note)
                if name in primary_here:
                    continue
                candidates.append(
                    {
                        **_candidate_entry(
                            time=time,
                            note=name,
                            source=source,
                            source_field="alternateGroupings",
                            rank=rank,
                            confidence=confidence,
                            reason=reason,
                        ),
                        "eventIndex": event_index,
                        "eventId": event.get("id"),
                    }
                )
                rank += 1

    for slot_index, slot in enumerate(payload.get("candidateSlots") or []):
        time = float(slot["startTime"])
        end_time = _safe_float(slot.get("endTime"))
        confidence = _safe_float(slot.get("confidence"), 0.0)
        reason = slot.get("dropReason", "")
        slot_notes: list[tuple[dict, str]] = []
        if slot.get("primaryNote"):
            slot_notes.append((slot["primaryNote"], "droppedPrimary"))
        for n in slot.get("candidates") or []:
            slot_notes.append((n, "droppedAlternate"))
        for rank_offset, (note, source) in enumerate(slot_notes, start=2):
            candidates.append(
                {
                    **_candidate_entry(
                        time=time,
                        end_time=end_time,
                        note=_note_name(note),
                        source=source,
                        source_field="candidateSlots",
                        rank=rank_offset,
                        confidence=confidence,
                        reason=reason,
                    ),
                    "slotIndex": slot_index,
                }
            )
    return candidates


def collect_debug_ranked_candidates(payload: dict) -> list[dict]:
    """Collect debug-only ranked/residual hypotheses for diagnostic recall@K."""
    debug = payload.get("debug") or {}
    ranked: list[dict] = []
    for seg_index, seg in enumerate(debug.get("segmentCandidates") or []):
        start = _safe_float(seg.get("startTime"))
        if start is None:
            continue
        end = _safe_float(seg.get("endTime"), start)
        for field in ("rankedCandidates", "residualCandidates"):
            for rank, cand in enumerate(seg.get(field) or [], start=1):
                name = cand.get("noteName") or cand.get("note")
                if not name:
                    continue
                ranked.append(
                    {
                        **_candidate_entry(
                            time=start,
                            end_time=end,
                            note=name,
                            source=field,
                            source_field=f"debug.segmentCandidates.{field}",
                            rank=rank,
                            confidence=_safe_float(cand.get("confidence")),
                            score=_safe_float(cand.get("score")),
                            reason=",".join(cand.get("reasons") or []),
                        ),
                        "segmentIndex": seg_index,
                    }
                )
    return ranked


def _source_counts(primary: list[dict], surface: list[dict], debug_ranked: list[dict]) -> dict:
    counts = {
        "oneBestNotes": len(primary),
        "alternateGroupingNotes": 0,
        "softAlternateNotes": 0,
        "droppedCandidateNotes": 0,
        "debugRankedCandidateNotes": 0,
        "debugResidualCandidateNotes": 0,
    }
    for cand in surface:
        if cand["sourceField"] == "alternateGroupings":
            counts["alternateGroupingNotes"] += 1
        if cand["source"] == "softAlternate":
            counts["softAlternateNotes"] += 1
        if cand["sourceField"] == "candidateSlots":
            counts["droppedCandidateNotes"] += 1
    for cand in debug_ranked:
        if cand["sourceField"].endswith(".rankedCandidates"):
            counts["debugRankedCandidateNotes"] += 1
        elif cand["sourceField"].endswith(".residualCandidates"):
            counts["debugResidualCandidateNotes"] += 1
    return counts


def _candidate_recall_counts(
    truth: list[dict],
    primary: list[dict],
    surface_candidates: list[dict],
    *,
    max_rank: int | None,
) -> tuple[int, list[dict], list[dict]]:
    """Return (hits, candidate-assisted hits, misses) for primary+surface pool."""
    used_primary = [False] * len(primary)
    used_candidates = [False] * len(surface_candidates)
    hits = 0
    candidate_hits: list[dict] = []
    misses: list[dict] = []
    for gt in sorted(truth, key=lambda p: p["time"]):
        if max_rank is None or max_rank >= 1:
            primary_idx = _nearest_match_index(primary, gt, used=used_primary)
            if primary_idx >= 0:
                used_primary[primary_idx] = True
                hits += 1
                continue
        cand_idx = _nearest_match_index(
            surface_candidates,
            gt,
            used=used_candidates,
            window=CANDIDATE_TIME_WINDOW_SEC,
            max_rank=max_rank,
        )
        if cand_idx >= 0:
            used_candidates[cand_idx] = True
            hits += 1
            candidate_hits.append({**gt, "candidate": surface_candidates[cand_idx]})
        else:
            misses.append(gt)
    return hits, candidate_hits, misses


def ranked_topk_recall(debug_ranked: list[dict], truth: list[dict]) -> dict:
    """Diagnostic recall@K from debug.segmentCandidates ranked hypotheses."""
    hits = {k: 0 for k in CANDIDATE_RECALL_K}
    hits["any"] = 0
    ranks: list[int] = []
    for gt in truth:
        covering = [
            cand
            for cand in debug_ranked
            if cand["sourceField"].endswith(".rankedCandidates")
            and cand["time"] - CANDIDATE_TIME_WINDOW_SEC
            <= gt["time"]
            <= cand.get("endTime", cand["time"]) + CANDIDATE_TIME_WINDOW_SEC
        ]
        best_rank_by_note: dict[str, int] = {}
        for cand in covering:
            rank = int(cand["rank"])
            note = cand["note"]
            if note not in best_rank_by_note or rank < best_rank_by_note[note]:
                best_rank_by_note[note] = rank
        rank = best_rank_by_note.get(gt["note"])
        if rank is not None:
            ranks.append(rank)
            hits["any"] += 1
            for k in CANDIDATE_RECALL_K:
                if rank <= k:
                    hits[k] += 1
    n = len(truth)
    return {
        "recallAtK": {str(k): (hits[k] / n if n else 1.0) for k in CANDIDATE_RECALL_K},
        "hitsAtK": {str(k): hits[k] for k in CANDIDATE_RECALL_K},
        "recallAny": hits["any"] / n if n else 1.0,
        "foundAny": hits["any"],
        "meanRankWhenFound": (sum(ranks) / len(ranks)) if ranks else None,
    }


def compute_candidate_metrics(
    truth: list[dict],
    primary: list[dict],
    surface_candidates: list[dict],
    debug_ranked: list[dict],
    *,
    event_count: int,
) -> dict:
    n = len(truth)
    hits_at_k: dict[str, int] = {}
    recall_at_k: dict[str, float] = {}
    for k in CANDIDATE_RECALL_K:
        hits, _, _ = _candidate_recall_counts(truth, primary, surface_candidates, max_rank=k)
        hits_at_k[str(k)] = hits
        recall_at_k[str(k)] = hits / n if n else 1.0

    all_hits, candidate_hits, hard_misses = _candidate_recall_counts(
        truth, primary, surface_candidates, max_rank=None,
    )
    return {
        "recallAt1": recall_at_k["1"],
        "recallAt3": recall_at_k["3"],
        "recallAt5": recall_at_k["5"],
        "hitsAtK": hits_at_k,
        "candidateSlotsPerEvent": len(surface_candidates) / event_count if event_count else 0.0,
        "surfaceCandidateNotes": len(surface_candidates),
        "hardMisses": len(hard_misses),
        "hardMissRate": len(hard_misses) / n if n else 0.0,
        "candidateAssistedHits": len(candidate_hits),
        "surfaceAugmentedRecall": all_hits / n if n else 1.0,
        "rankedDiagnostic": ranked_topk_recall(debug_ranked, truth),
        # Public list of the GT notes recoverable from no surfaced candidate.
        # Emitted (not ``_``-prefixed) so both --json and --verbose can surface
        # the heaviest-correction events as the #18 corpus gains harder takes.
        "hardMissNotes": [{"time": m["time"], "note": m["note"]} for m in hard_misses],
        "_candidateHits": candidate_hits,
        "_hardMisses": hard_misses,
    }


def estimate_correction_burden(match: dict, candidate_metrics: dict) -> dict:
    """Coarse semantic edit cost, aligned with the product-UX survey.

    Initial skeleton weights:
    - remove extra predicted note: 1;
    - enable candidate / add candidate note: 1;
    - manually insert a hard-missed note: 3.

    Merge/split, gesture relabel, notation-layer costs are kept as explicit zero
    placeholders so future UI logs can fill them without reshaping the JSON.
    """
    note_removes = len(match["falsePositives"])
    candidate_adds = int(candidate_metrics["candidateAssistedHits"])
    manual_inserts = int(candidate_metrics["hardMisses"])
    estimated_cost = note_removes + candidate_adds + manual_inserts * 3
    miss_fixes = candidate_adds + manual_inserts
    all_fixes = note_removes + miss_fixes
    return {
        "estimatedCost": estimated_cost,
        "costPerTruthNote": estimated_cost / match["truthNotes"] if match["truthNotes"] else 0.0,
        "candidateAssistedFixRate": candidate_adds / miss_fixes if miss_fixes else None,
        "insertions": manual_inserts,
        "deletions": note_removes,
        "noteAdds": candidate_adds + manual_inserts,
        "noteRemoves": note_removes,
        "candidateEnabled": candidate_adds,
        "manualInserts": manual_inserts,
        "mergeSplits": 0,
        "gestureRelabels": 0,
        "allFixes": all_fixes,
    }


def _candidate_is_real(candidate: dict, truth: list[dict]) -> bool:
    gt_like = {"time": candidate["time"], "note": candidate["note"], "tol": CANDIDATE_TIME_WINDOW_SEC}
    return _nearest_match_index(truth, gt_like, window=CANDIDATE_TIME_WINDOW_SEC) >= 0


def _candidate_calibration(surface_candidates: list[dict], truth: list[dict]) -> dict:
    scored = [
        (float(c.get("confidence", 0.0)), _candidate_is_real(c, truth))
        for c in surface_candidates
        if c.get("confidence") is not None
    ]
    bins = []
    ece = None
    if scored:
        total = len(scored)
        weighted_error = 0.0
        for lo, hi in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:]):
            bucket = [(conf, ok) for conf, ok in scored if lo <= conf < hi]
            if not bucket:
                bins.append({"range": [round(lo, 1), round(min(hi, 1.0), 1)], "count": 0})
                continue
            avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
            accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
            weighted_error += len(bucket) / total * abs(accuracy - avg_conf)
            bins.append(
                {
                    "range": [round(lo, 1), round(min(hi, 1.0), 1)],
                    "count": len(bucket),
                    "avgConfidence": avg_conf,
                    "accuracy": accuracy,
                }
            )
        ece = weighted_error
    high = [(conf, ok) for conf, ok in scored if conf >= HIGH_CONFIDENCE_THRESHOLD]
    low = [(conf, ok) for conf, ok in scored if conf <= LOW_CONFIDENCE_THRESHOLD]
    return {
        "candidateExpectedCalibrationError": ece,
        "candidateHighConfidenceWrongRate": (
            sum(1 for _, ok in high if not ok) / len(high) if high else None
        ),
        "candidateLowConfidenceCorrectRate": (
            sum(1 for _, ok in low if ok) / len(low) if low else None
        ),
        "candidateCalibrationBins": bins,
        "_candidateConfidenceCount": len(scored),
        "_candidateHighConfidenceCount": len(high),
        "_candidateHighConfidenceWrongCount": sum(1 for _, ok in high if not ok),
        "_candidateLowConfidenceCount": len(low),
        "_candidateLowConfidenceCorrectCount": sum(1 for _, ok in low if ok),
    }


def _event_confidence_flags(payload: dict, match: dict) -> dict:
    events = payload.get("events", [])
    event_count = len(events)
    if not events:
        return {
            "flaggedEventPrecision": None,
            "missedErrorRate": None,
            "highConfidenceWrongRate": None,
            "lowConfidenceCorrectRate": None,
            "_counts": {
                "events": 0,
                "flaggedEvents": 0,
                "errorEvents": 0,
                "flaggedErrorEvents": 0,
                "unflaggedEvents": 0,
                "unflaggedErrorEvents": 0,
                "flaggedCorrectEvents": 0,
            },
        }

    flagged = {
        i for i, event in enumerate(events)
        if event.get("alternateGroupings")
    }
    error_events: set[int] = set()
    for fp in match["falsePositives"]:
        if fp.get("eventIndex") is not None:
            error_events.add(int(fp["eventIndex"]))
    for fn in match["falseNegatives"]:
        nearest_index = None
        nearest_dt = None
        for i, event in enumerate(events):
            dt = abs(float(event["startTimeSec"]) - float(fn["time"]))
            if dt <= CANDIDATE_TIME_WINDOW_SEC and (nearest_dt is None or dt < nearest_dt):
                nearest_index = i
                nearest_dt = dt
        if nearest_index is not None:
            error_events.add(nearest_index)

    flagged_errors = flagged & error_events
    unflagged = set(range(event_count)) - flagged
    unflagged_errors = unflagged & error_events
    flagged_correct = flagged - error_events
    return {
        "flaggedEventPrecision": len(flagged_errors) / len(flagged) if flagged else None,
        "missedErrorRate": len(error_events - flagged) / len(error_events) if error_events else None,
        "highConfidenceWrongRate": len(unflagged_errors) / len(unflagged) if unflagged else None,
        "lowConfidenceCorrectRate": len(flagged_correct) / len(flagged) if flagged else None,
        "_counts": {
            "events": event_count,
            "flaggedEvents": len(flagged),
            "errorEvents": len(error_events),
            "flaggedErrorEvents": len(flagged_errors),
            "unflaggedEvents": len(unflagged),
            "unflaggedErrorEvents": len(unflagged_errors),
            "flaggedCorrectEvents": len(flagged_correct),
        },
    }


def compute_confidence_calibration(
    payload: dict,
    truth: list[dict],
    surface_candidates: list[dict],
    match: dict,
) -> dict:
    event_flags = _event_confidence_flags(payload, match)
    candidate = _candidate_calibration(surface_candidates, truth)
    return {
        "flaggedEventPrecision": event_flags["flaggedEventPrecision"],
        "missedErrorRate": event_flags["missedErrorRate"],
        "highConfidenceWrongRate": event_flags["highConfidenceWrongRate"],
        "lowConfidenceCorrectRate": event_flags["lowConfidenceCorrectRate"],
        "candidateExpectedCalibrationError": candidate["candidateExpectedCalibrationError"],
        "candidateHighConfidenceWrongRate": candidate["candidateHighConfidenceWrongRate"],
        "candidateLowConfidenceCorrectRate": candidate["candidateLowConfidenceCorrectRate"],
        "candidateCalibrationBins": candidate["candidateCalibrationBins"],
        "_eventCounts": event_flags["_counts"],
        "_candidateConfidenceCount": candidate["_candidateConfidenceCount"],
        "_candidateHighConfidenceCount": candidate["_candidateHighConfidenceCount"],
        "_candidateHighConfidenceWrongCount": candidate["_candidateHighConfidenceWrongCount"],
        "_candidateLowConfidenceCount": candidate["_candidateLowConfidenceCount"],
        "_candidateLowConfidenceCorrectCount": candidate["_candidateLowConfidenceCorrectCount"],
    }


def evaluate_payload(payload: dict, truth: list[dict]) -> dict:
    primary = collect_one_best(payload)
    match = match_pairs(truth, primary)
    surface_candidates = collect_surface_candidates(payload)
    debug_ranked = collect_debug_ranked_candidates(payload)
    candidate_metrics = compute_candidate_metrics(
        truth,
        primary,
        surface_candidates,
        debug_ranked,
        event_count=len(payload.get("events", [])),
    )
    correction = estimate_correction_burden(match, candidate_metrics)
    confidence = compute_confidence_calibration(payload, truth, surface_candidates, match)

    match["oneBest"] = {
        "onsetPrecision": match["precision"],
        "onsetRecall": match["recall"],
        "onsetF1": match["f1"],
        "offsetAwareF1": None,
        "averageOverlapRatio": None,
    }
    match["candidates"] = {
        key: value
        for key, value in candidate_metrics.items()
        if not key.startswith("_")
    }
    match["correction"] = correction
    match["confidence"] = {
        key: value
        for key, value in confidence.items()
        if not key.startswith("_")
    }
    match["candidateSources"] = _source_counts(primary, surface_candidates, debug_ranked)
    match["_confidenceCounts"] = {
        "events": confidence["_eventCounts"],
        "candidateConfidenceCount": confidence["_candidateConfidenceCount"],
        "candidateHighConfidenceCount": confidence["_candidateHighConfidenceCount"],
        "candidateHighConfidenceWrongCount": confidence["_candidateHighConfidenceWrongCount"],
        "candidateLowConfidenceCount": confidence["_candidateLowConfidenceCount"],
        "candidateLowConfidenceCorrectCount": confidence["_candidateLowConfidenceCorrectCount"],
    }
    return match


def discover_tx_ids() -> list[str]:
    discovered: set[str] = set()
    if FREE_PERFORMANCE_CORPUS_DIR.is_dir():
        discovered.update(
            d.name
            for d in FREE_PERFORMANCE_CORPUS_DIR.iterdir()
            if (d / "ground_truth.json").is_file()
            and (d / "audio.wav").is_file()
            and (d / "request.json").is_file()
        )
    if CAPTURES_DIR.is_dir():
        discovered.update(
            d.name
            for d in CAPTURES_DIR.iterdir()
            if (d / "ground_truth.json").is_file()
            and (transaction_dir_for(d.name) / "audio.wav").is_file()
            and (transaction_dir_for(d.name) / "request.json").is_file()
        )
    return sorted(discovered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Note-level F1 benchmark")
    parser.add_argument("tx_ids", nargs="*", help="transaction IDs (default: all with ground_truth.json)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="list FP/FN pairs per recording")
    args = parser.parse_args()

    tx_ids = args.tx_ids or discover_tx_ids()
    if not tx_ids:
        print("No ground_truth.json found under", CAPTURES_DIR, file=sys.stderr)
        return 1

    client = TestClient(app)
    results = []
    total = {
        "tp": 0,
        "truth": 0,
        "predicted": 0,
        "candidateHits": {str(k): 0 for k in CANDIDATE_RECALL_K},
        "hardMisses": 0,
        "surfaceCandidateNotes": 0,
        "events": 0,
        "estimatedCost": 0,
        "candidateEnabled": 0,
        "manualInserts": 0,
        "noteRemoves": 0,
        "allFixes": 0,
        "flaggedEvents": 0,
        "flaggedErrorEvents": 0,
        "errorEvents": 0,
        "unflaggedEvents": 0,
        "unflaggedErrorEvents": 0,
        "flaggedCorrectEvents": 0,
        "candidateConfidenceCount": 0,
        "candidateHighConfidenceCount": 0,
        "candidateHighConfidenceWrongCount": 0,
        "candidateLowConfidenceCount": 0,
        "candidateLowConfidenceCorrectCount": 0,
        "candidateBinCounts": [0 for _ in range(len(CALIBRATION_BINS) - 1)],
        "candidateBinConfidenceSums": [0.0 for _ in range(len(CALIBRATION_BINS) - 1)],
        "candidateBinCorrectCounts": [0.0 for _ in range(len(CALIBRATION_BINS) - 1)],
    }
    for tx_id in tx_ids:
        truth = load_ground_truth(ground_truth_path_for(tx_id))
        payload = transcribe_payload(client, tx_id, debug=True)
        outcome = evaluate_payload(payload, truth)
        outcome["txId"] = tx_id
        results.append(outcome)
        total["tp"] += outcome["tp"]
        total["truth"] += outcome["truthNotes"]
        total["predicted"] += outcome["predictedNotes"]
        for k in CANDIDATE_RECALL_K:
            total["candidateHits"][str(k)] += outcome["candidates"]["hitsAtK"][str(k)]
        total["hardMisses"] += outcome["candidates"]["hardMisses"]
        total["surfaceCandidateNotes"] += outcome["candidates"]["surfaceCandidateNotes"]
        total["events"] += len(payload.get("events", []))
        total["estimatedCost"] += outcome["correction"]["estimatedCost"]
        total["candidateEnabled"] += outcome["correction"]["candidateEnabled"]
        total["manualInserts"] += outcome["correction"]["manualInserts"]
        total["noteRemoves"] += outcome["correction"]["noteRemoves"]
        total["allFixes"] += outcome["correction"]["allFixes"]
        counts = outcome["_confidenceCounts"]["events"]
        total["flaggedEvents"] += counts["flaggedEvents"]
        total["flaggedErrorEvents"] += counts["flaggedErrorEvents"]
        total["errorEvents"] += counts["errorEvents"]
        total["unflaggedEvents"] += counts["unflaggedEvents"]
        total["unflaggedErrorEvents"] += counts["unflaggedErrorEvents"]
        total["flaggedCorrectEvents"] += counts["flaggedCorrectEvents"]
        total["candidateConfidenceCount"] += outcome["_confidenceCounts"]["candidateConfidenceCount"]
        total["candidateHighConfidenceCount"] += outcome["_confidenceCounts"]["candidateHighConfidenceCount"]
        total["candidateHighConfidenceWrongCount"] += outcome["_confidenceCounts"]["candidateHighConfidenceWrongCount"]
        total["candidateLowConfidenceCount"] += outcome["_confidenceCounts"]["candidateLowConfidenceCount"]
        total["candidateLowConfidenceCorrectCount"] += outcome["_confidenceCounts"]["candidateLowConfidenceCorrectCount"]
        for i, bucket in enumerate(outcome["confidence"]["candidateCalibrationBins"]):
            count = int(bucket["count"])
            if count <= 0:
                continue
            total["candidateBinCounts"][i] += count
            total["candidateBinConfidenceSums"][i] += float(bucket["avgConfidence"]) * count
            total["candidateBinCorrectCounts"][i] += float(bucket["accuracy"]) * count

    micro_p = total["tp"] / total["predicted"] if total["predicted"] else 0.0
    micro_r = total["tp"] / total["truth"] if total["truth"] else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    candidate_bins = []
    candidate_ece = None
    if total["candidateConfidenceCount"]:
        weighted_error = 0.0
        for i, (lo, hi) in enumerate(zip(CALIBRATION_BINS, CALIBRATION_BINS[1:])):
            count = total["candidateBinCounts"][i]
            if count:
                avg_conf = total["candidateBinConfidenceSums"][i] / count
                accuracy = total["candidateBinCorrectCounts"][i] / count
                weighted_error += count / total["candidateConfidenceCount"] * abs(accuracy - avg_conf)
                candidate_bins.append(
                    {
                        "range": [round(lo, 1), round(min(hi, 1.0), 1)],
                        "count": count,
                        "avgConfidence": avg_conf,
                        "accuracy": accuracy,
                    }
                )
            else:
                candidate_bins.append({"range": [round(lo, 1), round(min(hi, 1.0), 1)], "count": 0})
        candidate_ece = weighted_error
    summary = {
        # Python sources only — a kalimba_dsp (Rust) rebuild does NOT change
        # this value, so always read it together with kalimbaDspFingerprint.
        "recognizerFingerprint": recognizer_fingerprint(),
        "kalimbaDspFingerprint": kalimba_dsp_fingerprint(),
        "recordings": len(results),
        "microPrecision": micro_p,
        "microRecall": micro_r,
        "microF1": micro_f1,
        "oneBest": {
            "onsetPrecision": micro_p,
            "onsetRecall": micro_r,
            "onsetF1": micro_f1,
        },
        "candidates": {
            "recallAt1": total["candidateHits"]["1"] / total["truth"] if total["truth"] else 1.0,
            "recallAt3": total["candidateHits"]["3"] / total["truth"] if total["truth"] else 1.0,
            "recallAt5": total["candidateHits"]["5"] / total["truth"] if total["truth"] else 1.0,
            "candidateSlotsPerEvent": (
                total["surfaceCandidateNotes"] / total["events"] if total["events"] else 0.0
            ),
            "hardMissRate": total["hardMisses"] / total["truth"] if total["truth"] else 0.0,
            "surfaceCandidateNotes": total["surfaceCandidateNotes"],
            "hardMisses": total["hardMisses"],
        },
        "correction": {
            "estimatedCost": total["estimatedCost"],
            "costPerTruthNote": total["estimatedCost"] / total["truth"] if total["truth"] else 0.0,
            "candidateAssistedFixRate": (
                total["candidateEnabled"] / (total["candidateEnabled"] + total["manualInserts"])
                if (total["candidateEnabled"] + total["manualInserts"]) else None
            ),
            "candidateEnabled": total["candidateEnabled"],
            "manualInserts": total["manualInserts"],
            "noteRemoves": total["noteRemoves"],
            "allFixes": total["allFixes"],
        },
        "confidence": {
            "flaggedEventPrecision": (
                total["flaggedErrorEvents"] / total["flaggedEvents"] if total["flaggedEvents"] else None
            ),
            "missedErrorRate": (
                (total["errorEvents"] - total["flaggedErrorEvents"]) / total["errorEvents"]
                if total["errorEvents"] else None
            ),
            "highConfidenceWrongRate": (
                total["unflaggedErrorEvents"] / total["unflaggedEvents"]
                if total["unflaggedEvents"] else None
            ),
            "lowConfidenceCorrectRate": (
                total["flaggedCorrectEvents"] / total["flaggedEvents"] if total["flaggedEvents"] else None
            ),
            "candidateExpectedCalibrationError": candidate_ece,
            "candidateHighConfidenceWrongRate": (
                total["candidateHighConfidenceWrongCount"] / total["candidateHighConfidenceCount"]
                if total["candidateHighConfidenceCount"] else None
            ),
            "candidateLowConfidenceCorrectRate": (
                total["candidateLowConfidenceCorrectCount"] / total["candidateLowConfidenceCount"]
                if total["candidateLowConfidenceCount"] else None
            ),
            "candidateCalibrationBins": candidate_bins,
        },
    }

    if args.json:
        for r in results:
            r.pop("_confidenceCounts", None)
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return 0

    print(
        f"{'tx':38} {'GT':>4} {'pred':>4} {'TP':>4} {'P':>6} {'R':>6} {'F1':>6}"
        f" {'cR@3':>6} {'hard':>4} {'cost':>5}"
    )
    for r in results:
        print(
            f"{r['txId'][:36]:38} {r['truthNotes']:>4} {r['predictedNotes']:>4} {r['tp']:>4}"
            f" {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}"
            f" {r['candidates']['recallAt3']:6.3f} {r['candidates']['hardMisses']:>4}"
            f" {r['correction']['estimatedCost']:>5}"
        )
        if args.verbose:
            for fp in r["falsePositives"]:
                print(f"    FP {fp['time']:8.3f}s {fp['note']}")
            for fn in r["falseNegatives"]:
                print(f"    FN {fn['time']:8.3f}s {fn['note']}")
            for fn in r["candidates"].get("hardMissNotes", []):
                print(f"    HARD_MISS {fn['time']:8.3f}s {fn['note']}")
    print(
        f"\nmicro P={summary['microPrecision']:.3f} R={summary['microRecall']:.3f}"
        f" F1={summary['microF1']:.3f}  ({summary['recordings']} recordings,"
        f" recognizer {summary['recognizerFingerprint']}"
        f" dsp {summary['kalimbaDspFingerprint']})"
    )
    print(
        "candidate "
        f"R@1={summary['candidates']['recallAt1']:.3f}"
        f" R@3={summary['candidates']['recallAt3']:.3f}"
        f" R@5={summary['candidates']['recallAt5']:.3f}"
        f" hardMiss={summary['candidates']['hardMissRate']:.3f}"
        f" slots/event={summary['candidates']['candidateSlotsPerEvent']:.2f}"
    )
    cafr = summary["correction"]["candidateAssistedFixRate"]
    cafr_text = f"{cafr:.3f}" if cafr is not None else "n/a"
    print(
        "correction "
        f"cost={summary['correction']['estimatedCost']}"
        f" cost/GT={summary['correction']['costPerTruthNote']:.3f}"
        f" candidateFixRate={cafr_text}"
        f" manualInserts={summary['correction']['manualInserts']}"
        f" noteRemoves={summary['correction']['noteRemoves']}"
    )
    fep = summary["confidence"]["flaggedEventPrecision"]
    mer = summary["confidence"]["missedErrorRate"]
    fep_text = f"{fep:.3f}" if fep is not None else "n/a"
    mer_text = f"{mer:.3f}" if mer is not None else "n/a"
    print(f"confidence flaggedPrecision={fep_text} missedErrorRate={mer_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
