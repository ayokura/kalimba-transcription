#!/usr/bin/env python3
"""Triage: dedupe data/transactions/ by audio SHA-256, re-evaluate each unique
audio with the currently-loaded recognizer, and emit a priority-ranked report.

Usage:
  uv run python scripts/audio-analysis/triage_unique_audios.py [--verbose]
  uv run python scripts/audio-analysis/triage_unique_audios.py --json > /tmp/triage.json

Re-transcription uses dryRun=true + force=true (no server-side persistence,
no hash dedup). stored response.json is compared against the fresh run so
event-count shifts flag fixtures where the recognizer behavior changed.

Rough prioritization heuristic (higher is more urgent):
  +40 if new event count is 0 (recognizer broke)
  +30 if |delta| >= 10 events
  +20 if |delta / old_count| >= 0.3 (significant shift)
  +10 if audio is healthy (peak >= -12 dB) but new onset rate < 0.5 ev/s (under-detection)
  -20 if audio peak < -15 dB (recording issue, not recognizer's fault)
  -10 if this audio was already transcribed with the current recognizer
       (recognizerFingerprint == current), so no re-evaluation value
  -100 if a closing verdict (correct_detection / recording_issue / wontfix) is
       recorded — physically-verified recordings must not be re-flagged
       (the ev/s heuristic mis-flags slow performances; see 0002b267 / 4e0f6c49)

Verdicts persist in transaction-captures/triage_verdicts.json, keyed by
(audioSha256, tuningId). Record one after physical verification:

  uv run python scripts/audio-analysis/triage_unique_audios.py \\
      --set-verdict 0002b267 --tuning kalimba-17-c \\
      --verdict correct_detection --method energy_trace+ear_verified \\
      --comment "D4+D5 octave dyad x5; 6.46/11.8s onsets are mute contacts"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import (  # noqa: E402
    git_head_sha,
    kalimba_dsp_fingerprint,
    recognizer_fingerprint,
)
from apps.api.app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"
VERDICTS_PATH = (
    REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"
    / "triage_verdicts.json"
)

# Verdicts that close a triage item (sink to the bottom of the priority list).
# "recognizer_bug" keeps the item visible: the bug is confirmed and tracked,
# re-evaluation after recognizer changes is still meaningful.
CLOSING_VERDICTS = {"correct_detection", "recording_issue", "wontfix"}
VALID_VERDICTS = CLOSING_VERDICTS | {"recognizer_bug"}


def load_verdicts() -> dict[tuple[str, str], dict]:
    if not VERDICTS_PATH.is_file():
        return {}
    try:
        doc = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"[warn] failed to read {VERDICTS_PATH}", file=sys.stderr)
        return {}
    return {
        (v["audioSha256"], v["tuningId"]): v
        for v in doc.get("verdicts", [])
        if v.get("audioSha256") and v.get("tuningId")
    }


def save_verdict(entry: dict) -> None:
    doc = {"version": 1, "verdicts": []}
    if VERDICTS_PATH.is_file():
        doc = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    verdicts = [
        v for v in doc.get("verdicts", [])
        if (v.get("audioSha256"), v.get("tuningId"))
        != (entry["audioSha256"], entry["tuningId"])
    ]
    verdicts.append(entry)
    doc["verdicts"] = sorted(verdicts, key=lambda v: (v["audioSha256"], v["tuningId"]))
    VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICTS_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@dataclass
class TransactionMeta:
    tx_id: str
    mtime: float
    audio_bytes: bytes
    audio_sha256: str
    tuning: dict
    stored_response: dict
    peak_db: float
    duration_sec: float


def _audio_peak_db(audio_bytes: bytes) -> tuple[float, float]:
    """Return (peak_db, duration_sec) from WAV bytes. peak_db = 0 means full-scale int16."""
    import io
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    duration = n / sr if sr else 0.0
    if sw != 2:
        return (0.0, duration)
    import array
    a = array.array("h", raw)
    if not a:
        return (0.0, duration)
    peak = max(abs(v) for v in a)
    if peak == 0:
        return (float("-inf"), duration)
    db = 20 * math.log10(peak / 32768.0)
    return (round(db, 2), round(duration, 3))


def load_all_transactions() -> list[TransactionMeta]:
    results: list[TransactionMeta] = []
    if not DATA_DIR.exists():
        return results
    for tx_dir in sorted(DATA_DIR.iterdir()):
        if not tx_dir.is_dir():
            continue
        audio_path = tx_dir / "audio.wav"
        request_path = tx_dir / "request.json"
        response_path = tx_dir / "response.json"
        if not (audio_path.exists() and request_path.exists() and response_path.exists()):
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            response = json.loads(response_path.read_text(encoding="utf-8"))
            audio_bytes = audio_path.read_bytes()
        except (OSError, json.JSONDecodeError):
            continue
        audio_sha256 = request.get("audioSha256")
        if not audio_sha256:
            import hashlib
            audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        tuning = request.get("tuning") or {}
        peak_db, duration = _audio_peak_db(audio_bytes)
        results.append(
            TransactionMeta(
                tx_id=tx_dir.name,
                mtime=audio_path.stat().st_mtime,
                audio_bytes=audio_bytes,
                audio_sha256=audio_sha256,
                tuning=tuning,
                stored_response=response,
                peak_db=peak_db,
                duration_sec=duration,
            )
        )
    return results


def pick_representative(entries: list[TransactionMeta]) -> TransactionMeta:
    """Among duplicates of the same (hash, tuningId), pick the most recent."""
    return max(entries, key=lambda e: e.mtime)


def score_priority(
    old_events: int,
    new_events: int,
    onset_rate: float,
    peak_db: float,
    stored_recognizer_fp: str | None,
    current_recognizer_fp: str,
    verdict: dict | None = None,
) -> tuple[int, list[str]]:
    score = 0
    tags: list[str] = []
    delta = new_events - old_events
    if new_events == 0:
        score += 40
        tags.append("broken")
    if abs(delta) >= 10:
        score += 30
        tags.append(f"big-delta({delta:+d})")
    elif old_events > 0 and abs(delta / old_events) >= 0.3:
        score += 20
        tags.append(f"shift({delta:+d})")
    if peak_db >= -12 and onset_rate < 0.5:
        score += 10
        tags.append("under-detect")
    if peak_db < -15:
        score -= 20
        tags.append("low-gain")
    if stored_recognizer_fp == current_recognizer_fp:
        score -= 10
        tags.append("same-recognizer")
    if verdict is not None:
        tags.append(f"verdict:{verdict['verdict']}")
        if verdict["verdict"] in CLOSING_VERDICTS:
            score -= 100
    return score, tags


def main():
    parser = argparse.ArgumentParser(description="Dedupe + re-evaluate tester captures")
    parser.add_argument("--verbose", action="store_true", help="Show all transactions per hash, not just representative")
    parser.add_argument("--json", action="store_true", help="Emit JSON (not table)")
    parser.add_argument("--filter-tuning", type=str, default=None, help="Only process transactions with this tuning id")
    parser.add_argument("--set-verdict", type=str, default=None, metavar="SHA_PREFIX",
                        help="Record a verdict for the audio matching this SHA-256 prefix (requires --tuning, --verdict)")
    parser.add_argument("--tuning", type=str, default=None, help="tuning id for --set-verdict")
    parser.add_argument("--verdict", type=str, default=None, choices=sorted(VALID_VERDICTS),
                        help="verdict value for --set-verdict")
    parser.add_argument("--method", type=str, default=None,
                        help="verification method for --set-verdict (e.g. energy_trace+ear_verified)")
    parser.add_argument("--comment", type=str, default=None, help="verdict comment for --set-verdict")
    parser.add_argument("--issue", type=int, default=None, help="tracking issue number for --set-verdict")
    args = parser.parse_args()

    if args.set_verdict:
        if not (args.tuning and args.verdict):
            parser.error("--set-verdict requires --tuning and --verdict")
        matches = sorted(
            {t.audio_sha256 for t in load_all_transactions()
             if t.audio_sha256.startswith(args.set_verdict) and t.tuning.get("id") == args.tuning}
        )
        if len(matches) != 1:
            parser.error(
                f"SHA prefix {args.set_verdict!r} + tuning {args.tuning!r} matched "
                f"{len(matches)} unique audios (need exactly 1): {[m[:12] for m in matches]}"
            )
        import datetime
        entry = {
            "audioSha256": matches[0],
            "tuningId": args.tuning,
            "verdict": args.verdict,
            "verifiedAt": datetime.date.today().isoformat(),
        }
        if args.method:
            entry["method"] = args.method
        if args.comment:
            entry["comment"] = args.comment
        if args.issue:
            entry["issue"] = args.issue
        save_verdict(entry)
        print(f"recorded verdict {args.verdict} for {matches[0][:12]} ({args.tuning})"
              f" -> {VERDICTS_PATH.relative_to(REPO_ROOT)}")
        return

    current_recognizer_fp = recognizer_fingerprint()
    current_dsp_fp = kalimba_dsp_fingerprint()
    current_commit = git_head_sha()

    all_tx = load_all_transactions()
    if args.filter_tuning:
        all_tx = [t for t in all_tx if t.tuning.get("id") == args.filter_tuning]
    verdicts = load_verdicts()

    # Group by (audio_sha256, tuning_id) — same audio + different tuning = separate entry
    groups: dict[tuple[str, str | None], list[TransactionMeta]] = defaultdict(list)
    for tx in all_tx:
        key = (tx.audio_sha256, tx.tuning.get("id"))
        groups[key].append(tx)

    reps = [pick_representative(entries) for entries in groups.values()]

    client = TestClient(app)

    results: list[dict] = []
    for rep in reps:
        duplicate_ids = sorted(
            (e.tx_id for e in groups[(rep.audio_sha256, rep.tuning.get("id"))]),
            key=lambda tid: tid,
        )
        stored_request = {}
        stored_request_path = DATA_DIR / rep.tx_id / "request.json"
        try:
            stored_request = json.loads(stored_request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

        old_events = len(rep.stored_response.get("events") or [])

        # Re-transcribe with latest recognizer. dryRun + force = no persist, no cache hit.
        response = client.post(
            "/api/transcriptions",
            data={
                "tuning": json.dumps(rep.tuning),
                "debug": "true",
                "dryRun": "true",
                "force": "true",
            },
            files={"file": ("audio.wav", rep.audio_bytes, "audio/wav")},
        )
        if response.status_code != 200:
            print(f"[error] {rep.tx_id[:8]}: HTTP {response.status_code}", file=sys.stderr)
            continue
        new = response.json()
        new_events = len(new.get("events") or [])
        new_event_rate = round(new_events / rep.duration_sec, 3) if rep.duration_sec > 0 else 0.0

        verdict = verdicts.get((rep.audio_sha256, rep.tuning.get("id") or ""))
        score, tags = score_priority(
            old_events,
            new_events,
            new_event_rate,
            rep.peak_db,
            stored_request.get("recognizerFingerprint"),
            current_recognizer_fp,
            verdict=verdict,
        )

        results.append(
            {
                "audioSha256_prefix": rep.audio_sha256[:8],
                "tuningId": rep.tuning.get("id"),
                "representativeId": rep.tx_id,
                "duplicateIds": duplicate_ids,
                "durationSec": rep.duration_sec,
                "peakDb": rep.peak_db,
                "oldEvents": old_events,
                "newEvents": new_events,
                "delta": new_events - old_events,
                "newEventRate": new_event_rate,
                "storedCommit": stored_request.get("commitSha"),
                "storedRecognizerFp": stored_request.get("recognizerFingerprint"),
                "currentRecognizerFp": current_recognizer_fp,
                "priorityScore": score,
                "tags": tags,
                "verdict": verdict,
            }
        )

    results.sort(key=lambda r: -r["priorityScore"])

    if args.json:
        print(json.dumps({
            "currentCommit": current_commit,
            "currentRecognizerFp": current_recognizer_fp,
            "currentDspFp": current_dsp_fp,
            "uniqueAudioCount": len(results),
            "transactionCount": len(all_tx),
            "items": results,
        }, indent=2, ensure_ascii=False))
        return

    print(f"current commit: {current_commit}")
    print(f"current recognizer fp: {current_recognizer_fp}  dsp fp: {current_dsp_fp}")
    print(f"transactions: {len(all_tx)}  unique (hash, tuning) pairs: {len(results)}")
    print()
    print(f"{'prio':>4}  {'hash':8}  {'tuning':16}  {'dur':>6}  {'peak':>7}  {'old':>4}  {'new':>4}  {'Δ':>4}  {'rate':>5}  {'storedFp':10}  tags")
    print("-" * 110)
    for r in results:
        dur = f"{r['durationSec']:.1f}s"
        peak = f"{r['peakDb']:.1f}" if r['peakDb'] != float('-inf') else "-inf"
        stored_fp = (r['storedRecognizerFp'] or '--')[:8]
        delta_str = f"{r['delta']:+d}" if r['delta'] != 0 else "0"
        tags_str = ",".join(r['tags']) if r['tags'] else "-"
        print(
            f"{r['priorityScore']:>4}  "
            f"{r['audioSha256_prefix']:8}  "
            f"{(r['tuningId'] or '?'):16}  "
            f"{dur:>6}  "
            f"{peak:>7}  "
            f"{r['oldEvents']:>4}  "
            f"{r['newEvents']:>4}  "
            f"{delta_str:>4}  "
            f"{r['newEventRate']:>5}  "
            f"{stored_fp:10}  "
            f"{tags_str}"
        )
        if args.verbose and r['duplicateIds']:
            print(f"        transactions: {', '.join(tid[:8] for tid in r['duplicateIds'])}")


if __name__ == "__main__":
    main()
