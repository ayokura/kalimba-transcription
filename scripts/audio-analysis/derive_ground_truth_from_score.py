#!/usr/bin/env python3
"""Derive ground_truth.json for a score-known transaction recording.

For tester recordings of a piece we already have a completed fixture for
(e.g. the BWV147 playback uploads), the note content of the ground truth
comes from the fixture's `expected.json:assertions.expectedEventNoteSetsOrdered`
(score truth, post alignment-overrides) — independent of the recognizer.
Onset *times* are taken from the current recognizer output after verifying a
strict 1:1 sequence alignment between expected note sets and predicted events.

Caveat (by design): timings are recognizer-derived, so this GT primarily
tracks *regressions* (note-content changes, dropped/added events, >tolerance
timing shifts) rather than validating absolute onset times. Entries are
marked `method: "score_aligned"` with the source fingerprint recorded.

The tool refuses to write GT unless the alignment is perfect (every expected
event matched exactly once, in order, with an identical note set). Imperfect
recordings print a diff for human follow-up instead.

Usage:
  uv run python scripts/audio-analysis/derive_ground_truth_from_score.py \\
      <tx-id> --fixture kalimba-34l-c-bwv147-sequence-163-01 [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import recognizer_fingerprint  # noqa: E402
from apps.api.app.main import app  # noqa: E402

DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"
MANUAL_CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures"

DEFAULT_TOLERANCE_SEC = 0.08


_CHROMATIC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
              "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _pitch_key(note: str) -> tuple[int, int]:
    pitch_class = note.rstrip("0123456789")
    octave = int(note[len(pitch_class):])
    return (octave, _CHROMATIC[pitch_class])


def note_set_token(notes: list[str]) -> str:
    """Canonical token: notes sorted by pitch, so source ordering differences
    (e.g. 'F4+A4' vs 'A4+F4') never count as mismatches."""
    return "+".join(sorted(set(notes), key=_pitch_key))


def load_expected_tokens(fixture: str) -> list[str]:
    expected_path = MANUAL_CAPTURES_DIR / fixture / "expected.json"
    doc = json.loads(expected_path.read_text(encoding="utf-8"))
    ordered = doc.get("assertions", {}).get("expectedEventNoteSetsOrdered")
    if not isinstance(ordered, list) or not ordered:
        raise SystemExit(f"{fixture}: expected.json has no expectedEventNoteSetsOrdered")
    return [note_set_token(token.split("+")) for token in ordered]


def transcribe(client: TestClient, tx_id: str) -> list[dict]:
    tx_dir = DATA_DIR / tx_id
    audio_bytes = (tx_dir / "audio.wav").read_bytes()
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "false",
            "dryRun": "true",
            "force": "true",
        },
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    events = []
    for event in response.json()["events"]:
        notes = [f"{n['pitchClass']}{n['octave']}" for n in event["notes"]]
        events.append(
            {
                "time": float(event["startTimeSec"]),
                "notes": notes,
                "token": note_set_token(notes),
            }
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tx_id", help="transaction ID (data/transactions/<tx-id>/)")
    parser.add_argument(
        "--fixture", required=True,
        help="manual-capture fixture name supplying expectedEventNoteSetsOrdered",
    )
    parser.add_argument("--dry-run", action="store_true", help="print GT without writing")
    parser.add_argument("--force", action="store_true", help="overwrite existing ground_truth.json")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_SEC,
        help=f"GT toleranceSec (default {DEFAULT_TOLERANCE_SEC})",
    )
    args = parser.parse_args()

    gt_path = CAPTURES_DIR / args.tx_id / "ground_truth.json"
    if gt_path.is_file() and not args.force and not args.dry_run:
        print(f"SKIP: {gt_path.relative_to(REPO_ROOT)} exists — use --force to overwrite")
        return 1

    expected_tokens = load_expected_tokens(args.fixture)
    predicted = transcribe(TestClient(app), args.tx_id)
    predicted_tokens = [e["token"] for e in predicted]

    matcher = difflib.SequenceMatcher(a=expected_tokens, b=predicted_tokens, autojunk=False)
    opcodes = matcher.get_opcodes()
    mismatches = [op for op in opcodes if op[0] != "equal"]
    if mismatches:
        print(
            f"REFUSE: alignment is not 1:1 "
            f"(expected {len(expected_tokens)} events, predicted {len(predicted_tokens)}). Diff:"
        )
        for tag, i1, i2, j1, j2 in mismatches:
            exp = expected_tokens[i1:i2]
            pred = [
                f"{predicted[j]['token']}@{predicted[j]['time']:.2f}s" for j in range(j1, j2)
            ]
            print(f"  {tag:8} expected[{i1}:{i2}]={exp}  predicted[{j1}:{j2}]={pred}")
        print("Resolve by ear/spectrogram verification, then write GT manually.")
        return 1

    onsets = [
        {
            "timeSec": round(e["time"], 4),
            "notes": expected_tokens[i].split("+"),
            "method": "score_aligned",
        }
        for i, e in enumerate(predicted)
    ]
    gt = {
        "version": 1,
        "toleranceSec": args.tolerance,
        "source": {
            "type": "score-aligned",
            "fixture": args.fixture,
            "timingRecognizerFingerprint": recognizer_fingerprint()[:16],
            "comment": "notes from score truth (expectedEventNoteSetsOrdered); "
                       "times from recognizer output at the recorded fingerprint",
        },
        "onsets": onsets,
    }

    if args.dry_run:
        print(json.dumps(gt, ensure_ascii=False, indent=2))
        return 0

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {gt_path.relative_to(REPO_ROOT)} ({len(onsets)} onsets, 1:1 alignment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
