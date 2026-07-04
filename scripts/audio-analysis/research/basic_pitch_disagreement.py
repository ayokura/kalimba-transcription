"""Basic Pitch disagreement map (S6 bets #3, #199).

Runs Spotify basic-pitch (isolated Python 3.11 subprocess — see
basic_pitch_infer.py) against every GT recording and classifies each GT note
into: both-hit / recognizer-only / bp-only / both-miss.

**bp-only = the recognizer's blind spot that a second opinion could point
at.**  This is a REPORT-ONLY research tool: BP output is a disagreement map,
NOT a pseudo-label source (never promotes anything to completed without
human verification — see sprint plan S6).

Usage (from repo root):
    uv run python scripts/audio-analysis/research/basic_pitch_disagreement.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "transactions"
CORPUS_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "free-performance-corpus"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"
CACHE_DIR = REPO_ROOT / "data" / "basic_pitch_cache"

# Non-saturated GT recordings (F1 < 1.0 as of 2026-07-05) — the interesting
# targets for a blind-spot map.
TX_IDS = [
    "17ea7626-3c5d-450d-ae74-0116dea6e881",
    "4e1ae5c6-df9a-4876-917d-b7e47699c8e5",
    "9ce7df83-33a0-455d-bf86-c9392ce6f777",
    "a9e30986-5300-4401-8b69-152cba821042",
    "d7a82772-f77f-4820-9798-00133ae45f4e",
    "ea7edd71-e815-4638-a248-a47fe21e5061",
    "ebecf0c6-7e41-430b-bd60-8111a495185e",
]

NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()
BP_MATCH_TOLERANCE_SEC = 0.15  # BP onsets are model-quantized; be lenient.


def tx_dir_for(tx_id: str) -> Path:
    for base in (CORPUS_DIR, DATA_DIR):
        d = base / tx_id
        if (d / "audio.wav").is_file() and (d / "request.json").is_file():
            return d
    raise FileNotFoundError(tx_id)


def gt_path_for(tx_id: str) -> Path:
    for base in (CORPUS_DIR, CAPTURES_DIR):
        p = base / tx_id / "ground_truth.json"
        if p.is_file():
            return p
    raise FileNotFoundError(f"ground_truth.json for {tx_id}")


def midi_to_name(midi: int) -> str:
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def name_to_midi(name: str) -> int:
    # "C#5" / "C5" style
    pc = name[:-1]
    octave = int(name[-1])
    return NOTE_NAMES.index(pc) + (octave + 1) * 12


def run_basic_pitch(tx_id: str, audio_path: Path) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{tx_id}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    result = subprocess.run(
        [
            "uv", "run", "--python", "3.11", "--no-project",
            "--with", "basic-pitch", "--with", "setuptools<81",
            "python", str(REPO_ROOT / "scripts/audio-analysis/research/basic_pitch_infer.py"),
            str(audio_path),
        ],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"basic-pitch failed for {tx_id}: {result.stderr[-500:]}")
    rows = json.loads(result.stdout)
    cache.write_text(json.dumps(rows))
    return rows


def recognizer_notes(client: TestClient, tx_dir: Path) -> list[tuple[float, str]]:
    request = json.loads((tx_dir / "request.json").read_text())
    resp = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "false", "dryRun": "true", "force": "true",
        },
        files={"file": ("audio.wav", (tx_dir / "audio.wav").read_bytes(), "audio/wav")},
    )
    resp.raise_for_status()
    out = []
    for event in resp.json()["events"]:
        for note in event["notes"]:
            out.append((float(event["startTimeSec"]), f"{note['pitchClass']}{note['octave']}"))
    return out


def main() -> int:
    client = TestClient(app)
    totals = {"both": 0, "recognizer_only": 0, "bp_only": 0, "both_miss": 0}
    bp_extra_total = 0
    gt_total = 0
    print(f"{'tx':10s} {'GT':>3s} {'both':>5s} {'recOnly':>7s} {'bpOnly':>6s} {'miss':>4s} {'bpExtra':>7s}")
    blind_spots: list[str] = []
    for tx_id in TX_IDS:
        tx_dir = tx_dir_for(tx_id)
        gt = json.loads(gt_path_for(tx_id).read_text())
        tol = float(gt.get("toleranceSec", 0.08))
        bp = run_basic_pitch(tx_id, tx_dir / "audio.wav")
        rec = recognizer_notes(client, tx_dir)
        rec_used = [False] * len(rec)
        bp_used = [False] * len(bp)
        counts = {"both": 0, "recognizer_only": 0, "bp_only": 0, "both_miss": 0}
        for onset in gt["onsets"]:
            t = float(onset["timeSec"])
            names = onset.get("notes") or [onset.get("note")]
            for name in names:
                gt_total += 1
                midi = name_to_midi(name)
                rec_hit = None
                for i, (rt, rn) in enumerate(rec):
                    if not rec_used[i] and rn == name and abs(rt - t) <= tol:
                        rec_hit = i
                        break
                bp_hit = None
                for i, row in enumerate(bp):
                    if not bp_used[i] and row["midi"] == midi and abs(row["start"] - t) <= BP_MATCH_TOLERANCE_SEC:
                        bp_hit = i
                        break
                if rec_hit is not None:
                    rec_used[rec_hit] = True
                if bp_hit is not None:
                    bp_used[bp_hit] = True
                if rec_hit is not None and bp_hit is not None:
                    counts["both"] += 1
                elif rec_hit is not None:
                    counts["recognizer_only"] += 1
                elif bp_hit is not None:
                    counts["bp_only"] += 1
                    blind_spots.append(f"{tx_id[:8]} {t:8.3f}s {name}")
                else:
                    counts["both_miss"] += 1
        bp_extra = sum(1 for u in bp_used if not u)
        bp_extra_total += bp_extra
        n_gt = sum(len(o.get("notes") or [1]) for o in gt["onsets"])
        print(f"{tx_id[:8]:10s} {n_gt:3d} {counts['both']:5d} {counts['recognizer_only']:7d} {counts['bp_only']:6d} {counts['both_miss']:4d} {bp_extra:7d}")
        for k in totals:
            totals[k] += counts[k]
    print(f"{'TOTAL':10s} {gt_total:3d} {totals['both']:5d} {totals['recognizer_only']:7d} {totals['bp_only']:6d} {totals['both_miss']:4d} {bp_extra_total:7d}")
    print()
    print("=== recognizer blind spots pointed at by Basic Pitch (bp_only) ===")
    for line in blind_spots:
        print(" ", line)
    if not blind_spots:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
