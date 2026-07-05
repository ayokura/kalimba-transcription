#!/usr/bin/env python3
"""gt_verdict_seed.py — review UI の corrections を gt-review verdict の種にする。

通常 review (corrections.json) を済ませた録音の gt-review 裁定で、同じ判断を
二重入力しなくて済むように、時刻マッチする未裁定行へ verdict を seed する
(2026-07-05 ユーザー要望、d7a82772 が動機)。

規則:
  - 既に人間が裁定済みの行は**絶対に上書きしない**
  - corrected event と ±MATCH_TOL_SEC で 1:1 greedy マッチ (近い順)。
    notes が draftNotes と一致 → decision=accept、相違 → decision=fix + notes
  - マッチしない行は未裁定のまま残す (人間の注意をそこへ集中させる)
  - seed 行は "seeded": true を付け、gt_finalize は ear_verified ではなく
    user_corrected として GT 化する (耳確認と偽らない — ガードレール 8)。
    gt-review UI で人間が改めて裁定した行は seeded が外れ ear_verified 扱い

Usage:
    uv run python scripts/audio-analysis/gt_verdict_seed.py <tx8> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFTS_DIR = REPO_ROOT / "data" / "gt_drafts"
TX_DIR = REPO_ROOT / "data" / "transactions"
CORPUS_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "free-performance-corpus"

MATCH_TOL_SEC = 0.15


def find_corrections(tx_id: str) -> Path | None:
    for base in (TX_DIR, CORPUS_DIR):
        p = base / tx_id / "corrections.json"
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tx8")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows_path = DRAFTS_DIR / f"{args.tx8}.rows.json"
    if not rows_path.is_file():
        print(f"{args.tx8}: rows.json が無い (gt_draft.py 未実行?)", file=sys.stderr)
        return 1
    doc = json.loads(rows_path.read_text(encoding="utf-8"))
    tx_id = doc.get("txId") or doc.get("source", {}).get("transactionId")
    if not tx_id:
        # rows.json 直下に txId が無い世代は tx8 から transactions を引く
        hits = [d.name for d in TX_DIR.iterdir() if d.name.startswith(args.tx8)]
        if len(hits) != 1:
            print(f"{args.tx8}: transactionId を特定できない", file=sys.stderr)
            return 1
        tx_id = hits[0]

    corr_path = find_corrections(tx_id)
    if corr_path is None:
        print(f"{args.tx8}: corrections.json が見つからない (通常 review 未実施?)", file=sys.stderr)
        return 1
    corrected = json.loads(corr_path.read_text(encoding="utf-8"))["events"]

    verdict_path = DRAFTS_DIR / f"{args.tx8}.verdict.json"
    verdict = (
        json.loads(verdict_path.read_text(encoding="utf-8"))
        if verdict_path.is_file()
        else {"rows": {}, "unplaced": {}, "done": False}
    )
    verdict.setdefault("rows", {})

    undecided = [
        row for row in doc["rows"]
        if not (verdict["rows"].get(str(row["index"])) or {}).get("decision")
    ]

    # greedy 1:1: (行, corrected event) を時刻差昇順でペアリング
    pairs = sorted(
        (
            (abs(row["timeSec"] - ev["timeSec"]), ri, ei)
            for ri, row in enumerate(undecided)
            for ei, ev in enumerate(corrected)
            if abs(row["timeSec"] - ev["timeSec"]) <= MATCH_TOL_SEC
        ),
    )
    row_used: set[int] = set()
    ev_used: set[int] = set()
    counts = {"accept": 0, "fix": 0}
    for _, ri, ei in pairs:
        if ri in row_used or ei in ev_used:
            continue
        row_used.add(ri)
        ev_used.add(ei)
        row = undecided[ri]
        ev = corrected[ei]
        same = sorted(row["draftNotes"]) == sorted(ev["notes"])
        entry: dict = {
            "decision": "accept" if same else "fix",
            "seeded": True,
            "comment": "[seeded] review corrections "
            + ("と一致" if same else f"の notes を採用 (draft: {'+'.join(row['draftNotes'])})"),
        }
        if not same:
            entry["notes"] = list(ev["notes"])
        if ev.get("accompanimentOnly"):
            entry["accompanimentOnly"] = True
        verdict["rows"][str(row["index"])] = entry
        counts["accept" if same else "fix"] += 1

    unmatched_rows = [undecided[i] for i in range(len(undecided)) if i not in row_used]
    unmatched_evs = [corrected[i] for i in range(len(corrected)) if i not in ev_used]

    print(f"{args.tx8}: rows={len(doc['rows'])} 既裁定={len(doc['rows']) - len(undecided)}")
    print(f"  seeded: accept={counts['accept']} fix={counts['fix']}")
    print(f"  残り未裁定 (人間の裁定対象): {len(unmatched_rows)} 行")
    for row in unmatched_rows:
        print(f"    #{row['index']} {row['timeSec']:.3f}s draft={'+'.join(row['draftNotes'])} flag={row['flag']}")
    if unmatched_evs:
        print(f"  corrections 側の未対応 event (draft 行が無い — added 候補): {len(unmatched_evs)}")
        for ev in unmatched_evs:
            print(f"    {ev['timeSec']:.3f}s {'+'.join(ev['notes'])} origin={ev.get('origin')}")

    if args.dry_run:
        print("(dry-run: 書き込みなし)")
        return 0
    verdict["savedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    verdict_path.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"-> {verdict_path.relative_to(REPO_ROOT)} (done は変更していない)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
