#!/usr/bin/env python3
"""gt_finalize.py — /debug/gt-review の verdict から最終 ground_truth.json を生成する。

gt_draft.py が出した rows.json と、/debug/gt-review でユーザーが保存した
verdict.json を突き合わせ、ear_verified な ground_truth.json を
apps/api/tests/fixtures/transaction-captures/<tx-id>/ に書き出す
(gitignored — audio の rights review 前でもローカル benchmark に使える)。

Usage:
  uv run python scripts/audio-analysis/gt_finalize.py <tx8> [<tx8> ...] [--force]

決定規則:
  - 行: 明示 verdict (accept/fix/ignore) > 暗黙 accept (flag=ok)。
    未裁定行が残っている場合はエラー (--force でスキップ扱い)。
  - added: ユーザー手動追加。timing は再生位置由来で粗いので
    per-onset toleranceSec を広げる。
  - unplaced: place → onset (同じく広い tolerance)、discard → 破棄。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFTS_DIR = REPO_ROOT / "data" / "gt_drafts"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"

# gt_draft.py と同じ既定 (onset backtrack 由来のずれ余地)
ROW_TOLERANCE_SEC = 0.08
# 手動配置 (added / unplaced place) は再生位置由来で粗い
MANUAL_TOLERANCE_SEC = 0.25


def finalize(tx8: str, force: bool) -> bool:
    rows_path = DRAFTS_DIR / f"{tx8}.rows.json"
    verdict_path = DRAFTS_DIR / f"{tx8}.verdict.json"
    if not rows_path.is_file():
        print(f"{tx8}: rows.json が無い (gt_draft.py 未実行?)", file=sys.stderr)
        return False
    if not verdict_path.is_file():
        print(f"{tx8}: verdict.json が無い (/debug/gt-review 未裁定)", file=sys.stderr)
        return False
    doc = json.loads(rows_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    undecided: list[int] = []
    onsets: list[dict] = []
    counts = {"accept": 0, "fix": 0, "ignore": 0, "added": 0, "placed": 0, "discarded": 0}

    for row in doc["rows"]:
        rv = verdict.get("rows", {}).get(str(row["index"]), {})
        decision = rv.get("decision") or ("accept" if row["flag"] == "ok" else None)
        if decision is None:
            undecided.append(row["index"])
            continue
        counts[decision] += 1
        if decision == "ignore":
            continue
        notes = rv.get("notes") if decision == "fix" else row["draftNotes"]
        comment_parts = []
        if decision == "fix":
            comment_parts.append(f"user corrected (draft: {'+'.join(row['draftNotes'])})")
        if rv.get("comment"):
            comment_parts.append(rv["comment"])
        # gt_verdict_seed.py 由来の行は耳確認していない (review corrections の
        # 転記) ため user_corrected に留める — ear_verified と偽らない
        # (ガードレール 8)。UI で人間が押し直した行は seeded が外れている
        seeded = bool(rv.get("seeded"))
        onset = {
            "timeSec": row["timeSec"],
            "notes": notes,
            "method": "user_corrected" if seeded else "ear_verified",
            "comment": "; ".join(comment_parts) if comment_parts else (
                "seeded from review corrections" if seeded else "gt-review verdict"
            ),
        }
        # 主旋律を含まない onset (伴奏のみ)。旋律抽出評価で層別できるよう
        # role として構造化して残す (F1 benchmark は notes/timeSec のみ読む)
        if rv.get("accompanimentOnly"):
            onset["role"] = "accompaniment"
        onsets.append(onset)

    for entry in verdict.get("added") or []:
        counts["added"] += 1
        comment = "user-added via /debug/gt-review (見逃し onset, timing は再生位置由来)"
        if entry.get("comment"):
            comment += f"; {entry['comment']}"
        onset = {
            "timeSec": entry["timeSec"],
            "notes": entry["notes"],
            "method": "ear_verified",
            "toleranceSec": MANUAL_TOLERANCE_SEC,
            "comment": comment,
        }
        if entry.get("accompanimentOnly"):
            onset["role"] = "accompaniment"
        onsets.append(onset)

    undecided_unplaced: list[int] = []
    for u in doc.get("unplacedExpected") or []:
        uv = verdict.get("unplaced", {}).get(str(u["index"]))
        if uv is None:
            undecided_unplaced.append(u["index"])
            continue
        if uv.get("decision") == "place":
            counts["placed"] += 1
            onsets.append(
                {
                    "timeSec": uv["timeSec"],
                    "notes": u["notes"],
                    "method": "ear_verified",
                    "toleranceSec": MANUAL_TOLERANCE_SEC,
                    "comment": "user-placed via /debug/gt-review (未検出 expected)",
                }
            )
        else:
            counts["discarded"] += 1

    if (undecided or undecided_unplaced) and not force:
        print(
            f"{tx8}: 未裁定が残っている (rows: {undecided[:10]}{'...' if len(undecided) > 10 else ''}, "
            f"unplaced: {undecided_unplaced}) — --force で未裁定をスキップして生成",
            file=sys.stderr,
        )
        return False

    onsets.sort(key=lambda o: o["timeSec"])
    gt = {
        "version": 1,
        "toleranceSec": ROW_TOLERANCE_SEC,
        "source": {
            "type": "gt-review-verdict",
            "transactionId": doc["txId"],
            "generator": "scripts/audio-analysis/gt_finalize.py",
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdictSavedAt": verdict.get("savedAt"),
            "draftGeneratedAt": doc.get("generatedAt"),
            "expectedSource": doc.get("expectedSource"),
            "reviewDone": bool(verdict.get("done")),
            "comment": verdict.get("comment") or "",
        },
        "onsets": onsets,
    }
    out_dir = CAPTURES_DIR / doc["txId"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ground_truth.json"
    out_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    skipped = f" skipped_undecided={len(undecided) + len(undecided_unplaced)}" if force else ""
    print(f"{tx8}: onsets={len(onsets)} {counts}{skipped} -> {out_path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tx8s", nargs="+", help="transaction id prefix (8 hex)")
    parser.add_argument("--force", action="store_true", help="未裁定行をスキップして生成")
    args = parser.parse_args()
    ok = all([finalize(tx8, args.force) for tx8 in args.tx8s])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
