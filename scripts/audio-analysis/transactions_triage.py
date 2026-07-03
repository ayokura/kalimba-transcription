#!/usr/bin/env python3
"""Prescreen data/transactions for non-saturated GT candidates (第 2 期 S1).

「録音は待つものではなく作るもの」の第一歩: 手持ちの生録音バックログを
sha256 で dedupe し、unique 録音ごとに崩壊シグナルを集計して
「非飽和 held-out の GT 候補になりそうな順」にランキングする。

Per unique recording this collects:
- audio stats (duration / sample rate / peak dBFS)
- stored response summary (event count, warnings) + memo + review_status
- GT presence across layers (free-performance-corpus / transaction-captures /
  local ground_truth.json), matched by BOTH tx-id and audio sha
- a fresh transcription with the CURRENT recognizer (TestClient, dryRun) and
  its drift vs the stored response — recognizer が当時から動いた録音は
  再レビューの価値が高い

The ranking is a transparent heuristic (report-only — 回帰 gate や過適合
ゲートの n には使わない。sprint-plan-2026-07b ガードレール 7)。

Output: ranked table (stdout) + ``data/triage_summary.json`` (the /debug/triage
dev page and /api/dev/triage endpoint consume this file).

Usage:
  uv run python scripts/audio-analysis/transactions_triage.py
  uv run python scripts/audio-analysis/transactions_triage.py --no-retranscribe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import note_f1_benchmark as bench  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import recognizer_fingerprint  # noqa: E402

DATA_DIR = bench.DATA_DIR
CAPTURES_DIR = bench.CAPTURES_DIR
CORPUS_DIR = bench.FREE_PERFORMANCE_CORPUS_DIR
SUMMARY_PATH = DATA_DIR.parent / "triage_summary.json"

# 崩壊シグナルの重み (transparent heuristic; 調整は自由 — report-only)
SCORE_COLLAPSE = 5  # events <= 1 なのに演奏長がある = ほぼ確実に非飽和
SCORE_LOW_DENSITY = 2  # イベント密度が演奏として不自然に低い
SCORE_DRIFT = 2  # 現行 recognizer で event 数が大きく動いた
SCORE_BURIED_CORRECTIONS = 2  # corrections 済みなのに GT 未生成 (埋もれ)
SCORE_MEMO = 1  # ユーザーが何かに気づいてメモを残した
SCORE_WARNING = 1  # per warning (cap 2)
SCORE_LOW_PEAK = -3  # 低信号 → unusable の可能性が高い (負のシグナル)
SCORE_ALREADY_JUDGED = -10  # 人間が unusable / rerecord_needed と判定済み

LOW_PEAK_DBFS = -35.0
LOW_DENSITY_EVENTS_PER_SEC = 0.15
DRIFT_RATIO = 0.3
# 再エンコードでバイト列が変わった同一録音を捕まえる相関しきい値
DUPLICATE_CORRELATION = 0.999
CLOSED_STATUSES = {"unusable", "rerecord_needed"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stats(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    peak_dbfs = float(20.0 * np.log10(peak)) if peak > 0 else None
    return {
        "durationSec": round(len(audio) / sample_rate, 1),
        "sampleRate": int(sample_rate),
        "peakDbfs": round(peak_dbfs, 1) if peak_dbfs is not None else None,
    }


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def gt_layers_by_sha_and_id() -> tuple[dict[str, str], dict[str, str]]:
    """Map audio-sha -> layer and tx-id -> layer for existing ground truth."""
    by_sha: dict[str, str] = {}
    by_id: dict[str, str] = {}
    for layer, root in (("corpus", CORPUS_DIR), ("captures", CAPTURES_DIR), ("local", DATA_DIR)):
        if not root.is_dir():
            continue
        for tx_dir in root.iterdir():
            if not (tx_dir / "ground_truth.json").is_file():
                continue
            by_id.setdefault(tx_dir.name, layer)
            audio = tx_dir / "audio.wav"
            if audio.is_file():
                by_sha.setdefault(sha256_of(audio), layer)
    return by_sha, by_id


def _mono_samples(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio)


def collect_unique_recordings() -> dict[str, list[Path]]:
    """Group transaction dirs by recording identity.

    2 段階: (1) file sha256 (完全一致の再アップロード)、(2) 同一サンプル数の
    グループ間で正規化相関 > DUPLICATE_CORRELATION なら同一録音とみなして
    マージ (再エンコードでバイト列だけ変わったケース — 実例: 2772aa11 は
    2bf55c75 と corr 1.000 の同一録音)。
    """
    groups: dict[str, list[Path]] = {}
    for tx_dir in sorted(DATA_DIR.iterdir()):
        audio = tx_dir / "audio.wav"
        if audio.is_file():
            groups.setdefault(sha256_of(audio), []).append(tx_dir)

    # 相関ベースの二次マージ (同一サンプル数のグループのみ比較)
    by_length: dict[int, list[str]] = {}
    lengths: dict[str, int] = {}
    for sha, dirs in groups.items():
        n_samples = len(_mono_samples(dirs[0] / "audio.wav"))
        lengths[sha] = n_samples
        by_length.setdefault(n_samples, []).append(sha)
    for shas in by_length.values():
        if len(shas) < 2:
            continue
        samples = {sha: _mono_samples(groups[sha][0] / "audio.wav") for sha in shas}
        merged: set[str] = set()
        for i, sha_a in enumerate(shas):
            if sha_a in merged:
                continue
            for sha_b in shas[i + 1 :]:
                if sha_b in merged:
                    continue
                a, b = samples[sha_a], samples[sha_b]
                denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
                if denom == 0:
                    continue
                corr = float(np.dot(a, b)) / denom
                if corr > DUPLICATE_CORRELATION:
                    groups[sha_a].extend(groups.pop(sha_b))
                    merged.add(sha_b)

    # 代表 = 最新 mtime の dir (レビュー状態が最も進んでいる可能性が高い)
    for sha, dirs in groups.items():
        groups[sha] = sorted(dirs, key=lambda d: (d / "audio.wav").stat().st_mtime, reverse=True)
    return groups


def summarize_group(
    sha: str,
    dirs: list[Path],
    gt_by_sha: dict[str, str],
    gt_by_id: dict[str, str],
    client: TestClient | None,
) -> dict[str, Any]:
    primary = dirs[0]
    stats = audio_stats(primary / "audio.wav")

    response = read_json(primary / "response.json") or {}
    stored_events = len(response.get("events", [])) if response else None
    warnings = response.get("warnings", []) or []
    request = read_json(primary / "request.json") or {}
    tuning_id = (request.get("tuning") or {}).get("id")

    # 譜面の成否を試聴と突き合わせられるよう、認識結果のイベント列 (音名) と
    # 期待列 (expectedPerformance があれば) を持たせる
    recognized_events = [
        "+".join(f"{note['pitchClass']}{note['octave']}" for note in event.get("notes", []))
        for event in response.get("events", [])
    ][:64]
    expected_perf = request.get("expectedPerformance") or None
    expected_events = (
        [str(ev.get("display", "")).replace(" ", "") for ev in expected_perf.get("events", [])]
        if expected_perf
        else None
    )

    statuses = {}
    memo = None
    corrections = False
    for tx_dir in dirs:
        status = read_json(tx_dir / "review_status.json")
        statuses[tx_dir.name] = status.get("status") if status else None
        if memo is None and (tx_dir / "memo.txt").is_file():
            text = (tx_dir / "memo.txt").read_text(encoding="utf-8").strip()
            memo = text or None
        corrections = corrections or (tx_dir / "corrections.json").is_file()

    gt_layer = gt_by_sha.get(sha) or next(
        (gt_by_id[d.name] for d in dirs if d.name in gt_by_id), None
    )
    judged_closed = any(status in CLOSED_STATUSES for status in statuses.values())

    fresh_events = None
    if client is not None and gt_layer is None:
        try:
            payload = bench.transcribe_payload(client, primary.name, debug=False)
            fresh_events = len(payload.get("events", []))
        except Exception as exc:  # noqa: BLE001 - triage は落とさず記録する
            fresh_events = None
            warnings = [*warnings, f"retranscribe failed: {exc}"]

    signals: list[str] = []
    score = 0
    duration = stats["durationSec"]
    if stored_events is not None and stored_events <= 1 and duration > 10:
        score += SCORE_COLLAPSE
        signals.append(f"collapse: {stored_events} events / {duration}s")
    elif stored_events is not None and duration > 15:
        density = stored_events / duration
        if density < LOW_DENSITY_EVENTS_PER_SEC:
            score += SCORE_LOW_DENSITY
            signals.append(f"low-density: {density:.2f} ev/s")
    if fresh_events is not None and stored_events:
        drift = abs(fresh_events - stored_events) / max(stored_events, 1)
        if drift > DRIFT_RATIO:
            score += SCORE_DRIFT
            signals.append(f"drift: stored {stored_events} -> fresh {fresh_events}")
    if corrections and gt_layer is None:
        score += SCORE_BURIED_CORRECTIONS
        signals.append("corrections-without-gt")
    if memo:
        score += SCORE_MEMO
        signals.append("has-memo")
    if warnings:
        bump = min(len(warnings), 2) * SCORE_WARNING
        score += bump
        signals.append(f"warnings x{len(warnings)}")
    if stats["peakDbfs"] is not None and stats["peakDbfs"] < LOW_PEAK_DBFS:
        score += SCORE_LOW_PEAK
        signals.append(f"low-peak {stats['peakDbfs']} dBFS (unusable risk)")
    if judged_closed:
        score += SCORE_ALREADY_JUDGED
        signals.append("already-judged: unusable/rerecord")

    return {
        "sha16": sha[:16],
        "primaryTx": primary.name,
        "duplicateTxs": [d.name for d in dirs[1:]],
        **stats,
        "tuningId": tuning_id,
        "storedEvents": stored_events,
        "freshEvents": fresh_events,
        "recognizedEvents": recognized_events,
        "expectedEvents": expected_events,
        "warnings": warnings,
        "memo": memo,
        "reviewStatuses": statuses,
        "hasCorrections": corrections,
        "gtLayer": gt_layer,
        "score": score,
        "signals": signals,
    }


def build_summary(retranscribe: bool) -> dict[str, Any]:
    gt_by_sha, gt_by_id = gt_layers_by_sha_and_id()
    groups = collect_unique_recordings()
    client = TestClient(bench.app) if retranscribe else None

    rows = [
        summarize_group(sha, dirs, gt_by_sha, gt_by_id, client)
        for sha, dirs in groups.items()
    ]
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)

    status_counts: dict[str, int] = {}
    for row in rows:
        for status in row["reviewStatuses"].values():
            key = status or "unset"
            status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recognizerFingerprint": recognizer_fingerprint(),
        "totals": {
            "transactionDirs": sum(len(d) for d in groups.values()),
            "uniqueRecordings": len(groups),
            "withGt": sum(1 for r in rows if r["gtLayer"]),
            "statusCounts": status_counts,
        },
        "recordings": ranked,
    }


def print_table(summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    print(
        f"dirs={totals['transactionDirs']} unique={totals['uniqueRecordings']} "
        f"withGT={totals['withGt']} statuses={totals['statusCounts']}"
    )
    print(f"recognizer: {summary['recognizerFingerprint']}")
    print()
    header = f"{'score':>5}  {'tx (primary)':<38} {'dur':>6} {'peak':>6} {'ev':>4} {'GT':<8} signals"
    print(header)
    print("-" * len(header))
    for row in summary["recordings"]:
        peak = row["peakDbfs"] if row["peakDbfs"] is not None else "-"
        print(
            f"{row['score']:>5}  {row['primaryTx']:<38} {row['durationSec']:>6} {peak:>6} "
            f"{row['storedEvents'] if row['storedEvents'] is not None else '-':>4} "
            f"{row['gtLayer'] or '-':<8} {'; '.join(row['signals'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-retranscribe",
        action="store_true",
        help="skip fresh transcription (faster; drift signal unavailable)",
    )
    args = parser.parse_args()

    summary = build_summary(retranscribe=not args.no_retranscribe)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nwrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
