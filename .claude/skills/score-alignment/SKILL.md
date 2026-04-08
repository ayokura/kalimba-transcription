---
name: score-alignment
description: Diagnose recognition accuracy by aligning score_structure events with recognizer output
user_invocable: true
arguments:
  - name: fixture
    description: Fixture name (e.g., bwv147-sequence-163-01)
    required: true
  - name: options
    description: "Options: --verbose, --mode events|segments, --line L1"
    required: false
---

<command-name>score-alignment</command-name>

Compare expected events from score_structure.json with recognizer output using ordered matching within each line.

## Requirements
- Fixture must have a `score_structure.json` file

## Instructions

Run the alignment diagnosis script:
```bash
uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture> [--verbose] [--mode events|segments] [--line L1]
```

## Modes

| Mode | Description |
|------|-------------|
| `events` (default) | Post-processed final output (mergedEvents). Reflects all event-level corrections. |
| `segments` | Raw segmentCandidates before event post-processing. Shows segment_peaks output directly. |

## Output symbols
| Symbol | Meaning |
|--------|---------|
| ✓ | Exact match (all notes correct) |
| ⊂ | Subset — detected notes are correct but incomplete (POLYPHONY limit) |
| ⊃ | Superset — all expected notes detected plus extra |
| △ | Partial — some overlap but missing and extra notes |
| ✗ | Mismatch — no note overlap |
| ∅ | No matching segment found (onset missing) |

Rejection reasons for missing notes are shown in `[note→reason]` format (cross-referenced from segmentCandidates trail in both modes).

Per-line `+N extra segments:` lines list detected events that did not match any expected event in the line.  The bottom SUMMARY block reports the total via `Extra segments: N` — useful for spotting over-detection regressions across runs.

## Caching

スクリプトは transcription 結果を `apps/api/tests/.cache/score_alignment/` に永続化する。キャッシュキーには `audio bytes` + リクエストデータ + `apps/api/app/transcription/` 配下の `.py` ファイル全体のフィンガープリントが含まれる。

- 同じ fixture を `--line` 違いで繰り返し呼ぶと 2 回目以降は **~1秒** で完了 (フルパイプライン ~30s-3min を回避)
- recognizer コードを編集すると次回は自動的に miss → fresh run (stale を踏まない設計)
- 編集を revert すると元のキーに戻ってヒットする

無効化したいとき (デバッグ等): `SCORE_ALIGNMENT_NO_CACHE=1 uv run python scripts/audio-analysis/score_alignment_diagnosis.py ...`
キャッシュディレクトリ削除: `rm -rf apps/api/tests/.cache/score_alignment/` (累積したエントリを掃除したい場合)

stderr に `[cache hit] <key prefix>` / `[cache miss] <key prefix>` が出るので動作確認可能。

## Known Limitations

**個別イベントの failure 分析には使わないこと。** ordered matching はセグメント/イベント数と期待イベント数が一致しない場合にアライメントがずれ、誤った failure 原因を報告する。

- **全体精度（exact %）と failure カテゴリの大局把握には有効**
- **個別イベントの原因分析には `/audio-energy-trace` + debug 出力の手動確認を推奨**

## Example Usage
```
/score-alignment bwv147-sequence-163-01
/score-alignment bwv147-sequence-163-01 --verbose
/score-alignment bwv147-sequence-163-01 --mode segments
/score-alignment bwv147-sequence-163-01 --line R4
```
