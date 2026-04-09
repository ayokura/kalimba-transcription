---
name: score-alignment
description: Diagnose recognition accuracy by aligning expected events (score_structure.json or expectedPerformance fallback) with recognizer output
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

Compare expected events with recognizer output using ordered matching. If the fixture has a `score_structure.json` (multi-line score structure, e.g. BWV147 sequence-163), use that for per-line matching. Otherwise fall back to a synthetic single line built from a simpler metadata source so the script also works on simple fixtures (e.g. `c4-repeat-01`, `mixed-sequence-01`).

## Expected event source priority

1. **`score_structure.json`** — multi-line score with `lines[].id`, `eventRange`, `content`. Used when present (e.g. BWV147 sequence-163).
2. **`request.json:expectedPerformance.events`** — Web UI's clickable-kalimba-ui ordered event list. Set on every captured fixture. Synthesized into a single line `ALL`.
3. **`expected.json:assertions.expectedEventNoteSetsOrdered`** — last-resort fallback. Only set on promoted (completed) fixtures; derived from recognizer output, so it requires the recognizer to already match the truth.

The script logs which source was used via stderr `[synthetic-line] using <source> (N events)` when falling back.

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
