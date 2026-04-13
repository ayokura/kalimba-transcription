---
name: score-alignment
description: Diagnose recognition accuracy by aligning expected events (score_structure.json or expectedPerformance fallback) with recognizer output
user_invocable: true
arguments:
  - name: fixture
    description: Fixture name (e.g., bwv147-sequence-163-01)
    required: true
  - name: options
    description: "Options: --verbose, --mode events|segments|candidates, --line L1, --no-cache"
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
uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture> [--verbose] [--mode events|segments|candidates] [--line L1] [--no-cache]
```

## Modes

| Mode | Description |
|------|-------------|
| `events` (default) | Post-processed final output (mergedEvents). Reflects all event-level corrections. |
| `segments` | Raw segmentCandidates before event post-processing. Shows segment_peaks output directly. |
| `candidates` | All segments with full candidate lists. No expected data required. Designed for Free Performance fixtures. |

### candidates mode

Free Performance fixture 等、期待データのない fixture でも使用可能。各セグメントの全候補（採用・棄却とも）をスコア・onset証拠・棄却理由付きで一覧表示する。

- デフォルトでは active segments のみ表示。`--verbose` で dropped segments と NarrowFFT detail も表示
- 末尾に final events 一覧を出力
- evaluation scope cropping を適用せず full audio で認識を実行

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

transcription 結果は `apps/api/tests/.cache/score_alignment/` に永続化される。キャッシュキーには `audio bytes` + リクエストデータ + `apps/api/app/transcription/` 配下の `.py` ファイル全体のフィンガープリントが含まれる。

- 同じ fixture を `--line` 違いで繰り返し呼ぶと 2 回目以降は **~1秒** で完了 (フルパイプライン ~30s-3min を回避)
- recognizer コードを編集すると次回は自動的に miss → fresh run (stale を踏まない設計)
- 編集を revert すると元のキーに戻ってヒットする

`--no-cache` を指定するとキャッシュ読み取りをスキップして fresh run を強制するが、結果はキャッシュに書き込まれるので次回は高速。recognizer コードを変更した場合はフィンガープリントが自動で変わるため `--no-cache` は不要 — データファイル変更やキャッシュ破損を疑う場合にのみ使う。

SUMMARY ブロックの末尾に `Cache: hit/miss/fresh <key> (recognizer: <fp>)` が出力される。`recognizer:` の値で、どのコードバージョンの結果かを確認できる。

キャッシュディレクトリ削除: `rm -rf apps/api/tests/.cache/score_alignment/` (累積したエントリを掃除したい場合)

### Cache を直接読んで recognizer 内部挙動を解析する

cache JSON には `payload["debug"]["segmentCandidates"]` 以下に **全 segment の rich debug info** (primaryCandidate / rankedCandidates / secondaryDecisionTrail / 物理量) が含まれている。 これを inline Python で読むことで、 fixture を再実行せずに以下が可能:

- **特定 reason を持つ event を抽出**: `secondaryDecisionTrail` の `reasons` field で `score-below-threshold` / `contiguous-tertiary-extension` 等を grep
- **物理量比較**: false positive と true positive を `attackEnergy` / `sustainEnergy` / `fundamentalRatio` / `candidateOnsetGain` 等で比較し discriminator を発見
- **gate 設計の calibration data 収集**: 複数 fixture の cache を読んで threshold の境界を実証的に決定

詳細な構造、 inline Python パターン、 G3d (2026-04-10) での実証例は `memory/reference_score_alignment_cache.md` を参照。

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
/score-alignment kalimba-17-c-free-performance-01 --mode candidates
/score-alignment kalimba-17-c-free-performance-01 --mode candidates --verbose
```
