---
name: score-alignment
description: Diagnose recognition accuracy by aligning score_structure events with recognizer output
user_invocable: true
arguments:
  - name: fixture
    description: Fixture name (e.g., bwv147-sequence-163-01)
    required: true
  - name: verbose
    description: Show exact matches too (default false)
    required: false
---

<command-name>score-alignment</command-name>

Compare expected events from score_structure.json with recognizer output using ordered matching within each line.

## Requirements
- Fixture must have a `score_structure.json` file

## Instructions

Run the alignment diagnosis script:
```bash
uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture> [--verbose]
```

## Output symbols
| Symbol | Meaning |
|--------|---------|
| ✓ | Exact match (all notes correct) |
| ⊂ | Subset — detected notes are correct but incomplete (POLYPHONY limit) |
| ⊃ | Superset — all expected notes detected plus extra |
| △ | Partial — some overlap but missing and extra notes |
| ✗ | Mismatch — no note overlap |
| ∅ | No matching segment found (onset missing) |

Rejection reasons for missing notes are shown in `[note→reason]` format.

## Known Limitations

**個別イベントの failure 分析には使わないこと。** greedy ordered matching はセグメント数と期待イベント数が一致しない場合にアライメントがずれ、誤った failure 原因を報告する。例: multi-note セグメントが直前の single-note イベントに割り当てられ、本来の multi-note イベントに別セグメントが当たるケース（2026-03-29 bwv147-163 R4 E142 で確認）。

- **全体精度（exact %）と failure カテゴリの大局把握には有効**
- **個別イベントの原因分析には `/audio-energy-trace` + debug 出力の手動確認を推奨**

## Example Usage
```
/score-alignment bwv147-sequence-163-01
/score-alignment bwv147-sequence-163-01 --verbose
```
