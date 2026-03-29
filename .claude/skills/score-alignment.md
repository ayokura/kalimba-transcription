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

## Example Usage
```
/score-alignment bwv147-sequence-163-01
/score-alignment bwv147-sequence-163-01 --verbose
```
