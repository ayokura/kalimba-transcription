---
name: fixture-rejection-sweep
description: Sweep primary rejection thresholds (PRIMARY_REJECTION_MAX_SCORE / PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO) against the real pytest fixture suite
user_invocable: true
arguments:
  - name: thresholds
    description: "Optional: comma-separated `score:fr` pairs (e.g. `10:0.8 20:0.9 30:0.97`). Defaults to a built-in 8-value sweep."
    required: false
---

<command-name>fixture-rejection-sweep</command-name>

Sweep the primary rejection thresholds (`PRIMARY_REJECTION_MAX_SCORE` and `PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO` in `apps/api/app/transcription/constants.py`) against the real pytest fixture suite (`test_manual_capture_completed.py` + `test_manual_capture_pending.py`), and report pass/fail counts per threshold pair.

## Why this exists

**Always use this script (or the real pytest suite) to evaluate rejection threshold changes.** Ad-hoc event counting scripts that bypass `evaluationWindows` / `ignoredRanges` / `expectedEventNoteSetsOrdered` were the source of false "regression" reports in #66 / #74. This script wraps the real test suite so the assertions are enforced exactly as in CI.

## Mechanism

1. Read current threshold values from `apps/api/app/transcription/constants.py`
2. For each candidate `(score, fr)` pair:
   - Patch the constants in the source
   - Run `uv run pytest test_manual_capture_completed.py test_manual_capture_pending.py -q --tb=line`
   - Collect pass/fail counts
3. Restore the original source on exit (including on error / interrupt)

## Instructions

```bash
# Default sweep (8 built-in pairs)
uv run python scripts/audio-analysis/fixture_rejection_sweep.py

# Custom thresholds (space-separated, each is `score:fr`)
uv run python scripts/audio-analysis/fixture_rejection_sweep.py 10:0.8 20:0.9 30:0.97
```

## Cost

Each threshold pair runs the full fixture pytest, which takes **~3-4 minutes per pair**. The default 8-pair sweep takes **~25-30 minutes**. Use a smaller custom set if iterating quickly.

## Output

For each pair:
- `score=<X>, fr=<Y>: <pytest summary>` (e.g. `316 passed`, `308 passed, 8 failed`)
- Failure list (`FAILED ...` lines from pytest output) if any

## Safety

The script uses an exit handler to restore the original `constants.py` content even on:
- Normal completion
- Exception during patching
- KeyboardInterrupt (Ctrl+C)

If the script is killed forcibly (`kill -9`), the file may remain in a patched state. In that case, `git restore apps/api/app/transcription/constants.py` reverts cleanly.

## Example Usage

```
/fixture-rejection-sweep
/fixture-rejection-sweep 15:0.85 25:0.92
```
