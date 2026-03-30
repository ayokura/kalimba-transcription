#!/usr/bin/env python3
"""Sweep primary rejection thresholds against the real test suite.

Unlike ad-hoc analysis scripts that count raw events, this script
patches PRIMARY_REJECTION_MAX_SCORE / PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO
in the source and runs the actual pytest test suite, which correctly
handles evaluation windows, ignoredRanges, and expectedEventNoteSetsOrdered.

Usage:
    uv run python scripts/audio-analysis/fixture_rejection_sweep.py

Outputs pass/fail counts for each threshold combination.
"""

import json
import subprocess
import sys
from pathlib import Path

THRESHOLDS = [
    (5.0, 0.5),    # current
    (10.0, 0.8),
    (15.0, 0.85),
    (20.0, 0.9),
    (25.0, 0.9),
    (30.0, 0.95),
    (50.0, 0.95),
]

SRC = Path("apps/api/app/transcription.py")
ORIGINAL_LINE = "PRIMARY_REJECTION_MAX_SCORE = 5.0\nPRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO = 0.5"


def main() -> None:
    original = SRC.read_text()
    if ORIGINAL_LINE not in original:
        print("ERROR: cannot find original threshold line in source", file=sys.stderr)
        sys.exit(1)

    try:
        for max_score, max_fr in THRESHOLDS:
            patched = original.replace(
                ORIGINAL_LINE,
                f"PRIMARY_REJECTION_MAX_SCORE = {max_score}\n"
                f"PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO = {max_fr}",
            )
            SRC.write_text(patched)

            result = subprocess.run(
                [
                    "uv", "run", "pytest",
                    "apps/api/tests/test_manual_capture_completed.py",
                    "apps/api/tests/test_manual_capture_pending.py",
                    "-q", "--tb=line",
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            lines = result.stdout.strip().split("\n")
            summary = lines[-1] if lines else "?"
            failures = [line for line in lines if "FAILED" in line]

            print(f"score<{max_score} FR<{max_fr}: {summary}")
            for f in failures:
                print(f"  {f}")
    finally:
        SRC.write_text(original)
        print("\nRestored original thresholds")


if __name__ == "__main__":
    main()
