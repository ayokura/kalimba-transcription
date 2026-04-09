#!/usr/bin/env python3
"""Sweep primary rejection thresholds against the real test suite.

IMPORTANT: Always use this script (or the real pytest suite) to evaluate
rejection threshold changes. Never rely on ad-hoc event counting scripts
that bypass evaluation windows, ignoredRanges, and expectedEventNoteSetsOrdered.
Ad-hoc counting was the source of false "regression" reports in #66/#74.

This script:
1. Reads the current threshold values from transcription.py
2. For each candidate threshold pair, patches the source temporarily
3. Runs the actual pytest fixture tests (completed + pending)
4. Reports pass/fail counts per threshold
5. Restores the original source on exit (including on error/interrupt)

Usage:
    uv run python scripts/audio-analysis/fixture_rejection_sweep.py

    # Custom thresholds (comma-separated score:fr pairs)
    uv run python scripts/audio-analysis/fixture_rejection_sweep.py 10:0.8 20:0.9 30:0.97
"""

import re
import subprocess
import sys
from pathlib import Path

SRC = Path("apps/api/app/transcription/constants.py")

SCORE_PATTERN = re.compile(r"PRIMARY_REJECTION_MAX_SCORE\s*=\s*([\d.]+)")
FR_PATTERN = re.compile(r"PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO\s*=\s*([\d.]+)")

DEFAULT_THRESHOLDS = [
    (5.0, 0.5),
    (10.0, 0.8),
    (15.0, 0.85),
    (20.0, 0.9),
    (25.0, 0.9),
    (30.0, 0.95),
    (30.0, 0.97),
    (50.0, 0.97),
]


def read_current_thresholds(source: str) -> tuple[float, float]:
    score_match = SCORE_PATTERN.search(source)
    fr_match = FR_PATTERN.search(source)
    if not score_match or not fr_match:
        print("ERROR: cannot find threshold constants in source", file=sys.stderr)
        sys.exit(1)
    return float(score_match.group(1)), float(fr_match.group(1))


def patch_source(source: str, max_score: float, max_fr: float) -> str:
    patched = SCORE_PATTERN.sub(f"PRIMARY_REJECTION_MAX_SCORE = {max_score}", source)
    patched = FR_PATTERN.sub(f"PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO = {max_fr}", patched)
    return patched


def run_fixture_tests() -> tuple[str, list[str]]:
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
    return summary, failures


def parse_args() -> list[tuple[float, float]]:
    if len(sys.argv) <= 1:
        return DEFAULT_THRESHOLDS
    thresholds = []
    for arg in sys.argv[1:]:
        parts = arg.split(":")
        if len(parts) != 2:
            print(f"ERROR: invalid threshold format '{arg}', expected score:fr", file=sys.stderr)
            sys.exit(1)
        thresholds.append((float(parts[0]), float(parts[1])))
    return thresholds


def main() -> None:
    original = SRC.read_text()
    current_score, current_fr = read_current_thresholds(original)
    thresholds = parse_args()

    print(f"Current thresholds: score<{current_score} FR<{current_fr}")
    print(f"Testing {len(thresholds)} threshold combinations\n")

    try:
        for max_score, max_fr in thresholds:
            patched = patch_source(original, max_score, max_fr)
            SRC.write_text(patched)

            marker = " ← current" if max_score == current_score and max_fr == current_fr else ""
            summary, failures = run_fixture_tests()

            print(f"score<{max_score} FR<{max_fr}: {summary}{marker}")
            for f in failures:
                print(f"  {f}")
    finally:
        SRC.write_text(original)
        print("\nRestored original thresholds")


if __name__ == "__main__":
    main()
