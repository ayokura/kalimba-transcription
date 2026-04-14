"""Profile the transcription pipeline on representative fixtures.

Outputs wall-clock time, cProfile cumulative time breakdown, and
per-module aggregation to help identify Rust-ification candidates.

Usage:
    uv run python scripts/profiling/profile_transcription.py [fixture_name ...]

If no fixtures are given, runs a default small/medium/large set.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "apps" / "api" / "tests"
API_DIR = REPO_ROOT / "apps" / "api"
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from conftest import _transcribe_manual_capture_fixture  # noqa: E402


DEFAULT_FIXTURES = [
    "kalimba-17-c-d5-repeat-01",
    "kalimba-17-c-c4-to-e6-sequence-17-single-01",
    "kalimba-17-c-c4-to-e6-sequence-17-repeat-03-01",
    "kalimba-17-c-bwv147-sequence-163-01",
]


def _run_once(fixture_name: str) -> float:
    t0 = time.perf_counter()
    _transcribe_manual_capture_fixture(fixture_name, True, None, True)
    return time.perf_counter() - t0


def _profile_one(fixture_name: str, out_dir: Path) -> dict:
    _transcribe_manual_capture_fixture.cache_clear()
    t_first = _run_once(fixture_name)

    _transcribe_manual_capture_fixture.cache_clear()
    prof = cProfile.Profile()
    prof.enable()
    _run_once(fixture_name)
    prof.disable()

    stats_path = out_dir / f"{fixture_name}.pstats"
    prof.dump_stats(str(stats_path))

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
    top_cumulative = buf.getvalue()

    buf2 = io.StringIO()
    pstats.Stats(prof, stream=buf2).sort_stats("tottime").print_stats(40)
    top_tottime = buf2.getvalue()

    module_totals: dict[str, float] = defaultdict(float)
    pst = pstats.Stats(prof)
    for (filename, _lineno, _funcname), (_cc, _nc, tt, _ct, _callers) in pst.stats.items():
        key = _categorize(filename)
        module_totals[key] += tt
    total = sum(module_totals.values())

    lines = [
        f"=== {fixture_name} ===",
        f"wall-clock: first={t_first:.2f}s profiled={sum(module_totals.values()):.2f}s (tottime sum)",
        "",
        "Module category breakdown (tottime):",
    ]
    for key, tt in sorted(module_totals.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * tt / total if total else 0.0
        lines.append(f"  {key:<40s} {tt:8.2f}s  {pct:5.1f}%")
    summary = "\n".join(lines)
    return {
        "fixture": fixture_name,
        "wall_first": t_first,
        "summary": summary,
        "top_cumulative": top_cumulative,
        "top_tottime": top_tottime,
        "module_totals": dict(module_totals),
        "total_tottime": total,
    }


def _categorize(filename: str) -> str:
    if "transcription/peaks" in filename:
        return "app/transcription/peaks.py"
    if "transcription/events" in filename:
        return "app/transcription/events.py"
    if "transcription/segments" in filename:
        return "app/transcription/segments.py"
    if "transcription/pipeline" in filename:
        return "app/transcription/pipeline.py"
    if "transcription/noise_floor" in filename:
        return "app/transcription/noise_floor.py"
    if "transcription/per_note" in filename:
        return "app/transcription/per_note.py"
    if "transcription/patterns" in filename:
        return "app/transcription/patterns.py"
    if "/transcription/" in filename:
        return "app/transcription/(other)"
    if "/librosa/" in filename:
        return "librosa"
    if "numpy" in filename:
        return "numpy"
    if "scipy" in filename:
        return "scipy"
    if "soundfile" in filename or "_soundfile" in filename:
        return "soundfile"
    if "fastapi" in filename or "starlette" in filename or "pydantic" in filename:
        return "fastapi/starlette/pydantic"
    if filename.startswith("<"):
        return f"builtin {filename}"
    if "/app/" in filename:
        return "app/(other)"
    return "other/stdlib"


def main(argv: list[str]) -> int:
    fixtures = argv[1:] or DEFAULT_FIXTURES
    out_dir = REPO_ROOT / "docs" / "performance" / "profile-raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = REPO_ROOT / "docs" / "performance" / "profile-summary.txt"
    all_results = []
    for fx in fixtures:
        print(f"profiling {fx} ...", flush=True)
        result = _profile_one(fx, out_dir)
        all_results.append(result)
        print(result["summary"])
        print()

    with report_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(result["summary"])
            f.write("\n\nTop 40 by cumulative:\n")
            f.write(result["top_cumulative"])
            f.write("\nTop 40 by tottime:\n")
            f.write(result["top_tottime"])
            f.write("\n\n")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
