#!/usr/bin/env python3
"""Regenerate response.json for all manual-capture fixtures using the current API.

Usage:
    uv run python scripts/sync_fixture_responses.py [--no-debug] [FIXTURE_NAME ...]

If FIXTURE_NAME(s) are given, only those fixtures are updated.
Otherwise all fixtures under apps/api/tests/fixtures/manual-captures/ are processed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "api"))
FIXTURE_ROOT = ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync fixture response.json to current API output.")
    parser.add_argument("fixtures", nargs="*", help="Fixture directory names to update (default: all)")
    parser.add_argument("--no-debug", action="store_true", help="Omit debug payload from response")
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    if args.fixtures:
        fixture_dirs = []
        for name in args.fixtures:
            d = FIXTURE_ROOT / name
            if not d.is_dir():
                print(f"ERROR: fixture not found: {d}", file=sys.stderr)
                return 1
            fixture_dirs.append(d)
    else:
        fixture_dirs = sorted(p for p in FIXTURE_ROOT.iterdir() if p.is_dir())

    debug_flag = "false" if args.no_debug else "true"
    total = len(fixture_dirs)
    errors: list[str] = []

    for idx, fixture_dir in enumerate(fixture_dirs, 1):
        name = fixture_dir.name
        request_path = fixture_dir / "request.json"
        audio_path = fixture_dir / "audio.wav"

        if not request_path.exists() or not audio_path.exists():
            print(f"  [{idx}/{total}] SKIP {name} (missing request.json or audio.wav)")
            continue

        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        audio_bytes = audio_path.read_bytes()

        data: dict[str, str] = {
            "tuning": json.dumps(request_payload["tuning"]),
            "debug": debug_flag,
        }
        if request_payload.get("midPerformanceStart"):
            data["midPerformanceStart"] = "true"
        if request_payload.get("midPerformanceEnd"):
            data["midPerformanceEnd"] = "true"

        t0 = time.monotonic()
        response = client.post(
            "/api/transcriptions",
            data=data,
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        )
        elapsed = time.monotonic() - t0

        if response.status_code != 200:
            msg = f"{name}: HTTP {response.status_code}"
            errors.append(msg)
            print(f"  [{idx}/{total}] ERROR {msg}")
            continue

        result = response.json()
        event_count = len(result.get("events", []))
        response_path = fixture_dir / "response.json"
        response_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  [{idx}/{total}] OK {name} ({event_count} events, {elapsed:.1f}s)")

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"\nDone. {total} fixture(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
