from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

REQUIRED_FILES = ("audio.wav", "request.json", "response.json", "notes.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a manual capture ZIP into API test fixtures.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("fixture_id")
    parser.add_argument("--expected-note")
    parser.add_argument("--min-events", type=int)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-primary-occurrences", type=int)
    parser.add_argument("--max-primary-note", action="append", default=[])
    parser.add_argument("--max-primary-occurrences", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("apps/api/tests/fixtures/manual-captures"))
    args = parser.parse_args()

    has_expectation = any(
        value is not None and value != []
        for value in (args.expected_note, args.min_events, args.max_events, args.min_primary_occurrences, args.max_primary_note)
    )
    if not has_expectation and not args.allow_incomplete:
        raise SystemExit("Refusing to create a weak fixture. Pass explicit expectations or use --allow-incomplete.")

    target_dir = args.root / args.fixture_id
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as archive:
        names = archive.namelist()
        folder_prefix = _detect_prefix(names)
        for file_name in REQUIRED_FILES:
            source_name = f"{folder_prefix}{file_name}"
            if source_name not in names:
                raise FileNotFoundError(f"Missing {source_name} in archive")
            with archive.open(source_name) as source, (target_dir / file_name).open("wb") as destination:
                shutil.copyfileobj(source, destination)

    expected = {
        "pending": not has_expectation,
        "assertions": {
            "minEvents": args.min_events,
            "maxEvents": args.max_events,
            "requiredPrimaryNoteOccurrences": {},
            "maxPrimaryNoteOccurrences": {},
        },
    }
    if args.expected_note and args.min_primary_occurrences is not None:
        expected["assertions"]["requiredPrimaryNoteOccurrences"][args.expected_note] = args.min_primary_occurrences
    for note in args.max_primary_note:
        expected["assertions"]["maxPrimaryNoteOccurrences"][note] = args.max_primary_occurrences

    (target_dir / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def _detect_prefix(names: list[str]) -> str:
    for name in names:
        if name.endswith("audio.wav"):
            return name[: -len("audio.wav")]
    raise FileNotFoundError("audio.wav not found in archive")


if __name__ == "__main__":
    raise SystemExit(main())
