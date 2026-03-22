from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("audio.wav", "request.json", "response.json", "notes.md")
VALID_STATUSES = {"completed", "pending", "rerecord", "review_needed", "reference_only"}


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
    parser.add_argument("--required-event-note-set", action="append", default=[])
    parser.add_argument("--max-event-note-set", action="append", default=[])
    parser.add_argument("--status", choices=sorted(VALID_STATUSES))
    parser.add_argument("--reason")
    parser.add_argument("--audit-verdict")
    parser.add_argument("--recommended-recapture", action="append", default=[])
    parser.add_argument("--evaluation-window", action="append", default=[])
    parser.add_argument("--ignored-range", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("apps/api/tests/fixtures/manual-captures"))
    args = parser.parse_args()

    has_expectation = any(
        value is not None and value != []
        for value in (
            args.expected_note,
            args.min_events,
            args.max_events,
            args.min_primary_occurrences,
            args.max_primary_note,
            args.required_event_note_set,
            args.max_event_note_set,
        )
    )

    status = args.status or ("completed" if has_expectation else "pending")
    if status == "completed" and not has_expectation:
        raise SystemExit("Completed fixtures require explicit expectations.")
    if not has_expectation and not args.allow_incomplete and status != "completed":
        raise SystemExit("Refusing to create a weak fixture. Pass explicit expectations or use --allow-incomplete.")
    if status in {"pending", "rerecord", "review_needed", "reference_only"} and not args.reason:
        raise SystemExit(f"Status '{status}' requires --reason.")
    if status == "rerecord" and not args.recommended_recapture:
        raise SystemExit("Status 'rerecord' requires at least one --recommended-recapture entry.")
    if args.evaluation_window and args.ignored_range:
        raise SystemExit("Use either --evaluation-window or --ignored-range, not both.")

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

    expected: dict[str, Any] = {
        "pending": status != "completed",
        "status": status,
        "assertions": {
            "minEvents": args.min_events,
            "maxEvents": args.max_events,
            "requiredPrimaryNoteOccurrences": {},
            "maxPrimaryNoteOccurrences": {},
            "requiredEventNoteSetOccurrences": parse_note_set_assertions(args.required_event_note_set),
            "maxEventNoteSetOccurrences": parse_note_set_assertions(args.max_event_note_set),
        },
    }
    if args.expected_note and args.min_primary_occurrences is not None:
        expected["assertions"]["requiredPrimaryNoteOccurrences"][args.expected_note] = args.min_primary_occurrences
    for note in args.max_primary_note:
        expected["assertions"]["maxPrimaryNoteOccurrences"][note] = args.max_primary_occurrences

    if args.reason:
        expected["reason"] = args.reason
    if args.audit_verdict:
        expected["auditVerdict"] = args.audit_verdict
    if args.recommended_recapture:
        expected["recommendedRecapture"] = args.recommended_recapture
    if args.evaluation_window:
        expected["evaluationWindows"] = parse_range_specs(args.evaluation_window)
    if args.ignored_range:
        expected["ignoredRanges"] = parse_range_specs(args.ignored_range)

    (target_dir / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def parse_note_set_assertions(entries: list[str]) -> dict[str, int]:
    assertions: dict[str, int] = {}
    for entry in entries:
        note_set, separator, raw_count = entry.partition("=")
        if separator != "=" or not note_set or not raw_count:
            raise SystemExit(f"Invalid note-set assertion '{entry}'. Use NOTESET=COUNT, e.g. B4+D5=5")
        assertions[note_set] = int(raw_count)
    return assertions


def parse_range_specs(entries: list[str]) -> list[dict[str, float]]:
    ranges: list[dict[str, float]] = []
    previous_end = None
    for entry in entries:
        start_text, separator, end_text = entry.partition(":")
        if separator != ":" or not start_text or not end_text:
            raise SystemExit(f"Invalid range '{entry}'. Use START:END in seconds, e.g. 0.5:3.25")
        start_sec = float(start_text)
        end_sec = float(end_text)
        if start_sec < 0 or end_sec <= start_sec:
            raise SystemExit(f"Invalid range '{entry}'. END must be greater than START and both must be non-negative.")
        if previous_end is not None and start_sec < previous_end:
            raise SystemExit(f"Ranges must be non-overlapping and sorted: '{entry}'")
        ranges.append({"startSec": start_sec, "endSec": end_sec})
        previous_end = end_sec
    return ranges


def _detect_prefix(names: list[str]) -> str:
    for name in names:
        if name.endswith("audio.wav"):
            return name[: -len("audio.wav")]
    raise FileNotFoundError("audio.wav not found in archive")


if __name__ == "__main__":
    raise SystemExit(main())
