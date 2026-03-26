from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import soundfile as sf

VALID_STATUSES = {"completed", "pending", "rerecord", "review_needed", "reference_only"}


def build_summary(events: list[dict[str, Any]]) -> str:
    return " / ".join(event.get("display", "") for event in events)


def clip_audio_bytes(source_audio: Path, start_sec: float, end_sec: float) -> bytes:
    audio, sample_rate = sf.read(source_audio, always_2d=True)
    start_index = max(0, int(round(start_sec * sample_rate)))
    end_index = min(audio.shape[0], int(round(end_sec * sample_rate)))
    if end_index <= start_index:
        raise SystemExit("Selected audio range is empty")
    clipped = audio[start_index:end_index]
    import io
    buffer = io.BytesIO()
    sf.write(buffer, clipped, sample_rate, format="WAV")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a smaller manual-capture fixture from an existing source fixture.")
    parser.add_argument("source_fixture_dir", type=Path)
    parser.add_argument("target_fixture_dir", type=Path)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--start-event", type=int, required=True)
    parser.add_argument("--end-event", type=int, required=True)
    parser.add_argument("--context-before", type=float, default=0.0)
    parser.add_argument("--context-after", type=float, default=0.0)
    parser.add_argument("--status", choices=sorted(VALID_STATUSES), default="pending")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--scenario")
    args = parser.parse_args()

    if args.end_sec <= args.start_sec:
        raise SystemExit("--end-sec must be greater than --start-sec")
    if args.context_before < 0 or args.context_after < 0:
        raise SystemExit("--context-before/--context-after must be non-negative")
    if args.start_event < 1 or args.end_event < args.start_event:
        raise SystemExit("Invalid event range")

    source_dir = args.source_fixture_dir
    target_dir = args.target_fixture_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    request_payload = json.loads((source_dir / "request.json").read_text(encoding="utf-8"))
    expected_performance = request_payload.get("expectedPerformance") or {}
    source_events = expected_performance.get("events") or []
    selected_events = source_events[args.start_event - 1:args.end_event]
    if not selected_events:
        raise SystemExit("No expected events selected")

    reindexed_events: list[dict[str, Any]] = []
    for index, event in enumerate(selected_events, start=1):
        copied = json.loads(json.dumps(event, ensure_ascii=False))
        copied["index"] = index
        reindexed_events.append(copied)

    expected_performance["events"] = reindexed_events
    expected_performance["summary"] = build_summary(reindexed_events)
    request_payload["expectedNote"] = expected_performance["summary"]
    request_payload["expectedPerformance"] = expected_performance
    request_payload["scenario"] = args.scenario or target_dir.name

    (target_dir / "request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    request_audio = request_payload.get("audio") or {}
    source_duration = float(request_audio.get("durationSec") or 0.0)
    clip_start = max(0.0, args.start_sec - args.context_before)
    clip_end = args.end_sec + args.context_after
    if source_duration > 0:
        clip_end = min(source_duration, clip_end)

    audio_bytes = clip_audio_bytes(source_dir / "audio.wav", clip_start, clip_end)
    (target_dir / "audio.wav").write_bytes(audio_bytes)

    response_path = source_dir / "response.json"
    if response_path.exists():
        shutil.copy2(response_path, target_dir / "response.json")

    expected_payload = {
        "pending": args.status != "completed",
        "status": args.status,
        "assertions": {
            "minEvents": None,
            "maxEvents": None,
            "requiredPrimaryNoteOccurrences": {},
            "maxPrimaryNoteOccurrences": {},
            "requiredEventNoteSetOccurrences": {},
            "maxEventNoteSetOccurrences": {},
        },
        "reason": args.reason,
    }
    if clip_start != args.start_sec or clip_end != args.end_sec:
        expected_payload["evaluationWindows"] = [
            {
                "startSec": round(args.start_sec - clip_start, 4),
                "endSec": round(args.end_sec - clip_start, 4),
            }
        ]
    (target_dir / "expected.json").write_text(json.dumps(expected_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    notes = [
        "# Manual Notes",
        "",
        "- tester: manual",
        f"- verdict: {args.status}",
        f"- scenario: {request_payload['scenario']}",
        f"- expected note: {expected_performance['summary']}",
        f"- capture intent: {request_payload.get('captureIntent', 'unknown')}",
        f"- default capture intent: {expected_performance.get('defaultCaptureIntent', 'unknown')}",
        f"- source profile: {request_payload.get('sourceProfile', 'acoustic_real')}",
        f"- captured at: {request_payload.get('capturedAt', '')}",
        f"- source fixture: {source_dir.name}",
        f"- extracted range: {args.start_sec:.3f}s - {args.end_sec:.3f}s",
        f"- audio clip range: {clip_start:.3f}s - {clip_end:.3f}s",
        f"- extracted events: {args.start_event}-{args.end_event}",
        "",
        "## Expected Performance",
        "",
        f"- summary: {expected_performance['summary']}",
        "",
        "### Events",
        "",
    ]
    for event in reindexed_events:
        keys = event.get("keys") or []
        rendered_keys = ", ".join(f"{key['noteName']} (#{key['key']})" for key in keys)
        intent = event.get("intent")
        intent_suffix = f" [intent: {intent}]" if intent else ""
        notes.append(f"- {event['index']}. {event.get('display', '')}{intent_suffix} :: {rendered_keys}")
    notes.extend([
        "",
        "## Review",
        "",
        f"- summary: {args.reason}",
        f"- reason: {args.reason}",
    ])
    (target_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
