from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

VALID_STATUSES = {"completed", "pending", "rerecord", "review_needed", "reference_only"}


@dataclass
class ActivityRegion:
    start_sec: float
    end_sec: float


def fixture_status(expected: dict[str, Any]) -> str:
    status = expected.get("status")
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown fixture status: {status}")
        return status
    return "pending" if expected.get("pending") else "completed"


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def detect_activity_regions(audio: np.ndarray, sample_rate: int) -> list[ActivityRegion]:
    if audio.size == 0:
        return []
    frame_length = 4096
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
    noise_floor = float(np.percentile(rms, 35))
    threshold = max(noise_floor * 2.5, float(np.max(rms)) * 0.08, 1e-5)
    active = rms >= threshold

    regions: list[ActivityRegion] = []
    start_index: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            start_sec = max(0.0, float(times[start_index]) - 0.03)
            end_sec = float(times[index - 1]) + 0.08
            if end_sec - start_sec >= 0.08:
                regions.append(ActivityRegion(start_sec=start_sec, end_sec=end_sec))
            start_index = None
    if start_index is not None:
        start_sec = max(0.0, float(times[start_index]) - 0.03)
        end_sec = float(times[-1]) + 0.08
        if end_sec - start_sec >= 0.08:
            regions.append(ActivityRegion(start_sec=start_sec, end_sec=end_sec))
    return merge_adjacent_regions(regions)


def merge_adjacent_regions(regions: list[ActivityRegion]) -> list[ActivityRegion]:
    if not regions:
        return []
    merged = [regions[0]]
    for region in regions[1:]:
        previous = merged[-1]
        if region.start_sec - previous.end_sec <= 0.12:
            merged[-1] = ActivityRegion(start_sec=previous.start_sec, end_sec=max(previous.end_sec, region.end_sec))
        else:
            merged.append(region)
    return merged


def region_support(audio: np.ndarray, sample_rate: int, region: ActivityRegion, tuning_notes: dict[str, float], expected_notes: list[str]) -> dict[str, float]:
    start_index = max(0, int(region.start_sec * sample_rate))
    end_index = min(len(audio), int(region.end_sec * sample_rate))
    clip = audio[start_index:end_index]
    if clip.size < 1024:
        return {note: 0.0 for note in expected_notes}

    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip))))
    freqs = np.fft.rfftfreq(len(clip), 1.0 / sample_rate)
    support: dict[str, float] = {}
    for note in expected_notes:
        frequency = tuning_notes.get(note)
        if frequency is None:
            support[note] = 0.0
            continue
        support[note] = band_energy(freqs, spectrum, frequency)
    return support


def band_energy(freqs: np.ndarray, spectrum: np.ndarray, frequency: float) -> float:
    total = 0.0
    for multiple, weight in ((1.0, 1.0), (2.0, 0.55), (3.0, 0.3)):
        center = frequency * multiple
        half_width = max(12.0, center * 0.025)
        mask = (freqs >= center - half_width) & (freqs <= center + half_width)
        if np.any(mask):
            total += float(np.sum(spectrum[mask])) * weight
    return total


def extract_expected_notes(request_payload: dict[str, Any]) -> list[str]:
    performance = request_payload.get("expectedPerformance") or {}
    events = performance.get("events") or []
    note_names: list[str] = []
    for event in events:
        for key in event.get("keys") or []:
            note_name = key.get("noteName")
            if isinstance(note_name, str) and note_name not in note_names:
                note_names.append(note_name)
    return note_names


def format_ratio(value: float, peak: float) -> str:
    if peak <= 1e-9:
        return "0.00"
    return f"{value / peak:.2f}"


def audit_fixture(fixture_dir: Path) -> str:
    request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    status = fixture_status(expected)
    audio, sample_rate = load_audio(fixture_dir / "audio.wav")
    regions = detect_activity_regions(audio, sample_rate)
    tuning_notes = {note["noteName"]: float(note["frequency"]) for note in request_payload["tuning"]["notes"]}
    expected_notes = extract_expected_notes(request_payload)
    capture_intent = request_payload.get("captureIntent", "(unknown)")
    source_profile = request_payload.get("sourceProfile", "(unknown)")
    default_capture_intent = (request_payload.get("expectedPerformance") or {}).get("defaultCaptureIntent")
    expected_events = (request_payload.get("expectedPerformance") or {}).get("events") or []

    lines = [
        f"## {fixture_dir.name}",
        f"- status: {status}",
        f"- intent: {capture_intent}",
        f"- defaultCaptureIntent: {default_capture_intent or "(none)"}",
        f"- sourceProfile: {source_profile}",
        f"- durationSec: {request_payload['audio']['durationSec']}",
        f"- activityRegions: {len(regions)}",
    ]

    if expected_notes:
        lines.append(f"- expectedNotes: {' + '.join(expected_notes)}")
    if expected_events:
        event_summary = " / ".join(
            f"{event.get('index')}: {event.get('display')} [{event.get('intent') or "(none)"}]"
            for event in expected_events
        )
        lines.append(f"- expectedEvents: {event_summary}")
    if expected.get("reason"):
        lines.append(f"- currentReason: {expected['reason']}")

    if not regions:
        lines.append("- audit: no active regions detected")
        return "\n".join(lines)

    lines.append("- regionSupport:")
    for index, region in enumerate(regions, start=1):
        support = region_support(audio, sample_rate, region, tuning_notes, expected_notes)
        peak = max(support.values(), default=0.0)
        summary = ", ".join(f"{note}={format_ratio(value, peak)}" for note, value in support.items()) if support else "(no expected note map)"
        lines.append(f"  - {index}: {region.start_sec:.2f}s-{region.end_sec:.2f}s :: {summary}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manual capture fixtures from raw audio.")
    parser.add_argument("--root", type=Path, default=Path("apps/api/tests/fixtures/manual-captures"))
    parser.add_argument("--status", action="append", default=["pending", "rerecord", "review_needed"])
    parser.add_argument("--source-profile", action="append", default=[])
    args = parser.parse_args()

    allowed_profiles = set(args.source_profile) if args.source_profile else None

    targets = []
    for fixture_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
        if fixture_status(expected) not in set(args.status):
            continue
        request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
        source_profile = request_payload.get("sourceProfile", "(unknown)")
        if allowed_profiles and source_profile not in allowed_profiles:
            continue
        targets.append(fixture_dir)

    for index, fixture_dir in enumerate(targets):
        if index:
            print()
        print(audit_fixture(fixture_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
