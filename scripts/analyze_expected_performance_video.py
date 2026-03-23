from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if not (REPO_ROOT / "apps" / "api").exists():
    REPO_ROOT = REPO_ROOT.parent
API_ROOT = REPO_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.tunings import build_custom_tuning, get_default_tunings  # noqa: E402

DEFAULT_VIDEO = REPO_ROOT / ".codex-media" / "source-videos" / "ScreenRecording_03-23-2026_13-09-56_1.mov"
DEFAULT_OUTPUT = REPO_ROOT / ".codex-media" / "derived-analysis" / "expected-performance-analysis.json"


@dataclass(frozen=True)
class KeyRegion:
    physical_index: int
    x_start: int
    x_end: int
    key: int
    note_name: str


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    time_sec: float
    timecode: str
    app_scene: bool
    active_keys: tuple[int, ...]
    active_notes: tuple[str, ...]
    press_scores: tuple[float, ...]
    strongest_score: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze an app video frame-by-frame and project a candidate Expected Performance sequence."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Video file to analyze. Defaults to the sample under .codex-media/source-videos/.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Write JSON output to this path.")
    parser.add_argument("--tuning-id", default="kalimba-17-c", help="Preset tuning id from apps/api/app/tunings.py.")
    parser.add_argument("--custom-notes", help="Comma-separated note names. When set, overrides --tuning-id.")
    parser.add_argument("--custom-tuning-name", default="Custom Video Tuning")
    parser.add_argument("--key-region-y-start", type=int, default=120)
    parser.add_argument("--key-region-y-end", type=int, default=420)
    parser.add_argument("--white-threshold", type=float, default=240.0)
    parser.add_argument("--white-coverage-threshold", type=float, default=0.18)
    parser.add_argument("--press-threshold", type=float, default=18.0)
    parser.add_argument("--event-gap-sec", type=float, default=0.18)
    parser.add_argument("--projection-key-coverage", type=float, default=0.35)
    parser.add_argument("--peak-score-threshold", type=float, default=42.0)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    if args.key_region_y_start < 0 or args.key_region_y_end <= args.key_region_y_start:
        raise SystemExit("Invalid key analysis Y range.")

    tuning = load_tuning(args.tuning_id, args.custom_notes, args.custom_tuning_name)
    frame_times = load_frame_timestamps(args.video_path)
    metadata = load_video_metadata(args.video_path)
    baseline = decode_single_frame(args.video_path, metadata["width"], metadata["height"])
    key_regions = detect_key_regions(
        baseline,
        tuning["notes"],
        white_threshold=args.white_threshold,
        white_coverage_threshold=args.white_coverage_threshold,
    )
    observations = analyze_video_frames(
        video_path=args.video_path,
        width=metadata["width"],
        height=metadata["height"],
        frame_times=frame_times,
        key_regions=key_regions,
        baseline=baseline,
        y_start=args.key_region_y_start,
        y_end=args.key_region_y_end,
        white_threshold=args.white_threshold,
        white_coverage_threshold=args.white_coverage_threshold,
        press_threshold=args.press_threshold,
    )
    state_runs = build_state_runs(observations)
    events = build_projected_events(
        observations,
        state_runs,
        tuning["notes"],
        event_gap_sec=args.event_gap_sec,
        projection_key_coverage=args.projection_key_coverage,
        peak_score_threshold=args.peak_score_threshold,
    )
    projected_expected = build_expected_performance_projection(events)

    payload = {
        "schema": "video-expected-performance-analysis",
        "version": 1,
        "input": {
            "videoPath": str(args.video_path.resolve()),
            "videoFileName": args.video_path.name,
            "video": metadata,
            "analysisParameters": {
                "tuningId": tuning["id"],
                "keyRegionY": {"start": args.key_region_y_start, "end": args.key_region_y_end},
                "whiteThreshold": args.white_threshold,
                "whiteCoverageThreshold": args.white_coverage_threshold,
                "pressThreshold": args.press_threshold,
                "eventGapSec": args.event_gap_sec,
                "projectionKeyCoverage": args.projection_key_coverage,
                "peakScoreThreshold": args.peak_score_threshold,
            },
        },
        "tuning": tuning,
        "calibration": {
            "baselineFrame": {
                "frameIndex": 0,
                "timeSec": frame_times[0],
                "timecode": format_timecode(frame_times[0]),
            },
            "keyRegions": [
                {
                    "physicalIndex": region.physical_index,
                    "xStart": region.x_start,
                    "xEnd": region.x_end,
                    "key": region.key,
                    "noteName": region.note_name,
                }
                for region in key_regions
            ],
        },
        "observations": {
            "framesIncluded": args.include_frames,
            "frameCount": len(observations),
            "appSceneFrameCount": sum(1 for observation in observations if observation.app_scene),
            "stateRuns": state_runs,
            "frames": [frame_to_payload(frame) for frame in observations] if args.include_frames else [],
        },
        "scenario": {
            "events": events,
            "projectedExpectedPerformance": projected_expected,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote analysis JSON to {args.output}")
    if args.stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def load_tuning(tuning_id: str, custom_notes: str | None, custom_tuning_name: str) -> dict[str, Any]:
    if custom_notes:
        tuning = build_custom_tuning(custom_tuning_name, custom_notes.split(","))
        return json.loads(tuning.model_dump_json(by_alias=True))

    for tuning in get_default_tunings():
        if tuning.id == tuning_id:
            return json.loads(tuning.model_dump_json(by_alias=True))
    available = ", ".join(sorted(tuning.id for tuning in get_default_tunings()))
    raise SystemExit(f"Unknown tuning id '{tuning_id}'. Available presets: {available}")


def load_video_metadata(video_path: Path) -> dict[str, Any]:
    result = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration,avg_frame_rate,r_frame_rate,pix_fmt",
            "-of",
            "json",
            str(video_path),
        ]
    )
    stream = result["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nbFrames": int(stream.get("nb_frames") or 0),
        "durationSec": float(stream["duration"]),
        "avgFrameRate": stream["avg_frame_rate"],
        "rFrameRate": stream["r_frame_rate"],
        "pixelFormat": stream.get("pix_fmt"),
    }


def load_frame_timestamps(video_path: Path) -> list[float]:
    result = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video_path),
        ]
    )
    frame_times: list[float] = []
    for entry in result["frames"]:
        if "best_effort_timestamp_time" in entry:
            frame_times.append(float(entry["best_effort_timestamp_time"]))
    return frame_times


def decode_single_frame(video_path: Path, width: int, height: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    frame_size = width * height * 3
    if len(completed.stdout) != frame_size:
        raise SystemExit(f"Failed to decode the baseline frame from {video_path}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(height, width, 3)


def iter_video_frames(video_path: Path, width: int, height: int) -> Iterable[np.ndarray]:
    frame_size = width * height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-vsync",
        "0",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.wait()
        if process.returncode != 0:
            raise SystemExit(f"ffmpeg failed while decoding {video_path}")


def detect_key_regions(
    baseline: np.ndarray,
    tuning_notes: list[dict[str, Any]],
    *,
    white_threshold: float,
    white_coverage_threshold: float,
) -> list[KeyRegion]:
    brightness = baseline.mean(axis=2)
    white_coverage = (brightness >= white_threshold).mean(axis=0)
    segments = contiguous_segments(white_coverage >= white_coverage_threshold, min_width=8)
    if len(segments) != len(tuning_notes):
        raise SystemExit(
            f"Detected {len(segments)} key columns from the baseline frame, but tuning has {len(tuning_notes)} notes."
        )

    key_regions: list[KeyRegion] = []
    for physical_index, ((x_start, x_end), note) in enumerate(zip(segments, tuning_notes, strict=True), start=1):
        key_regions.append(
            KeyRegion(
                physical_index=physical_index,
                x_start=x_start,
                x_end=x_end,
                key=int(note["key"]),
                note_name=str(note["noteName"]),
            )
        )
    return key_regions


def analyze_video_frames(
    *,
    video_path: Path,
    width: int,
    height: int,
    frame_times: list[float],
    key_regions: list[KeyRegion],
    baseline: np.ndarray,
    y_start: int,
    y_end: int,
    white_threshold: float,
    white_coverage_threshold: float,
    press_threshold: float,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    for frame_index, (frame, time_sec) in enumerate(
        zip(iter_video_frames(video_path, width, height), frame_times, strict=True)
    ):
        app_scene = detect_app_scene(
            frame,
            expected_key_count=len(key_regions),
            white_threshold=white_threshold,
            white_coverage_threshold=white_coverage_threshold,
        )
        press_scores: list[float] = []
        active_keys: list[int] = []
        active_notes: list[str] = []
        strongest_score = 0.0

        if app_scene:
            for region in key_regions:
                baseline_roi = baseline[y_start:y_end, region.x_start : region.x_end + 1]
                frame_roi = frame[y_start:y_end, region.x_start : region.x_end + 1]
                score = float(baseline_roi.mean() - frame_roi.mean())
                press_scores.append(round(score, 3))
                strongest_score = max(strongest_score, score)
                if score >= press_threshold:
                    active_keys.append(region.key)
                    active_notes.append(region.note_name)
        else:
            press_scores = [0.0] * len(key_regions)

        observations.append(
            FrameObservation(
                frame_index=frame_index,
                time_sec=time_sec,
                timecode=format_timecode(time_sec),
                app_scene=app_scene,
                active_keys=tuple(active_keys),
                active_notes=tuple(active_notes),
                press_scores=tuple(press_scores),
                strongest_score=round(strongest_score, 3),
            )
        )
    return observations


def detect_app_scene(
    frame: np.ndarray,
    *,
    expected_key_count: int,
    white_threshold: float,
    white_coverage_threshold: float,
) -> bool:
    brightness = frame.mean(axis=2)
    white_coverage = (brightness >= white_threshold).mean(axis=0)
    segments = contiguous_segments(white_coverage >= white_coverage_threshold, min_width=8)
    return len(segments) >= max(15, expected_key_count - 2)


def build_state_runs(observations: list[FrameObservation]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for observation in observations:
        state = {
            "appScene": observation.app_scene,
            "activeKeys": list(observation.active_keys),
            "activeNotes": list(observation.active_notes),
        }
        if current is None or current["appScene"] != state["appScene"] or current["activeKeys"] != state["activeKeys"]:
            if current is not None:
                runs.append(finalize_run(current))
            current = {
                **state,
                "startFrame": observation.frame_index,
                "endFrame": observation.frame_index,
                "startTimeSec": observation.time_sec,
                "endTimeSec": observation.time_sec,
                "frameCount": 1,
                "maxScore": observation.strongest_score,
            }
            continue

        current["endFrame"] = observation.frame_index
        current["endTimeSec"] = observation.time_sec
        current["frameCount"] += 1
        current["maxScore"] = max(current["maxScore"], observation.strongest_score)

    if current is not None:
        runs.append(finalize_run(current))
    return runs


def finalize_run(run: dict[str, Any]) -> dict[str, Any]:
    run["durationSec"] = round(run["endTimeSec"] - run["startTimeSec"], 6)
    run["timecodeStart"] = format_timecode(run["startTimeSec"])
    run["timecodeEnd"] = format_timecode(run["endTimeSec"])
    run["maxScore"] = round(float(run["maxScore"]), 3)
    return run


def build_projected_events(
    observations: list[FrameObservation],
    state_runs: list[dict[str, Any]],
    tuning_notes: list[dict[str, Any]],
    *,
    event_gap_sec: float,
    projection_key_coverage: float,
    peak_score_threshold: float,
) -> list[dict[str, Any]]:
    note_name_by_key = {int(note["key"]): str(note["noteName"]) for note in tuning_notes}
    active_runs = [run for run in state_runs if run["appScene"] and run["activeKeys"]]
    grouped_runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_end = -math.inf
    for run in active_runs:
        gap = run["startTimeSec"] - previous_end if current else 0.0
        if current and gap > event_gap_sec:
            grouped_runs.append(current)
            current = []
        current.append(run)
        previous_end = run["endTimeSec"]
    if current:
        grouped_runs.append(current)

    projected: list[dict[str, Any]] = []
    all_keys = [int(note["key"]) for note in tuning_notes]
    for event_index, event_runs in enumerate(grouped_runs, start=1):
        start_frame = event_runs[0]["startFrame"]
        end_frame = event_runs[-1]["endFrame"]
        start_time = event_runs[0]["startTimeSec"]
        end_time = event_runs[-1]["endTimeSec"]
        event_frames = [
            observation
            for observation in observations
            if start_frame <= observation.frame_index <= end_frame and observation.app_scene
        ]
        frames_with_activity = [frame for frame in event_frames if frame.active_keys]
        frame_count = len(frames_with_activity)
        key_stats: dict[int, dict[str, Any]] = {}
        for frame in frames_with_activity:
            press_by_key = {key: score for key, score in zip(all_keys, frame.press_scores, strict=True)}
            for key in frame.active_keys:
                stats = key_stats.setdefault(
                    key,
                    {
                        "key": key,
                        "noteName": note_name_by_key[key],
                        "activeFrameCount": 0,
                        "maxScore": 0.0,
                        "firstFrame": frame.frame_index,
                        "lastFrame": frame.frame_index,
                        "firstTimeSec": frame.time_sec,
                        "lastTimeSec": frame.time_sec,
                    },
                )
                stats["activeFrameCount"] += 1
                stats["lastFrame"] = frame.frame_index
                stats["lastTimeSec"] = frame.time_sec
                stats["maxScore"] = max(stats["maxScore"], press_by_key[key])

        onset_frames = [stats["firstFrame"] for stats in key_stats.values() if stats["activeFrameCount"] > 0]
        onset_span = max(onset_frames) - min(onset_frames) if len(onset_frames) >= 2 else 0
        coverage_cutoff = max(1, math.ceil(frame_count * projection_key_coverage))
        candidate_keys = [
            key
            for key, stats in sorted(key_stats.items())
            if stats["activeFrameCount"] >= coverage_cutoff or stats["maxScore"] >= peak_score_threshold
        ]
        if not candidate_keys and key_stats:
            strongest = sorted(key_stats.values(), key=lambda item: (-item["maxScore"], -item["activeFrameCount"], item["key"]))
            candidate_keys = [strongest[0]["key"]]

        candidate_notes = [note_name_by_key[key] for key in candidate_keys]
        gesture = infer_gesture(candidate_keys, onset_span)
        state_path = [
            {
                "startFrame": run["startFrame"],
                "endFrame": run["endFrame"],
                "startTimeSec": run["startTimeSec"],
                "endTimeSec": run["endTimeSec"],
                "timecodeStart": run["timecodeStart"],
                "timecodeEnd": run["timecodeEnd"],
                "activeKeys": run["activeKeys"],
                "activeNotes": run["activeNotes"],
                "frameCount": run["frameCount"],
                "durationSec": run["durationSec"],
                "maxScore": run["maxScore"],
            }
            for run in event_runs
        ]

        projected.append(
            {
                "index": event_index,
                "startFrame": start_frame,
                "endFrame": end_frame,
                "startTimeSec": round(start_time, 6),
                "endTimeSec": round(end_time, 6),
                "timecodeStart": format_timecode(start_time),
                "timecodeEnd": format_timecode(end_time),
                "durationSec": round(end_time - start_time, 6),
                "frameCount": end_frame - start_frame + 1,
                "gesture": gesture,
                "projectedKeys": [{"key": key, "noteName": note_name_by_key[key]} for key in candidate_keys],
                "display": " + ".join(candidate_notes),
                "evidence": {
                    "activeFrameCount": frame_count,
                    "projectionCoverageThreshold": projection_key_coverage,
                    "projectionCoverageCutoffFrames": coverage_cutoff,
                    "peakScoreThreshold": peak_score_threshold,
                    "onsetSpanFrames": onset_span,
                    "keyStats": [
                        {
                            "key": stats["key"],
                            "noteName": stats["noteName"],
                            "activeFrameCount": stats["activeFrameCount"],
                            "maxScore": round(float(stats["maxScore"]), 3),
                            "firstFrame": stats["firstFrame"],
                            "lastFrame": stats["lastFrame"],
                            "firstTimeSec": round(float(stats["firstTimeSec"]), 6),
                            "lastTimeSec": round(float(stats["lastTimeSec"]), 6),
                        }
                        for stats in sorted(key_stats.values(), key=lambda item: item["key"])
                    ],
                },
                "statePath": state_path,
            }
        )

    return projected


def infer_gesture(candidate_keys: list[int], onset_span: int) -> str:
    if len(candidate_keys) <= 1:
        return "single_note"
    if onset_span <= 2:
        return "strict_chord_like"
    return "slide_chord_like"


def build_expected_performance_projection(events: list[dict[str, Any]]) -> dict[str, Any]:
    projected_events: list[dict[str, Any]] = []
    for event in events:
        projected_events.append(
            {
                "index": event["index"],
                "keys": event["projectedKeys"],
                "display": event["display"],
                "timing": {
                    "startFrame": event["startFrame"],
                    "endFrame": event["endFrame"],
                    "startTimeSec": event["startTimeSec"],
                    "endTimeSec": event["endTimeSec"],
                    "timecodeStart": event["timecodeStart"],
                    "timecodeEnd": event["timecodeEnd"],
                },
                "gesture": event["gesture"],
            }
        )

    summary_parts = []
    counts = Counter(event["display"] for event in projected_events if event["display"])
    ordered_labels: list[str] = []
    for event in projected_events:
        label = event["display"]
        if label and label not in ordered_labels:
            ordered_labels.append(label)
    for label in ordered_labels:
        count = counts[label]
        summary_parts.append(f"{label} x {count}" if count > 1 else label)

    return {
        "source": "video-frame-analysis",
        "version": 1,
        "summary": " / ".join(summary_parts),
        "events": projected_events,
    }


def frame_to_payload(frame: FrameObservation) -> dict[str, Any]:
    return {
        "frameIndex": frame.frame_index,
        "timeSec": round(frame.time_sec, 6),
        "timecode": frame.timecode,
        "appScene": frame.app_scene,
        "activeKeys": list(frame.active_keys),
        "activeNotes": list(frame.active_notes),
        "strongestScore": frame.strongest_score,
        "pressScores": list(frame.press_scores),
    }


def contiguous_segments(mask: Iterable[bool], *, min_width: int) -> list[tuple[int, int]]:
    values = list(mask)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(values):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if index - start >= min_width:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= min_width:
        segments.append((start, len(values) - 1))
    return segments


def format_timecode(time_sec: float) -> str:
    total_millis = int(round(time_sec * 1000))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
