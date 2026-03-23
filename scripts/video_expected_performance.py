from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / 'apps' / 'api'
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.tunings import get_default_tunings  # noqa: E402


@dataclass(frozen=True)
class TineRegion:
    key: int
    note_name: str
    x_start: int
    x_end: int

    @property
    def x_center(self) -> int:
        return (self.x_start + self.x_end) // 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analyze a kalimba app screen recording and project Expected Performance candidates from frame-level key activation.'
    )
    parser.add_argument('video_path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tuning-id', default='kalimba-17-c')
    parser.add_argument('--strong-threshold', type=float, default=80.0)
    parser.add_argument('--support-threshold', type=float, default=45.0)
    parser.add_argument('--support-ratio', type=float, default=0.72)
    parser.add_argument('--min-gap-frames', type=int, default=2)
    parser.add_argument('--min-event-frames', type=int, default=2)
    parser.add_argument('--max-frame-candidates', type=int, default=4)
    parser.add_argument('--max-event-candidates', type=int, default=6)
    parser.add_argument('--frame-limit', type=int)
    args = parser.parse_args()

    tuning = load_tuning(args.tuning_id)
    analysis = analyze_video(
        video_path=args.video_path,
        tuning=tuning,
        strong_threshold=args.strong_threshold,
        support_threshold=args.support_threshold,
        support_ratio=args.support_ratio,
        min_gap_frames=args.min_gap_frames,
        min_event_frames=args.min_event_frames,
        max_frame_candidates=args.max_frame_candidates,
        max_event_candidates=args.max_event_candidates,
        frame_limit=args.frame_limit,
    )

    rendered = json.dumps(analysis, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    else:
        sys.stdout.write(rendered)
    return 0


def load_tuning(tuning_id: str) -> dict[str, Any]:
    for tuning in get_default_tunings():
        if tuning.id != tuning_id:
            continue
        raw = tuning.model_dump()
        return {
            'id': raw['id'],
            'name': raw['name'],
            'keyCount': raw['key_count'],
            'notes': [
                {
                    'key': note['key'],
                    'noteName': note['note_name'],
                    'frequency': note['frequency'],
                }
                for note in raw['notes']
            ],
        }
    valid_ids = ', '.join(sorted(tuning.id for tuning in get_default_tunings()))
    raise SystemExit(f"Unknown tuning id '{tuning_id}'. Valid ids: {valid_ids}")


def analyze_video(
    *,
    video_path: Path,
    tuning: dict[str, Any],
    strong_threshold: float,
    support_threshold: float,
    support_ratio: float,
    min_gap_frames: int,
    min_event_frames: int,
    max_frame_candidates: int,
    max_event_candidates: int,
    frame_limit: int | None,
) -> dict[str, Any]:
    video_info = probe_video(video_path)
    frames_iter = iter_video_frames(video_path, video_info['width'], video_info['height'], frame_limit=frame_limit)
    try:
        reference_frame = next(frames_iter)
    except StopIteration as exc:
        raise SystemExit(f'No frames decoded from {video_path}') from exc

    regions = detect_tine_regions(reference_frame, tuning)
    scene_features: list[dict[str, float]] = []
    roi_brightness: list[list[float]] = [measure_roi_brightness(reference_frame, regions)]

    scene_features.append(measure_scene(reference_frame))
    for frame in frames_iter:
        scene_features.append(measure_scene(frame))
        roi_brightness.append(measure_roi_brightness(frame, regions))

    brightness_matrix = np.asarray(roi_brightness, dtype=np.float32)
    sharpness = np.asarray([entry['sharpness'] for entry in scene_features], dtype=np.float32)
    saturation = np.asarray([entry['saturation'] for entry in scene_features], dtype=np.float32)
    instrument_mask, sharpness_threshold, saturation_threshold = classify_instrument_frames(sharpness, saturation)

    if not np.any(instrument_mask):
        raise SystemExit('No instrument scene frames detected. Thresholds are too strict for this video.')

    baseline = np.percentile(brightness_matrix[instrument_mask], 95, axis=0)
    activation = np.clip(baseline - brightness_matrix, a_min=0.0, a_max=None)
    frame_top_candidates = build_frame_candidates(
        activation=activation,
        instrument_mask=instrument_mask,
        regions=regions,
        strong_threshold=strong_threshold,
        support_threshold=support_threshold,
        max_frame_candidates=max_frame_candidates,
        fps=video_info['fps'],
    )
    event_ranges = detect_event_ranges(
        activation=activation,
        instrument_mask=instrument_mask,
        support_threshold=support_threshold,
        min_gap_frames=min_gap_frames,
        min_event_frames=min_event_frames,
        strong_threshold=strong_threshold,
    )
    events = build_events(
        event_ranges=event_ranges,
        activation=activation,
        regions=regions,
        strong_threshold=strong_threshold,
        support_threshold=support_threshold,
        support_ratio=support_ratio,
        fps=video_info['fps'],
        max_event_candidates=max_event_candidates,
    )
    scene_timeline = build_scene_timeline(instrument_mask, video_info['fps'])
    frame_timeline = build_frame_timeline(
        scene_features=scene_features,
        frame_top_candidates=frame_top_candidates,
        instrument_mask=instrument_mask,
        fps=video_info['fps'],
    )
    expected_projection = build_expected_performance_projection(events)

    return {
        'schema': 'kalimba-video-expected-performance-analysis',
        'version': 1,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'sourceVideo': {
            'path': str(video_path.resolve()),
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': round(video_info['fps'], 6),
            'frameCount': len(scene_features),
            'durationSec': round(len(scene_features) / video_info['fps'], 6),
        },
        'tuning': tuning,
        'detector': {
            'name': 'kalimba-tine-darkening-v1',
            'sceneRules': {
                'sharpnessThreshold': round(float(sharpness_threshold), 3),
                'saturationThreshold': round(float(saturation_threshold), 3),
            },
            'eventRules': {
                'strongThreshold': strong_threshold,
                'supportThreshold': support_threshold,
                'supportRatio': support_ratio,
                'minGapFrames': min_gap_frames,
                'minEventFrames': min_event_frames,
            },
            'notes': [
                'Instrument frames are separated from OS overlays using sharpness and saturation.',
                'Per-key activation comes from brightness drop against a bright-frame baseline.',
                'Projected events are derived from contiguous activation windows and keep frame timecodes for later verification.',
            ],
        },
        'layout': {
            'referenceFrame': 0,
            'tineRegions': [
                {
                    'key': region.key,
                    'noteName': region.note_name,
                    'xStart': region.x_start,
                    'xEnd': region.x_end,
                    'xCenter': region.x_center,
                }
                for region in regions
            ],
        },
        'sceneTimeline': scene_timeline,
        'frames': frame_timeline,
        'events': events,
        'projections': {
            'expectedPerformanceDraft': expected_projection['draft'],
            'manualCaptureExpectedPerformanceV1Compat': expected_projection['v1Compat'],
        },
    }


def probe_video(video_path: Path) -> dict[str, Any]:
    payload = run_command(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height,r_frame_rate',
            '-of',
            'json',
            str(video_path),
        ]
    )
    stream = json.loads(payload)['streams'][0]
    fps = parse_ratio(stream['r_frame_rate'])
    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': fps,
    }


def parse_ratio(raw_value: str) -> float:
    numerator, denominator = raw_value.split('/')
    return float(numerator) / float(denominator)


def iter_video_frames(video_path: Path, width: int, height: int, frame_limit: int | None = None) -> Iterator[np.ndarray]:
    command = [
        'ffmpeg',
        '-v',
        'error',
        '-i',
        str(video_path),
        '-map',
        '0:v:0',
        '-vsync',
        '0',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgb24',
        '-',
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        frame_size = width * height * 3
        frame_index = 0
        while True:
            if frame_limit is not None and frame_index >= frame_limit:
                break
            raw = process.stdout.read(frame_size) if process.stdout is not None else b''
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(np.float32)
            frame_index += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        process.wait()
        if process.returncode != 0:
            raise SystemExit(f'ffmpeg failed while decoding {video_path}')


def detect_tine_regions(reference_frame: np.ndarray, tuning: dict[str, Any]) -> list[TineRegion]:
    brightness = reference_frame.mean(axis=2)
    mask = brightness[20:430] > 230
    coverage = mask.mean(axis=0)

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_tine in enumerate(coverage > 0.32):
        if is_tine and start is None:
            start = index
        elif not is_tine and start is not None:
            if index - start > 8:
                segments.append((start, index - 1))
            start = None
    if start is not None and coverage.shape[0] - start > 8:
        segments.append((start, coverage.shape[0] - 1))

    if len(segments) != tuning['keyCount']:
        raise SystemExit(
            f"Detected {len(segments)} tine regions, but tuning '{tuning['id']}' expects {tuning['keyCount']} keys."
        )

    return [
        TineRegion(
            key=note['key'],
            note_name=note['noteName'],
            x_start=start,
            x_end=end,
        )
        for note, (start, end) in zip(tuning['notes'], segments, strict=True)
    ]


def measure_scene(frame: np.ndarray) -> dict[str, float]:
    gray = frame.mean(axis=2)
    sharpness = float(np.abs(np.diff(gray[60:430, 60:980], axis=1)).mean())
    saturation = float((frame.max(axis=2) - frame.min(axis=2)).mean())
    return {'sharpness': sharpness, 'saturation': saturation}


def measure_roi_brightness(frame: np.ndarray, regions: list[TineRegion]) -> list[float]:
    row: list[float] = []
    for region in regions:
        left = min(region.x_start + 3, region.x_end)
        right = max(region.x_end - 2, left + 1)
        roi = frame[170:430, left:right]
        row.append(float(roi.mean()))
    return row


def classify_instrument_frames(sharpness: np.ndarray, saturation: np.ndarray) -> tuple[np.ndarray, float, float]:
    sharpness_floor = float(np.min(sharpness))
    sharpness_mid = float(np.median(sharpness))
    saturation_floor = float(np.min(saturation))
    saturation_mid = float(np.median(saturation))
    sharpness_threshold = max(sharpness_floor + (sharpness_mid - sharpness_floor) * 0.55, 5.4)
    saturation_threshold = max(saturation_floor + (saturation_mid - saturation_floor) * 0.55, 70.0)
    instrument_mask = (sharpness >= sharpness_threshold) & (saturation >= saturation_threshold)
    return instrument_mask, sharpness_threshold, saturation_threshold


def build_frame_candidates(
    *,
    activation: np.ndarray,
    instrument_mask: np.ndarray,
    regions: list[TineRegion],
    strong_threshold: float,
    support_threshold: float,
    max_frame_candidates: int,
    fps: float,
) -> list[list[dict[str, Any]]]:
    candidates: list[list[dict[str, Any]]] = []
    for frame_index in range(activation.shape[0]):
        if not instrument_mask[frame_index]:
            candidates.append([])
            continue
        scored = [
            {
                'key': region.key,
                'noteName': region.note_name,
                'activationScore': round(float(score), 3),
                'state': 'strong' if score >= strong_threshold else 'support',
                'frameIndex': frame_index,
                'timeSec': round(frame_index / fps, 6),
            }
            for region, score in zip(regions, activation[frame_index], strict=True)
            if score >= support_threshold
        ]
        scored.sort(key=lambda item: item['activationScore'], reverse=True)
        candidates.append(scored[:max_frame_candidates])
    return candidates


def detect_event_ranges(
    *,
    activation: np.ndarray,
    instrument_mask: np.ndarray,
    support_threshold: float,
    min_gap_frames: int,
    min_event_frames: int,
    strong_threshold: float,
) -> list[tuple[int, int]]:
    active = instrument_mask & (activation.max(axis=1) >= support_threshold)
    active = close_short_gaps(active, min_gap_frames)

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(active) - 1))

    filtered: list[tuple[int, int]] = []
    for start, end in ranges:
        frame_count = end - start + 1
        peak_score = float(activation[start : end + 1].max())
        if frame_count >= min_event_frames or peak_score >= strong_threshold:
            filtered.append((start, end))
    return filtered


def close_short_gaps(active: np.ndarray, gap_limit: int) -> np.ndarray:
    if gap_limit <= 0:
        return active.copy()
    filled = active.copy()
    length = len(filled)
    index = 0
    while index < length:
        if filled[index]:
            index += 1
            continue
        gap_start = index
        while index < length and not filled[index]:
            index += 1
        gap_end = index - 1
        gap_width = gap_end - gap_start + 1
        if gap_start == 0 or index >= length:
            continue
        if gap_width <= gap_limit and filled[gap_start - 1] and filled[index]:
            filled[gap_start:index] = True
    return filled


def build_events(
    *,
    event_ranges: list[tuple[int, int]],
    activation: np.ndarray,
    regions: list[TineRegion],
    strong_threshold: float,
    support_threshold: float,
    support_ratio: float,
    fps: float,
    max_event_candidates: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event_index, (start_frame, end_frame) in enumerate(event_ranges, start=1):
        window = activation[start_frame : end_frame + 1]
        peak_scores = window.max(axis=0)
        peak_offsets = window.argmax(axis=0)
        strongest_score = float(peak_scores.max())
        candidate_floor = max(support_threshold, strongest_score * support_ratio)
        projected_regions = [
            region
            for region, score in zip(regions, peak_scores, strict=True)
            if score >= candidate_floor and score >= strong_threshold * 0.85
        ]
        if not projected_regions:
            peak_key_index = int(np.argmax(peak_scores))
            projected_regions = [regions[peak_key_index]]

        candidates = []
        for region, peak_score, peak_offset in zip(regions, peak_scores, peak_offsets, strict=True):
            if peak_score < support_threshold:
                continue
            peak_frame = start_frame + int(peak_offset)
            candidates.append(
                {
                    'key': region.key,
                    'noteName': region.note_name,
                    'peakActivationScore': round(float(peak_score), 3),
                    'peakFrame': peak_frame,
                    'peakTimeSec': round(peak_frame / fps, 6),
                    'state': 'projected'
                    if any(projected.key == region.key for projected in projected_regions)
                    else ('strong' if peak_score >= strong_threshold else 'support'),
                }
            )
        candidates.sort(key=lambda item: (-item['peakActivationScore'], item['key']))

        projected_keys = [
            {'key': region.key, 'noteName': region.note_name}
            for region in sorted(projected_regions, key=lambda item: item.key)
        ]
        projected_display = ' + '.join(item['noteName'] for item in projected_keys)
        peak_frame = int(np.argmax(window.max(axis=1))) + start_frame
        confidence = round(min(1.0, strongest_score / max(strong_threshold, 1.0)), 3)
        events.append(
            {
                'eventIndex': event_index,
                'startFrame': start_frame,
                'endFrame': end_frame,
                'startTimeSec': round(start_frame / fps, 6),
                'endTimeSec': round(end_frame / fps, 6),
                'peakFrame': peak_frame,
                'peakTimeSec': round(peak_frame / fps, 6),
                'projectedKeys': projected_keys,
                'projectedDisplay': projected_display,
                'confidence': confidence,
                'candidates': candidates[:max_event_candidates],
            }
        )
    return events


def build_scene_timeline(instrument_mask: np.ndarray, fps: float) -> list[dict[str, Any]]:
    if instrument_mask.size == 0:
        return []
    timeline: list[dict[str, Any]] = []
    start = 0
    current = bool(instrument_mask[0])
    for index in range(1, instrument_mask.size):
        if bool(instrument_mask[index]) == current:
            continue
        timeline.append(
            {
                'label': 'instrument' if current else 'excluded_overlay',
                'startFrame': start,
                'endFrame': index - 1,
                'startTimeSec': round(start / fps, 6),
                'endTimeSec': round((index - 1) / fps, 6),
            }
        )
        start = index
        current = bool(instrument_mask[index])
    timeline.append(
        {
            'label': 'instrument' if current else 'excluded_overlay',
            'startFrame': start,
            'endFrame': instrument_mask.size - 1,
            'startTimeSec': round(start / fps, 6),
            'endTimeSec': round((instrument_mask.size - 1) / fps, 6),
        }
    )
    return timeline


def build_frame_timeline(
    *,
    scene_features: list[dict[str, float]],
    frame_top_candidates: list[list[dict[str, Any]]],
    instrument_mask: np.ndarray,
    fps: float,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_index, (scene, candidates) in enumerate(zip(scene_features, frame_top_candidates, strict=True)):
        frames.append(
            {
                'frameIndex': frame_index,
                'timeSec': round(frame_index / fps, 6),
                'scene': 'instrument' if bool(instrument_mask[frame_index]) else 'excluded_overlay',
                'sharpness': round(float(scene['sharpness']), 3),
                'saturation': round(float(scene['saturation']), 3),
                'activeCandidates': candidates,
            }
        )
    return frames


def build_expected_performance_projection(events: list[dict[str, Any]]) -> dict[str, Any]:
    compat_events = []
    draft_events = []
    counts: dict[str, int] = {}
    ordered_displays: list[str] = []
    for event in events:
        display = event['projectedDisplay'] or '(unknown)'
        if display not in counts:
            ordered_displays.append(display)
            counts[display] = 0
        counts[display] += 1
        compat_events.append(
            {
                'index': event['eventIndex'],
                'keys': event['projectedKeys'],
                'display': display,
            }
        )
        draft_events.append(
            {
                'index': event['eventIndex'],
                'keys': event['projectedKeys'],
                'display': display,
                'timing': {
                    'startFrame': event['startFrame'],
                    'endFrame': event['endFrame'],
                    'peakFrame': event['peakFrame'],
                    'startTimeSec': event['startTimeSec'],
                    'endTimeSec': event['endTimeSec'],
                    'peakTimeSec': event['peakTimeSec'],
                },
                'confidence': event['confidence'],
                'derivedFromEventIndex': event['eventIndex'],
            }
        )

    summary = ' / '.join(
        f"{display} x {counts[display]}" if counts[display] > 1 else display
        for display in ordered_displays
    )
    return {
        'draft': {
            'source': 'video-frame-analysis',
            'version': 1,
            'summary': summary,
            'events': draft_events,
        },
        'v1Compat': {
            'summary': summary,
            'events': compat_events,
        },
    }


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


if __name__ == '__main__':
    raise SystemExit(main())

