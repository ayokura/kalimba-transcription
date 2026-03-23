# Video Expected Performance Analysis

## Purpose

`[scripts/video_expected_performance.py](/C:/src/calimba-score/scripts/video_expected_performance.py)` analyzes an app screen recording frame by frame and emits a JSON bundle that keeps both:

- detailed frame and event evidence for later verification
- a projected Expected Performance event list that can be imported later

This is intentionally app-specific. For the current kalimba UI, the primary signal is tine darkening, not OCR.

## Why keep a separate analysis JSON

Do not overload the current `ManualCaptureExpectedPerformance` shape with frame-level evidence yet.

Current `ManualCaptureExpectedPerformance` in `[apps/web/src/lib/api.ts](/C:/src/calimba-score/apps/web/src/lib/api.ts)` is optimized for:

- a compact human-authored event list
- archive export into `request.json`
- simple UI editing and comparison

Frame-level video analysis needs additional data that is too verbose and too unstable for that shape:

- per-frame timecodes
- scene segmentation such as `instrument` vs OS overlay
- per-frame candidate keys and activation scores
- per-event provenance and confidence
- detector metadata and layout calibration

Recommended split:

1. Keep `ManualCaptureExpectedPerformance` as the compact event sequence used by the existing app flow.
2. Introduce a separate intermediate JSON for video-derived evidence.
3. Project from the intermediate JSON into the compact event sequence when importing.

This script follows that split already.

## Output structure

Top-level sections in the emitted JSON:

- `sourceVideo`: path, fps, frame count, duration
- `tuning`: resolved tuning used for lane mapping
- `detector`: thresholds and detector notes
- `layout`: detected tine regions
- `sceneTimeline`: instrument scene vs excluded overlay ranges
- `frames`: frame-level timecodes plus active key candidates
- `events`: contiguous activation windows with projected keys
- `projections.expectedPerformanceDraft`: richer event list with timing and provenance
- `projections.manualCaptureExpectedPerformanceV1Compat`: compact import-friendly event list

### Draft projection shape

```json
{
  "source": "video-frame-analysis",
  "version": 1,
  "summary": "E4 / G4 / ...",
  "events": [
    {
      "index": 1,
      "keys": [{ "key": 10, "noteName": "E4" }],
      "display": "E4",
      "timing": {
        "startFrame": 123,
        "endFrame": 129,
        "peakFrame": 126,
        "startTimeSec": 2.320755,
        "endTimeSec": 2.433962,
        "peakTimeSec": 2.377358
      },
      "confidence": 0.91,
      "derivedFromEventIndex": 1
    }
  ]
}
```

### V1-compatible projection shape

```json
{
  "summary": "E4 / G4 / ...",
  "events": [
    {
      "index": 1,
      "keys": [{ "key": 10, "noteName": "E4" }],
      "display": "E4"
    }
  ]
}
```

This second shape is the safe bridge to the current UI and archive flow.

## Future integration direction

If a future Web UI import feature is added, prefer this flow:

1. Import the full analysis JSON.
2. Show the detected `sceneTimeline` and `events` for review.
3. Let the user accept, edit, merge, split, or delete projected events.
4. Save only the reviewed compact event list back into `request.json.expectedPerformance`.
5. Keep the original analysis JSON as auxiliary provenance, not as the canonical request payload.

If a typed in-app representation is needed later, a separate type is cleaner than mutating the current one immediately.

Suggested direction:

```ts
export type VideoExpectedPerformanceAnalysis = {
  schema: "kalimba-video-expected-performance-analysis";
  version: 1;
  sourceVideo: { path: string; fps: number; frameCount: number; durationSec: number };
  sceneTimeline: Array<{ label: "instrument" | "excluded_overlay"; startFrame: number; endFrame: number }>;
  frames: Array<{ frameIndex: number; timeSec: number; scene: string; activeCandidates: Array<{ key: number; noteName: string; activationScore: number }> }>;
  events: Array<{ eventIndex: number; startFrame: number; endFrame: number; projectedKeys: Array<{ key: number; noteName: string }> }>;
  projections: {
    expectedPerformanceDraft: unknown;
    manualCaptureExpectedPerformanceV1Compat: {
      summary: string;
      events: Array<{ index: number; keys: Array<{ key: number; noteName: string }>; display: string }>;
    };
  };
};
```

## Current detector assumptions

For this kalimba app recording, the script assumes:

- 17-key layout with fixed tine positions
- a bright reference frame exists in the instrument scene
- pressed or touched tines appear darker than the baseline
- OS overlays can be excluded with sharpness and saturation thresholds

These assumptions are suitable for the current sample video, but they should be treated as detector configuration, not universal truth.

## Command

```powershell
.\.venv313\Scripts\python.exe scripts/video_expected_performance.py .codex-media/source-videos/ScreenRecording_03-23-2026_13-09-56_1.mov --output .codex-media/derived-analysis/kira-kira-expected-performance.json
```

## Toward a skill

To turn this task into a reusable skill later, keep these parts separate:

- `scripts/`: deterministic video/frame analysis code
- `references/`: app-specific notes about scene structure, ROI strategy, and expected overlay behavior
- `SKILL.md`: only the workflow for choosing a sample video, validating detection, and deciding whether projection quality is acceptable

The repeatable workflow is:

1. Collect a representative video and a few extracted frames.
2. Identify the real visual signal to track.
3. Build an app-specific detector that emits frame-level evidence.
4. Project detector output into a compact event list.
5. Validate the projection against the raw video before import.

That workflow is the part that should become the skill. The current script is one concrete implementation of step 3 and step 4.

## Current Local Reference Finding

For the local file:
- `C:\src\calimba-score\.codex-media\source-videos\ScreenRecording_03-23-2026_13-09-56_1.mov`

The later half should currently be treated as an `arpeggio candidate`, not as a `slide_chord` regression target.

Observed from `.codex-media/derived-analysis/kira-kira-expected-performance.json` and the derived audio:
- the main candidate region is about `15.41s-22.83s`
- the clearest sub-region is about `19.25s-20.89s`
- projected event sequence there is `F4+F5 / A4 / C5 / A5 / A4 / F4`
- the extracted audio in `15.2s-22.9s` contains about `25` distinct onset peaks, which is much more consistent with ordered broken-chord motion than a single `slide_chord` gesture

Operational conclusion:
- keep this material local and `reference_only`
- do not convert it into an acoustic regression fixture
- use it to plan future real-device `arpeggio` recordings and data-model work
