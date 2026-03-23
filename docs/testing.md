# Testing Guide

## Verified in this workspace

### Web

```powershell
npm install
npm run lint:web
npm run build:web
npm run dev:web
```

`lint:web` and `build:web` completed successfully in this environment.

### API dependencies

```powershell
py -3.13 -m pip install -r apps/api/requirements.txt
```

Python 3.14 caused a `pydantic-core` build issue. Use Python 3.13 for the API.

## Recommended local run

### API

```powershell
$env:PYTHONPATH = "C:\src\calimba-score\apps\api"
py -3.13 -m uvicorn app.main:app --reload --app-dir apps/api
```

### Web

```powershell
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:8000"
npm run dev:web
```

## Minimum manual test cases

1. Open the web app and confirm tunings load from `/api/tunings`.
2. Start and stop microphone recording, then run transcription.
3. Confirm the vertical Do-Re-Mi view renders events with start and duration beats.
4. Switch to numbered and western notation and confirm the same event count is preserved.
5. Use a custom tuning like `C4,D4,E4,G4,A4,C5,E5,G5` and confirm the API accepts it.
6. In the editor, add a note, remove the last note, merge with next, and split an event.
7. Test error handling for microphone denial, silent audio, and malformed custom tuning.

## Manual capture pack workflow

After each analysis, use the workflow panel to build a single zip.

Saved files inside the zip:

- `audio.wav`: exact WAV sent to API
- `request.json`: capture time, tuning, scenario, expected note summary, expected performance events, memo, audio metadata
- `response.json`: raw API response body
- `notes.md`: tester notes template with expected performance summary

### Expected performance entry

Use the clickable kalimba key UI in the workflow panel.

1. Select `録音の意図` first.
2. Click one or more keys to define the current expected event.
3. Set `追加回数` if the same event repeats.
4. Press `...を ... 回追加` to append it to the expected performance sequence.
5. Use `最後を取り消す` or `すべてクリア` to fix mistakes.

This avoids typing note names by hand and records both physical key numbers and note names in `request.json`. The selected capture intent is also stored, so later review can distinguish strict chords from `slide_chord` gestures.

Recommended case naming:

- The debug capture UI now auto-generates a case ID when the field is left blank.
- Format is roughly `YYYY-MM-DD-<auto-scenario>-<tuning-id>`.
- You can still override it manually when needed.

## Fixture Statuses

`expected.json` should carry a `status` field.

- `completed`: regression target, executed by pytest with strict assertions
- `pending`: expected is provisional, keep for recognition work
- `rerecord`: keep the case, but prioritize re-recording before treating it as a regression target
- `review_needed`: metadata or expected performance needs human review first
- `reference_only`: keep as a difficult example, but do not run in regression

Current pytest behavior:

- `completed` fixtures run as strict regression tests
- `pending` fixtures run as smoke probes only
- `rerecord`, `review_needed`, and `reference_only` run through metadata validation only unless a dedicated probe test exists

## Fixture Evaluation Scope

Keep the original `audio.wav` untouched. If only part of the recording should count toward regression, describe that in `expected.json`.

Supported optional fields:

- `evaluationWindows`: explicit list of `{ "startSec": <float>, "endSec": <float> }`
- `ignoredRanges`: explicit list of `{ "startSec": <float>, "endSec": <float> }`

Rules:

- use either `evaluationWindows` or `ignoredRanges`, never both
- ranges are in seconds against the original `audio.wav`
- prefer this over trimming the source file, so audits can still inspect the full recording
- only `completed` fixtures should rely on these ranges for regression assertions

## Manual Capture Fixtures

A saved browser capture pack can be turned into an API fixture.

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --min-events 5 --max-events 5 --required-event-note-set B4+D5=5
```

For a pending capture that should be stored but not executed yet:

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --status pending --reason "expected looks plausible but recognizer still fragments the gesture" --allow-incomplete
```

For a rerecord target with explicit guidance:

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --status rerecord --reason "strict chord ground truth is weak" --recommended-recapture "Record 5 clearly simultaneous takes." --recommended-recapture "Leave 1 second of silence between takes." --allow-incomplete
```

For a completed fixture that should only evaluate part of the source audio:

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --status completed --min-events 5 --max-events 5 --required-event-note-set C4+E4+G4=5 --evaluation-window 0.35:5.90
```

The importer extracts:

- `audio.wav`
- `request.json`
- `response.json`
- `notes.md`
- generated `expected.json`

into `apps/api/tests/fixtures/manual-captures/<fixture-id>/`.

Then run:

```powershell
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python -m pytest apps/api/tests
```

## Recording Request Format

When asking for new manual recordings, always include the purpose of the capture.

Recommended template:

1. `goal`: what failure mode or ambiguity this recording is meant to test
2. `gesture`: the intended performance style
   - strict simultaneous chord
   - `slide_chord`
   - `slide_chord` / sweep
   - separated single notes
3. `notes`: exact target note-set or sequence
4. `repetitions`: how many times to play it
5. `spacing`: whether to leave clear silence between repetitions
6. `success criteria`: what the recognizer should ideally output

Example:

- goal: distinguish strict four-note chord from slow `slide_chord`
- gesture: strict simultaneous chord
- notes: `E4 + G4 + B4 + D5`
- repetitions: 5
- spacing: leave clear silence between takes
- success criteria: 5 events, each with `E4+G4+B4+D5`

This makes later fixture review and independent audit much easier.

## Next automated tests to add

1. API endpoint tests for `/api/health`, `/api/tunings`, and `/api/transcriptions` using WAV fixtures.
2. Pure function tests for tuning parsing, note formatting, and notation rebuilding.
3. Frontend interaction tests for notation mode switching and event editing.
4. End-to-end browser test covering record or upload substitute, analyze, and edit flow.

## Independent Audit Cadence

After an independent audit, update fixture status as needed.

- keep as `completed` when the expected performance is trustworthy and suitable for regression
- keep as `pending` when the case is still a recognition target
- move to `rerecord` when recording or performance quality is the main blocker
- move to `review_needed` when the expected metadata is suspect`r`n- move to `reference_only` when a legacy capture is still informative but should no longer drive recognizer work

After each recognition-improvement round stabilizes, run an independent audit on the remaining pending fixtures.

Independent audit should explicitly separate `recording quality` from `performance quality`.

- Use raw waveform / STFT / FFT inspection and `request.json.captureIntent` as the primary evidence. Do not let the app's current prediction decide the verdict by itself.
- Mark `pending` when the input looks musically valid and the recognizer is still the main problem.
- Mark `rerecord` when clipping, noise, unstable attack timing, or inconsistent execution is the main problem.
- Mark `review_needed` when the expected labels, intent, or metadata are likely wrong.
- When audit changes fixture status, update both `expected.json` and `notes.md` so they do not drift.
- If only part of the file is a trustworthy regression target, prefer `evaluationWindows` or `ignoredRanges` over trimming the source audio.

- Use a separate analysis path based on direct waveform / FFT inspection, not the app's current predicted events.
- Verify whether `expectedPerformance` still looks plausible from the raw audio.
- Classify each pending case as one of:
  - expected looks correct, keep improving recognition
  - likely recording or performance issue, re-record first
  - likely labeling or input mistake, fix fixture metadata first

This audit should happen after a small cluster of improvements lands, not after every tiny tweak.

Use the helper script when you want a fast raw-audio audit over the non-completed fixtures:

```powershell
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python scripts/audit_manual_captures.py
```

This does not change fixture status by itself. It gives a quick waveform/STFT-based view of activity regions and rough target-note support, so you can decide whether a case is still `pending`, should move to `rerecord`, or should be narrowed with `evaluationWindows` / `ignoredRanges`.

## Current caveat

The web app compiles cleanly. API syntax compiles cleanly with `py -3.13 -m compileall apps/api/app apps/api/tests`, and the canonical runtime path is now:

```powershell
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python -m pytest apps/api/tests
```

`.pytest_cache` still emits an access denied warning in this Windows environment, but the test run itself is valid.

## Next manual capture priorities

These are the next recommended recordings to collect after the current four-note `slide_chord` regression fixes.

1. `E4 + G4 + B4 + D5` strict simultaneous chord re-record
   - Goal: replace the current `rerecord` strict fixture with a clean simultaneous four-note ground truth
   - Gesture: strict chord
   - Repetitions: 5
   - Spacing: leave about 1 second of silence between takes
   - Success criteria: `5 events`, each with `E4+G4+B4+D5`
   - Performance note: keep the four attacks as simultaneous as possible; do not intentionally roll the chord
2. 3-note strict reference chord if 4-note strict remains physically unstable
   - Goal: preserve a clean simultaneous reference even if 4-note strict chord remains ergonomically difficult
   - Gesture: strict chord
   - Notes: `C4 + E4 + G4` or `A4 + C5 + E5`
   - Repetitions: 5
   - Spacing: leave about 1 second of silence between takes
3. Legacy four-note fixture follow-up only if needed
   - Target: `kalimba-17-c-e4-g4-b4-d5-four-note-repeat-01`
   - Goal: either recover original intent from notes/audio history or keep it as `review_needed` / `reference_only`
   - This is lower priority than the explicit strict re-record above
4. Additional `slide_chord` speed variants only after strict rerecord is stable
   - `E4 + G4 + B4 + D5` `slide_chord` at a second speed
   - `E4 -> G4 -> B4 -> D5` `slide_chord` at a second speed
   - These are no longer the immediate bottleneck because the current `slide_chord` fixtures already regress cleanly


