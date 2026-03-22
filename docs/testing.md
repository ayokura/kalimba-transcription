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

This avoids typing note names by hand and records both physical key numbers and note names in `request.json`. The selected capture intent is also stored, so later review can distinguish strict chords from rolled chords or gliss-like gestures.

Recommended case naming:

- The debug capture UI now auto-generates a case ID when the field is left blank.
- Format is roughly `YYYY-MM-DD-<auto-scenario>-<tuning-id>`.
- You can still override it manually when needed.

## Fixture Statuses

`expected.json` should carry a `status` field.

- `completed`: regression target, executed by pytest
- `pending`: expected is provisional, keep for recognition work
- `rerecord`: keep the case, but prioritize re-recording before treating it as a regression target
- `review_needed`: metadata or expected performance needs human review first
- `reference_only`: keep as a difficult example, but do not run in regression

Current pytest behavior should execute only `completed` fixtures.

## Manual Capture Fixtures

A saved browser capture pack can be turned into an API regression fixture.

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --min-events 5 --max-events 5 --required-event-note-set B4+D5=5
```

For a pending capture that should be stored but not executed yet:

```powershell
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --allow-incomplete
```

This extracts:

- `audio.wav`
- `request.json`
- `response.json`
- `notes.md`
- generated `expected.json`

into `apps/api/tests/fixtures/manual-captures/<fixture-id>/`.

Then run:

```powershell
$env:PYTHONPATH='C:\src\calimba-score\.pydeps;C:\src\calimba-score\apps\api'
py -3.13 -m pytest apps/api/tests
```

## Recording Request Format

When asking for new manual recordings, always include the purpose of the capture.

Recommended template:

1. `goal`: what failure mode or ambiguity this recording is meant to test
2. `gesture`: the intended performance style
   - strict simultaneous chord
   - rolled chord
   - gliss / sweep
   - separated single notes
3. `notes`: exact target note-set or sequence
4. `repetitions`: how many times to play it
5. `spacing`: whether to leave clear silence between repetitions
6. `success criteria`: what the recognizer should ideally output

Example:

- goal: distinguish strict four-note chord from slow rolled chord
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
- move to `review_needed` when the expected metadata is suspect

After each recognition-improvement round stabilizes, run an independent audit on the remaining pending fixtures.

Independent audit should explicitly separate `recording quality` from `performance quality`.

- Use raw waveform / STFT / FFT inspection and `request.json.captureIntent` as the primary evidence. Do not let the app's current prediction decide the verdict by itself.
- Mark `pending` when the input looks musically valid and the recognizer is still the main problem.
- Mark `rerecord` when clipping, noise, unstable attack timing, or inconsistent execution is the main problem.
- Mark `review_needed` when the expected labels, intent, or metadata are likely wrong.
- When audit changes fixture status, update both `expected.json` and `notes.md` so they do not drift.

- Use a separate analysis path based on direct waveform / FFT inspection, not the app's current predicted events.
- Verify whether `expectedPerformance` still looks plausible from the raw audio.
- Classify each pending case as one of:
  - expected looks correct, keep improving recognition
  - likely recording or performance issue, re-record first
  - likely labeling or input mistake, fix fixture metadata first

This audit should happen after a small cluster of improvements lands, not after every tiny tweak.

## Current caveat

The web app compiles cleanly. API syntax compiles cleanly with `py -3.13 -m compileall apps/api/app apps/api/tests`, but full runtime API tests were not completed because Python package resolution is inconsistent in this Windows environment outside the direct install flow.

## Next manual capture priorities

These are the next recommended recordings to collect after the current mixed-sequence fix.

1. `D4` single-note repeats
   - Goal: baseline low-note decay and octave misread behavior
   - Likely failures: `D5/E4` mis-snap, over-segmentation, tail pickup
2. `C4` single-note repeats
   - Goal: lowest-range decay stability
   - Likely failures: `C5` octave alias, false events in silence
3. `D4 -> D5` separated single notes
   - Goal: confirm low-note decay does not pollute the next higher note
   - Likely failures: false dyads, carryover events
4. `C4 + C5` octave dyad repeats
   - Goal: preserve a true octave dyad while still suppressing octave alias errors
   - Likely failures: one side disappearing, high-note bias
5. `D4 + D5` octave dyad repeats
   - Goal: verify octave handling in the lower register
   - Likely failures: collapse to `D5`, spurious split events
6. `B4 + D5` non-octave dyad repeats
   - Goal: keep a normal dyad stable after the octave-focused fixes
   - Likely failures: `D5` dominance, `B4` dropout
7. Short mixed phrase: `C4 -> C4+C5 -> B4+D5 -> D4`
   - Goal: evaluate transitions once atomic cases are stable
   - Likely failures: decay contaminating the next event, segmentation drift

