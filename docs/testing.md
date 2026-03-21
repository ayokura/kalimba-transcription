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

1. Click one or more keys to define the current expected event.
2. Set `追加回数` if the same event repeats.
3. Press `...を ... 回追加` to append it to the expected performance sequence.
4. Use `最後を取り消す` or `すべてクリア` to fix mistakes.

This avoids typing note names by hand and records both physical key numbers and note names in `request.json`.

Recommended case naming:

- `YYYY-MM-DD-<scenario>-<index>`
- Example: `2026-03-21-b4-d5-double-note-01`

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

## Next automated tests to add

1. API endpoint tests for `/api/health`, `/api/tunings`, and `/api/transcriptions` using WAV fixtures.
2. Pure function tests for tuning parsing, note formatting, and notation rebuilding.
3. Frontend interaction tests for notation mode switching and event editing.
4. End-to-end browser test covering record or upload substitute, analyze, and edit flow.

## Current caveat

The web app compiles cleanly. API syntax compiles cleanly with `py -3.13 -m compileall apps/api/app apps/api/tests`, but full runtime API tests were not completed because Python package resolution is inconsistent in this Windows environment outside the direct install flow.
