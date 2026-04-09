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

```bash
# WSL/Linux (primary)
uv sync
```

```powershell
# Windows (legacy)
py -3.13 -m pip install -r apps/api/requirements.txt
```

Python 3.14 caused a `pydantic-core` build issue. Use Python 3.13 for the API (Windows only; WSL uses uv-managed Python 3.13).

## Recommended local run

### API

```bash
# WSL/Linux (primary)
uv run uvicorn app.main:app --reload --app-dir apps/api
```

```powershell
# Windows (legacy)
$env:PYTHONPATH = "C:\src\calimba-score\apps\api"
py -3.13 -m uvicorn app.main:app --reload --app-dir apps/api
```

### Web

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev:web
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
- `request.json`: capture time, tuning, scenario, expected note summary, expected performance events, memo, source profile, audio metadata
- `response.json`: raw API response body
- `notes.md`: tester notes template with expected performance summary

### Expected performance entry

Use the clickable kalimba key UI in the workflow panel.

1. Select `録音の意図` first.
2. Click one or more keys to define the current expected event.
3. Set `追加回数` if the same event repeats.
4. Press `...を ... 回追加` to append it to the expected performance sequence.
5. Use `最後を取り消す` or `すべてクリア` to fix mistakes.

This avoids typing note names by hand and records both physical key numbers and note names in `request.json`. The selected capture intent is stored both as a recording-level field and, for new captures, on each expected event. Browser-recorded captures currently default to `sourceProfile = acoustic_real`.

String import also supports:

- simple sequences like `C4 / D4 / E4`
- per-event intent prefixes like `slide_chord:E4 + G4 + B4 + D5 x 4`
- JSON for mixed or compound events when one event combines multiple gesture parts

Example JSON import:

```json
{
  "defaultCaptureIntent": "separated_notes",
  "events": [
    { "notes": ["C4"] },
    {
      "parts": [
        { "intent": "slide_chord", "notes": ["C5", "D5"] },
        { "intent": "strict_chord", "notes": ["F4"] }
      ]
    }
  ]
}
```

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

Pytest markers:

- `manual_capture`: tests that exercise saved real-audio fixtures
- `slow`: long-running regression and integration tests

Useful commands:

```bash
# WSL/Linux (primary environment)
# Full test suite
uv run pytest apps/api/tests -q

# Fast inner loop: skip slow audio regressions
uv run pytest apps/api/tests -m "not slow" -q

# Focus on manual-capture regressions only
uv run pytest apps/api/tests -m manual_capture -q
```

```powershell
# Windows PowerShell (legacy)
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python -m pytest apps/api/tests -m "not slow"
```

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

```bash
# WSL/Linux (primary)
uv run python scripts/import_manual_capture.py <zip-path> <fixture-id> --min-events 5 --max-events 5 --required-event-note-set B4+D5=5
```

```powershell
# Windows (legacy)
py -3.13 scripts/import_manual_capture.py <zip-path> <fixture-id> --min-events 5 --max-events 5 --required-event-note-set B4+D5=5
```

For a pending capture that should be stored but not executed yet:

```bash
uv run python scripts/import_manual_capture.py <zip-path> <fixture-id> --status pending --reason "expected looks plausible but recognizer still fragments the gesture" --allow-incomplete
```

For a rerecord target with explicit guidance:

```bash
uv run python scripts/import_manual_capture.py <zip-path> <fixture-id> --status rerecord --reason "strict chord ground truth is weak" --recommended-recapture "Record 5 clearly simultaneous takes." --recommended-recapture "Leave 1 second of silence between takes." --allow-incomplete
```

For a completed fixture that should only evaluate part of the source audio:

```bash
uv run python scripts/import_manual_capture.py <zip-path> <fixture-id> --status completed --min-events 5 --max-events 5 --required-event-note-set C4+E4+G4=5 --evaluation-window 0.35:5.90
```

The importer extracts:

- `audio.wav`
- `request.json`
- `response.json`
- `notes.md`
- generated `expected.json`

into `apps/api/tests/fixtures/manual-captures/<fixture-id>/`.

Then run:

```bash
uv run pytest apps/api/tests -q
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

## Test Architecture (3-Tier Model)

テストは3層に分かれる。詳細は `AGENTS.md` の `## Test Architecture` を参照。

### Tier 1: Mechanism Tests

個別の recognizer 関数（`simplify_short_secondary_bleed`, `collapse_same_start_primary_singletons` 等）を構築した `RawEvent`/`NoteCandidate` 入力でテスト。

- ファイル: `test_event_processing.py`, `test_segment_peaks.py`, `test_detect_segments.py`, `test_repeated_patterns.py`
- 入力: 直接構築、または `fixtures/mechanism-snapshots/` の marshal データ
- アサーション: 関数の戻り値（ノート名、イベント数、etc.）
- **`payload["debug"]` のサブフィールドをアサートしない**

### Tier 2: Fixture Regression Tests

`test_manual_capture_completed.py` が全 `completed` fixture を自動的にパラメタライズドテスト。

- `expected.json` のアサーション（イベント数、ノートセット、順序）で検証
- `ground_truth.json` が存在する fixture はタイミングも検証
- **個別の fixture テストを追加する前に `expected.json` で表現できないか検討する**

### Tier 3: Ablation / Variant Tests

`test_ablation_pure.py`, `test_gap_filter.py` 等。フィーチャーフラグの ON/OFF で既存 fixture が壊れないかを検証。

### テスト追加ガイドライン

1. 新しい recognizer 関数 → Tier 1 mechanism test を追加
2. 新しい fixture → `expected.json` に適切なアサーションを記載（`expectedEventNoteSetsOrdered` 推奨）
3. 特定の偽検出を防ぎたい → `maxEventNoteSetOccurrences` や `expectedEventNoteSetsOrdered` で表現
4. タイミングの正しさを検証したい → `ground_truth.json` を作成

### ground_truth.json

人間が耳やスペクトログラムで確認した onset 時刻を記録するオプショナルファイル。

```json
{
  "version": 1,
  "toleranceSec": 0.05,
  "onsets": [
    {"timeSec": 2.08, "notes": ["D5"], "method": "spectrogram_verified"},
    {"timeSec": 4.60, "notes": ["D5"], "method": "ear_verified", "toleranceSec": 0.08}
  ]
}
```

- `timeSec`: audio.wav 先頭からの絶対秒（librosa 非依存）
- `toleranceSec`: デフォルト50ms、onset ごとにオーバーライド可能
- `method`: `ear_verified`, `spectrogram_verified`, `aubio_cross_checked`
- `test_manual_capture_completed.py` が自動的にチェック

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
- move to `reference_only` when a legacy capture is still informative but should no longer drive recognizer work

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

```bash
# WSL/Linux (primary)
uv run python scripts/audit_manual_captures.py
```

```powershell
# Windows (legacy)
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python scripts/audit_manual_captures.py
```

You can also limit the audit to one source profile:

```bash
uv run python scripts/audit_manual_captures.py --source-profile acoustic_real
```

To inspect corpus balance instead of per-fixture audio details:

```bash
uv run python scripts/audit_manual_captures.py --status completed --status pending --status rerecord --status review_needed --status reference_only --taxonomy-summary --summary-only
```

Current taxonomy buckets are heuristic and intended for collection planning, not for hard gating:

- `single_event_repeat`
- `small_repeated_phrase`
- `mixed_phrase`
- `free_performance`
- `no_expected_events`

This does not change fixture status by itself. It gives a quick waveform/STFT-based view of activity regions and rough target-note support, so you can decide whether a case is still `pending`, should move to `rerecord`, or should be narrowed with `evaluationWindows` / `ignoredRanges`.

## Current caveat

The canonical runtime path is:

```bash
# WSL/Linux (primary)
uv run pytest apps/api/tests -q
```

```powershell
# Windows (legacy)
$env:PYTHONPATH='C:\src\calimba-score\apps\api'
.\.venv313\Scripts\python -m pytest apps/api/tests
```

## Next manual capture priorities

There is no single blocking capture pattern right now. The main immediate bottlenecks are `arpeggio` design and explainability, not missing core regression samples.

Recommended collection strategy:

1. Free performance captures are acceptable
   - Prefer short, musically coherent phrases rather than random isolated notes
   - Always include `Expected Performance`
   - Add a brief memo when a section is intentionally `strict_chord`, `slide_chord`, `arpeggio`, or plain `separated_notes`
   - Expect these to land as `pending` or `reference_only` first, not necessarily `completed`

2. Targeted captures are still useful when they serve a design question
   - Highest-value future target: paired `arpeggio` vs `slide_chord` samples for the same note set
   - Example note sets: `C4 + E4 + G4`, `A4 + C5 + E5`, `E4 + G4 + B4 + D5`
   - Goal: make the category boundary testable without changing pitch content

3. Use intent notes when one recording mixes techniques
   - The current schema still stores recording-level `captureIntent`, but future design assumes event-level intent
   - Until event-level editing exists, note mixed-intent sections in the memo so fixture review can recover them later

4. Only request explicit re-records when a case blocks recognition work
   - Example: a capture is the best candidate for a strict baseline but the attacks are too staggered
   - Otherwise, prefer collecting broader musical material and triaging it with `pending / review_needed / reference_only`

