# Free-Performance Corpus

Repository-managed recordings for free-form kalimba transcription evaluation.
These are different from `manual-captures/` (score-known regression fixtures)
and from local `transaction-captures/` (developer/tester transaction scratch
space).

## Inclusion policy

Only recordings that passed a human rights/copyright review may be committed
here. A corpus item must include:

- `audio.wav`
- `request.json`
- `ground_truth.json`
- `metadata.json`
- usually `corrections.json` and `review_status.json` for provenance

`metadata.json.rightsReview.status` must be `approved_for_repository`.
`metadata.json.copyright.status` must be `original_performance`,
`public_domain`, or another explicitly cleared status. Copyright-unknown or
copyrighted recordings should remain local-only under `data/transactions/` plus
ignored `transaction-captures/` metadata.

## Metadata notes

`metadata.json.recording.device` / `microphone` should record capture hardware
when known (for example: `Samsung Galaxy S25 Ultra`, `internal microphone`).
This is not used by the recognizer yet, but is useful for future recording-profile
calibration.

Review-UI-derived `ground_truth.json` is score-level ground truth, not
spectral-grade onset annotation. `method: "user_corrected"` means note identity,
order, and grouping were corrected by a tester; `timeSec` values remain
approximate because they come from recognizer event starts, dropped segment
boundaries, or manual timeline placement. Use `toleranceSec` and re-verify with
spectral analysis before timing-sensitive training.

## Benchmark discovery

`scripts/audio-analysis/note_f1_benchmark.py` discovers this directory first.
Each corpus item carries its own `audio.wav` and `request.json`, so benchmark
results are reproducible without local `data/transactions/` state.
