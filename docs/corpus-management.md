# Corpus Management Policy

This project keeps two distinct free-performance corpus layers.

## 1. Repository-managed corpus

Path:

```text
apps/api/tests/fixtures/free-performance-corpus/<transaction-id>/
```

Purpose:

- Reproducible free-performance benchmark corpus.
- Safe to use in CI and by other contributors.
- Contains audio and teacher/ground-truth data.

Required files:

- `audio.wav`
- `request.json`
- `ground_truth.json`
- `metadata.json`
- `corrections.json` when promoted from review UI corrections
- `review_status.json` when promoted from the review workflow

Human review requirement:

- Repository inclusion requires a human copyright/rights decision.
- `metadata.json.rightsReview.status` must be `approved_for_repository`.
- `metadata.json.copyright.status` must be `original_performance`,
  `public_domain`, or another explicitly cleared status.
- If copyright status is unknown or questionable, do **not** commit the audio or
  teacher data to this directory.

Recommended metadata:

- recording device and microphone, when known
- sample rate, channel count, duration, peak/RMS level
- selected tuning
- recognizer event count, corrected event count, candidate slot count
- correction origin counts (`recognizer`, `edited`, `inserted-slot`,
  `inserted-manual`)
- review status and provenance

Example recording metadata:

```json
{
  "recording": {
    "device": "Samsung Galaxy S25 Ultra",
    "microphone": "internal microphone"
  }
}
```

## 2. Local-only / non-repository corpus

Paths:

```text
data/transactions/<transaction-id>/
apps/api/tests/fixtures/transaction-captures/<transaction-id>/
```

Purpose:

- Useful local data that may be copyrighted, unknown, or not yet rights-reviewed.
- Developer-specific benchmark and investigation data.
- Not assumed available in CI or other clones.

Rules:

- Do not commit audio or teacher data unless a human review has approved
  repository management.
- Valuable but non-clearable data can remain local and still be used in this
  environment.
- If local corrections are later cleared, promote them into the repository-managed
  corpus with a complete `metadata.json`.

## Promotion workflow

For review-UI corrections:

1. User/tester records audio.
2. Review UI saves `corrections.json`.
3. Review UI saves `review_status.json`.
4. Only `review_completed` recordings are candidates for ground-truth promotion.
5. A human confirms copyright/rights suitability for repository management.
6. Add an item under `free-performance-corpus/<transaction-id>/` with audio,
   request, corrections, review status, ground truth, and metadata.

`promote_corrections_to_ground_truth.py` converts reviewed corrections into
`ground_truth.json`. The generated `ground_truth.json` should keep
`source.provenance = "tester_corrected"`; this is distinct from
human-verified onset annotation methods such as `ear_verified` or
`spectrogram_verified`.

## Benchmark behavior

`note_f1_benchmark.py` discovers repository-managed corpus items first and then
local transaction-capture ground truth. Repository-managed corpus items carry
their own audio and request data, so they are reproducible outside the original
recording environment.

## Regression baseline (Tier 4 gate)

`free-performance-corpus/benchmark_baseline.json` records, per recording, the
note-F1 floor (`minF1`) and hard-miss ceiling (`maxHardMisses`) the current
recognizer must maintain. `apps/api/tests/test_free_performance_corpus.py`
asserts these in CI for every git-tracked corpus recording, plus the
governance requirements above (`rightsReview.status`, `copyright.status`).

Update discipline:

- Baselines move in the **improvement direction only**. After a recognizer
  improvement, run
  `uv run python scripts/audio-analysis/note_f1_benchmark.py --write-baseline`
  and commit the updated baseline together with the change that earned it.
- `--write-baseline` refuses to lower `minF1` / raise `maxHardMisses`. A
  lowering update requires `--allow-baseline-regression` and an explicit
  tradeoff decision recorded like a fixture-policy decision. Never lower a
  baseline just to make CI pass.
- When adding a recording to the corpus, add its baseline entry in the same
  commit (the test suite fails on a corpus recording without one).
- Local-only recordings (transaction-captures) also carry baseline entries but
  are invisible to CI; after recognizer changes, run
  `note_f1_benchmark.py --check-baseline` locally to cover them.

## Timing accuracy caveat (review-corrected ground truth)

Ground truth promoted from the web review UI (`method: "user_corrected"`,
`source.provenance: "tester_corrected"`) has a known accuracy profile:

- **Reliable**: note identity, ordering, and combination/grouping. The score is
  generally correct (apart from genuine misplays), because the tester confirmed
  what was played.
- **Approximate**: onset timing. The `timeSec` values are not spectral-grade.
  - recognizer-origin events use the recognizer's detected event start, which can
    itself deviate slightly from the perceptual/spectral onset;
  - `inserted-slot` events use a dropped-segment boundary;
  - `inserted-manual` events are hand-placed on the review timeline.

Implications:

- Fine for score-level and note-identity evaluation (the main free-performance
  benchmark use).
- Be cautious using these onset times for timing-sensitive purposes: training,
  tempo/rhythm modeling, onset-time learning, or anything needing spectral-grade
  timing.
- Per-onset `ground_truth.json` already records `toleranceSec`, and review-derived
  onsets get a wider tolerance for `inserted-slot` / `inserted-manual` origins.

Optional refinement:

- An agent or human may re-verify onset times with spectral analysis. If done,
  raise the affected per-onset `method` to `spectrogram_verified` /
  `aubio_cross_checked` and note it in the corpus `metadata.json`.
- This is not required unless a timing-sensitive use actually needs it; the
  review-corrected timeline is sufficient for score-level evaluation as-is.
