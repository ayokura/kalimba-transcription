# Recognition Roadmap

## Current State (2026-04-06)

### Fixture カバレッジ

43 fixtures total:
- **30 completed** — strict regression target
- **9 pending** — recognizer 改善待ち
- **2 reference_only** — 参照用（regression 対象外）
- **1 review_needed** — メタデータ要確認
- **1 rerecord** — 再録音優先

### Gesture families (completed)
- single notes（C4, D4, D5 等の繰り返し）
- octave dyads（C4+C5, D4+D5）
- triads（C4+E4+G4, A4+D4+F4, E4+G4+B4）
- four-note strict / rolled / gliss chords（E4+G4+B4+D5）
- ascending sequences（C4→E6 17音、C6→E6 13/15音）
- descending sequences（E6→C4 17/51音、E6→G4 6音、D6→E6 10音）
- mixed phrases（混合シーケンス）
- BWV147 scoped phrases（6 sub-fixtures: late-upper-tail, lower-context-roll, lower-mixed-roll, upper-mixed-cluster, restart-prefix, restart-tail 等）

### BWV147 practical coverage
- **17-key 163-event sequence**: pending（full-sequence 認識はまだ未完）
- **34-key 163-event sequence**: pending（初の multi-layer kalimba fixture）
- 6 scoped BWV sub-fixtures: 4 completed, 2 pending

### Explicitly not a current acoustic regression target
- legacy four-note fixture with broken metadata (`reference_only`)
- smartphone app reference video/audio

## Current Bottleneck

30 completed fixtures が安定した regression baseline を形成している。主な残課題:

- **peaks redesign / chord selector** ([#111](https://github.com/ayokura/kalimba-transcription/issues/111)): `_evidence_rescue_gate` の複雑化、sequential accept loop の構造的制約。3-note chord の検出が restart-prefix / restart-tail の pending 理由
- **ranked candidate 不在問題** ([#125](https://github.com/ayokura/kalimba-transcription/issues/125)): segment 全体の FFT で primary 倍音に吸収される genuine note の検出。onset-focused FFT window 等の spectral acquisition 改善が必要
- **BWV147 full-sequence** (163-event fixture × 2 が pending): onset detection 層の問題（34-key で 4 events が NO MATCH）と post-processing の fixture-specific debt
- **`arpeggio` modeling** ([#6](https://github.com/ayokura/kalimba-transcription/issues/6)): `slide_chord` との分離。Phase 1 設計は [arpeggio-design.md](arpeggio-design.md) に記載

## Active Fixture Policy

### ステータス分布

- `completed` (30): strict regression target。`test_manual_capture_completed.py` が自動検証
- `pending` (9): recognizer 改善待ち。smoke probe のみ実行
- `reference_only` (2): 参照用。regression 対象外
- `review_needed` (1): メタデータ要確認
- `rerecord` (1): 再録音優先

詳細なステータス定義は [testing.md](testing.md) を参照。

### Historical context
- strict four-note reference (`four-note-strict-repeat-02/03/04`): 初期の認識精度検証に使用。現在は 30 completed fixtures のうちの一部
- legacy four-note reference (`four-note-repeat-01`): `reference_only` — broken scenario metadata
- BWV147 fixture-specific ルールの管理は [recognizer-local-rules.md](recognizer-local-rules.md) を参照

## User-Facing Gesture Families

Current canonical user-facing families are:
- `strict_chord`
- `slide_chord`
- `separated_notes`
- `ambiguous`

For kalimba semantics, `rolled_chord` and `gliss` are unified as `slide_chord`.

## Future Boundary: Arpeggio

`arpeggio` should **not** be folded into `slide_chord`.

Why:
- `slide_chord` still resolves to one chord gesture family
- `arpeggio` is an ordered broken-chord pattern with distinct time structure
- later notation and editing will need note order and direction, not just the harmonic set

Planned direction:
- keep `arpeggio` separate from `slide_chord`
- attach it to a main harmonic event or explicitly link it to one
- avoid double-counting one musical idea as several unrelated top-level chord events

Tracking issue:
- [#6](https://github.com/ayokura/kalimba-transcription/issues/6)

## Local Rule Debt

For fixture-specific or very local recognizer rules that may become debt for future free-performance transcription, see [recognizer-local-rules.md](recognizer-local-rules.md).
For Strategy B gap-candidate design and the current candidate/promotion prototype, see [strategy-b-gap-candidates.md](strategy-b-gap-candidates.md).

## Immediate Next Engineering Tasks

1. 30 completed fixtures の regression baseline を維持する
2. peaks redesign (#111): chord selector による sequential accept loop の構造改善
3. ranked candidate 不在問題 (#125): onset-focused FFT window 等の spectral acquisition 改善
4. BWV147 full-sequence の pending 解消（onset detection 層 + post-processing debt）
5. `arpeggio` Phase 1 の vocabulary 導入（[arpeggio-design.md](arpeggio-design.md) 参照）

## Suggested Future Sample Matrix

When collection resumes, prefer paired samples for the same pitch set:

### For the same note set
- `strict_chord`
- `slide_chord`
- `arpeggio`

### Suggested note sets
- `C4 + E4 + G4`
- `A4 + C5 + E5`
- `E4 + G4 + B4 + D5`

This will make family boundaries testable without changing pitch content.

Longer term, intent should move from recording-level metadata to event-level metadata. The current one-intent-per-capture model is acceptable only because fixtures are deliberately collected as one-gesture-per-take.
## Next Recording Priorities

The current acoustic recognizer no longer has a strong repeated-pattern redesign blocker. The next useful data is directional and register-sensitive practical material.

### Priority 1: Descending separated-note runs

Purpose:
- validate whether ascending-only local carryover cleanup generalizes
- measure whether descending runs need their own logic or only more data

Suggested captures:
- E6 -> D6 -> C6 -> B5 -> A5 -> G5 -> F5 -> E5 -> D5 -> C5 -> B4 -> A4 -> G4 -> F4 -> E4 -> D4 -> C4
- single pass
- same phrase repeated 3 times without intentional silence between notes

### Priority 2: High-register short-tine coverage

Purpose:
- measure whether D6/E6 need note-specific handling or only better fixture support
- capture the shorter sustain / different timbre the user identified near the top of the instrument

Suggested captures:
- alternating D6 / E6 single notes, 5 cycles
- C6 / D6 / E6 ascending and descending, each 5 cycles
- short phrase endings such as A5 / B5 / C6 / D6 / E6

### Priority 3: High-register mixed phrase endings

Purpose:
- test whether phrase tails near D6/E6 are still robust when they follow denser mid-register material

Suggested captures:
- C4 -> ... -> B5 -> C6 -> D6 -> E6
- E6 -> C4 restart after a brief gap
- one or two realistic phrase endings rather than isolated unit patterns only

Collection rule:
- prefer real-device fixtures
- keep articulation natural unless the goal specifically requires silence between notes
- when a phrase mixes techniques, add intent notes in the memo even if the current schema is still recording-level
## Future Input Source Profiles

For a detailed comparison between the current acoustic recognizer and the local app/synth reference audio, see [app-synth-audio-gap-analysis.md](C:/src/calimba-score/docs/app-synth-audio-gap-analysis.md).


Long term, the recognizer should not assume one acoustic environment.
Different input sources should be represented explicitly.

Initial profile split:
- `acoustic_real`: real kalimba recorded by microphone
- `app_synth`: smartphone or software kalimba app audio/video

Likely future dimensions inside `acoustic_real`:
- close mic vs room mic
- quiet room vs noisy room
- different phone / laptop microphone responses
- different kalimba models and resonance behavior

Policy:
- primary regression stays on `acoustic_real`
- app-derived material stays `reference_only` until a separate profile exists
- source profile should affect fixture status, evaluation policy, and future feature normalization

Why this matters:
- phone app audio can still be useful for pattern discovery and symbolic references
- but mixing it directly into the real-device regression pool will blur recognizer tuning decisions

## Smartphone App Reference Video

Local path:
- `C:\src\calimba-score\.codex-media\source-videos\ScreenRecording_03-23-2026_13-09-56_1.mov`

Use it only as:
- visual vocabulary reference
- possible future `reference_only` UI material

Do **not** use it as:
- acoustic regression input
- real-device performance ground truth

Reason:
- it reflects app rendering and phone capture behavior, not real kalimba acoustics or hand technique

## App-Video Arpeggio Candidate

The later half of the local smartphone-app reference is not best understood as `slide_chord`.
It is better treated as an `arpeggio candidate` / broken-chord reference.

Current evidence from `.codex-media/derived-analysis/kira-kira-expected-performance.json`:
- strongest candidate block: about `15.41s-22.83s`
- clearest arpeggio-like sub-block: about `19.25s-20.89s`
- projected sequence in that sub-block: `F4+F5 / A4 / C5 / A5 / A4 / F4`

Interpretation:
- this is useful as a reference for future `arpeggio` semantics and sample planning
- it is not a current acoustic regression target
- it should not be folded into `slide_chord` semantics

Near-term use:
- keep the source media local under `.codex-media/`
- use the derived event sequence only as `reference_only` design evidence
- collect equivalent real-device samples before recognizer work starts

## Future Real-Device Sample Families

The app-video analysis suggests these real-device families are worth collecting later:
- wide-register dyads such as `G5 + C4`, `C5 + G4`, `F4 + F5`
- wide-register triads such as `G5 + E4 + G4`
- chord-to-single continuations such as `B4 + D5 -> D5`
- broken-chord / `arpeggio` patterns such as `F4+F5 / A4 / C5 / A5 / A4 / F4`

These should be recorded as real-device fixtures, not derived directly from the app video.

## Recording Request Template

Use this structure whenever asking for new manual data:

- `goal`
- `gesture`
- `notes`
- `repetitions`
- `spacing`
- `success criteria`

For the current strict four-note rerecord:
- goal: rebuild clean simultaneous four-note reference
- gesture: `strict_chord`
- notes: `E4 + G4 + B4 + D5`
- repetitions: `5`
- spacing: about `1s` silence between takes
- success criteria: `5 events`, each `E4+G4+B4+D5`

