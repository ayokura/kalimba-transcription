# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-22-e4-plus-g4-plus-b4-plus-d5-repeat-05-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 5
- capture intent: strict_chord
- captured at: 2026-03-22T15:52:46.153Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 5

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 5. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)

## Review

- summary: 一部の note-set または event 数がずれています。
- reason: 録音意図: 同時和音 / 検出傾向: 同時和音。認識改善の対象です。必要なら expected と detected の差分を見て、演奏意図の再確認も行ってください。

## Recapture Guidance

- 各反復で対象キーを同時に弾き、指をずらして順に入れない。
- 各反復の間に明確な無音を入れる。
- 和音の開始を揃え、slow slide_chord にならないようにする。

## Memo

(empty)

## Fixture Import Notes

- imported as strict-chord boundary sample on 2026-03-23
- current API output: 6 events, fragmented with mixed strict/slide_chord-like hints
- kept as `rerecord` rather than `pending` because the strict intent is explicit but the performance is materially staggered

## Independent Audit (2026-03-23)

- raw band-energy audit found all four target notes in 4 of 5 takes
- onset spread stayed small on most takes, so strict intent is plausible
- one take weakens B4, but the main blocker is recognizer fragmentation rather than recording quality
- fixture status promoted from `rerecord` to `pending`


## Independent Audit Update (2026-03-23 late)

- stricter review: recording quality is good, but the performed takes do not consistently establish a clean strict four-note ground truth
- D5 dominates repeatedly, while E4 / G4 / B4 are too weak or staggered in some takes
- status set back to `rerecord`


## Fixture Position Update (2026-03-23)

- superseded by `kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03` as the main strict 4-note rerecord target
- retained only as `reference_only` evidence
