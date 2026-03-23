# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-23-e4-plus-g4-plus-b4-plus-d5-repeat-06-strict-chord-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 6
- capture intent: strict_chord
- source profile: acoustic_real
- captured at: 2026-03-23T12:08:24.941Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 6

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 5. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 6. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)

## Review

- summary: strict 4音の 6 反復が期待どおり検出され、completed fixture に昇格しました。
- reason: human ear-check で x6 に訂正した take を recognizer が 6 event の strict chord として安定復元できることを確認しました。

## Memo

- 2026-03-23: strict four-note normalization により 6 event / full chord x6 を確認


## Fixture Import Notes

- imported on 2026-03-23 as the best current strict 4-note rerecord candidate
- raw audit found 6 activity regions with strong four-note support across all regions
- 2026-03-23 update: recognizer now restores the take as `E4+G4+B4+D5 x 6`, so this fixture is `completed`


## Expected Count Correction (2026-03-23)

- human ear-check confirmed this take contains 6 intended repetitions, not 5
- final status is `completed` after the recognizer was updated to match the corrected x6 expectation
