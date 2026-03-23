# Manual Notes

- tester: manual
- verdict: pending
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

## Review

- summary: event の分割または束ね方に大きな差があります。
- reason: 同時和音 の想定に対して差が大きく、検出側は 同時和音 優勢でした。録音意図と diff を確認してから fixture status を決めてください。

## Recapture Guidance

- 各反復で対象キーを同時に弾き、指をずらして順に入れない。
- 各反復の間に明確な無音を入れる。
- 和音の開始を揃え、スライド和音にならないようにする。
- 再録音前に expected と detected の差分を見て、演奏意図自体が正しいか確認する。

## Memo

(empty)


## Fixture Import Notes

- imported on 2026-03-23 as the best current strict 4-note rerecord candidate
- raw audit found 6 activity regions with strong four-note support across all regions
- current recognizer still fragments the performance, so this remains `rerecord` rather than `completed`


## Expected Count Correction (2026-03-23)

- human ear-check confirmed this take contains 6 intended repetitions, not 5
- status changed from `rerecord` to `pending` because the metadata correction removes the main rerecord rationale
- the remaining blocker is recognizer fragmentation into 7 events, not recording quality
