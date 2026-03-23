# Manual Notes

- tester: manual
- verdict: reference_only
- scenario: 2026-03-23-e4-plus-g4-plus-b4-plus-d5-repeat-05-strict-chord-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 5
- capture intent: strict_chord
- source profile: acoustic_real
- captured at: 2026-03-23T12:10:21.768Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 5

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 5. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)

## Review

- summary: event の分割または束ね方に大きな差があります。
- reason: 同時和音 の想定に対して差が大きく、検出側は 要確認 優勢でした。録音意図と diff を確認してから fixture status を決めてください。

## Recapture Guidance

- 各反復で対象キーを同時に弾き、指をずらして順に入れない。
- 各反復の間に明確な無音を入れる。
- 和音の開始を揃え、スライド和音にならないようにする。
- 再録音前に expected と detected の差分を見て、演奏意図自体が正しいか確認する。

## Memo

(empty)


## Fixture Import Notes

- imported on 2026-03-23 as an alternate strict 4-note rerecord attempt
- weaker than `strict-repeat-03` in both raw support balance and recognizer output
- retained as `reference_only` evidence only
