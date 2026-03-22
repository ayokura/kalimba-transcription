# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-22-e4-plus-g4-plus-b4-plus-d5-repeat-05-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 5
- capture intent: rolled_chord
- captured at: 2026-03-22T15:55:33.337Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 5

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 5. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)

## Review

- summary: 録音意図に対して segmentation がまだ粗いです。
- reason: rolled chord としては現実的な崩れ方です。検出側は rolled chord 優勢でした。認識改善ターゲットとして保持してください。

## Recapture Guidance

- 毎回ほぼ同じ方向と速さで順に鳴らす。
- 和音の各音の入り方が毎回大きく変わらないようにする。
- 反復の間に明確な無音を入れる。

## Memo

(empty)

## Fixture Import Notes

- imported as rolled-chord target on 2026-03-23
- current API output: 6 events, first 5 takes reconstruct as full 4-note rolled chords, final tail fragment remains
- kept as `pending` because this is a realistic user input pattern worth improving

## Independent Audit (2026-03-23)

- Independent audit found monotonic rolled attacks across takes; E4/G4 are weaker than upper notes, but the input pattern is realistic and should remain a recognition target.

