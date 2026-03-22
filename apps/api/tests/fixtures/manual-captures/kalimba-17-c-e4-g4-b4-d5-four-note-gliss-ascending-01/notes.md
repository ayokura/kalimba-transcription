# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-22-e4-plus-g4-plus-b4-plus-d5-repeat-05-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 5
- capture intent: gliss
- captured at: 2026-03-22T15:56:41.076Z

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
- reason: gliss / sweep としては現実的な崩れ方です。検出側は ambiguous 優勢でした。認識改善ターゲットとして保持してください。

## Recapture Guidance

- 1 gesture ごとに一方向へ連続してなぞり、途中で止めない。
- 各 gesture の間に無音を入れて区切る。
- 狙ったキー範囲だけを sweep する。

## Memo

(empty)

## Fixture Import Notes

- imported as ascending four-note gliss target on 2026-03-23
- current API output: 6 events, with two reconstructed 4-note events but remaining G4+B4+D5 / D5 / E4+F4 / G4+D5 fragments
- kept as `pending` because slow gliss is a realistic user pattern and should remain an improvement target

## Independent Audit (2026-03-23)

- Independent audit found monotonic ascending sweeps in several takes; E4 is often weak but the gliss intent is still plausible and should remain a recognition target.

