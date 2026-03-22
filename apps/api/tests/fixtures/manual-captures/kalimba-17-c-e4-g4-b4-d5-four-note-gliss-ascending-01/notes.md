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

- summary: note-set の復元は通りましたが、gesture はまだ gliss ではなく rolled chord 側へ寄っています。
- reason: 5回とも E4+G4+B4+D5 へ再構成できていますが、gesture-aware 判定は未解決です。

## Recapture Guidance

- 1 gesture ごとに一方向へ連続してなぞり、途中で止めない。
- 各 gesture の間に無音を入れて区切る。
- 狙ったキー範囲だけを sweep する。

## Memo

(empty)

## Fixture Import Notes

- imported as ascending four-note gliss target on 2026-03-23
- current API output: 5 events, all reconstruct as E4+G4+B4+D5
- kept as `pending` because gesture classification is still rolled_chord-biased instead of gliss

## Independent Audit (2026-03-23)

- Independent audit found monotonic ascending sweeps in several takes; E4 is often weak but the gliss intent is still plausible. Note reconstruction is now acceptable, but gesture labeling is still open.
