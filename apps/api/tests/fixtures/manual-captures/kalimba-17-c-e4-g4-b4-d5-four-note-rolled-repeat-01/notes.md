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

- summary: 先頭5回は rolled chord として妥当ですが、focused region でも認識はまだ割れます。
- reason: 最後の別 take は評価範囲から外しましたが、先頭5回だけを見ても認識はまだ 1 chord に安定していません。

## Recapture Guidance

- 毎回ほぼ同じ方向と速さで順に鳴らす。
- 和音の各音の入り方が毎回大きく変わらないようにする。
- 反復の間に明確な無音を入れる。

## Memo

(empty)

## Fixture Import Notes

- imported as rolled-chord target on 2026-03-23
- evaluation scope excludes the final separated fragment via ignoredRanges: 24.5s-34.14s
- first 5 takes are isolated as the evaluation scope, but the fixture remains pending until reconstruction stabilizes

## Independent Audit (2026-03-23)

- Independent audit found monotonic rolled attacks across takes; E4/G4 are weaker than upper notes, and the first five takes are good focused material, but the recognizer still fragments them, so the fixture remains pending.
