# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-22-e4-plus-g4-plus-b4-plus-d5-repeat-05-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 4 (scoped)
- capture intent: slide_chord
- captured at: 2026-03-22T15:55:33.337Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 4 (scoped)

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)


## Review

- summary: 4つの valid take は slide_chord として成立し、scope 内で completed に昇格した。
- reason: 4回目 take と最後の別 take は引き続き ignoredRanges で除外し、そのほかの4 takeを completed regression として固定した。

## Recapture Guidance

- 毎回ほぼ同じ方向と速さで順に鳴らす。
- 和音の各音の入り方が毎回大きく変わらないようにする。
- 反復の間に明確な無音を入れる。

## Memo

(empty)

## Fixture Import Notes

- imported as slide_chord target on 2026-03-23
- evaluation scope excludes the fourth in-scope take via ignoredRanges: 17.3s-19.0s
- evaluation scope excludes the final separated fragment via ignoredRanges: 24.5s-34.14s
- practical slide_chord sample completed on four scoped takes; do not distort recognizer behavior to force the excluded fourth take into a rolled/slide interpretation

## Independent Audit

- 2026-03-23: Independent audit found monotonic slide_chord attacks across takes, but later regression review showed that the fourth in-scope take contains an extra trailing note and should not be treated as a valid rolled/slide target.

