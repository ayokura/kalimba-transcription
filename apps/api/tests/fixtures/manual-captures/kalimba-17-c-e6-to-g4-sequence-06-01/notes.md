# Manual Notes

- tester: manual
- verdict: review_needed
- scenario: 2026-03-25-e6-to-g4-sequence-06-separated-notes-kalimba-17-c
- expected note: E6 / C4 / D4 / E4 / F4 / G4
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:57:23.176Z

## Expected Performance

- summary: E6 / C4 / D4 / E4 / F4 / G4

### Events

- 1. E6 [intent: separated_notes] :: E6 (#17)
- 2. C4 [intent: separated_notes] :: C4 (#9)
- 3. D4 [intent: separated_notes] :: D4 (#8)
- 4. E4 [intent: separated_notes] :: E4 (#10)
- 5. F4 [intent: separated_notes] :: F4 (#7)
- 6. G4 [intent: separated_notes] :: G4 (#11)

## Review

- summary: event の分割または束ね方に大きな差があります。
- reason: 単音列 の想定に対して差が大きく、検出側は 要確認 優勢でした。録音意図と diff を確認してから fixture status を決めてください。

## Recapture Guidance

- 各音をはっきり区切り、次の音まで十分に待つ。
- 低音残響が長い場合でも、新しい打鍵の間隔を広めに取る。
- 再録音前に expected と detected の差分を見て、演奏意図自体が正しいか確認する。

## Memo

(empty)
