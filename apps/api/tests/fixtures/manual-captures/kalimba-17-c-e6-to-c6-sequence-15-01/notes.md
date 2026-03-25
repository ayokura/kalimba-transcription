# Manual Notes

- tester: manual
- verdict: review_needed
- scenario: 2026-03-25-e6-to-c6-sequence-15-separated-notes-kalimba-17-c
- expected note: E6 x 5 / D6 x 5 / C6 x 5
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:49:14.911Z

## Expected Performance

- summary: E6 x 5 / D6 x 5 / C6 x 5

### Events

- 1. E6 [intent: separated_notes] :: E6 (#17)
- 2. D6 [intent: separated_notes] :: D6 (#1)
- 3. C6 [intent: separated_notes] :: C6 (#16)
- 4. E6 [intent: separated_notes] :: E6 (#17)
- 5. D6 [intent: separated_notes] :: D6 (#1)
- 6. C6 [intent: separated_notes] :: C6 (#16)
- 7. E6 [intent: separated_notes] :: E6 (#17)
- 8. D6 [intent: separated_notes] :: D6 (#1)
- 9. C6 [intent: separated_notes] :: C6 (#16)
- 10. E6 [intent: separated_notes] :: E6 (#17)
- 11. D6 [intent: separated_notes] :: D6 (#1)
- 12. C6 [intent: separated_notes] :: C6 (#16)
- 13. E6 [intent: separated_notes] :: E6 (#17)
- 14. D6 [intent: separated_notes] :: D6 (#1)
- 15. C6 [intent: separated_notes] :: C6 (#16)

## Review

- summary: event の分割または束ね方に大きな差があります。
- reason: 単音列 の想定に対して差が大きく、検出側は 要確認 優勢でした。録音意図と diff を確認してから fixture status を決めてください。

## Recapture Guidance

- 各音をはっきり区切り、次の音まで十分に待つ。
- 低音残響が長い場合でも、新しい打鍵の間隔を広めに取る。
- 再録音前に expected と detected の差分を見て、演奏意図自体が正しいか確認する。

## Memo

(empty)
