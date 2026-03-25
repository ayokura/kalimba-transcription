# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-25-c6-to-e6-sequence-15-separated-notes-kalimba-17-c
- expected note: C6 x 5 / D6 x 5 / E6 x 5
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:48:14.575Z

## Expected Performance

- summary: C6 x 5 / D6 x 5 / E6 x 5

### Events

- 1. C6 [intent: separated_notes] :: C6 (#16)
- 2. D6 [intent: separated_notes] :: D6 (#1)
- 3. E6 [intent: separated_notes] :: E6 (#17)
- 4. C6 [intent: separated_notes] :: C6 (#16)
- 5. D6 [intent: separated_notes] :: D6 (#1)
- 6. E6 [intent: separated_notes] :: E6 (#17)
- 7. C6 [intent: separated_notes] :: C6 (#16)
- 8. D6 [intent: separated_notes] :: D6 (#1)
- 9. E6 [intent: separated_notes] :: E6 (#17)
- 10. C6 [intent: separated_notes] :: C6 (#16)
- 11. D6 [intent: separated_notes] :: D6 (#1)
- 12. E6 [intent: separated_notes] :: E6 (#17)
- 13. C6 [intent: separated_notes] :: C6 (#16)
- 14. D6 [intent: separated_notes] :: D6 (#1)
- 15. E6 [intent: separated_notes] :: E6 (#17)

## Review

- summary: 一部の note-set または event 数がずれています。
- reason: 録音意図: 単音列 / 検出傾向: 要確認。認識改善の対象です。必要なら expected と detected の差分を見て、演奏意図の再確認も行ってください。

## Recapture Guidance

- 各音をはっきり区切り、次の音まで十分に待つ。
- 低音残響が長い場合でも、新しい打鍵の間隔を広めに取る。

## Memo

(empty)
