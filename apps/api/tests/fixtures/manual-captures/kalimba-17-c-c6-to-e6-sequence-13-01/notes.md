# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-25-c6-to-e6-sequence-13-separated-notes-kalimba-17-c
- expected note: C6 x 3 / D6 x 5 / E6 x 4 / C6 + E6
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:56:19.004Z

## Expected Performance

- summary: C6 x 3 / D6 x 5 / E6 x 4 / C6 + E6

### Events

- 1. C6 [intent: separated_notes] :: C6 (#16)
- 2. D6 [intent: separated_notes] :: D6 (#1)
- 3. E6 [intent: separated_notes] :: E6 (#17)
- 4. C6 [intent: separated_notes] :: C6 (#16)
- 5. D6 [intent: separated_notes] :: D6 (#1)
- 6. E6 [intent: separated_notes] :: E6 (#17)
- 7. E6 [intent: separated_notes] :: E6 (#17)
- 8. D6 [intent: separated_notes] :: D6 (#1)
- 9. C6 [intent: separated_notes] :: C6 (#16)
- 10. D6 [intent: separated_notes] :: D6 (#1)
- 11. C6 + E6 [intent: strict_chord] :: C6 (#16), E6 (#17)
- 12. D6 [intent: separated_notes] :: D6 (#1)
- 13. E6 [intent: separated_notes] :: E6 (#17)

## Review

- summary: 一部の note-set または event 数がずれています。
- reason: 録音意図: 単音列 / 検出傾向: 要確認。認識改善の対象です。必要なら expected と detected の差分を見て、演奏意図の再確認も行ってください。

## Recapture Guidance

- 各音をはっきり区切り、次の音まで十分に待つ。
- 低音残響が長い場合でも、新しい打鍵の間隔を広めに取る。

## Memo

(empty)
