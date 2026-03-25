# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-25-e6-to-c4-sequence-51-separated-notes-kalimba-17-c
- expected note: E6 x 3 / D6 x 3 / C6 x 3 / B5 x 3 / A5 x 3 / G5 x 3 / F5 x 3 / E5 x 3 / D5 x 3 / C5 x 3 / B4 x 3 / A4 x 3 / G4 x 3 / F4 x 3 / E4 x 3 / D4 x 3 / C4 x 3
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:45:26.760Z

## Expected Performance

- summary: E6 x 3 / D6 x 3 / C6 x 3 / B5 x 3 / A5 x 3 / G5 x 3 / F5 x 3 / E5 x 3 / D5 x 3 / C5 x 3 / B4 x 3 / A4 x 3 / G4 x 3 / F4 x 3 / E4 x 3 / D4 x 3 / C4 x 3

### Events

- 1. E6 [intent: separated_notes] :: E6 (#17)
- 2. D6 [intent: separated_notes] :: D6 (#1)
- 3. C6 [intent: separated_notes] :: C6 (#16)
- 4. B5 [intent: separated_notes] :: B5 (#2)
- 5. A5 [intent: separated_notes] :: A5 (#15)
- 6. G5 [intent: separated_notes] :: G5 (#3)
- 7. F5 [intent: separated_notes] :: F5 (#14)
- 8. E5 [intent: separated_notes] :: E5 (#4)
- 9. D5 [intent: separated_notes] :: D5 (#13)
- 10. C5 [intent: separated_notes] :: C5 (#5)
- 11. B4 [intent: separated_notes] :: B4 (#12)
- 12. A4 [intent: separated_notes] :: A4 (#6)
- 13. G4 [intent: separated_notes] :: G4 (#11)
- 14. F4 [intent: separated_notes] :: F4 (#7)
- 15. E4 [intent: separated_notes] :: E4 (#10)
- 16. D4 [intent: separated_notes] :: D4 (#8)
- 17. C4 [intent: separated_notes] :: C4 (#9)
- 18. E6 [intent: separated_notes] :: E6 (#17)
- 19. D6 [intent: separated_notes] :: D6 (#1)
- 20. C6 [intent: separated_notes] :: C6 (#16)
- 21. B5 [intent: separated_notes] :: B5 (#2)
- 22. A5 [intent: separated_notes] :: A5 (#15)
- 23. G5 [intent: separated_notes] :: G5 (#3)
- 24. F5 [intent: separated_notes] :: F5 (#14)
- 25. E5 [intent: separated_notes] :: E5 (#4)
- 26. D5 [intent: separated_notes] :: D5 (#13)
- 27. C5 [intent: separated_notes] :: C5 (#5)
- 28. B4 [intent: separated_notes] :: B4 (#12)
- 29. A4 [intent: separated_notes] :: A4 (#6)
- 30. G4 [intent: separated_notes] :: G4 (#11)
- 31. F4 [intent: separated_notes] :: F4 (#7)
- 32. E4 [intent: separated_notes] :: E4 (#10)
- 33. D4 [intent: separated_notes] :: D4 (#8)
- 34. C4 [intent: separated_notes] :: C4 (#9)
- 35. E6 [intent: separated_notes] :: E6 (#17)
- 36. D6 [intent: separated_notes] :: D6 (#1)
- 37. C6 [intent: separated_notes] :: C6 (#16)
- 38. B5 [intent: separated_notes] :: B5 (#2)
- 39. A5 [intent: separated_notes] :: A5 (#15)
- 40. G5 [intent: separated_notes] :: G5 (#3)
- 41. F5 [intent: separated_notes] :: F5 (#14)
- 42. E5 [intent: separated_notes] :: E5 (#4)
- 43. D5 [intent: separated_notes] :: D5 (#13)
- 44. C5 [intent: separated_notes] :: C5 (#5)
- 45. B4 [intent: separated_notes] :: B4 (#12)
- 46. A4 [intent: separated_notes] :: A4 (#6)
- 47. G4 [intent: separated_notes] :: G4 (#11)
- 48. F4 [intent: separated_notes] :: F4 (#7)
- 49. E4 [intent: separated_notes] :: E4 (#10)
- 50. D4 [intent: separated_notes] :: D4 (#8)
- 51. C4 [intent: separated_notes] :: C4 (#9)

## Review

- summary: expected どおりの 51 event を安定して回収しています。
- reason: practical な下降 separated-notes run として completed 化。third-cycle の bridge overlap と sparse-gap-tail の C4 欠落は recognizer 側で回収済み。

## Recapture Guidance

- なし。現時点では再録音不要。

## Memo

(empty)
