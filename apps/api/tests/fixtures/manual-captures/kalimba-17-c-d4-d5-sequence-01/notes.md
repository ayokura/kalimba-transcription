# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-21-D4-D5-kalimba-17-c
- expected note: D4 x 3 / D5 x 3
- captured at: 2026-03-21T15:00:53.853Z

## Expected Performance

- summary: D4 x 3 / D5 x 3

### Events

- 1. D4 :: D4 (#8)
- 2. D5 :: D5 (#13)
- 3. D4 :: D4 (#8)
- 4. D5 :: D5 (#13)
- 5. D4 :: D4 (#8)
- 6. D5 :: D5 (#13)

## Audio Quality

- 0s-3s contains chair noise (not kalimba)
- actual kalimba performance begins at approximately 3.49s
- the chair noise attack cluster around 1.81s-1.84s was previously promoted as a false gap event; current upstream gap-onset rejection now removes it

## Memo

- remains pending: the early chair-noise path is now rejected upstream and the merged output is the correct 6 events
- remaining issue: raw G5+B5 misdetection at 11.82s (actually D4) is still dropped_after_raw by normalization
- next resolution target: correct peak detection at the 11.82s active range boundary
