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

- 0s-3s contains chair noise (not kalimba), which the recognizer detects as D6/C4/E4
- actual kalimba performance begins at approximately 3.49s
- the chair noise has strong attack profiles (gain=28.5, flux=3.1) indistinguishable from kalimba by current thresholds

## Memo

- downgraded to pending: fixture passes through pattern normalization removing chair noise events and a G5+B5 misdetection at 11.82s (actually D4), coincidentally producing the correct 6 events
- resolution requires: upstream non-instrument sound rejection, and correct peak detection at 11.82s active range boundary
