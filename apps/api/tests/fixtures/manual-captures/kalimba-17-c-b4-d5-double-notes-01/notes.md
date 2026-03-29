# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-21-b4-d5-double-notes
- expected note: B4 + D5 x 5
- captured at: 2026-03-21T08:44:21.800Z

## Memo

The saved pack originally labeled this as A5 + D5. During review, the user confirmed the played chord was actually B4 + D5 by physical key position. The regression fixture uses the corrected chord identity.

## Review (2026-03-30)

- 0-3.5s contains male voice noise, not kalimba playing. Original C4+G4 event was voice misidentification.
- Evaluation window set to 3.5-18.0s to exclude noise.
- seg 7 (@5.77s) F5 is sympathetic vibration from D5+B4 chord (F5 key14 adjacent to D5 key13), confirmed false positive.
