# App Synth Audio Gap Analysis

## Scope

Local reference source:
- Video: `C:\src\calimba-score\.codex-media\source-videos\ScreenRecording_03-23-2026_13-09-56_1.mov`
- Derived audio: `C:\src\calimba-score\.codex-media\derived-audio\kira-kira-reference.wav`
- Derived expected-performance analysis: `C:\src\calimba-score\.codex-media\derived-analysis\kira-kira-expected-performance.json`

These assets are local-only and are not tracked in git. The purpose of this note is to compare the current acoustic recognizer against the app/synth reference and decide whether any recognizer changes are justified.

## Expected Structure From Video Analysis

The video-derived analysis projects 39 expected events from approximately `2.66s` to `22.83s`.

Pattern families present in the expected sequence:
- single-note ostinato around `E4`, `G4`, `C5`
- wide-register dyads such as `C5+G4`, `G5+C4`, `F4+F5`
- one broad triad-like sonority `G5+E4+G4`
- a late section that behaves like broken-chord / arpeggio material rather than strict vertical chord attacks

This makes the source useful as a pattern-discovery reference, but not as acoustic ground truth.

## Current Recognizer Output On Derived Audio

Running the current recognizer directly on `kira-kira-reference.wav` with the `17 Key C Major` tuning produces only 2 detected events:
- `A4+C5`
- `D5+G4`

Additional debug observations:
- `segments = 2`
- `activeRanges = 2`
- `segmentCandidates = 2`
- `mergedEvents = 2`
- gesture mix is effectively meaningless because the front-end failed first

This is a catastrophic coverage miss relative to the 39 expected events.

## Root Cause Analysis

The failure is not primarily pitch snapping.

The dominant mismatch is the segmentation front-end:
- normalized RMS median is about `0.126`
- current threshold becomes `max(max_rms * 0.18, median_rms * 2.2, 0.01)`
- for this source that evaluates to about `0.277`
- only the top ~`0.5%` of frames exceed that threshold
- result: only 2 active regions survive

By contrast, onset analysis on the same audio sees many more plausible attacks:
- onset detector finds about `59` onsets across the clip
- onset times line up with the expected event density far better than RMS gating does

Interpretation:
- app/synth audio has a relatively elevated and stable energy floor
- the current `acoustic_real` active-range heuristic assumes stronger contrast between note attacks and background sustain/noise
- that assumption fails for screen-recorded synth audio

## Differences Observed

### Coverage Differences
- expected: `39` events
- detected: `2` events
- missing coverage is broad, not local

### Structural Differences
- expected opening mixes dyads/triads followed by single-note echoes
- recognizer collapses the entire early/middle phrase into nothing
- later pulse-like sections around `E4/G4/C5` are also missed except for 2 isolated peaks

### Timing Differences
- expected activity spans most of the instrument scene (`2.66s` to `22.83s`)
- detected activity appears only around `8.5s` and `11.2s`

### Gesture Differences
- no meaningful strict/slide/arpeggio classification can be inferred because segmentation fails first

## Should The Recognizer Be Improved For This?

### For the current acoustic recognizer
No direct change should be made solely to accommodate this source.

Reason:
- the current recognizer is tuned for `sourceProfile = acoustic_real`
- changing its segmentation thresholds globally would risk regressions on real microphone recordings
- the failure mode is clearly source-profile-specific

### For future `app_synth` support
Yes. The result is strong evidence that `app_synth` should get a separate front-end.

Recommended improvements for a future `app_synth` path:
1. onset-driven or hybrid onset/RMS segmentation instead of RMS-active-range gating alone
2. lower reliance on median-RMS-based thresholds
3. weaker carryover / resonance assumptions than `acoustic_real`
4. separate evaluation expectations for synthetic sustain envelopes
5. optional post-pass for broken-chord / arpeggio-like app playback

## Practical Product Conclusion

This source does expose useful future work, but that work belongs under source-profile separation rather than the main acoustic recognizer.

Short version:
- **Do not patch `acoustic_real` to make this audio pass.**
- **Do use this source as evidence for `app_synth` profile design.**
- **Do treat the late broken-chord section as relevant input for future arpeggio handling.**
