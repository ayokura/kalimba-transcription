# Metamorphic Alarm v0 (S2 instrumentation lane)

**REPORT-ONLY / NON-BLOCKING.** Not a regression gate, no CI wiring, no baseline file (AGENTS.md guardrail 7, docs/sprint-plan-2026-07c.md S2). Ground-truth-free: compares each recording's own transcription before/after a benign transform.

- Generated: 2026-07-05T06:08:34+00:00
- Recognizer fingerprint: `29eaaf1daa45ca68`
- kalimba_dsp fingerprint: `b5688f4db575441d`
- Tolerance: 0.05s
- WARN threshold: diff > max(2 notes, 5% of baseline note count)
- Recordings: 17ea7626-3c5d-450d-ae74-0116dea6e881, 4e1ae5c6-df9a-4876-917d-b7e47699c8e5, 9ce7df83-33a0-455d-bf86-c9392ce6f777, bbd6797f-da44-4e89-b2bf-21b803f3f129, d7a82772-f77f-4820-9798-00133ae45f4e, ea7edd71-e815-4638-a248-a47fe21e5061, ebecf0c6-7e41-430b-bd60-8111a495185e
- Any WARN this run: **yes**

## Matrix (recording x condition)

| txId | family | level | baseline notes | transformed notes | added | dropped | threshold | WARN |
|---|---|---|---|---|---|---|---|---|
| 17ea7626-3c5d-450d-ae74-0116dea6e881 | identity | control | 45 | 45 | 0 | 0 | 2.25 | - |
| 17ea7626-3c5d-450d-ae74-0116dea6e881 | gain | +6dB | 45 | 45 | 0 | 0 | 2.25 | - |
| 17ea7626-3c5d-450d-ae74-0116dea6e881 | lowpass | 8000hz | 45 | 46 | 1 | 0 | 2.25 | - |
| 4e1ae5c6-df9a-4876-917d-b7e47699c8e5 | identity | control | 5 | 5 | 0 | 0 | 2.00 | - |
| 4e1ae5c6-df9a-4876-917d-b7e47699c8e5 | gain | +6dB | 5 | 5 | 0 | 0 | 2.00 | - |
| 4e1ae5c6-df9a-4876-917d-b7e47699c8e5 | lowpass | 8000hz | 5 | 5 | 0 | 0 | 2.00 | - |
| 9ce7df83-33a0-455d-bf86-c9392ce6f777 | identity | control | 5 | 5 | 0 | 0 | 2.00 | - |
| 9ce7df83-33a0-455d-bf86-c9392ce6f777 | gain | +6dB | 5 | 5 | 0 | 0 | 2.00 | - |
| 9ce7df83-33a0-455d-bf86-c9392ce6f777 | lowpass | 8000hz | 5 | 5 | 1 | 1 | 2.00 | - |
| bbd6797f-da44-4e89-b2bf-21b803f3f129 | identity | control | 20 | 20 | 0 | 0 | 2.00 | - |
| bbd6797f-da44-4e89-b2bf-21b803f3f129 | gain | +6dB | 20 | 20 | 0 | 0 | 2.00 | - |
| bbd6797f-da44-4e89-b2bf-21b803f3f129 | lowpass | 8000hz | 20 | 20 | 0 | 0 | 2.00 | - |
| d7a82772-f77f-4820-9798-00133ae45f4e | identity | control | 41 | 41 | 0 | 0 | 2.05 | - |
| d7a82772-f77f-4820-9798-00133ae45f4e | gain | +6dB | 41 | 41 | 0 | 0 | 2.05 | - |
| d7a82772-f77f-4820-9798-00133ae45f4e | lowpass | 8000hz | 41 | 41 | 0 | 0 | 2.05 | - |
| ea7edd71-e815-4638-a248-a47fe21e5061 | identity | control | 19 | 19 | 0 | 0 | 2.00 | - |
| ea7edd71-e815-4638-a248-a47fe21e5061 | gain | +6dB | 19 | 19 | 0 | 0 | 2.00 | - |
| ea7edd71-e815-4638-a248-a47fe21e5061 | lowpass | 8000hz | 19 | 19 | 1 | 1 | 2.00 | - |
| ebecf0c6-7e41-430b-bd60-8111a495185e | identity | control | 21 | 21 | 0 | 0 | 2.00 | - |
| ebecf0c6-7e41-430b-bd60-8111a495185e | gain | +6dB | 21 | 21 | 0 | 0 | 2.00 | - |
| ebecf0c6-7e41-430b-bd60-8111a495185e | lowpass | 8000hz | 21 | 17 | 1 | 5 | 2.00 | **WARN** |

## WARN detail

### ebecf0c6-7e41-430b-bd60-8111a495185e / lowpass-8000hz

diff=6 > threshold=2.00

Added (present after transform, absent at baseline):
- 8.747s D5
Dropped (present at baseline, missing after transform):
- 2.603s F5
- 3.184s D5
- 3.184s F5
- 8.480s D5
- 8.805s D5

## Notes

- Transform set (v0, see script docstring for full rationale): ``identity/control``
  (determinism canary — re-transcribes byte-identical audio and should never
  WARN), ``gain +6dB`` (headroom), ``lowpass 8000hz`` (mic-distance/muffling
  proxy). All three showed a negligible F1/predicted-count delta on both
  recordings in the 2026-07-03 ``augmentation_robustness.py`` run; reverb and
  additive-noise families are excluded from v0 because their mildest tested
  levels already move F1 on the non-saturated reference recording — using
  them here would conflate "known DSP degradation" with "patch-boundary
  fragility".
- A WARN here is a *lead*, not a verdict: it means a benign transform changed
  the recognizer's own output, which is consistent with (but not proof of)
  AGENTS.md's guardrail-11 trigger 1 ("patch が衝突する"). Follow up with the
  usual audio-diagnose / energy-trace tools on the specific added/dropped
  onsets before concluding a pass conflict.

