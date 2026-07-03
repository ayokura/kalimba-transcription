# DSP Augmentation Robustness Map (S2 bets #1)

**REPORT-ONLY.** Not a regression gate. Augmented recordings do not count toward the overfitting-gate or S5-branch non-saturated-recording n (AGENTS.md guardrail 7, docs/sprint-plan-2026-07b.md S2 bets #1).

- Generated: 2026-07-03T20:40:31+00:00
- Recognizer fingerprint: `0c21e79da21194ae`
- kalimba_dsp fingerprint: `b5688f4db575441d`
- Base seed: `20260704`
- Recordings: 17ea7626-3c5d-450d-ae74-0116dea6e881, bbd6797f-da44-4e89-b2bf-21b803f3f129

## Baseline reproduction check

| txId | note_f1_benchmark.py F1 | augmentation-script (no-op) F1 | match |
|---|---|---|---|
| 17ea7626-3c5d-450d-ae74-0116dea6e881 | 0.8632 | 0.8632 | yes |
| bbd6797f-da44-4e89-b2bf-21b803f3f129 | 1.0000 | 1.0000 | yes |

## 17ea7626-3c5d-450d-ae74-0116dea6e881

| family | level | params | GT | pred | TP | F1 | clipped |
|---|---|---|---|---|---|---|---|
| none | baseline | - | 52 | 43 | 41 | 0.8632 | - |
| gain | -30dB | gainDb=-30.0 | 52 | 2 | 0 | 0.0000 | - |
| gain | -20dB | gainDb=-20.0 | 52 | 2 | 0 | 0.0000 | - |
| gain | -10dB | gainDb=-10.0 | 52 | 43 | 37 | 0.7789 | - |
| gain | 0dB | gainDb=0.0 | 52 | 43 | 41 | 0.8632 | - |
| gain | +6dB | gainDb=6.0 | 52 | 45 | 42 | 0.8660 | - |
| noise_white | snr30 | snrDb=30.0, color=white | 52 | 26 | 25 | 0.6410 | - |
| noise_white | snr20 | snrDb=20.0, color=white | 52 | 23 | 17 | 0.4533 | - |
| noise_white | snr10 | snrDb=10.0, color=white | 52 | 16 | 14 | 0.4118 | - |
| noise_white | snr5 | snrDb=5.0, color=white | 52 | 19 | 16 | 0.4507 | - |
| noise_white | snr0 | snrDb=0.0, color=white | 52 | 13 | 13 | 0.4000 | - |
| noise_pink | snr30 | snrDb=30.0, color=pink | 52 | 37 | 31 | 0.6966 | - |
| noise_pink | snr20 | snrDb=20.0, color=pink | 52 | 24 | 21 | 0.5526 | - |
| noise_pink | snr10 | snrDb=10.0, color=pink | 52 | 16 | 15 | 0.4412 | - |
| noise_pink | snr5 | snrDb=5.0, color=pink | 52 | 16 | 14 | 0.4118 | - |
| noise_pink | snr0 | snrDb=0.0, color=pink | 52 | 25 | 16 | 0.4156 | - |
| reverb | light | rt60Sec=0.3, wet=0.2 | 52 | 39 | 34 | 0.7473 | - |
| reverb | medium | rt60Sec=0.6, wet=0.3 | 52 | 39 | 34 | 0.7473 | - |
| reverb | heavy | rt60Sec=1.2, wet=0.4 | 52 | 37 | 29 | 0.6517 | - |
| reverb | extreme | rt60Sec=2.0, wet=0.5 | 52 | 33 | 24 | 0.5647 | - |
| lowpass | 8000hz | cutoffHz=8000.0, order=4 | 52 | 44 | 41 | 0.8542 | - |
| lowpass | 4000hz | cutoffHz=4000.0, order=4 | 52 | 47 | 40 | 0.8081 | - |
| lowpass | 2000hz | cutoffHz=2000.0, order=4 | 52 | 47 | 36 | 0.7273 | - |
| lowpass | 1000hz | cutoffHz=1000.0, order=4 | 52 | 24 | 21 | 0.5526 | - |

## bbd6797f-da44-4e89-b2bf-21b803f3f129

| family | level | params | GT | pred | TP | F1 | clipped |
|---|---|---|---|---|---|---|---|
| none | baseline | - | 20 | 20 | 20 | 1.0000 | - |
| gain | -30dB | gainDb=-30.0 | 20 | 4 | 0 | 0.0000 | - |
| gain | -20dB | gainDb=-20.0 | 20 | 18 | 15 | 0.7895 | - |
| gain | -10dB | gainDb=-10.0 | 20 | 20 | 16 | 0.8000 | - |
| gain | 0dB | gainDb=0.0 | 20 | 20 | 20 | 1.0000 | - |
| gain | +6dB | gainDb=6.0 | 20 | 20 | 20 | 1.0000 | 0.0% |
| noise_white | snr30 | snrDb=30.0, color=white | 20 | 20 | 16 | 0.8000 | - |
| noise_white | snr20 | snrDb=20.0, color=white | 20 | 21 | 13 | 0.6341 | - |
| noise_white | snr10 | snrDb=10.0, color=white | 20 | 21 | 9 | 0.4390 | - |
| noise_white | snr5 | snrDb=5.0, color=white | 20 | 19 | 11 | 0.5641 | - |
| noise_white | snr0 | snrDb=0.0, color=white | 20 | 15 | 0 | 0.0000 | - |
| noise_pink | snr30 | snrDb=30.0, color=pink | 20 | 20 | 20 | 1.0000 | - |
| noise_pink | snr20 | snrDb=20.0, color=pink | 20 | 20 | 8 | 0.4000 | - |
| noise_pink | snr10 | snrDb=10.0, color=pink | 20 | 21 | 9 | 0.4390 | - |
| noise_pink | snr5 | snrDb=5.0, color=pink | 20 | 18 | 5 | 0.2632 | - |
| noise_pink | snr0 | snrDb=0.0, color=pink | 20 | 20 | 0 | 0.0000 | - |
| reverb | light | rt60Sec=0.3, wet=0.2 | 20 | 20 | 20 | 1.0000 | - |
| reverb | medium | rt60Sec=0.6, wet=0.3 | 20 | 20 | 20 | 1.0000 | - |
| reverb | heavy | rt60Sec=1.2, wet=0.4 | 20 | 20 | 16 | 0.8000 | - |
| reverb | extreme | rt60Sec=2.0, wet=0.5 | 20 | 18 | 16 | 0.8421 | - |
| lowpass | 8000hz | cutoffHz=8000.0, order=4 | 20 | 20 | 20 | 1.0000 | - |
| lowpass | 4000hz | cutoffHz=4000.0, order=4 | 20 | 20 | 16 | 0.8000 | - |
| lowpass | 2000hz | cutoffHz=2000.0, order=4 | 20 | 19 | 15 | 0.7692 | - |
| lowpass | 1000hz | cutoffHz=1000.0, order=4 | 20 | 17 | 13 | 0.7027 | - |

## Mean F1 by family/level (across recordings)

| family | level | mean F1 | recordings |
|---|---|---|---|
| none | baseline | 0.9316 | 2 |
| gain | -30dB | 0.0000 | 2 |
| gain | -20dB | 0.3947 | 2 |
| gain | -10dB | 0.7895 | 2 |
| gain | 0dB | 0.9316 | 2 |
| gain | +6dB | 0.9330 | 2 |
| noise_white | snr30 | 0.7205 | 2 |
| noise_white | snr20 | 0.5437 | 2 |
| noise_white | snr10 | 0.4254 | 2 |
| noise_white | snr5 | 0.5074 | 2 |
| noise_white | snr0 | 0.2000 | 2 |
| noise_pink | snr30 | 0.8483 | 2 |
| noise_pink | snr20 | 0.4763 | 2 |
| noise_pink | snr10 | 0.4401 | 2 |
| noise_pink | snr5 | 0.3375 | 2 |
| noise_pink | snr0 | 0.2078 | 2 |
| reverb | light | 0.8736 | 2 |
| reverb | medium | 0.8736 | 2 |
| reverb | heavy | 0.7258 | 2 |
| reverb | extreme | 0.7034 | 2 |
| lowpass | 8000hz | 0.9271 | 2 |
| lowpass | 4000hz | 0.8040 | 2 |
| lowpass | 2000hz | 0.7483 | 2 |
| lowpass | 1000hz | 0.6277 | 2 |

## Interpretation: predicted recognizer weak points (2026-07-04 baseline run)

No condition thinning was needed: the full grid (24 conditions x 2 recordings
= 48 recognizer calls) completed in well under a minute, far inside the
30-minute budget in the task brief.

1. **Gain scaling is a cliff, not a slope.** Mean F1 is essentially flat
   across -10dB/0dB/+6dB (~0.79-0.93) then collapses to ~0.39 at -20dB and to
   0.00 at -30dB on both recordings. This is not the pipeline's silence guard
   (`read_audio` rejects only when peak amplitude < 1e-4; at -30dB the
   quieter recording's peak is still ~30x that floor). Both recordings also
   surface the pipeline's own `"Only a small number of note events were
   detected."` warning exactly at the two collapsing levels. Because most
   calibrated onset constants in `constants.py` are expressed as *ratios*
   (`ONSET_GATE_MIN_ONSET_GAIN`, backward-attack-gain thresholds, etc.), a
   clean ratio-based design should degrade smoothly with gain; the observed
   cliff instead points to some absolute-magnitude-dependent stage in the
   broadband onset/spectral-flux pipeline. This corroborates the existing
   project note `feedback_gain_vs_attack_profile.md` ("gain絶対量はマイク距離
   変化に弱い") — quiet or mic-distant captures are a predicted practical
   failure mode, not just louder/quieter versions of the same transcription.

2. **Additive noise degrades fast even at nominal high SNR.** Mean F1 drops
   from ~0.93 (clean) to ~0.72-0.85 already at SNR 30dB (white/pink) and
   continues to ~0.20-0.30 by SNR 0dB, close to monotonic for both colors.
   SNR here is defined against whole-recording RMS; a supplementary check
   confirmed the RMS in +/-100ms windows around actual onsets is within ~2dB
   of the whole-file RMS for both recordings, so this is not a
   silence-diluted-SNR artifact — the effective SNR near real note attacks is
   close to nominal. Predicted-note *counts* drop under noise rather than
   balloon (e.g. 17ea7626: 43 predicted at baseline vs 13-26 under noise),
   meaning noise mostly suppresses true attacks rather than only adding false
   positives. This predicts broadband spectral-flux onset detection is more
   noise-floor-sensitive than gain-sensitive: room hiss, HVAC noise, or
   electrical hum during real capture is a meaningfully larger practical risk
   than the current (comparatively controlled) fixture/corpus recordings
   would suggest.

3. **Reverb is the gentlest-sloped family — a likely surface/kill-condition
   mismatch.** Mean F1 barely moves at "light"/"medium" RT60 (0.87, at or
   above the additive-noise floor) and only reaches ~0.70 at "extreme"
   (RT60=2.0s, wet=0.5 — an unrealistically large hall for solo kalimba).
   This is weaker than the task brief's working hypothesis ("reverb wet 高で
   F1 が崩れる → carryover 判別が弱い"). The likely reason is the IR family
   used here (exponential-decay *white noise*) has no coherent per-note
   spectral structure, unlike real kalimba sympathetic-tine carryover/decay,
   which retains the ringing note's own partials. **This is the sharpest kill
   condition to watch**: if a real adversarial "carryover" recording (Mech2,
   per the S1/S2 敵対的セルフ録音 menu) shows large F1 collapse that this
   synthetic reverb sweep did not predict, that is direct evidence this
   augmentation family should be downgraded to invariance-testing use only
   (per the task brief's explicit kill condition) rather than used to predict
   real reverberant-carryover weaknesses.

4. **Low-pass filtering degrades close to linearly with cutoff** — the most
   "expected", least surprising family. Mean F1 ~0.93 at an 8kHz cutoff
   (near baseline; fundamentals and first partials of essentially all kalimba
   notes survive) down to ~0.63 at 1kHz (cuts into the fundamental/first
   partial region for mid/high notes). Predicted-note counts stay closer to
   truth-note counts than in the noise family, suggesting this failure mode
   is more about note-identity substitution (per-tine partial matching /
   narrow-FFT confusion) than dropped onsets — a natural follow-up would be a
   false-positive/false-negative breakdown, out of scope for this
   report-only pass.

**Overall predicted risk ranking** (steepest to gentlest F1 collapse in this
surface): gain (cliff at quiet levels) > additive noise > low-pass/muffling >
reverb. This ranking is a testable prediction against the S1/S2 adversarial
recording menu once those takes have human-reviewed ground truth: if the
ranking or the reverb/carryover mismatch above does not hold on real
recordings, downgrade this bet to an invariance-testing tool only, per the
task brief's kill condition (docs/sprint-plan-2026-07b.md 頑健性マップの
「予測される弱点機構」が実新録音の弱点と一致しなければ不変性テスト用途に
格下げ).

Transforms were applied as independent single-family sweeps (no pairwise/grid
combinations) to keep the surface interpretable and the runtime small;
combined conditions (e.g. noise + reverb) are a natural extension if this bet
graduates past the kill condition above.

