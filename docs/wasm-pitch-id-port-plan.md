# Overnight Goal Spec — Browser-side Pitch Identification (chunk_spectrum + rank_tuning_candidates -> Rust/WASM)

> This is the working spec the overnight `/goal` follows. Nailed down from a 5-agent code
> investigation (2026-06-15). Do NOT re-derive these decisions — they are resolved. Verify exact
> line numbers against the cited files as you go, but the algorithm/constants/ABI below are
> authoritative. If reality contradicts this doc, STOP and report — do not improvise.

## 0. Branch & commit discipline (do this first)

- Create and switch to branch `claude/wasm-pitch-id` (`git checkout -b claude/wasm-pitch-id`).
- Commit `docs/wasm-pitch-id-port-plan.md` as the first commit on that branch.
- Commit after EACH numbered task with a clear message (reviewable granularity).
- Do NOT push. Leave everything local for morning review. Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- If a gate fails and cannot be fixed correctly, STOP and write a blocker report.
  NEVER loosen a tolerance, skip a parity case, or revert an import to make a gate go green.

## 1. Scope (what "done" means)

Deliver browser-side pitch identification end-to-end on `claude/wasm-pitch-id`:
1. f64 `chunk_spectrum` in Rust (pyo3 + wasm bindings).
2. `rank_tuning_candidates` integer-comb branch in Rust (pyo3 + wasm bindings).
3. Native parity (pytest: rust == numpy) AND wasm parity (`check_wasm.sh`: wasm == native).
4. Re-vendored wasm bundle into `apps/web/src/lib/wasm/`.
5. `pitch.ts` wrapper + `/wasm-demo` page rendering a note name per onset, lint/build green.

Completion is the OBJECTIVE gates in section 8, NOT the visual correctness of the notes.
Visual/listening correctness of displayed notes is a HUMAN morning check (no automated oracle
exists for the demo path). If notes look wrong, it is a window/n_fft knob (section 7), not a port bug.

## 2. Hard invariants (the silent-failure traps — non-negotiable)

1. f64 FFT, not f32. `chunk_spectrum` MUST use `rustfft::FftPlanner<f64>` + `Complex<f64>`. Do NOT
   reuse the existing `Complex32`/`FftPlanner<f32>` path (lib.rs note_band_energy ~46/56/147, onset.rs
   ~171). rustfft 6.4 is generic over `FftNum` -> f64 needs ZERO new deps. Use a SEPARATE thread-local
   `FFT_PLANNER_F64`. Downstream peak_energy_near/batch_peak_energies are f64 and the numpy reference
   is f64; f32 silently drifts and flips borderline notes.
2. Symmetric Hann window. Apply `np.hanning(len(chunk))` = `0.5 - 0.5*cos(2*pi*i/(n-1))` (same as
   `lib.rs::cached_hanning`), applied in f64, at the chunk's own length, then zero-pad to `n_fft`. Do
   NOT use onset.rs periodic Hann (`2*pi*i/n_fft`, that is for librosa STFT parity).
3. Tolerance is FIXED. Native parity for the spectrum: `np.allclose(rtol=1e-9, atol=1e-12)`.
   Frequencies (rfftfreq): `np.array_equal` (exact). rank scores: `np.allclose(rtol=1e-9, atol=1e-12)`.
   These values are locked. rustfft != numpy pocketfft only at the last bit (~1e-13..1e-12), so 1e-9 is
   safe AND tight enough to catch an f32 mistake (which diverges at ~1e-6). Do not change them.
4. Integer-comb branch only. Port the `use_per_tine_partial_scoring == False` path (peaks.py ~226-327),
   the production default (settings.py:65). Skip the partials branch (~190-225). It is self-contained:
   depends only on note frequencies, batch_peak_energies (already Rust f64), the 8 constants in
   section 4, the inline literals, and the <40 Hz sub-harmonic floor.
5. No recognizer delegation. Add pyo3 bindings for the parity tests, but do NOT change profiles.py /
   peaks.py to call the Rust versions. The Python recognizer path stays on numpy, so the 471-test
   suite and F1=1.000 are trivially protected. The pyo3 binding exists purely so pytest can
   `import kalimba_dsp; kalimba_dsp.chunk_spectrum(...)`.
6. Two parity layers, both required. pytest proves rust == numpy; check_wasm.sh proves wasm == native.
   A green `wasm equivalence: NN/NN` alone does NOT prove correctness — add BOTH a pytest parity test
   AND a `wasm_reference.py` case for each new function.
7. Re-vendor is a manual 4-file copy. After `wasm-pack build --target web`, copy ONLY
   `kalimba_dsp.js`, `kalimba_dsp.d.ts`, `kalimba_dsp_bg.wasm`, `kalimba_dsp_bg.wasm.d.ts` from `pkg/`
   into `apps/web/src/lib/wasm/`. NEVER `rm -rf` that dir or clobber the hand-written `onset.ts`.
   `crates/kalimba-dsp/pkg/` is STALE — always rebuild before copying. Treat
   `apps/web/src/lib/wasm/*.d.ts` as the source of truth for what is exported.
8. Stable sort. rank output is `sorted(..., key=score, reverse=True)` with Python's STABLE sort (ties
   keep tuning order). Use a stable sort in Rust with original-index tiebreak.
9. n_fft is an explicit argument to `chunk_spectrum` (the caller computes it via the already-exported
   `adaptive_n_fft`). Do not hardcode an n_fft rule inside `chunk_spectrum`.

## 3. Algorithm: chunk_spectrum (ref: profiles.py:323-327, audio.py:18-29)

```
chunk_spectrum(chunk: f32[], sample_rate: i64, n_fft: usize) -> (freqs: f64[], magnitudes: f64[])
  window   = hanning_f64(len(chunk))      # 0.5-0.5*cos(2pi*i/(n-1)); n==1 -> [1.0]; n==0 -> []
  buf      = [ (chunk[i] as f64) * window[i] for i in 0..len(chunk) ]   # then zero-pad to n_fft
  spectrum = | rfft_f64(buf, n_fft) |      # magnitude sqrt(re^2+im^2), bins 0..=n_fft/2
  freqs    = [ k * sample_rate / n_fft for k in 0..=n_fft/2 ]           # == np.fft.rfftfreq
  return (freqs, spectrum)                 # both f64, length n_fft/2+1
```
Magnitude is `abs(rfft)`, NOT power. No normalization. Implement via full `FftPlanner<f64>` FFT over an
`n_fft`-length `Complex<f64>` buffer, take the first `n_fft/2+1` magnitudes (mirror note_band_energy_inner
structure but in f64).

## 4. Algorithm: rank_tuning_candidates integer-comb (ref: peaks.py:228-309)

Inputs: `frequencies: f64[]`, `spectrum: f64[]`, `note_freqs: f64[]`, `band_cents = 40.0`.

```
n = len(note_freqs)
harmonic_targets = concat([ note_freqs * m  for m in 1..=4 ])        # len 4n
sub_half  = [ f/2 if f/2 >= 40.0 else 0.0  for f in note_freqs ]     # len n
sub_third = [ f/3 if f/3 >= 40.0 else 0.0  for f in note_freqs ]     # len n
all_targets  = concat([harmonic_targets, sub_half, sub_third])       # len 6n
all_energies = batch_peak_energies(frequencies, spectrum, all_targets, band_cents=40.0)  # Rust f64
H  = all_energies[0 .. 4n]    # row-major (4, n): H[h][i] = all_energies[h*n + i]
SH = all_energies[4n .. 5n]   # sub_half energy per note
ST = all_energies[5n .. 6n]   # sub_third energy per note

for i in 0..n:
  e = [H[0][i], H[1][i], H[2][i], H[3][i]]              # harmonic energies m=1..4
  w = HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]
  fundamental_energy = e[0]
  overtone_energy    = w[1]*e[1] + w[2]*e[2] + w[3]*e[3]
  harmonic_support   = fundamental_energy + overtone_energy
  fundamental_ratio  = fundamental_energy / max(harmonic_support, 1e-9)
  subharmonic_alias_energy = 0.7*SH[i] + 0.45*ST[i]
  octave_alias_energy      = SH[i]
  octave_alias_ratio       = octave_alias_energy / max(fundamental_energy, 1e-9)
  octave_alias_penalty = 0.0
  if octave_alias_ratio >= 1.15 and fundamental_ratio <= 0.34:
       octave_alias_penalty = octave_alias_energy * 0.85
  score = harmonic_support*(0.2 + 0.8*fundamental_ratio)
          + 0.45*fundamental_energy
          - 0.6*subharmonic_alias_energy
          - octave_alias_penalty
  if fundamental_ratio < 0.18:
       score -= 0.0 * overtone_energy        # no-op now (OVERTONE_DOMINANT_PENALTY_WEIGHT=0); keep
  scores[i] = score

return scores[]      # per note, in note order; JS sorts/argmax and maps to note name
```
Constants (verify against constants.py:449,450,556,566-570): HARMONIC_WEIGHTS=[1.0,0.55,0.3,0.15],
HARMONIC_BAND_CENTS=40.0, MAX_HARMONIC_MULTIPLE=4, OVERTONE_DOMINANT_FUNDAMENTAL_RATIO=0.18,
OVERTONE_DOMINANT_PENALTY_WEIGHT=0.0, OCTAVE_ALIAS_RATIO_THRESHOLD=1.15,
OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO=0.34, OCTAVE_ALIAS_PENALTY=0.85.
The pytest parity test MUST import these from constants.py, not re-hardcode, so future drift is caught.

## 5. ABI / marshalling (no serde dependency)

- chunk_spectrum: wasm `(audio: &[f32], sample_rate: i64, n_fft: usize) -> Vec<f64>` returning the
  magnitude spectrum (Float64Array). Frequencies are a deterministic ramp — compute JS-side OR add a
  trivial `rfftfreq_f64(n_fft, sample_rate) -> Vec<f64>` helper. pyo3 wrapper returns PyArray1<f64>.
- rank_tuning_candidates: wasm `(frequencies: &[f64], spectrum: &[f64], note_freqs: &[f64],
  band_cents: f64) -> Vec<f64>` returning per-note scores in input note order. Note names live in JS
  (recovered from the tuning the caller passed). Do NOT return strings or structs from Rust.
  Optionally also return a `Uint32Array` of the stable-sorted descending index order.
- Register every new pyo3 fn in the `#[pymodule] fn kalimba_dsp` add_function list (lib.rs ~800-815).
  wasm exports auto-collect via `#[wasm_bindgen]`.
- Module layout: put chunk_spectrum inline in lib.rs (small). Before landing rank_tuning (~150 lines),
  create `crates/kalimba-dsp/src/pitch.rs` (mirror the onset.rs split) so lib.rs stays < ~1100 lines.

## 6. Browser tuning data (kalimba-17-c, frequencies only — integer-comb ignores partials)

Generate `apps/web/src/lib/wasm/tuning-17c.ts` from the server tuning. Read `apps/api/app/tunings.py`
for the exact build_tuning API, then dump kalimba-17-c as `{ key:number, noteName:string,
frequency:number }[]` (partials NOT needed for the integer-comb path). A tuning selector is out of
scope (interactive follow-up); hardcode 17-c for the demo.

## 7. Demo per-onset window / n_fft policy (refinable — not the gate)

For each detected onset i (from existing detectOnsetsInBrowser):
- chunk = samples[onset_i_sample : min(onset_{i+1}_sample, onset_i_sample + round(0.25*sr))]
- n_fft = adaptive_n_fft(sr, min(note_freqs), len(chunk))   (already a wasm export)
- scores = rank_tuning_candidates(freqs, chunk_spectrum(chunk, sr, n_fft), note_freqs, 40.0)
- display note = tuning[argmax(scores)].noteName
Reasonable demo heuristic; the exact recognizer windowing (peaks.py ~1693-1809) is a morning
refinement if notes look off. Top-1 per onset only (no polyphony).

## 8. Objective gates (the goal's completion conditions — all must hold)

Baselines (verified live): pytest 471 passed; check_wasm.sh 51/51; F1 1.000.

1. `cargo check` (default/python) AND `cargo check --no-default-features --features wasm` exit 0 (from
   crates/kalimba-dsp).
2. New `apps/api/tests/test_chunk_spectrum_rust.py`: kalimba_dsp.chunk_spectrum vs
   profiles._chunk_spectrum -> spectrum np.allclose(rtol=1e-9, atol=1e-12), freqs array_equal. Plus a
   rank parity test: rust scores vs Python rank_tuning_candidates (default settings) np.allclose
   (rtol=1e-9, atol=1e-12), same argmax/order.
3. `TMPDIR=/tmp uv run pytest apps/api/tests -q` -> >=471 passed, 0 failed.
4. `bash crates/kalimba-dsp/check_wasm.sh` -> `wasm equivalence: NN/NN cases passed`, NN > 51,
   pass==total, exit 0 (new chunk_spectrum AND rank cases added to tools/wasm_reference.py).
5. `grep -c chunk_spectrum apps/web/src/lib/wasm/kalimba_dsp.d.ts` >= 1 and the rank export present;
   `git status` shows only the 4 generated files changed in that dir, onset.ts untouched.
6. `npm --prefix apps/web run lint` (tsc --noEmit) exits 0; pitch.ts wrapper added; /wasm-demo
   page.tsx renders a note name per onset (compiles).
7. `uv run python scripts/audio-analysis/note_f1_benchmark.py` -> F1 = 1.000 (additive change).
8. All work committed to `claude/wasm-pitch-id`, nothing pushed.

## 9. If blocked

Write a short blocker note (what failed, the exact command output, your hypothesis) and STOP. Do not:
loosen tolerances, delete parity cases, skip the re-vendor, port the partials branch, or push. A
partial-but-correct landing (e.g. chunk_spectrum done, rank blocked) reported clearly is better than a
green-but-wrong full run.
