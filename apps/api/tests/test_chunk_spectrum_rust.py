"""Mechanism tests: Rust pitch DSP (kalimba_dsp) == numpy reference.

`kalimba_dsp.chunk_spectrum` is the f64 magnitude-spectrum primitive the browser
pitch-ID path uses; it must match the recognizer's numpy `_chunk_spectrum`
(`profiles.py`). `kalimba_dsp.rank_tuning_candidates` ports the integer-harmonic-comb
branch of `peaks.rank_tuning_candidates` (the production default,
`use_per_tine_partial_scoring=False`) and must produce the same per-note scores and
the same top candidate as the Python implementation. Constants come from the live
Python modules (not re-hardcoded) so a future constant change is caught here. The
wasm equivalence harness (`crates/kalimba-dsp/check_wasm.sh`) extends both
guarantees to the .wasm build.
"""

import numpy as np
import pytest

import kalimba_dsp as K
from app.transcription import peaks
from app.transcription.profiles import _chunk_spectrum
from app.transcription.peaks import rank_tuning_candidates, HARMONIC_BAND_CENTS
from app.tunings import get_default_tunings


# rustfft (f64) vs numpy pocketfft (f64) agree to ~1e-13 relative; 1e-9 is a tight
# bound that still rejects an accidental f32 implementation (which diverges ~1e-6).
RTOL = 1e-9
ATOL = 1e-12


def _tuning(tuning_id: str):
    for t in get_default_tunings():
        if t.id == tuning_id:
            return t
    raise KeyError(tuning_id)


class TestChunkSpectrum:
    @pytest.mark.parametrize("sr,n_fft", [(44100, 4096), (48000, 4096), (96000, 8192), (48000, 8192)])
    @pytest.mark.parametrize("chunk_len", [512, 2048, 3528, 4096])
    def test_matches_numpy(self, sr, n_fft, chunk_len):
        rng = np.random.default_rng(sr + n_fft + chunk_len)
        chunk = (0.5 * rng.standard_normal(chunk_len)).astype(np.float32)
        ref_freqs, ref_spec = _chunk_spectrum(chunk, sr, n_fft)
        got = np.asarray(K.chunk_spectrum(chunk, sr, n_fft))
        assert got.shape == ref_spec.shape == (n_fft // 2 + 1,)
        assert np.allclose(got, ref_spec, rtol=RTOL, atol=ATOL), \
            f"sr={sr} n_fft={n_fft} chunk_len={chunk_len} max|d|={np.abs(got - ref_spec).max():.3e}"

    @pytest.mark.parametrize("sr,n_fft", [(44100, 4096), (48000, 8192), (96000, 16384)])
    def test_frequency_ramp_matches_rfftfreq(self, sr, n_fft):
        # chunk_spectrum returns magnitudes only; the browser computes freqs as
        # k*sr/n_fft. Confirm that formula matches numpy's rfftfreq the test path uses.
        js_freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
        ref_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        assert np.allclose(js_freqs, ref_freqs, rtol=0, atol=1e-6)

    def test_tone_peak_is_correct_bin(self):
        sr, n_fft, freq = 48000, 8192, 440.0
        t = np.arange(int(sr * 0.1)) / sr
        chunk = (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        spec = np.asarray(K.chunk_spectrum(chunk, sr, n_fft))
        peak_bin = int(np.argmax(spec))
        peak_freq = peak_bin * sr / n_fft
        assert abs(peak_freq - freq) < sr / n_fft  # within one bin


class TestRankTuningCandidates:
    @staticmethod
    def _structured_spectrum(note_freqs, sr, n_fft, seed):
        rng = np.random.default_rng(seed)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float64)
        spec = np.zeros_like(freqs)
        # put harmonic energy on a few notes so scores are non-trivial / distinct
        for nf in note_freqs[:: max(1, len(note_freqs) // 4)]:
            for m in (1, 2, 3):
                b = int(round(nf * m / (sr / n_fft)))
                if 0 <= b < len(spec):
                    spec[b] += 1.0 / m
        spec += 0.01 * np.abs(rng.standard_normal(len(freqs)))
        return np.ascontiguousarray(freqs), np.ascontiguousarray(spec)

    @pytest.mark.parametrize("tuning_id", ["kalimba-17-c", "kalimba-17-g-low", "kalimba-34l-c"])
    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_scores_match_numpy(self, tuning_id, seed):
        assert peaks.settings.get().use_per_tine_partial_scoring is False
        tuning = _tuning(tuning_id)
        note_freqs = np.array([n.frequency for n in tuning.notes], dtype=np.float64)
        sr, n_fft = 48000, 8192
        freqs, spec = self._structured_spectrum(note_freqs, sr, n_fft, seed)

        py = rank_tuning_candidates(freqs, spec, tuning)
        py_by_key = {h.candidate.key: h.score for h in py}
        py_scores = np.array([py_by_key[n.key] for n in tuning.notes], dtype=np.float64)
        rust_scores = np.asarray(
            K.rank_tuning_candidates(freqs, spec, note_freqs, float(HARMONIC_BAND_CENTS))
        )
        assert rust_scores.shape == py_scores.shape
        assert np.allclose(py_scores, rust_scores, rtol=RTOL, atol=ATOL), \
            f"{tuning_id} seed={seed} max|d|={np.abs(py_scores - rust_scores).max():.3e}"
        # top candidate (argmax) must agree — the load-bearing pitch-ID output
        assert tuning.notes[int(np.argmax(rust_scores))].key == py[0].candidate.key

    def test_flat_spectrum_all_finite(self):
        tuning = _tuning("kalimba-17-c")
        note_freqs = np.array([n.frequency for n in tuning.notes], dtype=np.float64)
        sr, n_fft = 48000, 4096
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float64)
        spec = np.ones_like(freqs)
        rust_scores = np.asarray(
            K.rank_tuning_candidates(freqs, spec, note_freqs, float(HARMONIC_BAND_CENTS))
        )
        py = rank_tuning_candidates(freqs, spec, tuning)
        py_by_key = {h.candidate.key: h.score for h in py}
        py_scores = np.array([py_by_key[n.key] for n in tuning.notes], dtype=np.float64)
        assert np.all(np.isfinite(rust_scores))
        assert np.allclose(py_scores, rust_scores, rtol=RTOL, atol=ATOL)
