"""Mechanism tests: Rust onset DSP (kalimba_dsp) == numpy reference (segments.py).

The Rust ports in `crates/kalimba-dsp/src/onset.rs` are the browser-pipeline
front end (WebAudio Float32Array -> wasm). They are NOT delegated from the
server recognizer — `segments.py`'s numpy implementations stay the source of
truth for the frame-index-sensitive onset path. These tests pin the Rust port to
that reference so the two never drift; the wasm equivalence harness
(`crates/kalimba-dsp/check_wasm.sh`) extends the same guarantee to the .wasm
build.

Constructed inputs only (Test Architecture rule 2): a synthetic multi-onset
signal exercises the full chain without fixture I/O.
"""

import numpy as np
import pytest

import kalimba_dsp as K
import app.transcription.segments as S
from app.transcription.constants import FRAME_LENGTH, HOP_LENGTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_note(freq: float, sr: int, dur: float = 0.45) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    env = np.exp(-4.5 * t)
    sig = np.sin(2 * np.pi * freq * t)
    for k, w in enumerate((0.7, 0.45, 0.25), start=2):
        sig += w * np.sin(2 * np.pi * freq * k * t)
    return (sig * env).astype(np.float32)


def _multi_onset_audio(sr: int) -> np.ndarray:
    """Five decaying-harmonic notes at known times + faint noise."""
    notes = [(261.63, 0.3), (329.63, 1.1), (392.0, 1.9), (523.25, 2.7), (329.63, 3.4)]
    n = int(sr * 4.2)
    a = np.zeros(n, dtype=np.float32)
    for freq, ts in notes:
        s = _synth_note(freq, sr)
        i = int(ts * sr)
        seg = s[: max(0, min(len(s), n - i))]
        a[i : i + len(seg)] += seg
    a += (0.002 * np.random.default_rng(1).standard_normal(n)).astype(np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


# ---------------------------------------------------------------------------
# mel_filterbank — bit-exact (f64 internal, f32 output, like numpy)
# ---------------------------------------------------------------------------

class TestMelFilterbank:
    @pytest.mark.parametrize("sr,n_fft,n_mels", [
        (44100, 2048, 128),
        (96000, 2048, 128),
        (48000, 1024, 64),
        (44100, 4096, 128),
    ])
    def test_matches_numpy(self, sr, n_fft, n_mels):
        ref = S._mel_filterbank(sr, n_fft, n_mels)
        got = np.asarray(K.mel_filterbank(sr, n_fft, n_mels)).reshape(ref.shape)
        # f64 math rounded to f32 in both -> identical.
        assert np.array_equal(ref, got), f"max|d|={np.abs(ref - got).max():.3e}"


# ---------------------------------------------------------------------------
# rms / onset_strength — f32 numerical equivalence
# ---------------------------------------------------------------------------

class TestRmsOnsetStrength:
    @pytest.mark.parametrize("sr", [44100, 96000])
    def test_rms(self, sr):
        a = _multi_onset_audio(sr)
        ref = S._rms_numpy(a, FRAME_LENGTH, HOP_LENGTH)
        got = np.asarray(K.rms(a, FRAME_LENGTH, HOP_LENGTH))
        assert got.shape == ref.shape
        assert np.allclose(ref, got, rtol=1e-4, atol=1e-5), f"max|d|={np.abs(ref - got).max():.3e}"

    @pytest.mark.parametrize("sr", [44100, 96000])
    def test_onset_strength(self, sr):
        a = _multi_onset_audio(sr)
        ref = S._onset_strength_numpy(a, sr, HOP_LENGTH)
        got = np.asarray(K.onset_strength(a, sr, HOP_LENGTH, FRAME_LENGTH, 128))
        assert got.shape == ref.shape
        # numpy STFT is single precision (complex64); rustfft f32 matches to ~1e-5.
        assert np.allclose(ref, got, rtol=1e-3, atol=1e-3), f"max|d|={np.abs(ref - got).max():.3e}"
        assert int(ref.argmax()) == int(got.argmax())


# ---------------------------------------------------------------------------
# peak_pick / onset_backtrack — integer outputs, exact match
# ---------------------------------------------------------------------------

class TestPeakPickBacktrack:
    def test_peak_pick_constructed(self):
        x = np.array(
            [0.0, 0.1, 0.9, 0.2, 0.05, 0.3, 0.95, 0.4, 0.1, 0.0, 0.8, 0.2],
            dtype=np.float32,
        )
        kw = dict(pre_max=2, post_max=2, pre_avg=3, post_avg=3, delta=0.05, wait=2)
        ref = S._peak_pick_numpy(x, **kw)
        got = np.asarray(
            K.peak_pick(x, kw["pre_max"], kw["post_max"], kw["pre_avg"],
                        kw["post_avg"], kw["delta"], kw["wait"])
        )
        assert np.array_equal(ref, got), f"ref={list(ref)} got={list(got)}"

    def test_onset_backtrack_constructed(self):
        energy = np.array(
            [0.5, 0.2, 0.4, 0.9, 0.3, 0.1, 0.6, 0.95, 0.2, 0.7],
            dtype=np.float32,
        )
        events = np.array([3, 7, 9], dtype=np.intp)
        ref = S._onset_backtrack_numpy(events, energy)
        got = np.asarray(K.onset_backtrack([int(e) for e in events], energy))
        assert np.array_equal(ref, got), f"ref={list(ref)} got={list(got)}"


# ---------------------------------------------------------------------------
# onset_detect — full chain, FRAME-EXACT (the load-bearing test)
# ---------------------------------------------------------------------------

class TestOnsetDetectFrameExact:
    @pytest.mark.parametrize("sr", [44100, 96000])
    @pytest.mark.parametrize("backtrack", [True, False])
    def test_full_chain_frame_exact(self, sr, backtrack):
        a = _multi_onset_audio(sr)
        onset_env = S._onset_strength_numpy(a, sr, HOP_LENGTH)
        ref = S._onset_detect_numpy(onset_env, sr, HOP_LENGTH, backtrack=backtrack)
        got = np.asarray(
            K.onset_detect(onset_env.astype(np.float32), sr, HOP_LENGTH, backtrack)
        )
        assert np.array_equal(ref, got), f"ref={list(ref)} got={list(got)}"

    @pytest.mark.parametrize("sr", [44100, 96000])
    def test_end_to_end_from_audio(self, sr):
        """audio -> Rust onset_strength -> Rust onset_detect vs the full numpy chain."""
        a = _multi_onset_audio(sr)
        ref = S._onset_detect_numpy(
            S._onset_strength_numpy(a, sr, HOP_LENGTH), sr, HOP_LENGTH, backtrack=True
        )
        rust_env = np.asarray(K.onset_strength(a, sr, HOP_LENGTH, FRAME_LENGTH, 128)).astype(np.float32)
        got = np.asarray(K.onset_detect(rust_env, sr, HOP_LENGTH, True))
        assert np.array_equal(ref, got), f"ref={list(ref)} got={list(got)}"
