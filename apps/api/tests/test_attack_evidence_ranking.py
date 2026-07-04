"""analyze_attack_evidence_at_onset の mechanism テスト (#178 S3)。

残響下の弱打 (masked note): 窓内スペクトルの最大は鳴り続ける masker なので
score 順では埋もれるが、fresh-attack gain 順なら弱打側が top に来ることを
構築入力 (合成音) で pin する。
"""

from __future__ import annotations

import numpy as np

from app.transcription.audio import build_custom_tuning
from app.transcription.peaks import (
    analyze_attack_evidence_at_onset,
    analyze_spectrum_at_onset,
)

SR = 48000


def _masked_attack_audio() -> np.ndarray:
    """0s から B4 が強く鳴り続け、0.5s に弱い C5 の fresh attack が乗る。"""
    t = np.arange(int(SR * 1.2)) / SR
    b4 = 0.5 * np.exp(-0.8 * t) * np.sin(2 * np.pi * 493.883 * t)
    c5 = np.zeros_like(t)
    onset = int(SR * 0.5)
    tc = t[: len(t) - onset]
    c5[onset:] = 0.25 * np.exp(-2.0 * tc) * np.sin(2 * np.pi * 523.251 * tc)
    return (b4 + c5).astype(np.float32)


def test_attack_gain_ranking_surfaces_masked_note():
    tuning = build_custom_tuning("test", ["B4", "C5", "G4", "E5"])
    audio = _masked_attack_audio()

    by_score = analyze_spectrum_at_onset(audio, SR, 0.5, tuning)
    by_gain = analyze_attack_evidence_at_onset(audio, SR, 0.5, tuning)

    assert by_score, "score 順分析が空"
    assert by_gain, "gain 順分析が空"
    # score 順: 鳴り続ける B4 が top (masked note は埋もれる)
    assert by_score[0].note_name == "B4"
    # gain 順: fresh attack を持つ C5 が top に来る
    assert by_gain[0].note_name == "C5"
    assert (by_gain[0].onset_gain or 0) > 5.0


def test_attack_gain_ranking_returns_empty_for_silence():
    tuning = build_custom_tuning("test", ["B4", "C5"])
    silent = np.zeros(SR, dtype=np.float32)
    assert analyze_attack_evidence_at_onset(silent, SR, 0.3, tuning) == []
