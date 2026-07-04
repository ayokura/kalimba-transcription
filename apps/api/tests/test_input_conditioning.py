"""入力条件付け (read_audio) の mechanism テスト。

優勢チャンネル選択と peak 正規化 (-6dBFS 目標)。テスター録音 (iPhone 内蔵
マイク、片チャンネル無音 + peak -26〜-33dBFS) で認識が崩壊した実測
(a9e30986 ほか、2026-07-05) を受けた入力レベル不変化。
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from app.transcription.audio import NORMALIZE_TARGET_PEAK, read_audio


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """samples: float32 (-1..1), shape (n,) or (n, channels)."""
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _upload(data: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename="audio.wav")


def _tone(peak: float, n: int = 1600) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    return (peak * np.sin(2 * np.pi * 440 * t / 16000)).astype(np.float32)


@pytest.mark.asyncio
async def test_quiet_audio_is_normalized_to_target_peak():
    audio, sr, conditioning = await read_audio(_upload(_wav_bytes(_tone(0.02))))
    assert sr == 16000
    assert float(np.max(np.abs(audio))) == pytest.approx(NORMALIZE_TARGET_PEAK, rel=1e-3)
    # 0.02 peak = -34dBFS → 約 +28dB の正規化 gain が provenance に残る
    assert conditioning["inputPeakDbfs"] == pytest.approx(-34.0, abs=0.5)
    assert conditioning["normalizationGainDb"] == pytest.approx(28.0, abs=0.5)


@pytest.mark.asyncio
async def test_loud_audio_is_left_untouched():
    # 増幅のみ: 減衰まで行う完全正規化は 0〜-3dBFS の既存 fixture を回帰させた
    audio, _, conditioning = await read_audio(_upload(_wav_bytes(_tone(0.95))))
    assert float(np.max(np.abs(audio))) == pytest.approx(0.95, rel=1e-2)
    assert conditioning["normalizationGainDb"] == 0.0


@pytest.mark.asyncio
async def test_dominant_channel_is_selected_for_right_only_stereo():
    left = np.zeros(1600, dtype=np.float32)
    right = _tone(0.3)
    stereo = np.stack([left, right], axis=1)
    audio, _, conditioning = await read_audio(_upload(_wav_bytes(stereo)))
    assert conditioning["selectedChannel"] == 1
    # 従来の「常に ch0」だと無音 422 だったケースが正規化まで到達する
    assert float(np.max(np.abs(audio))) == pytest.approx(NORMALIZE_TARGET_PEAK, rel=1e-3)
    assert conditioning["channelPeaksDbfs"][0] < -75
    assert conditioning["channelPeaksDbfs"][1] == pytest.approx(-10.5, abs=1.0)


@pytest.mark.asyncio
async def test_balanced_stereo_keeps_channel_zero():
    # 両チャンネルが生きた通常ステレオでは従来どおり ch0 (G-low fixture の教訓:
    # L/R 差 0.04dB で ch1 に切り替えると fixture 回帰する)
    left = _tone(0.5)
    right = _tone(0.52)
    stereo = np.stack([left, right], axis=1)
    _, _, conditioning = await read_audio(_upload(_wav_bytes(stereo)))
    assert conditioning["selectedChannel"] == 0


@pytest.mark.asyncio
async def test_silent_audio_still_rejected_not_normalized():
    silent = np.full(1600, 5e-5, dtype=np.float32)
    with pytest.raises(HTTPException) as exc:
        await read_audio(_upload(_wav_bytes(silent)))
    assert exc.value.status_code == 422
