from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from ..models import InstrumentTuning
from ..tunings import build_custom_tuning, get_default_tunings
from .models import Note, NoteCandidate


@lru_cache(maxsize=64)
def cached_hanning(length: int) -> np.ndarray:
    window = np.hanning(length)
    window.setflags(write=False)
    return window


@lru_cache(maxsize=64)
def cached_rfftfreq(n_fft: int, sample_rate: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    freqs.setflags(write=False)
    return freqs


def parse_tuning_json(tuning_json: str) -> InstrumentTuning:
    try:
        payload: Any = json.loads(tuning_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid tuning JSON.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Tuning JSON must be an object.")

    notes = payload.get("notes", [])
    if not isinstance(notes, list) or not notes:
        raise HTTPException(status_code=400, detail="Tuning must contain at least one note.")

    note_names: list[Any] = []
    for note in notes:
        if not isinstance(note, dict):
            raise HTTPException(status_code=400, detail="Each tuning note must be an object.")
        if "noteName" not in note:
            raise HTTPException(status_code=400, detail="Each tuning note must include noteName.")
        note_names.append(note["noteName"])

    name = payload.get("name", "Custom Tuning")
    if not isinstance(name, str):
        raise HTTPException(status_code=400, detail="Tuning name must be a string.")

    # If the request's tuning id matches a known default tuning and the note
    # names also match, return the server-side default tuning directly so that
    # per-tine partial configurations (see `apps/api/app/tunings.py`) are
    # applied.  Requests built from the web client typically send standard
    # note sets with the matching id, so this path is the common case.
    tuning_id = payload.get("id")
    if (
        isinstance(tuning_id, str)
        and tuning_id
        and all(isinstance(n, str) for n in note_names)
    ):
        for default in get_default_tunings():
            if default.id != tuning_id:
                continue
            default_names = [n.note_name for n in default.notes]
            if default_names == note_names:
                return default
            # id matched but notes diverged — treat as a custom variant
            break

    return build_custom_tuning(name, note_names)


# 入力条件付け (2026-07-05, テスターFB起点):
# - 優勢チャンネル選択: iPhone 系録音は片チャンネルがデジタル無音 (例: a9e30986
#   L=-33dBFS / R=-81dBFS)。従来の「常に ch0」は右のみ録音で無音 422 になる。
# - peak 正規化: テスター録音 (iPhone 内蔵マイク) は peak -26〜-33dBFS に集中し、
#   認識器は peak < 約 -25dBFS で急崩壊する (a9e30986: 1 event → +12dB で 27 events。
#   2cc06261/01fc3b8b/47902d34/70cc6637 も同傾向、2026-07-05 実測)。認識器入力を
#   レベル不変にするため peak を固定目標へ正規化し、元レベルは conditioning として
#   debug へ残す (recording-profile 較正 #173 の材料)。
NORMALIZE_TARGET_PEAK = 0.5  # -6.02 dBFS
# 増幅のみ (gain >= 1)。減衰も行う完全正規化は 0〜-3dBFS の既存 fixture 6 件 +
# corpus 1 件を回帰させた (2026-07-05 実測) — 認識器の絶対定数は大音量録音で
# 較正されており、健全レベルを触る必要はない。静かな録音の救済だけが目的。
SILENCE_PEAK_FLOOR = 1e-4  # -80 dBFS。これ未満は無音扱いで正規化しない
CHANNEL_SWITCH_MIN_DOMINANCE_DB = 20.0  # ch0 以外を採用する最小 L/R 差


def _peak_dbfs(peak: float) -> float:
    return 20.0 * math.log10(max(peak, 1e-10))


def condition_input_audio(audio: np.ndarray) -> tuple[np.ndarray, dict, float]:
    """優勢チャンネル選択 + 増幅のみ peak 正規化 (pure numpy)。

    戻り値の peak は正規化前の値。無音判定 (SILENCE_PEAK_FLOOR 未満) は
    呼び出し側の責務 — その場合 gain は掛からない。gt_draft.py など
    サーバー外の分析ツールも server parity のためにこれを使う。
    """
    conditioning: dict = {}
    if audio.ndim > 1:
        channel_peaks = np.max(np.abs(audio), axis=0)
        # 既定は従来どおり ch0。別チャンネルが圧倒的 (20dB 以上) に大きい場合のみ
        # 切り替える — iPhone 系の片チャンネル無音 (L/R 差 ~48dB) を救済しつつ、
        # 両チャンネルが生きた通常ステレオ (例: G-low fixture、L/R 差 0.04dB) の
        # 従来挙動を変えない。
        channel = 0
        loudest = int(np.argmax(channel_peaks))
        if loudest != 0 and (
            _peak_dbfs(float(channel_peaks[loudest])) - _peak_dbfs(float(channel_peaks[0]))
            >= CHANNEL_SWITCH_MIN_DOMINANCE_DB
        ):
            channel = loudest
        conditioning["channelPeaksDbfs"] = [
            round(_peak_dbfs(float(p)), 1) for p in channel_peaks
        ]
        conditioning["selectedChannel"] = channel
        audio = audio[:, channel]

    peak = float(np.max(np.abs(audio)))
    conditioning["inputPeakDbfs"] = round(_peak_dbfs(peak), 2)
    gain = max(1.0, NORMALIZE_TARGET_PEAK / peak) if peak >= SILENCE_PEAK_FLOOR else 1.0
    if gain > 1.0:
        audio = audio * gain
    conditioning["normalizationGainDb"] = round(20.0 * math.log10(gain), 2)
    return np.ascontiguousarray(audio, dtype=np.float32), conditioning, peak


async def read_audio(upload: UploadFile) -> tuple[np.ndarray, int, dict]:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Send WAV audio from the web client.") from exc

    audio, conditioning, peak = condition_input_audio(np.asarray(audio))
    if peak < SILENCE_PEAK_FLOOR:
        raise HTTPException(status_code=422, detail="Audio appears to be silent.")

    return audio, sample_rate, conditioning


def cents_distance(freq_a: float, freq_b: float) -> float:
    return abs(1200.0 * math.log2(freq_a / freq_b))


def snap_frequency_to_tuning(freq: float, tuning: InstrumentTuning) -> NoteCandidate | None:
    best_note = None
    best_distance = float("inf")

    for note in tuning.notes:
        distance = cents_distance(freq, note.frequency)
        if distance < best_distance:
            best_note = note
            best_distance = distance

    if best_note is None or best_distance > 80:
        return None

    return NoteCandidate(
        key=best_note.key,
        note=Note.from_name(best_note.note_name),
    )
