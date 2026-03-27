"""
各fixtureの「最初の本当のカリンバ音」を特定する。
- expected.jsonの期待される音程と一致するか
- 十分なgainがあるか
- 倍音構造があるか
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

REPO_ROOT = Path("/home/ayokura/kalimba-transcription")
FIXTURE_ROOT = REPO_ROOT / "apps/api/tests/fixtures/manual-captures"
sys.path.insert(0, str(REPO_ROOT / "apps/api"))

# カリンバ基音
KALIMBA_FUNDAMENTALS = {
    "C4": 261.6, "D4": 293.7, "E4": 329.6, "F4": 349.2,
    "G4": 392.0, "A4": 440.0, "B4": 493.9,
    "C5": 523.3, "D5": 587.3, "E5": 659.3, "F5": 698.5,
    "G5": 784.0, "A5": 880.0, "B5": 987.8,
    "C6": 1046.5, "D6": 1174.7, "E6": 1318.5,
}


def get_onset_profiles(fixture_name: str) -> tuple:
    import app.transcription as mod
    mod.USE_ATTACK_VALIDATED_GAP_COLLECTOR = True

    from fastapi.testclient import TestClient
    from app.main import app

    fixture_dir = FIXTURE_ROOT / fixture_name
    client = TestClient(app)
    request_payload = json.loads((fixture_dir / "request.json").read_text())
    audio_bytes = (fixture_dir / "audio.wav").read_bytes()

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    debug = data.get("debug") or {}
    profiles_dict = debug.get("onsetAttackProfiles", {})
    active_ranges = debug.get("activeRanges", [])

    profiles = [{"time": float(k), **v} for k, v in profiles_dict.items()]
    profiles.sort(key=lambda x: x["time"])

    first_active_start = active_ranges[0][0] if active_ranges else None

    return profiles, first_active_start


def detect_pitch(audio: np.ndarray, sr: int, onset_time: float) -> str:
    """onset後50-100msの区間でピッチを検出"""
    start = int(sr * (onset_time + 0.03))  # 30ms後から
    end = int(sr * (onset_time + 0.10))    # 100msまで

    if end > len(audio):
        return "N/A"

    segment = audio[start:end]

    # 自己相関でピッチ検出
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]

    # 最初のピークを探す（基本周波数に対応）
    # 低周波数を除外（100Hz以上 = sr/100サンプル以下）
    min_lag = int(sr / 1500)  # 1500Hz以下
    max_lag = int(sr / 100)   # 100Hz以上

    if max_lag > len(corr):
        return "N/A"

    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return "N/A"

    peak_idx = np.argmax(search_region) + min_lag
    freq = sr / peak_idx if peak_idx > 0 else 0

    # 最も近いカリンバ音を特定
    closest_note = None
    closest_dist = float('inf')
    for note, fund in KALIMBA_FUNDAMENTALS.items():
        dist = abs(freq - fund) / fund
        if dist < closest_dist:
            closest_dist = dist
            closest_note = note

    if closest_dist < 0.1:  # 10%以内
        return f"{closest_note} ({freq:.0f}Hz)"
    else:
        return f"? ({freq:.0f}Hz)"


FIXTURES = [
    ("kalimba-17-c-d5-repeat-01", "D5"),
    ("kalimba-17-c-c4-repeat-01", "C4"),
    ("kalimba-17-c-c4-e4-g4-triad-repeat-01", "C4/E4/G4"),
    ("kalimba-17-c-mixed-sequence-01", "mixed"),
    ("kalimba-17-c-d4-repeat-01", "D4"),
    ("kalimba-17-c-a4-d4-f4-triad-repeat-01", "A4/D4/F4"),
    ("kalimba-17-c-e4-g4-b4-triad-repeat-01", "E4/G4/B4"),
]


def main():
    print("=" * 130)
    print("FIRST REAL KALIMBA NOTE DETECTION")
    print("=" * 130)
    print()
    print(f"{'Fixture':<45} {'Expected':<12} {'1st Active':<12} {'Onset':<8} {'Gain':<8} {'Detected':<15} {'Match?':<8}")
    print("-" * 130)

    for fixture_name, expected_notes in FIXTURES:
        profiles, first_active_start = get_onset_profiles(fixture_name)

        fixture_dir = FIXTURE_ROOT / fixture_name
        audio_path = fixture_dir / "audio.wav"
        audio, sr = librosa.load(audio_path, sr=None)

        # gainが高いonsetを探す（最初の3つ）
        high_gain_onsets = sorted(
            [p for p in profiles if p.get("broadbandOnsetGain", 0) > 10],
            key=lambda x: x["time"]
        )[:3]

        for i, p in enumerate(high_gain_onsets):
            onset_time = p["time"]
            gain = p.get("broadbandOnsetGain", 0)

            detected = detect_pitch(audio, sr, onset_time)

            # 期待される音と一致するか
            match = "YES" if any(note in detected for note in expected_notes.split("/")) else "NO"

            if i == 0:
                print(f"{fixture_name:<45} {expected_notes:<12} {first_active_start:<12.4f} "
                      f"{onset_time:<8.4f} {gain:<8.1f} {detected:<15} {match:<8}")
            else:
                print(f"{'':<45} {'':<12} {'':<12} "
                      f"{onset_time:<8.4f} {gain:<8.1f} {detected:<15} {match:<8}")


if __name__ == "__main__":
    main()
