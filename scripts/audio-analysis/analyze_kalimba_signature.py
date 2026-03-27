"""
カリンバ特有の周波数パターンを検出する。
カリンバの特徴：
1. 特定の基音周波数（C4〜E6）
2. 整数倍の倍音構造
3. タイン固有の共鳴パターン

これと椅子の音、環境ノイズなどを区別できるか検証。
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

REPO_ROOT = Path("/home/ayokura/kalimba-transcription")
FIXTURE_ROOT = REPO_ROOT / "apps/api/tests/fixtures/manual-captures"
sys.path.insert(0, str(REPO_ROOT / "apps/api"))

# 17キーカリンバの基音周波数（C調）
KALIMBA_17_FUNDAMENTALS = {
    "C4": 261.6, "D4": 293.7, "E4": 329.6, "F4": 349.2,
    "G4": 392.0, "A4": 440.0, "B4": 493.9,
    "C5": 523.3, "D5": 587.3, "E5": 659.3, "F5": 698.5,
    "G5": 784.0, "A5": 880.0, "B5": 987.8,
    "C6": 1046.5, "D6": 1174.7, "E6": 1318.5,
}


def get_onset_profiles(fixture_name: str) -> list:
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

    profiles = [{"time": float(k), **v} for k, v in profiles_dict.items()]
    profiles.sort(key=lambda x: x["time"])

    return profiles


def find_harmonic_match(spectrum: np.ndarray, freqs: np.ndarray,
                        fundamental: float, n_harmonics: int = 4) -> dict:
    """基音とその倍音のエネルギーを計算"""
    total_harmonic_energy = 0
    matched_harmonics = []

    for h in range(1, n_harmonics + 1):
        target_freq = fundamental * h
        # ±3%の範囲で探索
        freq_low = target_freq * 0.97
        freq_high = target_freq * 1.03
        mask = (freqs >= freq_low) & (freqs <= freq_high)

        if np.any(mask):
            energy = np.max(spectrum[mask])
            if energy > 0:
                total_harmonic_energy += energy
                matched_harmonics.append(h)

    return {
        "fundamental": fundamental,
        "harmonic_energy": total_harmonic_energy,
        "matched_harmonics": matched_harmonics,
    }


def analyze_kalimba_signature(audio: np.ndarray, sr: int, onset_time: float,
                               window_ms: float = 30) -> dict:
    """onset直後のスペクトルがカリンバの倍音構造に一致するか"""
    onset_sample = int(sr * onset_time)
    # onset直後〜30msを分析（基音が立ち上がった後）
    start = onset_sample + int(sr * 0.010)  # 10ms後から
    window_samples = int(sr * window_ms / 1000)
    end = min(len(audio), start + window_samples)
    segment = audio[start:end]

    if len(segment) < 256:
        return None

    n_fft = 4096  # 高い周波数分解能
    spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    total_energy = np.sum(spectrum ** 2)
    if total_energy == 0:
        return None

    # 各カリンバ基音に対して倍音一致度を計算
    best_match = None
    best_score = 0

    for note, fundamental in KALIMBA_17_FUNDAMENTALS.items():
        match = find_harmonic_match(spectrum, freqs, fundamental)
        score = match["harmonic_energy"]
        if score > best_score:
            best_score = score
            best_match = {
                "note": note,
                "fundamental": fundamental,
                "score": score,
                "harmonics": match["matched_harmonics"],
            }

    # カリンバらしさの指標
    # 1. 最も一致した基音のスコア / 全エネルギー
    kalimba_ratio = best_score / np.sqrt(total_energy) if total_energy > 0 else 0

    # 2. ピーク周波数がカリンバの基音に近いか
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    closest_note = None
    closest_dist = float('inf')
    for note, fundamental in KALIMBA_17_FUNDAMENTALS.items():
        dist = abs(peak_freq - fundamental) / fundamental
        if dist < closest_dist:
            closest_dist = dist
            closest_note = note

    peak_matches_kalimba = closest_dist < 0.05  # 5%以内

    # 3. スペクトルの「整数倍倍音らしさ」（harmonicity）
    # ピーク周波数の整数倍の位置にエネルギーがあるか
    harmonicity = 0
    if peak_freq > 100:
        for h in range(2, 5):
            target = peak_freq * h
            if target < freqs[-1]:
                idx = np.argmin(np.abs(freqs - target))
                harmonicity += spectrum[idx]

    return {
        "best_match_note": best_match["note"] if best_match else None,
        "best_match_fundamental": best_match["fundamental"] if best_match else None,
        "best_match_harmonics": best_match["harmonics"] if best_match else [],
        "kalimba_ratio": float(kalimba_ratio),
        "peak_freq": float(peak_freq),
        "peak_matches_kalimba": peak_matches_kalimba,
        "closest_note": closest_note,
        "closest_dist_pct": float(closest_dist * 100),
        "harmonicity": float(harmonicity),
    }


# テスト対象
FIXTURES = [
    # ノイズと判定されたもの
    ("kalimba-17-c-c4-e4-g4-triad-repeat-01", 0.0560, "NOISE?"),
    ("kalimba-17-c-mixed-sequence-01", 0.0587, "NOISE?"),
    ("kalimba-17-c-d4-repeat-01", 0.0613, "NOISE?"),
    # 実音と判定されたもの
    ("kalimba-17-c-c4-e4-g4-triad-repeat-01", 0.0293, "REAL?"),
    ("kalimba-17-c-d5-repeat-01", 0.0587, "REAL"),
    ("kalimba-17-c-a4-d4-f4-triad-repeat-01", 0.0613, "REAL"),
    ("kalimba-17-c-e4-g4-b4-triad-repeat-01", 0.0587, "REAL"),
]


def main():
    print("=" * 120)
    print("KALIMBA SIGNATURE ANALYSIS - カリンバ特有の倍音構造検出")
    print("=" * 120)
    print()
    print(f"{'Fixture':<45} {'Onset':<8} {'Type':<8} {'BestNote':<8} {'PeakFreq':<10} {'ClosestNote':<12} {'Dist%':<8} {'Harmonicity':<12}")
    print("-" * 120)

    for fixture_name, onset_time, expected in FIXTURES:
        fixture_dir = FIXTURE_ROOT / fixture_name
        audio_path = fixture_dir / "audio.wav"
        audio, sr = librosa.load(audio_path, sr=None)

        result = analyze_kalimba_signature(audio, sr, onset_time)
        if result:
            print(f"{fixture_name:<45} {onset_time:<8.4f} {expected:<8} "
                  f"{result['best_match_note']:<8} {result['peak_freq']:<10.0f} "
                  f"{result['closest_note']:<12} {result['closest_dist_pct']:<8.1f} "
                  f"{result['harmonicity']:<12.1f}")

    # 追加：確実な楽音との比較
    print()
    print("=" * 120)
    print("COMPARISON: 確実な楽音（gain > 100）")
    print("=" * 120)
    print()

    for fixture_name in ["kalimba-17-c-d5-repeat-01", "kalimba-17-c-c4-repeat-01"]:
        profiles = get_onset_profiles(fixture_name)
        high_gain = [p for p in profiles if p.get("broadbandOnsetGain", 0) > 100]

        fixture_dir = FIXTURE_ROOT / fixture_name
        audio_path = fixture_dir / "audio.wav"
        audio, sr = librosa.load(audio_path, sr=None)

        for p in high_gain[:2]:
            onset_time = p["time"]
            result = analyze_kalimba_signature(audio, sr, onset_time)
            if result:
                print(f"{fixture_name:<45} {onset_time:<8.4f} {'REAL':<8} "
                      f"{result['best_match_note']:<8} {result['peak_freq']:<10.0f} "
                      f"{result['closest_note']:<12} {result['closest_dist_pct']:<8.1f} "
                      f"{result['harmonicity']:<12.1f}")


if __name__ == "__main__":
    main()
