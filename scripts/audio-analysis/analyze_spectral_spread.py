"""
onset直後のスペクトル帯域幅（spread）を測定する。
ノイズ：広帯域にエネルギーが分散
楽音：特定帯域にエネルギーが集中
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

REPO_ROOT = Path("/home/ayokura/kalimba-transcription")
FIXTURE_ROOT = REPO_ROOT / "apps/api/tests/fixtures/manual-captures"
sys.path.insert(0, str(REPO_ROOT / "apps/api"))


def get_onset_profiles(fixture_name: str) -> tuple:
    """APIを呼んでonset_attack_profilesを取得"""
    import app.transcription as mod
    mod._legacy.USE_ATTACK_VALIDATED_GAP_COLLECTOR = True

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


def compute_spectral_spread(audio: np.ndarray, sr: int, onset_time: float,
                            window_ms: float = 15) -> dict:
    """onset直後のスペクトル特性を計算"""
    onset_sample = int(sr * onset_time)
    window_samples = int(sr * window_ms / 1000)

    # onset直後のウィンドウ
    start = onset_sample
    end = min(len(audio), onset_sample + window_samples)
    segment = audio[start:end]

    if len(segment) < 256:
        return None

    # FFT
    n_fft = 2048
    spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # エネルギー正規化
    total_energy = np.sum(spectrum ** 2)
    if total_energy == 0:
        return None

    norm_spectrum = spectrum ** 2 / total_energy

    # スペクトル重心（centroid）
    centroid = np.sum(freqs * norm_spectrum) / np.sum(norm_spectrum)

    # スペクトル帯域幅（bandwidth）- エネルギーの90%が含まれる帯域
    cumsum = np.cumsum(norm_spectrum)
    idx_05 = np.searchsorted(cumsum, 0.05)
    idx_95 = np.searchsorted(cumsum, 0.95)
    bandwidth_90 = freqs[idx_95] - freqs[idx_05]

    # スペクトル拡散（spread）- 重心からの標準偏差
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * norm_spectrum))

    # 高周波比率（2kHz以上のエネルギー比率）
    hf_mask = freqs >= 2000
    hf_ratio = np.sum(spectrum[hf_mask] ** 2) / total_energy

    # 超高周波比率（8kHz以上）
    vhf_mask = freqs >= 8000
    vhf_ratio = np.sum(spectrum[vhf_mask] ** 2) / total_energy

    # ピーク周波数
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    return {
        "centroid": float(centroid),
        "bandwidth_90": float(bandwidth_90),
        "spread": float(spread),
        "hf_ratio": float(hf_ratio),
        "vhf_ratio": float(vhf_ratio),
        "peak_freq": float(peak_freq),
    }


# テスト対象
FIXTURES = [
    # ノイズと思われる
    ("kalimba-17-c-c4-e4-g4-triad-repeat-01", "NOISE?"),
    ("kalimba-17-c-mixed-sequence-01", "NOISE?"),
    ("kalimba-17-c-d4-repeat-01", "NOISE?"),
    # 実音
    ("kalimba-17-c-d5-repeat-01", "REAL"),
    ("kalimba-17-c-c4-c5-octave-dyad-01", "REAL"),
    ("kalimba-17-c-b4-d5-double-notes-01", "REAL"),
    # 境界線上（gain高いので実音かも）
    ("kalimba-17-c-a4-d4-f4-triad-repeat-01", "BOUNDARY"),
    ("kalimba-17-c-e4-g4-b4-triad-repeat-01", "BOUNDARY"),
]


def main():
    print("=" * 100)
    print("SPECTRAL SPREAD ANALYSIS - onset直後15msのスペクトル特性")
    print("=" * 100)
    print()
    print(f"{'Fixture':<45} {'Type':<10} {'Onset':<8} {'Gain':<8} {'Centroid':<10} {'BW90':<10} {'Spread':<10} {'VHF%':<8}")
    print("-" * 100)

    for fixture_name, expected_type in FIXTURES:
        profiles, first_active_start = get_onset_profiles(fixture_name)

        # 最初の0.1秒以内のis_valid=Trueのonset
        early_onsets = [p for p in profiles if p["time"] < 0.1 and p.get("isValidAttack")]

        fixture_dir = FIXTURE_ROOT / fixture_name
        audio_path = fixture_dir / "audio.wav"
        audio, sr = librosa.load(audio_path, sr=None)

        for p in early_onsets[:1]:  # 最初の1つだけ
            onset_time = p["time"]
            gain = p.get("broadbandOnsetGain", 0)

            result = compute_spectral_spread(audio, sr, onset_time)
            if result:
                print(f"{fixture_name:<45} {expected_type:<10} {onset_time:<8.4f} {gain:<8.1f} "
                      f"{result['centroid']:<10.0f} {result['bandwidth_90']:<10.0f} "
                      f"{result['spread']:<10.0f} {result['vhf_ratio']*100:<8.2f}")

    print()
    print("=" * 100)
    print("COMPARISON: 実音の2番目以降のonset（確実な楽音）")
    print("=" * 100)
    print()

    for fixture_name in ["kalimba-17-c-d5-repeat-01", "kalimba-17-c-c4-repeat-01"]:
        profiles, _ = get_onset_profiles(fixture_name)

        # gainが高いonset（確実な楽音）
        high_gain_onsets = [p for p in profiles if p.get("broadbandOnsetGain", 0) > 50]

        fixture_dir = FIXTURE_ROOT / fixture_name
        audio_path = fixture_dir / "audio.wav"
        audio, sr = librosa.load(audio_path, sr=None)

        for p in high_gain_onsets[:2]:
            onset_time = p["time"]
            gain = p.get("broadbandOnsetGain", 0)

            result = compute_spectral_spread(audio, sr, onset_time)
            if result:
                print(f"{fixture_name:<45} {'REAL':<10} {onset_time:<8.4f} {gain:<8.1f} "
                      f"{result['centroid']:<10.0f} {result['bandwidth_90']:<10.0f} "
                      f"{result['spread']:<10.0f} {result['vhf_ratio']*100:<8.2f}")


if __name__ == "__main__":
    main()
