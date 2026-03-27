# Audio Analysis Scripts

音声分析用スクリプト集。Claude Code skills (`/audio-*`) から呼び出される。

## 必要なツール

- **sox**: スペクトログラム生成
- **aubio**: onset検出、ピッチ検出
- **praat**: 高精度ピッチ分析
- **librosa**: スペクトル特徴量計算 (Python)

## スクリプト一覧

### spectrum_stats.py
onset時刻でのスペクトル特性を計算。

```bash
uv run python spectrum_stats.py <audio_file> <onset_time> [window_ms]
```

出力:
- centroid_hz: スペクトル重心
- bandwidth_90_hz: 90%エネルギー帯域幅
- spread_hz: スペクトル拡散
- hf_ratio_pct: 高周波比率 (>2kHz)
- vhf_ratio_pct: 超高周波比率 (>8kHz)
- classification: NOISE / KALIMBA / UNCLEAR

### pitch_detect.praat
Praatを使用した高精度ピッチ検出。

```bash
praat --run pitch_detect.praat <audio_file> <start> <duration> <step> <min_pitch> <max_pitch>
```

出力: 時刻、周波数、最も近いカリンバ音、偏差%

### analyze_spectral_spread.py
複数fixtureのスペクトル帯域幅を比較分析。

### analyze_kalimba_signature.py
カリンバ特有の倍音構造を検出。

### find_real_first_note.py
各fixtureの「最初の本当のカリンバ音」を特定。

## 判定基準

### ノイズ vs 楽音の判定

| 特徴 | ノイズ | カリンバ音 |
|------|--------|------------|
| BW90 | >6000 Hz | <2000 Hz |
| Centroid | >3000 Hz | <1000 Hz |
| VHF% | >2% | <1% |
| ピッチ偏差 | >10% | <5% |

## Claude Code Skills

これらのスクリプトは以下のskillsから呼び出される:

- `/audio-visualize` - スペクトログラム生成 (sox)
- `/audio-onset` - onset検出 (aubio)
- `/audio-pitch` - ピッチ検出 (praat)
- `/audio-spectrum` - スペクトル特徴量 (librosa)
- `/audio-diagnose` - 統合診断

## 関連ドキュメント

- [Issue #43 分析レポート](../../docs/issue-43-leading-gap-noise-analysis.md)
