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

### onset_separation_analysis.py
onset群を比較し、分離に最も有効な特徴量を特定。Cohen's d + overlap分析。

```bash
# 簡易モード（1ファイル内の2群比較）
uv run python scripts/audio-analysis/onset_separation_analysis.py \
  --audio c4-repeat-01 --real 1.87,3.15,5.06 --compare 4.16

# JSON設定（複数ファイル・複数群の包括分析）
uv run python scripts/audio-analysis/onset_separation_analysis.py --config samples.json
```

出力:
- 40以上の特徴量のCohen's d（分離度）をランキング表示
- CLEAN分離（群間overlap無し）の自動検出
- 上位特徴量の生値テーブル

JSON設定フォーマット:
```json
{
  "groups": {
    "real": [{"audio": "fixture-name", "onset": 1.87, "label": "E5"}],
    "noise": [{"audio": "fixture-name", "onset": 16.97, "label": "trailing"}]
  },
  "reference_group": "real"
}
```

### energy_trace.py
指定 audio の time window における per-note 帯域エネルギーを step ごとにトレース。
rescue / suppression 設計の前提検証で頻用。

```bash
uv run python scripts/audio-analysis/energy_trace.py <audio> <start> <duration> [--notes G4,G5] [--step 0.05] [--band 15]
```

### note_peak_track.py
指定ノート帯域の peak 周波数と cents ずれを時間的に追跡。 tuning drift の検出や
FFT 分解能問題の診断に使用。

```bash
uv run python scripts/audio-analysis/note_peak_track.py <audio> <start> <duration> --notes D4,B4,G4
```

### fixture_rejection_sweep.py
primary rejection 閾値を **実テストスイート (pytest)** に対して sweep。
ad-hoc な event count 比較は evaluation window / ignoredRanges /
expectedEventNoteSetsOrdered を無視するため、 偽の「回帰」を報告する。
本スクリプトは実 fixture テストを走らせて pass/fail を集計する。

```bash
uv run python scripts/audio-analysis/fixture_rejection_sweep.py
uv run python scripts/audio-analysis/fixture_rejection_sweep.py 10:0.8 20:0.9 30:0.97
```

### score_alignment_diagnosis.py
期待 events と recognizer 出力を ordered matching で整列・差分表示。
fixture が `score_structure.json` を持つ場合は per-line matching、 持たない
シンプル fixture では `request.json:expectedPerformance` を fallback として
synthetic 単一 line で動作する。

```bash
uv run python scripts/audio-analysis/score_alignment_diagnosis.py <fixture> [--verbose] [--mode events|segments] [--line L1]
```

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
- `/audio-separate` - onset群の特徴量分離分析
- `/audio-energy-trace` - per-note 帯域エネルギートレース
- `/audio-peak-track` - per-note peak 周波数 + cents ずれ追跡
- `/score-alignment` - 期待 events と recognizer 出力の整列
- `/fixture-rejection-sweep` - rejection 閾値 sweep (実 pytest 経由)

## 関連ドキュメント

- [Issue #43 分析レポート](../../docs/issue-43-leading-gap-noise-analysis.md)
