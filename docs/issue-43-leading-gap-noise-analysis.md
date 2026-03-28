# Issue #43: Leading Gap Noise Analysis

## 概要

`USE_ATTACK_VALIDATED_GAP_COLLECTOR=True` で7つのfixtureがregressionを起こす原因を調査した記録。

## 問題の症状

attack-validated gap collectorを有効にすると、以下の7 fixtureで偽のイベントが生成される：
- kalimba-17-c-a4-d4-f4-triad-repeat-01
- kalimba-17-c-bwv147-restart-prefix-01
- kalimba-17-c-c4-e4-g4-triad-repeat-01
- kalimba-17-c-d4-d5-octave-dyad-01
- kalimba-17-c-d4-repeat-01
- kalimba-17-c-e4-g4-b4-triad-repeat-01
- kalimba-17-c-mixed-sequence-01

偽イベントは `startBeat=0.0` または非常に小さい値で出力される。

## 調査の経緯

### Phase 1: ロジック探索

leading gap（0秒〜first_active_start）内のonsetを拾うロジックを特定：
- `_valid_attack_gap_onsets()` (transcription.py:610-623)
- `collect_attack_validated_gap_segments()` (transcription.py:977-986)

条件: `0.05 < onset_time < gap_end - 0.05` かつ `is_valid_attack=True`

### Phase 2: is_valid_attack判定の分析

`is_valid_attack` の判定条件:
```python
high_band_flux >= 1.5 or (broadband_gain >= 10.0 and high_band_flux >= 0.5)
```

affected fixtureの冒頭onset:
| fixture | onset_time | gain | hf_flux | is_valid |
|---------|------------|------|---------|----------|
| c4-e4-g4-triad @0.056s | 2.5 | 3.6 | True |
| mixed-sequence @0.059s | 2.0 | 424.8 | True |
| d4-repeat @0.061s | 1.2 | 276.8 | True |

gainは低いがhf_fluxが高いため `is_valid_attack=True` になっている。

### Phase 3: スペクトル特性の分析

onset直後15msのスペクトル帯域幅（BW90）を測定：

| タイプ | BW90 | Centroid | 特徴 |
|--------|------|----------|------|
| ノイズ候補 | 7312-11156 Hz | 2298-4401 Hz | 広帯域 |
| 実音 | 375-1031 Hz | 283-544 Hz | 狭帯域 |

**発見**: ノイズは全周波数帯に広がるブロードバンドパターン、楽音は特定周波数に集中。

### Phase 4: スペクトログラム確認

soxでスペクトログラムを生成し視覚的に確認：
- c4-e4-g4-triad @0.055-0.065s: **縦方向（全帯域）に広がるスパイク** → ノイズ
- d5-repeat @0.055-0.065s: **水平方向のバンド（特定周波数）** → 楽音

### Phase 5: ピッチ検出による検証

onset直後の音程を自己相関で検出し、期待される音と比較：

| Fixture | 期待音 | 冒頭onset検出音 | 一致 |
|---------|--------|-----------------|------|
| d5-repeat @0.059s | D5 | D6(1200Hz) | **NO** |
| c4-repeat @0.392s | C4 | ?(105Hz) | **NO** |
| c4-e4-g4-triad @0.029s | C4/E4/G4 | C6(1103Hz) | **NO** |
| a4-d4-f4-triad @0.061s | A4/D4/F4 | ?(1500Hz) | **NO** |
| e4-g4-b4-triad @0.059s | E4/G4/B4 | ?(1500Hz) | **NO** |

**結論**: 冒頭0.1秒以内のonsetは**カリンバの音ではない**。

### Phase 6: 最初の本当のカリンバ音の特定

各fixtureの最初の正しいカリンバ音は `first_active_start` 付近から始まる：

| Fixture | 1st Active | 正しい音のonset | 検出音 |
|---------|------------|-----------------|--------|
| d5-repeat | 2.063s | 2.080s | D5 ✓ |
| c4-repeat | 2.553s | 2.573s | C4 ✓ |
| a4-d4-f4-triad | 3.425s | 3.435s | D4 ✓ |

## 結論

1. **冒頭0.1秒以内のonsetはカリンバの音ではない**
   - 録音開始時のノイズ（マイククリック、環境音）
   - 椅子の音、録音準備の動作音など
   - カリンバの基音周波数に一致しない

2. **first_active_start（RMSベース検出）が実際の演奏開始を正しく捉えている**
   - leading gapは本来「無音」であるべき区間
   - そこにある「onset」は演奏ではない

3. **問題の根本原因**
   - leading gap内の録音ノイズが `is_valid_attack=True` と判定される
   - 高周波フラックスは高い（ブロードバンドノイズ）がカリンバの音ではない

## 検討した解決策

### 案1: 冒頭0.1秒カット（時間ベース）
- **メリット**: 実装が最もシンプル、WebAssembly移植容易
- **デメリット**: 録音開始が早いケースで冒頭の音を失う可能性

### 案2: gain/hf_flux比率閾値
- **メリット**: 相対値なので録音環境に依存しにくい、差が明確（ノイズ: ~0.005, 実音: ~60）
- **デメリット**: gainの絶対値を使う点がやや環境依存

### 案3: スペクトル帯域幅（BW90）閾値
- **メリット**: 物理的に意味のある区別（ブロードバンド vs 楽音）
- **デメリット**: 計算コストがやや高い、閾値設定が難しい

### 案4: カリンバ基音一致判定
- **メリット**: 最も正確（カリンバの音かどうかを直接判定）
- **デメリット**: onset直後は基音が立ち上がっていないため判定困難

## 推奨

**案1（冒頭0.1秒カット）** が最もシンプルで確実。

理由:
- 実際の録音で冒頭0.1秒に有効な演奏が入ることはほぼない
- AGENTS.mdの方針（streaming対応、WebAssembly移植性）と整合
- 実装・テストが容易

補助的に **案2（gain/hf_flux比率）** を組み合わせることで、0.1秒より後のノイズも除外可能。

## Phase 7: 波形特徴量による trailing noise フィルタ（2026-03-28）

### 手法: fake種類別の特徴量分離分析

単一の「fake vs real」比較では有効な閾値が見つからなかった。
**fake を種類別に分類し直した**ところ、trailing noise に対して6つのメトリクスが CLEAN 分離を示した。

| fake種類 | 物理的原因 | 分離性 |
|----------|-----------|--------|
| noise_trailing | 演奏停止後の環境ノイズ/機器ノイズ | 高（6メトリクスCLEAN） |
| noise_leading | 録音開始時のノイズ | 中（1サンプルのため一般化注意） |
| residual | カリンバ残響の減衰 | 低（スペクトルが実音と本質的に同一） |

### trailing noise に有効な特徴量

| メトリック | Cohen's d | 分離 | 閾値方向 |
|-----------|----------|------|---------|
| kurtosis_20ms | 3.31 | CLEAN | noise > 2.0, real < 1.0 |
| post_crest_20ms | 2.59 | CLEAN | noise > 3.8, real < 3.6 |
| crest_change | 2.01 | CLEAN | noise > 1.0, real < 0.6 |
| gain_positive_frac | 1.90 | CLEAN | noise > 0.77, real < 0.68 |
| broadband_gain_80ms | 2.87 | CLEAN | noise > 8.3, real < 3.7 |
| rms_ratio_80ms | 2.51 | CLEAN | noise > 2.9, real < 1.9 |

### 実装結果

| フィルタ | 定数 | 値 | 状態 | 理由 |
|---------|------|-----|------|------|
| broadband_onset_gain | `GAP_ONSET_MIN_BROADBAND_GAIN` | 0.95 | 有効 | 既存、residual decay対策 |
| kurtosis | `GAP_ONSET_MAX_KURTOSIS` | 2.0 | **有効** | 最もロバスト（振幅/ダイナミクス非依存） |
| post_crest | `GAP_ONSET_MAX_POST_CREST` | 0.0 | 無効化可能 | 3.8で有効化可能 |
| gain_positive_frac | — | — | 未実装 | マージン狭い、改善案は #44 |

### 実装上の注意: candidates の分離

kurtosis フィルタを `attack_validated_gap_candidates` に適用すると、フィルタされた candidates が `collect_multi_onset_gap_segments` にも漏れ、意図しない回帰を引き起こした。

**対策**: unfiltered candidates（共有パス用）と filtered candidates（gap collector専用）を分離。

## Onset分類の標準手順

今回確立した手法（再利用可能）:

1. **可視化で仮説を立てる**: `/audio-spectrum`, `/audio-visualize`
2. **特徴量を網羅的に計算**: `onset_separation_analysis.py` or `/audio-separate`
3. **fake種類別に分離度を評価**: 混合分析は避ける
4. **Cohen's d + overlap分析で客観的に閾値を決定**
5. 閾値が CLEAN 分離を示すか確認（overlap があれば false positive リスクを検討）

## 関連ファイル

分析スクリプト:
- `scripts/audio-analysis/onset_separation_analysis.py` — 汎用onset群分離分析
- `.claude/skills/audio-separate.md` — 分離分析スキル

旧分析スクリプト（/tmp/に作成、参考用）:
- analyze_is_valid_attack.py
- analyze_first_2sec.py
- analyze_attack_spectrum.py
- analyze_attack_evolution.py
- analyze_spectral_spread.py
- analyze_kalimba_signature.py
- find_real_first_note.py
- per_type_separation.py（onset_separation_analysis.pyの元）
- crest_impulse_analysis.py

## 関連Issue

- #43: attack-validated gap collector regressions
- #44: gain_positive_frac による広帯域ノイズ検出

## 更新履歴

- 2026-03-27: 初版作成（Issue #43調査結果）
- 2026-03-28: Phase 7追加（波形特徴量フィルタ実装、onset分類手法の確立）
