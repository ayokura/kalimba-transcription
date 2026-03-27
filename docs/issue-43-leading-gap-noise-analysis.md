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

## 関連ファイル

分析スクリプト（/tmp/に作成）:
- analyze_is_valid_attack.py
- analyze_first_2sec.py
- analyze_attack_spectrum.py
- analyze_attack_evolution.py
- analyze_spectral_spread.py
- analyze_kalimba_signature.py
- find_real_first_note.py

## 更新履歴

- 2026-03-27: 初版作成（Issue #43調査結果）
