@AGENTS.md

# Claude Code-Specific Overrides and Additions

- This file is maintained by Claude Code. Other agents should not update it.
- Rules shared with other agents belong in AGENTS.md, not here.

## Audio Analysis Skills

音声分析用のスキルが `.claude/skills/` に定義されている:

| Skill | Tool | 用途 |
|-------|------|------|
| `/audio-visualize` | sox | スペクトログラム生成 |
| `/audio-onset` | aubio | onset検出（複数アルゴリズム） |
| `/audio-pitch` | praat | 高精度ピッチ検出 |
| `/audio-spectrum` | librosa | スペクトル特徴量（BW90, centroid等） |
| `/audio-diagnose` | 統合 | onset判定（ノイズ vs カリンバ音） |
| `/audio-separate` | librosa | onset群の特徴量分離分析（Cohen's d） |

### 使用例

```
/audio-visualize d5-repeat-01 0 0.2
/audio-onset d5-repeat-01 hfc
/audio-pitch d5-repeat-01 2.0 1.0
/audio-spectrum d5-repeat-01 0.059
/audio-diagnose d5-repeat-01
/audio-separate bwv147-restart-prefix-01 --real 1.87,3.15,5.06 --compare 4.16
```

fixture名（例: `d5-repeat-01`）は自動的にフルパスに展開される。

## Audio Analysis Scripts

`scripts/audio-analysis/` にヘルパースクリプトがある:

- `spectrum_stats.py` - スペクトル特徴量計算
- `pitch_detect.praat` - Praatピッチ検出スクリプト
- `analyze_spectral_spread.py` - 複数fixtureのスペクトル比較
- `analyze_kalimba_signature.py` - カリンバ倍音構造検出
- `find_real_first_note.py` - 最初の本当のカリンバ音を特定
- `onset_separation_analysis.py` - onset群の特徴量分離分析
- `fixture_rejection_sweep.py` - rejection閾値探索（実テストスイート使用）

### Fixture 影響評価の注意

rejection 閾値やフィルタ変更の影響を評価する際は、**必ず実テストスイート（pytest）を使うこと**。
ad-hoc な event count 比較は evaluation window / ignoredRanges / expectedEventNoteSetsOrdered を無視するため、偽の「回帰」を報告する。`fixture_rejection_sweep.py` はこの教訓から作成されたツール。

詳細は `scripts/audio-analysis/README.md` を参照。
