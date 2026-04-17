@AGENTS.md

# Claude Code-Specific Overrides and Additions

- This file is maintained by Claude Code. Other agents should not update it.
- Rules shared with other agents belong in AGENTS.md, not here.

## Dev server startup

API サーバーは常に `--reload` 付きで起動すること。ファイル変更時に自動リロードされ、コード反映のたびにサーバーを kill/restart する必要がない:

```
uv run uvicorn app.main:app --app-dir apps/api --reload --host 0.0.0.0 --port 8000
```

バックグラウンドで起動した server の停止は、`kill` ではなく **`TaskStop` tool** を使うこと (run_in_background で起動したタスクの task_id を渡す)。`kill` は Claude Code がフリーズする原因になる場合がある。

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
| `/audio-peak-track` | numpy | ノート帯域ピーク周波数・centsずれ追跡 |
| `/audio-energy-trace` | librosa | per-note 帯域エネルギーの時間トレース |
| `/score-alignment` | recognizer | 期待 events と recognizer 出力の整列 |
| `/fixture-rejection-sweep` | pytest | rejection 閾値 sweep（実 fixture テスト経由）|

### 使用例

```
/audio-visualize d5-repeat-01 0 0.2
/audio-onset d5-repeat-01 hfc
/audio-pitch d5-repeat-01 2.0 1.0
/audio-spectrum d5-repeat-01 0.059
/audio-diagnose d5-repeat-01
/audio-separate bwv147-restart-prefix-01 --real 1.87,3.15,5.06 --compare 4.16
/audio-peak-track bwv147-sequence-163-01 40.5 1.5 --notes D4,B4,G4
```

fixture名（例: `d5-repeat-01`）は自動的にフルパスに展開される。

## Audio Analysis Scripts

`scripts/audio-analysis/` にヘルパースクリプトがある:

- `spectrum_stats.py` - スペクトル特徴量計算
- `pitch_detect.praat` - Praatピッチ検出スクリプト
- `onset_separation_analysis.py` - onset群の特徴量分離分析
- `note_peak_track.py` - ノート帯域ピーク周波数・centsずれ追跡
- `energy_trace.py` - ノート帯域エネルギーの時間推移トレース (rescue/suppression 設計の前提検証で頻用)
- `fixture_rejection_sweep.py` - rejection閾値探索（実テストスイート使用）
- `score_alignment_diagnosis.py` - expected events (`score_structure.json` / `request.json:expectedPerformance` / `expected.json:expectedEventNoteSetsOrdered` の優先順で fallback) と recognizer 出力の整列・差分表示

### Fixture 影響評価の注意

rejection 閾値やフィルタ変更の影響を評価する際は、**必ず実テストスイート（pytest）を使うこと**。
ad-hoc な event count 比較は evaluation window / ignoredRanges / expectedEventNoteSetsOrdered を無視するため、偽の「回帰」を報告する。`fixture_rejection_sweep.py` はこの教訓から作成されたツール。

詳細は `scripts/audio-analysis/README.md` を参照。

### score_alignment_diagnosis.py のキャッシュ挙動

`scripts/audio-analysis/score_alignment_diagnosis.py` のキャッシュキーには **recognizer source code (`apps/api/app/transcription/*.py`) の SHA256 fingerprint** が含まれる (`_recognizer_code_fingerprint()`)。コードを変更すると自動 invalidate されるので、`--no-cache` を常時付ける必要はない。iterative 修正 → diagnosis 走らせる workflow ではキャッシュが効いて時短になる。`--no-cache` はキャッシュ読み取りをスキップするが結果は書き込むので、次回は高速。データファイル変更やキャッシュ破損を疑う場合にのみ使う。SUMMARY 末尾の `Cache: hit/miss/fresh (recognizer: ...)` で結果の由来を確認可能。

## GitHub interaction conventions

### Commit SHA は GitHub 上ではバッククォートで囲わない

GitHub に投稿する content (issue / PR comment, issue / PR body, commit message body 等) で commit SHA を参照する時は、**バッククォートで囲わないこと**。GitHub はバッククォートで囲った文字列を inline code として扱い、自動 commit リンクを生成しないため、せっかくの SHA がクリックできない form になる。

- 悪い例: 修正は `35bca12` で入った
- 良い例: 修正は 35bca12 で入った

これは `gh issue comment`, `gh pr comment`, `gh issue create --body`, `gh pr create --body`, `git commit -m` 等、最終的に GitHub UI に表示される **すべての content** に適用する。

ローカルファイル (CLAUDE.md, AGENTS.md, `memory/*.md`, `docs/*.md` 等) では GitHub のリンク化対象外なので、code 表現としてのバッククォートを残してよい。
