@AGENTS.md

# Claude Code-Specific Overrides and Additions

- This file is maintained by Claude Code. Other agents should not update it.
- Rules shared with other agents belong in AGENTS.md, not here.

## Dev server startup

prod の API/Web/tunnel は systemd user service として常駐し、**127.0.0.1:8000 (api) と 127.0.0.1:3000 (web) を占有している**。dev で素の uvicorn を 8000 に立てると prod とポート競合で落ちるため、用途で使い分ける:

- **prod へコード反映**: prod は `~/kalimba-prod` worktree (detached HEAD) から serve される。**web は zero-downtime デプロイ**: `~/bin/deploy-web.sh` (systemd-socket-proxyd + blue-green standalone backend。`systemctl --user restart` 単独では反映されない)。**API は** `cd ~/kalimba-prod && git fetch origin && git checkout -f --detach origin/main` → `systemctl --user restart kalimba-api`。実値・詳細手順・ロールバック (L1-L3) は `.runtime-local/deploy.md` (host-local の single source) を読むこと。
- **dev / ブランチ検証で別インスタンスを立てる**: prod (8000/3000) と衝突しない **port 8001** で起動する。`--reload` 付きにすればファイル変更時に自動リロードされ、コード反映のたびにサーバーを kill/restart する必要がない:

```
uv run uvicorn app.main:app --app-dir apps/api --reload --host 0.0.0.0 --port 8001
```

prod と共有の `data/` を汚したくない実験では `KALIMBA_DATA_DIR` を一時ディレクトリに向ける。

バックグラウンドで起動した server の停止は、`kill` ではなく **`TaskStop` tool** を使うこと (run_in_background で起動したタスクの task_id を渡す)。`kill` は Claude Code がフリーズする原因になる場合がある。

## Tool 呼び出しの言語安全策 (Opus 4.8 の既知バグ対応、全モデルで無害)

Opus 4.8 には「**複数行の日本語 + コードを含む tool 引数**」で tool 呼び出しの直列化が壊れる既知バグがある (malformed tool_use / 沈黙 turn。GitHub issues #63604/#64658/#68510 等。4.7/Sonnet では非再現)。モデルを問わず次を守ること:

- **日本語の長文 (issue/PR 本文、コメント、レポート) を Bash 引数の heredoc に直接埋めない。** Write でファイルに書き、`gh issue comment --body-file <path>` / `--body-file` 系で渡す
- 日本語ドキュメントの大きな書き換えは、長い日本語 old_string/new_string の Edit 連打より、Write (全置換) か Python パッチスクリプト経由を優先する
- commit メッセージは従来どおり英語 (グローバル規約) — 変更不要
- 症状 (tool call parse 失敗・応答が捨てられ沈黙) が出たら: ユーザーが rewind → 正しい tool 形式で再試行を指示。**発生した事例は model-roles memory の実地観察に記録する** (頻発時は当該セッションのみ 4.7 へ一時切替が既知の回避策 — 恒久採用の変更ではない)

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
- `note_f1_benchmark.py` - free-performance 録音の note-level F1 評価（ground_truth.json ベース、自由演奏の改善追跡用）
- `score_alignment_diagnosis.py` - expected events (`score_structure.json` / `request.json:expectedPerformance` / `expected.json:expectedEventNoteSetsOrdered` の優先順で fallback) と recognizer 出力の整列・差分表示

### Fixture 影響評価の注意

規範は AGENTS.md「Test Architecture」参照 (実 pytest 必須、ad-hoc event count 比較禁止)。Claude では `/fixture-rejection-sweep` skill を使うのが最短。詳細は `scripts/audio-analysis/README.md`。

### score_alignment_diagnosis.py のキャッシュ挙動

`scripts/audio-analysis/score_alignment_diagnosis.py` のキャッシュキーには **recognizer source code (`apps/api/app/transcription/*.py`) の SHA256 fingerprint** が含まれる (`_recognizer_code_fingerprint()`)。コードを変更すると自動 invalidate されるので、`--no-cache` を常時付ける必要はない。iterative 修正 → diagnosis 走らせる workflow ではキャッシュが効いて時短になる。`--no-cache` はキャッシュ読み取りをスキップするが結果は書き込むので、次回は高速。データファイル変更やキャッシュ破損を疑う場合にのみ使う。SUMMARY 末尾の `Cache: hit/miss/fresh (recognizer: ...)` で結果の由来を確認可能。

## GitHub interaction conventions

AGENTS.md「GitHub Conventions」に移設済み (SHA バッククォート禁止、gh-attach-image.sh)。全 agent 共通規範のためここには再掲しない。
