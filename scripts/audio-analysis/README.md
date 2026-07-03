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

### transactions_triage.py
data/transactions の生録音バックログを sha256 + サンプル相関で dedupe し、unique
録音ごとに崩壊シグナル (events≤1 / 低密度 / 現行 recognizer との drift / 埋もれ
corrections / 低 peak / 人間判定済み) を集計して「非飽和 GT 候補になりそうな順」に
ランキングする (第 2 期 S1 の計器修理)。report-only — 回帰 gate や過適合ゲートの
n には使わない。出力 `data/triage_summary.json` は /debug/triage ページの供給源。

```bash
uv run python scripts/audio-analysis/transactions_triage.py [--no-retranscribe]
```

### note_f1_benchmark.py
free-performance 録音に対する layered benchmark。既存の note-level Precision / Recall / F1
を one-best onset-only 指標として維持しつつ、survey 反映の skeleton として
Candidate Recall@K、粗い Correction Burden、HardMissRate、ConfidenceCalibration 初期集計も出す。
fixture 回帰スイートが「楽譜との完全一致」を assert するのに対し、本ベンチマークは
人間検証済み ground truth (`ground_truth.json`, AGENTS.md スキーマ) との
onset 時刻 ± tolerance の note 単位マッチングで「実演奏にどれだけ近いか」を測る。
自由演奏転写の改善追跡用 (完全一致が定義できない録音向け)。
追加レイヤは analysis script 限定で、本番 response schema は広げない。benchmark 実行時のみ
debug 出力から `rankedCandidates` を拾い、通常 response の `alternateGroupings` / `candidateSlots`
(soft alternate / dropped candidate) と合わせて JSON に source counts と recall/cost を出す。

ground truth の置き場所:

- repo 管理: `apps/api/tests/fixtures/free-performance-corpus/<tx-id>/ground_truth.json`
  - `audio.wav` / `request.json` も同じ corpus item 内から読む
- local/dev only: `apps/api/tests/fixtures/transaction-captures/<tx-id>/ground_truth.json`
  - audio と tuning は `data/transactions/<tx-id>/` から読む

repo 管理 corpus への昇格には human rights/copyright review が必要。
詳細は `docs/corpus-management.md`。

```bash
uv run python scripts/audio-analysis/note_f1_benchmark.py              # GT のある全録音
uv run python scripts/audio-analysis/note_f1_benchmark.py <tx-id> --verbose  # FP/FN 明細
uv run python scripts/audio-analysis/note_f1_benchmark.py --json
```

### candidate_recall_benchmark.py
multi-candidate 出力の有効性を測る (#178 Phase 2)。note_f1_benchmark が primary 出力のみを
測るのに対し、本ツールは「primary が外した GT 音を、surfaced 候補 (event `alternateGroupings`
\+ `candidateSlots`) から 1 タップで復元できるか」=編集コスト削減効果と、候補レイヤが追加する
ノイズ (実音に対応しない候補=要却下) + confidence が real/noise を分離できているかを測る。
診断として `debug.segmentCandidates.rankedCandidates` 由来の ranked top-K recall も出す
(正解音が生 segment scoring で top-K に入るか。Phase 3 calibration ターゲット特定用)。
GT は note_f1_benchmark と同じ置き場所。現コーパスは primary recall ~1.0 のため recovery は
飽和、当面の signal は候補ノイズ率 + ranked 診断 (harder な #18 録音追加で recall@K が効く)。

```bash
uv run python scripts/audio-analysis/candidate_recall_benchmark.py            # GT のある全録音
uv run python scripts/audio-analysis/candidate_recall_benchmark.py <tx-id> --verbose
uv run python scripts/audio-analysis/candidate_recall_benchmark.py --json
```

### promote_corrections_to_ground_truth.py
review UI のユーザー修正 (`data/transactions/<tx-id>/corrections.json`) を
F1 ベンチマーク用 `ground_truth.json` に変換する。「テスター修正 = GT 収集装置」の
ループを閉じるツール。既存 GT (特に人間検証済み) は `--force` なしで上書きしない。
同一 audio SHA-256 の GT が他 tx にあれば二重カウント防止で skip。

review-status ゲート: 既定では `review_status.json` が `review_completed` の録音だけを
promote する (テスターが確認・修正を終えた録音のみを GT 候補にする)。
`recorded_only` / `rerecord_needed` / `unusable` / `uncertain` や status 未設定は
既定で skip される。`--require-status <status>` でゲートを変更、`--ignore-status` で
ゲートを無効化 (status ファイルの無い旧 corrections 用)。生成される GT の
`source.provenance` は `tester_corrected` で、人間検証 (ear/spectrogram) tier とは
区別される。

この script は `ground_truth.json` 生成用の staging tool。repo 管理 corpus に昇格する場合は、
human rights/copyright review 後に `apps/api/tests/fixtures/free-performance-corpus/<tx-id>/`
へ `audio.wav` / `request.json` / `ground_truth.json` / `metadata.json` などを揃えて追加する。

### review_priority_report.py
benchmark 指標 (note_f1_benchmark) と review status (`review_status.json`) を結合し、
「次に確認すべき録音」を優先度順に出す。優先度が高いのは、(a) まだ人手確認が必要な状態
(`recorded_only` / `review_started` / `uncertain` / 未設定) かつ (b) recognizer が苦戦
している (onset F1 が低い・HardMissRate が高い・Correction Burden が大きい) 録音。
`review_completed` / `unusable` / `rerecord_needed` は de-prioritize される
(確認しても次の改善に効きにくいため)。スコアリング (`compute_priority`) は IO を持たない
純関数で、`apps/api/tests/test_review_priority_report.py` で単体テストしている。

```bash
uv run python scripts/audio-analysis/review_priority_report.py
uv run python scripts/audio-analysis/review_priority_report.py --json
```

```bash
uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py          # 候補一覧
uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py <tx-id>  # promote
uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py --all --dry-run
```

### derive_ground_truth_from_score.py
score 既知の録音 (BWV147 playback 等) の GT を、対応 fixture の
`expectedEventNoteSetsOrdered` (score 真実) + recognizer timing から導出する。
厳密 1:1 alignment が成立しない場合は diff を表示して拒否 (人間検証へ回す)。
timing は recognizer 由来なので回帰追跡用 (`method: "score_aligned"`)。

```bash
uv run python scripts/audio-analysis/derive_ground_truth_from_score.py <tx-id> \
    --fixture kalimba-34l-c-bwv147-sequence-163-01 [--dry-run] [--force]
```

### calibrate_tuning_mismatch.py
`apps/api/app/transcription/tuning_check.py` (tuning mismatch 警告) の閾値較正。
テスターコーパス全録音に対して選択 tuning の pitch-class coverage と最良代替を表示する。
閾値を変更する際は必ずこれで分離を再確認すること。

```bash
uv run python scripts/audio-analysis/calibrate_tuning_mismatch.py
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
