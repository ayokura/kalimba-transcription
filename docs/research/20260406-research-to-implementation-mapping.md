# 研究サーベイ → 現行パイプライン適用マッピング

更新日: 2026-04-06

> ## ⚠️ バイアス警告 (2026-06-26 追記)
>
> 本ドキュメントおよびソースの `20260406-deep-research-report.md` /
> `20260406-kalimba_amt_survey.md` は **LLM 調査レポートを一次ソースとした二次資料**であり、
> 以下のバイアス・品質上の問題が確認された。設計判断の根拠に使う前に必ず留意すること。
>
> 1. **一次文献の未検証**: 引用 (`[Chapman2012]`, `[Hawthorne2017]` 等) は LLM 出力をそのまま
>    取り込んでおり、原典に当たって検証した形跡がない。`deep-research-report.md` には
>    `citeturn..search` 形式の **LLM 検索トークンが約 60 箇所、未編集で残存**している
>    (= 人手編集・検証を経ていない物的証拠)。
> 2. **「インパクト: 大/中」ランクが未検証**: 各提案の影響度は ablation 等の実測ではなく
>    レポート著者 (LLM) の主観。現コードベースでの効果は測られていない。
> 3. **実装現状とのズレ**: 当時の「現行」記述は陳腐化している。例: 項目2「onset gate は
>    未実装」→ 現在は `use_onset_gate=True` で**実装済み (ただし broadband/per-note/
>    backward-attack の 3 証拠が全滅した時だけ棄却する弱 AND)**。項目1「整数 comb のみ」→
>    per-tine partial scoring は実装済みだが `use_per_tine_partial_scoring=False` で
>    既定無効 (#149 の tine 衝突問題のため)。
> 4. **NN 系の一律退け**: 「大規模 NN は不整合」と結論しているが、Basic Pitch / PESTO の
>    ような軽量・instrument-agnostic・streamable 路線を teacher / baseline として使う選択肢を
>    検討していない。
>
> **バイアスを除去して再収集・再評価した結果は
> [`20260626-unbiased-amt-reassessment.md`](./20260626-unbiased-amt-reassessment.md) を参照。**
> 本ファイルは経緯保存のため残すが、新規の設計判断は再評価版を優先すること。

## 概要

2つの LLM 調査レポートの知見を現行 recognizer パイプラインと照合し、
生かせるポイントを影響度順に整理したもの。

ソースレポート:
- [20260406-deep-research-report.md](./20260406-deep-research-report.md) — 横断的サーベイ
- [20260406-kalimba_amt_survey.md](./20260406-kalimba_amt_survey.md) — 実装向け調査メモ

## 現行パイプラインが既に整合している点

| レポートの推奨 | 現行の実装 |
|---|---|
| onset を別系統で検証する | attack profile (kurtosis, crest factor, spectral flux) で onset の質を評価 |
| 再打鍵の物理モデル | mute-dip 検出 (per_note.py) |
| residual decay 判定 | onset gain による直近ノート残響の区別 |
| octave alias 抑制 | subharmonic エネルギーによる減点 |
| 既存 heuristics は補助特徴に | flatness/centroid/BW90 は実際に補助的に使用されている |

## 生かせるポイント

### 1. 倍音比の非整数化 (インパクト: 大)

**現行**: `rank_tuning_candidates()` が整数倍 (2x, 3x, 4x) の harmonic comb でスコアリング。

**研究知見**: カリンバの倍音は梁振動由来で非整数比。tine ごとに異なる。
整数倍を前提にすると本来の倍音を拾えず、他ノートの整数倍を誤って拾うリスクがある。
[Chapman2012]

**提案**: 各 tine の実測 partial table (17音分) を用意し、
`HARMONIC_WEIGHTS` の周波数位置を tine ごとの実測比に変える。
キャリブレーション録音からの自動抽出も可能。

**関連 issue**: #149 (per-tine partial table), #111 (peaks redesign), #78 (algorithm exploration)

### 2. onset gate の強化 — "onset がなければ note-on にしない" (インパクト: 大)

**現行**: attack profile は onset 時刻調整や gap onset フィルタに使用されるが、
`_resolve_primary()` での候補選択を onset 有無でゲートする設計ではない。

**研究知見**: Onsets and Frames 系の知見として、onset detector が同意しない限り
新規 note-on を許可しない構成が、共鳴起因 ghost note 抑制に最も効果的。
[Hawthorne2017, Cheuk2021]

**提案**: `_resolve_primary()` に onset confidence を渡し、閾値未満なら棄却
または `RESONANCE_ONLY` フラグを付与する。
現行の broadband gain + high-band flux はそのまま onset confidence の構成要素として使える。

**関連 issue**: #141 (per-note onset detection umbrella) — この方向性を既に含む

> **(2026-04-16 更新)** PR #186 で gap-rise rescue が実装され、per-note onset detection の gap rescue 側 (mute-dip + gap-rise) が一段進展した。ただしこれは gap 内の rescue であり、本項で述べている onset gate (「onset detector が同意しない限り新規 note-on を許可しない」) は #141 §2 として未実装のまま。onset gate は `_resolve_primary()` の候補選択をゲートする設計であり、gap-rise rescue とは異なるレイヤーの改善。

### 3. band-segmented spectral flatness (インパクト: 中)

**現行**: BW90, centroid, flatness を全帯域で計算。倍音豊富なカリンバ音で
genuine note も広帯域に散る問題がある。

**研究知見**: 帯域セグメントごとの flatness を使う改良が提案されている。
ゲイン不変 (幾何平均/算術平均比) で録音距離変化にも強い。
周波数セグメント化により「倍音で広帯域」問題を緩和。
[SFM 定義, Peeters 2004 系]

**提案**: flatness を「基音帯域」「倍音帯域」「高域ノイズ帯域」の3区間で算出。
genuine note は基音帯域で低 flatness、ノイズは全帯域で高 flatness。

**関連 issue**: #150 (band-segmented flatness), #96 (broadband vs per-note 分離), #34 (is_valid_attack flux 移行)

### 4. attack / body / late_decay の note-state 明示化 (インパクト: 中)

**現行**: residual decay 判定はあるが、ノートごとの状態遷移モデルは暗黙的。

**研究知見**: `OFF → ATTACK → BODY → LATE_DECAY (→ RESONANCE_ONLY)` の
状態機械を pitch ごとに持ち、遷移条件で onset 要求度を変える。
[BenetosWeyde2015, Cheng2016]

**提案**: Note dataclass (#142 で導入済み) を拡張して state を持たせ、
セグメントごとの判定で状態遷移ルールを適用。
`recent_note_names` による追跡ロジックを自然に吸収できる。

**関連 issue**: #141 (sub-issue として追加可能), #142 (Note dataclass)

### 5. HPSS による onset/pitch の入力分離 (インパクト: 中, 実装容易)

**現行**: 同一の FFT 結果から onset 検出と pitch 推定の両方を行っている。

**研究知見**: HPSS で percussive 成分 (onset 検出用) と harmonic 成分 (pitch 推定用) を分離。
カリンバの attack は percussive 寄りなので onset 検出の精度向上が期待できる。
ただし撥弦 attack が percussive 側に回る副作用には注意。
[Fitzgerald 2010, Driedger+ 2014]

**提案**: `librosa.effects.hpss()` で分離し、onset detection を percussive 成分に、
pitch 推定を harmonic 成分に分ける。1行追加レベルで試験可能。

**関連 issue**: #148 (HPSS 試験)

### 6. CQT/VQT front-end (インパクト: 大, 中期的)

**現行**: FFT (STFT) ベース。`_adaptive_n_fft()` で周波数解像度を確保。

**研究知見**: CQT/VQT は音楽的対数周波数スケールで、低音の周波数解像度と
高音の時間解像度を自然に両立。17音固定なら resonator filterbank も選択肢。
VQT は低音側でさらに改善。WebAudio/WASM への移植性も良い。
[BenetosWeyde2015, Brown 1991]

**提案**: CQT を並行計算して候補スコアリングの一つとして試す。
全面置換ではなく dual-scoring で比較する形が安全。

**関連 issue**: #78 (algorithm exploration)

## 現行で不要な変更

| レポートの推奨 | 現行で十分な理由 |
|---|---|
| flatness/centroid/BW90 を補助に回す | 既にそう使われている |
| IS divergence 導入 | 現時点ではマイク距離変化が主要ボトルネックではない |
| 大規模 NN モデル (MT3 等) | ストリーミング/WASM 制約と不整合 |

## 推奨ロードマップ

### 短期 (既存 issue の延長で対応可能)

1. onset gate 強化 → #141 の一部として自然に実装
2. note-state 明示化 → #142 Note dataclass 拡張
3. HPSS 試験 → #148

### 中期

4. band-segmented flatness → #150
5. per-tine partial table → #149

### 長期

6. CQT/VQT front-end → #78 の一部として検討
