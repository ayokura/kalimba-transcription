# AMT 研究 再評価 — バイアス除去版 (2026-06-26)

## このドキュメントの目的

`20260406-*` の研究サーベイ群は **LLM 調査レポートを一次ソースとした二次資料**で、
一次文献の未検証・影響度ランクの主観性・実装現状とのズレといったバイアスを含んでいた
(詳細は [`20260406-research-to-implementation-mapping.md`](./20260406-research-to-implementation-mapping.md)
冒頭のバイアス警告を参照)。

本ドキュメントは、そのバイアスを意識的に外したうえで、(1) 現実装の**コードで確認した事実**、
(2) 最新研究・隣接手法、(3) 両者を突き合わせた進め方、を再整理する。

**前提となる方法論的姿勢:**
- 「研究的に美しい」ことを採用根拠にしない。**ablation で測ってから採用**する。
- 「物理的に正しい」ことも単独では採用根拠にしない (例: 非整数 partial)。実装上の副作用を測る。
- NN を「置換」候補にしない代わりに「teacher / baseline / candidate generator」として現実的に位置づける。
- fixture の成功と free-performance readiness を**別レイヤ**として扱う。

---

## 1. 現実装の事実 (2026-06-26 コード確認)

| 項目 | 事実 | 根拠 |
|---|---|---|
| ピッチ同定 | 本番は整数倍 comb (note×1..4)。`MAX_HARMONIC_MULTIPLE=4` | `peaks.py rank_tuning_candidates`, `constants.py` |
| per-tine partial | 定義済み (`KALIMBA_DEFAULT_PARTIALS`, 1.5× 等) だが scoring は既定無効 | `settings.use_per_tine_partial_scoring=False` |
| partial の実用途 | 正の scoring ではなく `suppress_harmonics` の抑制位置にのみ使用 | `peaks.py:78-175` |
| onset gate | **実装済み・既定有効**。ただし broadband(<2.0)/per-note(<1.8)/backward-attack(<20.0) が**全滅した時だけ棄却**する弱 AND | `settings.use_onset_gate=True`, `peaks.py` onset-gate ブロック |
| confirmed_primary | mute-dip / gap-rise rescue で挿入された segment は gate を skip | `per_note.py`, `_resolve_confirmed_primary` |
| note-state machine | OFF→ATTACK→BODY→LATE_DECAY の明示状態機械は**不在**。segment 単位 FFT + evidence 関数の二値/連続判定 | コード全走査で該当なし |
| post-processing | events.py 約27パス。**ほぼ局所/因果的** (隣接1〜2 event 参照)。コーパス全体書き換えは patterns.py の repeated triad / four-note の2つのみ | `events.py`, `patterns.py` |
| multi-candidate | `alternateGroupings` は通常 response に出る。`rankedCandidates`/`secondaryDecisionTrail`/`residualCandidates` は **debug 限定** | `pipeline.py`, `app/models.py` |
| fixture | 35 total / 33 completed / 1 pending / 1 rerecord (実測) | `expected.json` 集計 |
| 評価 | API テスト 501 pass (xdist 20 並列で ~37s)。F1 ベンチは既知フレーズで飽和傾向 | `pytest.ini`, `note_f1_benchmark.py` |

**旧サーベイとの主要な差分 (=旧サーベイのバイアス):**
- 旧: 「onset gate 未実装」→ 実: 実装済み (弱 AND)。
- 旧: 「整数 comb のみ」→ 実: 非整数 partial は実装済みだが既定無効 (#149 衝突)。
- → つまり旧サーベイの「インパクト: 大」上位2項目は、**すでに着手済み or 意図的に無効化済み**。
  未着手の新規大改修として扱うのは誤り。

---

## 2. バイアスを外した最新研究の読み直し

出典は手法名レベルで示す。各項目は「研究の示唆 / このプロジェクトでの扱い」で記述。

### 2.1 カリンバ/mbira の物理 (非整数倍音)
- **示唆**: tine は梁振動由来で倍音が非整数比、tine ごと・bridge 位置依存。整数 harmonicity/comb を
  そのまま使うと破綻しやすい (Chapman 系の kalimba 音響、mbira の overtone 報告)。
- **扱い**: 方向としては正しいが、現実装で per-tine partial を既定無効にしている理由 (#149: 非整数
  partial が隣接 tine の基音と衝突し legitimate chord を誤罰) こそが本丸。**「整数 comb を捨てる」
  ではなく「partial の帰属問題を解く」**問題として再定義する。整数 comb は実用近似として残し、
  partial は追加 evidence / 占有ベース帰属として ablation する。
- **補遺**: 巻号・ページ・タイトルの検証、未確認の倍音比レンジ、attack 区間で pitch 確定を急がない
  実装含意、fixture/corpus 収集項目は
  [`20260626-kalimba-acoustics-survey.md`](./20260626-kalimba-acoustics-survey.md) を優先参照する。

### 2.2 軽量・instrument-agnostic AMT (Basic Pitch)
- **示唆**: Spotify Basic Pitch は polyphonic・instrument-agnostic・ブラウザ実行可・pitch bend 対応。
  出力は onset/note/contour の 3 ヘッドで、後段で note event へ整形する設計。「一度に 1 楽器」が得意。
- **扱い**: **置換ではなく** (a) free-performance baseline、(b) candidate generator、(c) browser-side
  比較対象、(d) pitch-bend/contour teacher として使う。出力構造が本プロジェクトの
  rankedCandidates/alternateGroupings + onset evidence と近く、接続コストが低い。
- **補遺**: Basic Pitch TS / PESTO / ONNX Runtime Web / AudioWorklet / WASM 分担は
  [`20260626-browser-realtime-implementation-survey.md`](./20260626-browser-realtime-implementation-survey.md) を参照。

### 2.3 自己教師あり・低レイテンシ pitch (PESTO)
- **示唆**: VQT frame 入力の自己教師あり pitch 推定。~13万 params、cross-dataset 汎化、streamable VQT、
  低レイテンシ。「raw compute が速いだけでは不十分で causal に逐次処理できる必要」と明示。低音側 VQT
  window が latency 主因。基本 monophonic。
- **扱い**: STFT 整数 comb を即 VQT 置換しない。research line で (a) log-frequency front-end の比較、
  (b) attack 周辺の dominant-pitch evidence、(c) streaming/causal 設計の参考、として検証。

### 2.4 大規模 token-based AMT (MT3 / YourMT3+ / hFT-Transformer)
- **示唆**: SOTA は token seq2seq。低リソース前提で複数データセット/楽器を扱う方向 (MT3)、
  hierarchical attention・MoE・cross-dataset stem augmentation でデータ不足に対応 (YourMT3+)。
  典型失敗は instrument leakage / polyphonic confusion / hallucination。近年の challenge でも
  polyphony と timbre variation は難点として残存。
- **扱い**: カリンバ単体・ブラウザ・streaming・説明可能性という本プロジェクト制約とはズレる。
  **本体置換にはしない**。ただし offline teacher / baseline / corpus 作成補助としては有用。
  旧サーベイの「一律退け」も、この「teacher として使う」可能性を見落としていた点でバイアス。

### 2.5 AMT 評価と corpus bias
- **示唆**: deep AMT は特定データセット (piano/MAESTRO 的) に偏りやすく、sound/genre/polyphony の
  distribution shift で note-level F1 が大きく落ちる。precision/recall だけでなく timing nuance /
  articulation / dynamics など musically-informed metrics の重要性も指摘されている。
- **扱い**: 本プロジェクトでも fixture exact-match の成功は free-performance の成功を保証しない。
  評価を**多層**に分ける (下記 §3.2)。
- **補遺**: mir_eval / MIREX / AMT Challenge / onset annotation と、Candidate Recall /
  Correction Burden の具体設計は
  [`20260626-amt-evaluation-survey.md`](./20260626-amt-evaluation-survey.md) を参照。

---

## 3. あらためての進め方

### 3.1 結論

> **認識器の大改造ではなく、自由演奏評価ループ + multi-candidate / correction UX を先に固める。**

- note-state machine 化は有望だが **今の本線ではない** (research spike として検証)。
- per-tine partial の既定化も **今の本線ではない** (ablation で勝ってから)。
- Basic Pitch / PESTO / MT3 系は **置換せず**、外部比較・teacher・candidate generator として使う。
- まず「自由演奏で何がどれだけ困るか」を**測れる状態**を作る。

### 3.2 評価指標を多層化する (最優先)

現状 F1=1.000 飽和は「成功」ではなく「次の指標が要る」サイン。以下を別レイヤで持つ:

1. **fixture exact-match** (既存。リファクタの命綱として維持)
2. **note-level F1** (既存ベンチ。OOD 録音で別途測る)
3. **Candidate Recall@K** — 正解 note が 1-best ∪ alternates ∪ dropped candidates の Top-K に入る率
4. **Correction Burden** — 予測 event 列 → ground_truth event 列の編集距離 (ユーザー修正手数の代理)
5. **Confidence calibration** — 低 confidence とした箇所が実際に誤りやすいか
6. **Drop-to-candidate rate** — 従来捨てていた音を候補として残せている率

> Candidate Recall は候補乱発で水増しできるため、必ず Correction Burden と**対で**監視する。

### 3.3 ロードマップ (now / next / later)

各項目に**測定可能な exit criteria** を付す。日付ではなく条件で進める。

#### NOW — 測れる状態を作る (低リスク)
- docs 現状化 (本コミットで roadmap / research-mapping は対応済み。free-performance-readiness も追随)。
- `note_f1_benchmark.py` に Candidate Recall + Correction Burden を追加し CI 配線。
- onset gate の棄却を「drop」ではなく「低 confidence の candidate slot へ降格」に統一。
- **exit:** 自由演奏 GT 20件以上で「F1 / Candidate Recall / Correction Burden」が自動計測され
  ベースライン確立。

#### NEXT — 候補保持の作り込み + 物理モデルは検証のみ
- #178: debug 限定の rankedCandidates / dropped segment を**本出力 + review UI** に昇格。
- #18: 自由演奏 corpus を増やす (mixed / strict / slide / arpeggio-like、録音環境を変える、
  `ground_truth.json` の `method` を明記)。
- per-tine partial / note-state machine は **research branch の spike** (dual-run、main 投入は判断保留)。
- **exit:** 境界事例で「safe drop ではなく候補保持」が回り、Candidate Recall がベースライン超え。
  spike は exact-match 非劣化 + 自由演奏指標改善 + suppression pass 削減を同時に満たした時だけ merge 検討。

#### LATER — Causal/WASM 整合 + teacher 配線
- `peak_pick + backtrack` の lookahead を有界固定遅延 (≤50ms) の causal onset に置換、Rust も同型化。
- log-frequency (CQT/VQT) front-end を dual-scoring で比較開始。
- Basic Pitch / PESTO を **offline teacher** として GT ブートストラップに配線
  (teacher 由来 GT は `method: "model_suggested"` 等で厳格区別、人手 verify 前に completed 昇格しない)。
- **exit:** ブラウザ (WebAudio+WASM) で固定遅延発火、Python バッチ版との F1 誤差 ±2% 以内。

#### UX / correction workflow
- 自動採譜 product の correction workflow と Correction Burden の UI 操作単位は
  [`20260626-transcription-product-ux-survey.md`](./20260626-transcription-product-ux-survey.md) を参照。

### 3.4 やらない方がよいこと
- events.py に新規 suppression pass を追加し続ける (限界収益が低下、自由演奏で破綻しやすい)。
- per-tine partial を信念で既定化する (#149 衝突。ablation で勝ってから)。
- MT3 / YourMT3+ を本体置換候補にする (制約とズレ。teacher なら可)。
- F1=1.000 を成功指標として追い続ける。

---

## 4. 一次確認した参考ソース

旧サーベイの問題は「URLがあること」ではなく、**LLM検索トークンが未編集で残り、一次文献に
戻れる形になっていなかったこと**である。今後の設計判断では、少なくとも以下の安定した論文ID /
DOI / 公式実装へ戻って確認する。

- Kalimba / mbira / lamellophone 音響:
  - 詳細補遺: [`20260626-kalimba-acoustics-survey.md`](./20260626-kalimba-acoustics-survey.md)
  - Chapman, *The tones of the kalimba (African thumb piano)*, JASA 131(1): 945–950 (2012), DOI: `10.1121/1.3651090`
  - McNeil & Mitran, *Vibrational frequencies and tuning of the African mbira*, JASA 123(2): 1169–1178 (2008), DOI: `10.1121/1.2828063`
- Spotify Basic Pitch:
  - Paper: *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation*, arXiv: `2203.09893`
  - TypeScript/browser implementation: `github.com/spotify/basic-pitch-ts`
  - Note creation / onset+frame decoding reference: `github.com/spotify/basic-pitch`
- Browser / realtime implementation:
  - 詳細補遺: [`20260626-browser-realtime-implementation-survey.md`](./20260626-browser-realtime-implementation-survey.md)
  - ONNX Runtime Web
  - WebAudio AudioWorklet
  - WASM / Worker / ring buffer patterns
- Product UX / correction workflow:
  - 詳細補遺: [`20260626-transcription-product-ux-survey.md`](./20260626-transcription-product-ux-survey.md)
- MT3: *Multi-Task Multitrack Music Transcription*, arXiv: `2111.03017`
- YourMT3+: *Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation*, arXiv: `2407.04822`
- hFT-Transformer: *Automatic Piano Transcription with Hierarchical Frequency-Time Transformer*, arXiv: `2307.04305`
- PESTO:
  - ISMIR 2023 / arXiv: `2309.02265`
  - Real-time streamable VQT version: arXiv: `2508.01488`
- 2025 AMT Challenge: *Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge*, arXiv: `2603.27528`
- AMT evaluation / benchmark:
  - 詳細補遺: [`20260626-amt-evaluation-survey.md`](./20260626-amt-evaluation-survey.md)
  - `mir_eval.transcription`
  - MIREX Multiple Fundamental Frequency Estimation & Tracking
- AMT corpus bias / OOD evaluation:
  - Marták, Hu, Widmer, *Sound and music biases in deep music transcription models: a systematic analysis*, Journal on Audio, Speech, and Music Processing 2026, DOI: `10.1186/s13636-025-00428-z`

長い引用は置かない。必要な設計判断ごとに、上記原典の該当箇所へ当たり、実装側の ablation とセットで判断する。

## 履歴
- 2026-06-26: 新規作成。`20260406-*` のバイアス除去・現実装事実との再突合・最新研究 (Basic Pitch /
  PESTO / MT3 系 / AMT bias) の再収集を反映。
