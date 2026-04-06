# カリンバ自動採譜向けアルゴリズム調査メモ

更新日: 2026-04-06

## 結論（先に要点）

カリンバ採譜では、**固定17音の辞書を持つテンプレート分解系**と、**onset を別系統で強く見る note-state モデル**の組み合わせが最も有望です。

現時点で有望度が高い順に並べると、次の通りです。

1. **半教師あり CNMF / 多テンプレート NMF・PLCA 系**
   - 単音サンプルを少量録るだけで始めやすい
   - 固定音高・固定音域・固定 timbre というカリンバの条件に合う
   - 録音ごとのテンプレート適応がしやすい
2. **attack / sustain / decay を分ける sound-state モデル**
   - カリンバの「鋭い attack + 長い減衰」に直接対応できる
   - sympathetic resonance と「新規 pluck」を分けやすい
3. **onset detector を使って note-on をゲートする構成**
   - 共鳴起因の ghost note を大きく減らせる可能性が高い
4. **HCQT + 軽量ニューラル（Basic Pitch 系、salience 系）**
   - 専用データが揃えば強い
   - ただし少量データ・録音条件差・可搬性では、最初の主力にはしにくい

要するに、**「多重ピッチ推定」だけで押し切るより、onset / state / decay を明示的に持つ方がカリンバには合う**、という整理です。

---

## 1. 音響的に何を前提にすべきか

カリンバ（lamellophone）の直接的な音響研究として重要なのは Chapman の `The tones of the kalimba` です。ここでは、

- 基音 `f1` に対する高次成分の比は**理想的な整数倍に固定されない**
- overtone 比は tine ごとに異なる
- 支持条件やブリッジ位置がスペクトル構造に効く

ことが示されています。また、lamellophone 系では、attack では非整数倍寄りの成分が目立ち、時間が経つとより純音的に見えやすくなる、という理解が妥当です。[Chapman2012]

この性質から、カリンバ採譜では以下が重要になります。

- **理想倍音列を前提にしすぎない**
- 1 note を **attack / sustain / decay** の複数状態で扱う
- note ごとに **独自の部分音表（partial table）** を持つ
- energy だけでなく **新規 attack の証拠** を別系統で持つ

---

## 2. 直接有望なアルゴリズム群

### 2.1 半教師あり CNMF / 多テンプレート NMF・PLCA

#### なぜ有望か

カリンバは

- 音高集合が固定（17キー）
- 音域が狭い
- 単音サンプルが取りやすい
- 録音ごとに timbre が少し変わる

ので、**固定辞書をもつ分解系**と非常に相性が良いです。

Wu らの半教師あり CNMF は、**各音の単独録音を 1 つずつ用意するだけで開始できる**構成です。[Wu2022]

Kirchhoff らの multi-template svNMD は、**1 pitch に複数テンプレート**を持たせることで、ダイナミクス差や time-varying spectral envelope を扱います。実験では、少なくとも上限性能の評価で、**1 template より 3 template の方が一貫して精度向上**していました。[Kirchhoff2012]

#### カリンバ向けの読み替え

- 各 tine に対して `attack`, `early sustain`, `late sustain/decay` の複数テンプレートを持つ
- さらに `hard pluck`, `soft pluck` のような強さ違いテンプレートを許す
- 初期辞書は単音録音から作り、本番録音内の高信頼 note で辞書を少し更新する

#### 特に効きそうな点

- **noise-floor FP** への耐性
- **コード**や**高速分散和音**での重なり処理
- **録音条件差**への追従

---

### 2.2 sound-state モデル（attack / sustain / decay）

Benetos & Weyde の temporally constrained probabilistic model、および Benetos & Dixon の shift-invariant model は、**sound state spectral templates** を用い、各 note を `attack / sustain / decay` のような状態遷移として扱います。[BenetosWeyde2015][BenetosDixon2013]

Cheng らの attack/decay model ではさらに、

- **attack component**
- **harmonic decay component**
- **指数減衰**
- **spike-shaped note activation**

を明示していて、これはカリンバの pluck 音にかなり近い発想です。[Cheng2016]

#### カリンバ向けの拡張案

piano 向け論文をそのまま使うのではなく、カリンバでは note state を例えば以下に分けるのが自然です。

1. `off`
2. `attack`
3. `body`
4. `late_decay`
5. `resonance_only`（任意）

この `resonance_only` は論文の既存 state ではなく、**sympathetic resonance を genuine note-on と分けるための設計上の追加案**です。

---

### 2.3 onset detection を別系統に立てる

Benetos & Stylianou は、auditory spectrum 上で spectral flux / group delay / F0 estimator を組み合わせる pitched-instrument onset detection を提案し、DFT ベース特徴より良い精度を報告しています。[BenetosStylianou2010]

Böck らは online onset detection で、

- **log magnitude**
- **adaptive whitening**
- 適切な peak-picking

が有効で、**online でも offline と同程度の性能帯まで持っていける**と報告しています。[Boeck2012]

Onsets and Frames は、framewise pitch 推定とは別に onset を予測し、**新規 note の開始は onset detector が同意した場合だけ許可する**構成です。[Hawthorne2017]

Cheuk らの再検討でも、**onset が最重要の attentive feature**であり、強い post-processing が重要だと示されています。[Cheuk2021]

#### カリンバ向けに何が効くか

これはほぼそのまま current issue に刺さります。

- sympathetic resonance による secondary activation が出ても、**attack の証拠がなければ note-on にしない**
- 既発音 note の残響は `active` のまま維持しつつ、新規開始だけを厳しくする
- spectral flux だけでなく、**局所 rise time / group delay / high-frequency novelty** を併用する

---

### 2.4 VQT/CQT と salience 系 multi-F0

Benetos & Weyde は、CQT より **VQT representation の方が lower pitch detection に改善**をもたらしたと報告しています。[BenetosWeyde2015]

Bittner らの deep salience model は、polyphonic multi-f0 に対する **salience representation** を学習します。[Bittner2017]

Spotify の lightweight instrument-agnostic model（Basic Pitch 論文）は、**frame-wise onset / multipitch / note posteriorgram を同時出力**し、さらに HCQT 的な harmonic 配置を扱う軽量設計です。[BittnerEtAl2022]

#### カリンバ向けの位置づけ

- 辞書系が先に立ち上がるなら、salience 系は**比較対象**または**後段候補**
- カリンバ単音から synthetic mixture を大量生成できるなら、HCQT 系 lightweight model は十分有望
- WebAudio / WASM を考えると、**小型モデル + ルールベース post-process** はかなり現実的

---

## 3. 今の課題に対して、どのアルゴリズムが効くか

| 現在の課題 | 今の特徴量だけでは弱い理由 | 推奨する主対策 | 補助対策 |
|---|---|---|---|
| noise-floor FP | flatness / BW90 / centroid は note の物理構造そのものを表しきれない | **辞書分解 + reconstruction residual** で genuine note らしさを見る | log-mag + adaptive whitening, boundary mask |
| sympathetic resonance による spurious secondary | energy はあるので閾値だけでは切りにくい | **onset gate** により新規 note-on を抑制 | resonance-only state, hysteresis |
| 先行音の残響 vs 新規 onset | sustain 中に partial が増減して紛らわしい | **attack state** を明示した note-state model | local group delay, high-band novelty |
| chord / polyphony | partial overlap で fundamental 単独判定が壊れやすい | **multi-template NMF / PLCA / svNMD** | pitch-wise HMM, sparse prior |
| マイク距離変化 | 絶対振幅ベースは壊れやすい | **IS divergence / scale-invariant な比較** | loudness normalization, whitening |
| streaming 化 | 双方向文脈に依存すると遅延が大きい | **online onset detector + causal dictionary decoder** | 小型 HCQT model |

---

## 4. 推奨パイプライン（実装向け）

以下は、現実に実装しやすく、かつ研究の当たりもよい構成です。

### Phase 1: front-end

- STFT と並行して **VQT か高解像度 CQT** を計算
- 振幅は `log(1 + a*x)` 系で圧縮
- 周波数方向に軽い whitening / equalization をかける

### Phase 2: calibration / template acquisition

各 tine について単音録音を集め、以下を学習します。

- partial frequency table
- partial amplitude envelope
- note-wise decay constant
- attack template
- body / late-decay template

ここで重要なのは、**partial を整数倍に丸めない**ことです。各 tine に固有の inharmonicity をそのまま持たせます。[Chapman2012]

### Phase 3: onset branch

onset detector は別系統で持ちます。

入力候補:

- auditory spectral flux
- log-mag spectral flux
- local group delay
- 高域 novelty
- pitch-band rise ratio

出力:

- `onset_prob[pitch, frame]`

### Phase 4: activation / decomposition branch

候補は 2 つです。

#### A. 第一推奨: semi-supervised CNMF / multi-template PLCA

- 各 pitch に複数テンプレート
- `attack`, `body`, `late_decay` を別テンプレート
- activation には sparsity と temporal smoothness を付与
- cost は **IS divergence** を優先して比較開始

#### B. 比較対象: HCQT + lightweight neural model

- 入力は HCQT
- 出力は `onset`, `frame-active`, `offset(optional)`
- 小型 CNN / CRNN で十分

### Phase 5: note-state decoding

pitch ごとに小さい状態機械を持ちます。

推奨状態:

- `OFF`
- `ATTACK`
- `BODY`
- `LATE_DECAY`
- `RESONANCE_ONLY`（任意）

遷移例:

- `OFF -> ATTACK` は onset が十分高いときだけ許可
- `BODY -> ATTACK` は refractory を短く設けたうえで再打鍵可能
- `LATE_DECAY -> ATTACK` は高めの onset を要求
- `BODY / LATE_DECAY -> RESONANCE_ONLY` は decomposition 上のエネルギーはあるが onset がないときに遷移

### Phase 6: post-processing

- boundary 近傍は **別ルール** を入れる
- note 長の下限を pitch 依存で置く
- 高音ほど decay が短いので、**pitch 依存 duration prior** を入れる
- 同一 pitch の再 onset は 30–60ms 程度の最小間隔を試す

---

## 5. まず試すべきアブレーション

優先順位順に並べます。

### 実験 1: 現行器に onset gate だけ足す

目的:

- sympathetic resonance 起因の false onset を減らせるか確認

比較:

- baseline
- baseline + spectral-flux onset gate
- baseline + spectral-flux + group-delay gate

### 実験 2: 単音録音から 17音辞書を作る

目的:

- 音高ごとの partial table と decay template の有効性確認

比較:

- harmonic comb ベース
- fixed learned template 1個/音
- multi-template 3個/音

### 実験 3: attack/body/late-decay の 3 state 化

目的:

- 残響中の note tracking と新規 onset 分離の改善確認

### 実験 4: CQT vs VQT

目的:

- 低音側の安定性と全体精度の差を見る

### 実験 5: IS divergence vs KL divergence

目的:

- gain 変動や録音距離差への頑健性を見る

---

## 6. 実装上の具体提案

### 6.1 いまの heuristics をどう活かすか

既存の

- spectral flatness
- centroid
- BW90
- sustain ratio
- onset energy gain

は捨てなくてよいですが、**主判定器ではなく補助特徴**に回すのがよいです。

おすすめは次の使い方です。

- `flatness`: noise / resonance-only の補助
- `centroid`: attack 強さの補助
- `BW90`: attack vs late decay の補助
- `sustain ratio`: offset 判定の補助
- `onset energy gain`: onset detector の一部

### 6.2 genuine note 判定の主役

主役にするのは以下です。

1. **テンプレート再構成誤差**
2. **note-state 遷移の整合性**
3. **onset の存在**

つまり「その帯域にエネルギーがあるか」ではなく、

- その音高の辞書でちゃんと説明できるか
- それは新規 attack なのか
- 状態遷移として自然か

で見る方がよいです。

---

## 7. ストリーミング / WebAudio / WASM の観点

Böck らの結果から、**online onset detection は十分現実的**です。[Boeck2012]

また、Benetos & Weyde の VQT/PLCA 系は、計算を工夫すれば高速で、従来系より高速化できることが示されています。[BenetosWeyde2015]

したがって streaming 版の現実的な道筋は以下です。

- onset は online spectral flux 系
- active note 推定は causal な template decoder
- post-process は短い遅延の state machine

将来的に neural を入れるとしても、まずは

- 小型 HCQT model
- onset / frame / note の multi-head
- 最後は rule-based decoding

の形がよいです。

---

## 8. 研究の追い方（継続調査の軸）

今後継続的に追うべき文献領域は以下です。

### A. Instrument-specific AMT

特に piano transcription のうち、以下のキーワードが重要です。

- `attack decay model transcription`
- `sound state templates transcription`
- `temporally constrained PLCA transcription`
- `semi-supervised CNMF transcription`
- `multiple templates per pitch transcription`

### B. Multi-F0 / salience estimation

- `HCQT multi-pitch estimation`
- `deep salience multi-f0`
- `polyphonic note transcription lightweight`

### C. Pitched onset detection

- `auditory spectral flux onset detection`
- `group delay onset pitched instruments`
- `online onset detection adaptive whitening`

### D. Lamellophone acoustics / idiophone acoustics

- `kalimba acoustics`
- `mbira overtone ratios`
- `lamellophone inharmonicity`
- `sympathetic resonance idiophone`

### E. Low-resource / self-supervised AMT

- `low-resource polyphonic transcription`
- `weakly supervised multi-pitch estimation`
- `few-shot transcription instrument specific`

---

## 9. 推奨ロードマップ

### 直近（最優先）

1. 単音録音から 17音テンプレート辞書を作る
2. onset branch を別立てする
3. note-on を onset gate で制御する
4. attack/body/late-decay の 3 state 化を試す

### 次点

5. multi-template 化（1音あたり 3 template）
6. IS divergence 導入
7. VQT front-end 比較

### その後

8. HCQT + lightweight neural model を比較導入
9. streaming 用 causal decoder を実装
10. resonance-only state の導入

---

## 10. 最終提案

現状の課題設定を見る限り、第一候補は次です。

> **VQT/CQT front-end + 半教師あり multi-template CNMF/PLCA + onset detector + pitch-wise note-state decoder**

この構成を推す理由は、

- カリンバの**固定音高**に強い
- **sympathetic resonance** を新規 onset と分けやすい
- **attack / decay** の物理に沿う
- 少量データで始めやすい
- streaming 化しやすい

からです。

ニューラル系は十分有望ですが、現段階では **比較対象・第2候補**として進めるのがよさそうです。

---

## 参考文献

- [Chapman2012] David M. F. Chapman, *The tones of the kalimba (African thumb piano)*, Journal of the Acoustical Society of America, 2012.
- [BenetosDixon2013] Emmanouil Benetos, Simon Dixon, *Multiple-instrument polyphonic music transcription using a temporally constrained shift-invariant model*, JASA, 2013.
- [BenetosWeyde2015] Emmanouil Benetos, Tillman Weyde, *An Efficient Temporally-Constrained Probabilistic Model for Multiple-Instrument Music Transcription*, ISMIR 2015.
- [Cheng2016] Tian Cheng, Matthias Mauch, Emmanouil Benetos, Simon Dixon, *An Attack/Decay Model for Piano Transcription*, ISMIR 2016.
- [Wu2022] Haoran Wu, Axel Marmoret, Jérémy E. Cohen, *Semi-Supervised Convolutive NMF for Automatic Piano Transcription*, 2022.
- [Kirchhoff2012] Henning Kirchhoff et al., *Multi-template Shift-Variant Non-Negative Matrix Deconvolution for Semi-Automatic Music Transcription*, ISMIR 2012.
- [BenetosStylianou2010] Emmanouil Benetos, Yannis Stylianou, *Auditory Spectrum-Based Pitched Instrument Onset Detection*, IEEE TASLP, 2010.
- [Boeck2012] Sebastian Böck, Florian Krebs, Markus Schedl, *Evaluating the Online Capabilities of Onset Detection Methods*, ISMIR 2012.
- [Hawthorne2017] Curtis Hawthorne et al., *Onsets and Frames: Dual-Objective Piano Transcription*, 2017.
- [Cheuk2021] Kin Wai Cheuk et al., *Revisiting the Onsets and Frames Model with Additive Attention*, 2021.
- [Bittner2017] Rachel M. Bittner et al., *Deep Salience Representations for F0 Estimation in Polyphonic Music*, ISMIR 2017.
- [BittnerEtAl2022] Rachel M. Bittner et al., *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation*, ICASSP 2022.

