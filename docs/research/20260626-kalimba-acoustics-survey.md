# Kalimba / Mbira 音響サーベイ補遺 (2026-06-26)

## 目的

`20260626-unbiased-amt-reassessment.md` のうち、カリンバ / mbira / lamellophone / idiophone
の物理音響に関する部分を、一次ソースに戻れる形で補強する。

この補遺の主眼は、**実装定数に焼き込んではいけない未検証数値を排除し、実測すべき量を明確化する**
ことである。特に per-tine partial / note-state / resonance handling の将来 spike で参照する。

---

## 1. 引用メタデータの修正と扱い

旧サーベイやエージェント調査過程で、巻号・ページ・タイトルに複数の誤りが見つかった。
以下は現時点での扱いを整理したもの。

### 確実に参照してよい一次ソース

1. **Chapman, "The tones of the kalimba (African thumb piano)," JASA 131(1): 945–950 (2012)**,
   DOI `10.1121/1.3651090`
   - 最重要。
   - 以前の「131, 645」はページ誤り。**945–950** が正しい。
   - tine を一端 clamped・他端 free・中間点 bridge supported に近い Euler–Bernoulli 梁として扱う。

2. **Chapman, "Characterizing the sound of an African thumb piano (kalimba)," JASA 123(5_Suppl): 3806 (2008)**,
   DOI `10.1121/1.2935513`
   - 会議要旨。巻号・ページは正しいが、査読フル論文ではない。
   - 測定された第1倍音比レンジの出典として扱えるが、実装定数化には不足。

3. **McNeil & Mitran, "Vibrational frequencies and tuning of the African mbira," JASA 123(2): 1169–1178 (2008)**,
   DOI `10.1121/1.2828063`
   - 正式題名はこれ。旧記述の "The acoustic spectrum of the mbira" は題名ではない。
   - mbira の倍音比・放射効率・FEM 解析を見る一次ソース。

4. **Ludwigsen, "Measurements of the acoustics of the kalimba," JASA 116(4_Suppl): 2593 (2004)**,
   DOI `10.1121/1.4785344`
   - 旧記述の「123, 3806 (2004)」は誤り。
   - スペクトル情報・モード解析・ギター / マレット楽器との比較に関する会議要旨。

5. **"Kalimba tine boundary condition models," JASA 153(3_Suppl): A227 (2023)**
   - 旧記述の「153, 1841」はページ誤り。**A227** が正しい。
   - 会議要旨。共鳴箱モード、bridge 点の機械インピーダンス、thin-beam を超える境界条件の手がかり。

6. **Waltham & Kotlicki, JASA 124(3): 1774–1780 (2008)**,
   DOI `10.1121/1.2956479`
   - mbira 関連一次ソースとして確認済み。ただし内容は未精査のため、設計根拠に使う前に読むこと。

### 要確認として残すもの

- **"Modes of the kalimba resonator box," JASA (Oct 2008)**
  - 共鳴箱モード / sympathetic resonance に関係する可能性が高いが、現時点で巻号・ページ未確認。
- **"Free vibration of a kalimba tine model beam with offset boundary condition"**
  - 題名は確認されているが、旧記述の「148, 2697 (2020)」は独立確認できていない。
  - 引用・実装根拠化の前に巻号・ページを確認する。

---

## 2. 実装に効く物理事実

### 2.1 非整数倍音は確実。ただし固定レンジを焼き込まない

確実に言えること:

- カリンバ / mbira の tine は梁振動由来で、整数 harmonic comb とは異なる非整数 partial を持つ。
- 倍音比は bridge / support 条件に依存し、tine ごとに変わる。
- したがって、**全キー共通の固定 partial 比をハードコードする設計は危険**。

現時点で採用してよい数値:

- Chapman 2008 要旨で確認できる第1倍音 / 基音比は、おおむね **5.3–5.9**。
- mbira では主要倍音がおおよそ基音の **5倍** と **14倍** 付近に現れる、という報告がある。
- 理想一様 clamped-free 梁の教科書的理論値として `f2/f1≈6.27`, `f3/f1≈17.5` は参考になるが、
  実カリンバは中間支持・bridge 条件により一致しない。

採用してはいけないもの:

- `f2/f1=5.3–6.8`
- `f3/f1=8.9–18.5`

これらは今回の確認では出典を確定できなかった。calibration prior や constants に焼き込まないこと。

### 2.2 横 (曲げ) モードが放射を支配する

実装上は、まず横モード由来のピーク追跡に集中してよい。
縦モードやねじれモードの厳密モデル化は、マイク録音からの採譜では投資対効果が低い。

### 2.3 アタック区間のピッチ推定を信用しすぎない

今回の補遺で最も重要な実装含意:

> **アタック区間には、薄板梁モデルが予測しない余分な spectral content が乗る。**
> そのため、アタック中は pitch 確定を急がず、onset evidence として扱い、BODY 区間で pitch を確定する設計が安全。

これは `OFF → ATTACK → BODY → LATE_DECAY` のような note-state model を将来検証する場合の、
最も強い物理的根拠である。

ただし、これは「即 note-state machine を main に入れるべき」という意味ではない。
現時点では、次のような低リスクな実装検証に落とす:

- onset 周辺 FFT と BODY 側 FFT の candidate rank 差を可視化する。
- weak attack / gap-rise rescue で、pitch 確定窓を attack 直後から少し後ろへずらす ablation を試す。
- Candidate Recall / Correction Burden で改善を測る。

### 2.4 sympathetic resonance の扱いは「設計推論」と明記する

一次ソースで確実に支えられるのは:

- bridge が振動を resonator box へ伝える。
- bridge 点の機械インピーダンスは複雑な境界条件になる。

一方で:

- 「sympathetic resonance は広帯域アタックスパイクを欠く」
- 「したがって onset 有無を note-on gate に使う」

これは物理的に整合するが、今回確認した一次ソースの直接記述ではない。
ドキュメント上は **設計推論** と明示する。

---

## 3. 実装へ直結しない注意点

- 縦・ねじれモードの厳密モデル化は当面不要。
- モード別減衰時間は一次ソースに測定値がないため、理論値で決めない。
- 伝統 mbira の微分音・just intonation 的チューニングは、現対象が市販 C major kalimba である限り表現層の問題。
- buzzer / rattle ノイズは高域非定常ノイズとして別扱いでよく、pitch recognizer の中心課題にしない。

---

## 4. fixture / corpus 収集に反映すべき観測項目

### 4.1 全キー単音プロファイル

各 tine の isolated single-note hit を全数録音し、少なくとも以下を測る:

- 基音周波数
- dominant partial の比 (`partial_freq / fundamental`)
- partial weight
- attack window と body window の rank 差

目的:

- `KALIMBA_DEFAULT_PARTIALS` を固定値ではなく、楽器・録音環境ごとの calibration に寄せる。
- #149 の「partial が隣接 tine 基音と衝突する」問題を、実データで分類する。

### 4.2 モード別減衰時間

各 tine で、attack 後に高次 partial が noise floor へ落ちるまでの時間を測る。
これは一次ソースに値がないため、`ATTACK/BODY/LATE_DECAY` の時間定数は自前 corpus で決める。

### 4.3 アタック過渡スペクトル

アタック中に薄板梁モデル外の spectral content がどの程度出るかを記録する。
pitch 確定を BODY まで遅延するかどうかの ablation 根拠に使う。

### 4.4 共鳴 / クロストークのみのサンプル

例:

- 1キーを強打して即ミュート
- 別キーの residual / box resonance だけを観測
- mute あり / なしで比較

目的:

- onset gate の FP 抑制根拠を自前データで測る。
- 「共鳴音はアタックスパイクを欠く」という設計推論を実測で検証する。

---

## 5. 現行方針への反映

この補遺は、以下の現在方針を補強する。

- per-tine partial は正しい方向だが、**既定化は ablation 後**。
- note-state machine は有望だが、**本線ではなく research spike**。
- attack evidence と pitch evidence は分ける。
- 将来の corpus では、自由演奏だけでなく **全キー単音 calibration / decay / resonance impulse** を必ず集める。

## 履歴

- 2026-06-26: 追加。楽器音響担当サーベイの引用検証・物理記述の修正・fixture 収集項目を反映。
