# 位相追跡 onset 検出の ROC 較正 (第 3 期 S3)

- 実施: 2026-07-05。スクリプト: `scripts/audio-analysis/research/phase_tracking_roc.py`、生データ: [`phase-tracking-roc.json`](phase-tracking-roc.json)
- 基盤: きらきら星 2 本の ear_verified GT + spectral pin (90a8c0f、#201 v1 意味論)。pinned は ±60ms、unpinned は ±80ms で note-level greedy 1:1 マッチ
- 対象: S0 位相追跡プローブ (branch claude/s5-agenda4-bg-reattack-rescue, fcedeb3) の PHASE_BAR × JERK_BAR を 6×6 sweep

## 結果 (要点 4 つ)

### 1. 閾値では precision を買えない (プローブ結論の GT 定量化)

全 36 combo で **precision ≤ 0.29** (70cc6637 best: P=0.238 / R=0.766、47902d34 best: P=0.289 / R=0.771)。ROC 曲面は「recall ほぼ平坦 (0.70-0.77)・precision 低位固定」で、単発時点の 2 スカラー閾値では残 FP (鳴り続ける隣接 tine の skirt bleed) を分離できないことが GT で確認された。cross-tine explaining-away (= tracker 本体) が必要という S0 結論はそのまま。

### 2. 閾値を緩めると recall が**下がる** (非単調性 — 単発時点 guard の構造的害)

pooled で loosest (0.3/20) R=0.704 < tightest 側 (0.7/150) R=0.772。緩めるほど spurious イベントが増え、winner-take-all の cross-tine guard (dominance / harmonic-parent) が真イベントを殺すため。**「候補を増やして後段で選ぶ」が単発時点排他では成立しない**ことを示す — 状態ベースの explaining-away に置き換える動機の独立した証拠。

### 3. FN の構造: 倍音一致メロディ音の系統的抑圧 (実測 partial table が修正手段)

best combo の FN (70cc6637: 33 / 47902d34: 11) をピン状態とクロス集計すると:

- **FN の大半は pinned (綺麗な band rise がある音)**: 25/33、10/11
- 内訳は C5×12・G5×9・F5×7 など**メロディ上音に集中**。きらきら星は C4+C5 オクターブ重ねと G4 伴奏が常時鳴る構造のため、integer 倍音 (m=2,3,4 ±50c) 前提の harmonic-parent guard が「C4 の 2 倍音 = C5」「C4 の 3 倍音 ≈ G5」として真のメロディ打鍵を捨てている
- しかし実測 partial table (2075529) によれば **カリンバの第 2 モードは ×2.8-3.0 であり ×2.0 の partial はそもそも存在しない**。さらに実測 bleed 振幅は C4→G5 で median 0.002 と微小。integer 倍音 guard は前提が物理的に誤っており、**実測 partial 比 + 期待 bleed 振幅による定量 explaining-away に置換すべき** — tracker 観測モデルの設計入力として最重要

### 4. carryover-mask 型は既に回収できている (K2 最低獲物の到達可能性)

spectral pin で unpinnable (帯域 rise が 6dB 未満 = 鳴り残り中の再打鍵) と判定された 38 音のうち、**位相追跡は 30 音 (79%) を検出**。FN はむしろ pinned 側に偏る。単発時点検出が構造的に取れない領域を位相追跡が取れていることの直接証拠で、kill 条件 K2 (carryover-mask 回収) の到達可能性を支持する。

## S4 判定材料への含意

- GO 側の観測モデル設計は「位相 RMS + jerk (検出)」×「実測 partial 比 + 減衰状態 (explaining-away)」の 2 層が明確になった。integer 倍音 guard と winner-take-all 排他は tracker では使わない
- 較正済み動作点 (単発時点での上限): pooled F1 0.375 @ PHASE=0.7/JERK=150。tracker はこの上限を precision 側で大きく超える必要がある (recall は 0.77 が単発上限)
- ユーザー仮説 (2026-07-05): マイク距離揺らぎによる音量変動への頑健性は位相系の期待利点 (440Hz で 1cm ≈ 0.08 rad、PHASE_BAR 0.7 に対し余裕大)。slow AM メタモルフィックテストで振幅系と比較検証する (統合設計文書に記載)
