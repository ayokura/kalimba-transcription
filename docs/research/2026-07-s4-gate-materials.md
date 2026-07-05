# S4 実装ゲート判定資料 — 反証 3 系統の統合 (第 3 期)

- 作成: 2026-07-05 (S4)。判定対象: **per-tine tracker 実装本体 (S5) の GO/NO-GO** (ユーザー判断、ガードレール 11)
- kill/GO 条件: [`2026-07-per-tine-kill-criteria.md`](2026-07-per-tine-kill-criteria.md)。**条件の種類・閾値は S0 固定のまま、判定分母は同文書の S4 基準値追記 (n=9、29eaaf1daa45ca68) が正** — K2 = 10/16→14/16、K3 = R≥0.741 + CI 条件、K4 = FP≤61
- 本資料は GO 支持材料と反対材料 (counter-evidence) を併記する (確証バイアス防止、第 3 期転換 4)。別系モデルの敵対的監査を経てからユーザーが判定する
- 測定条件: recognizer 29eaaf1daa45ca68 / 596 tests green / GT レビュー済み録音 15 本 (非飽和 9)

## (a) 反証系 1: 録音多様化

**結果: per-tine の前提を崩す録音は出現していない。under-detection 構造は多様化後も維持。**

- corpus は S1 で repo 3→7、GT 済み全 15 本 (非飽和 9)。テスター録音 (他者楽器・iPhone 収録) は計 7 本と確定 (2026-07-05 ユーザー情報)、うち 5 本 GT 化済み、2 本 (1955b5bd/98019f67) ドラフト裁定待ち
- **headline: 非飽和 (n=9) micro F1=0.784 CI95=[0.666, 0.849]** (micro P=0.852 / R=0.726)

| tx | truth | tp | FP | FN | P | R | F1 | 性格 |
|---|---|---|---|---|---|---|---|---|
| a9e30986 | 81 | 36 | 20 | 45 | 0.643 | 0.444 | 0.526 | チューリップ (多声・最難) |
| 4e1ae5c6 | 8 | 5 | 0 | 3 | 1.000 | 0.625 | 0.769 | carryover-mask 敵対テイク |
| 9ce7df83 | 8 | 5 | 0 | 3 | 1.000 | 0.625 | 0.769 | 同上 |
| 47902d34 | 48 | 36 | 4 | 12 | 0.900 | 0.750 | 0.818 | きらきら星・高速 |
| 70cc6637 | 141 | 111 | 14 | 30 | 0.888 | 0.787 | 0.835 | きらきら星・標準 (多声) |
| d7a82772 | 45 | 36 | 5 | 9 | 0.878 | 0.800 | 0.837 | First Noel |
| ea7edd71 | 18 | 16 | 3 | 2 | 0.842 | 0.889 | 0.865 | 敵対テイク |
| 17ea7626 | 52 | 42 | 3 | 10 | 0.933 | 0.808 | 0.866 | 初の非飽和テスター録音 |
| ebecf0c6 | 18 | 17 | 4 | 1 | 0.810 | 0.944 | 0.872 | 敵対テイク |

- **弱点分布 (非飽和 FN=115)**: C5 23 / E4 16 / G4 15 / C4 14 / G5 13 — 伴奏反復 tine (G4)・オクターブ/倍音一致のメロディ音 (C5/G5)・中低域に集中。carryover-mask と倍音抑圧の 2 型が支配的 (ROC の FN 構造分析と整合)

## (b) 反証系 2: 用途検証

**結果: 「精度が律速」(R1 成立) — ただし n=1 のため provisional (監査指摘 6)。ガードレール 14 (粗い転写で足りる) は非発火。**

- dogfooding 第 1 回 (1307ad97、S2): 修正必要音率 22.2% > 10% で R1 成立。A1/A3/A4/A5 成立・A2 (修正率 ≤5%) のみ不成立。弾き戻し 2/2 成功
- **G2 ✓ は provisional 扱いとする**: n=1・18 音中 touched 4 音の実データであり、確定には長め・別難度の dogfooding 1-2 本の追加が必要 (人間アクション)。判定様式自体は S2 前に事前定義済み (事後解釈の混入なし)

## (c) 反証系 3: GT 除染後の recall 盲点実態

**結果: 除染後も FN=115 > FP=53 (G3 充足)。盲点は「単発時点検出の構造的死角」に集中。**

- bp-only 23 件の人手 verify → GT 統合済み (S2)。それでも under-detection が主要因子
- きらきら星 ear_verified GT + spectral pin により盲点の物理構造が判明: 70cc6637 の GT 141 音中 38 音 (27%) は帯域 rise 6dB 未満の unpinnable (= 鳴り残り中の再打鍵) で、単発時点検出が原理的に苦手な領域
- **現行 recognizer の FN のうち、位相追跡プローブが既に検出している割合 (combo 0.7/150・full guard で統一、算出 artifact: [`fn-overlap-artifact.json`](fn-overlap-artifact.json)): 70cc6637 = 23/30 (77%、偶然一致期待 ≈2.9 件)、47902d34 = 6/12 (50%、期待 ≈0.5 件)** — 偶然では説明できない。ただし位相追跡単体の precision は 0.25-0.29 のまま、という強い条件付き (guard off なら生 recall 0.97 だが P=0.16)

## S3 research 成果と GO 条件

G1-G4 **全充足** (詳細は #203 の S3 exit コメント):

- G1 ✓ partial table 自壊せず (fresh 実衝突は 2 ペアのみ)
- G2 ✓ 精度が律速 (b)
- G3 ✓ FN > FP (c)
- G4 ✓ spectral pin 済み録音 2 本

## GO を支持する材料

1. G1-G4 全充足 + kill 条件 (K2-K4/K6, C1-C2) は S4 基準値追記により判定式が一意 (種類・閾値は S0 固定のまま)
2. 現行 FN の 77% (70cc6637: 23/30) / 50% (47902d34: 6/12) を位相追跡が既に検出 — 偶然一致は null で棄却済み (recall 側の獲物は実在する)
3. unpinnable (carryover-mask) 38 音中 30 回収 — K2 最低獲物 (0.875) の到達可能性を支持
4. guard mode 対照実験 (監査対応) により、**検出器の生 recall は 0.97 で「上限 0.77」は guard 設計の人工物**と判明 — 獲物の総量は当初想定より大きい。FN 主因は dominance guard (23 件) と harmonic-parent guard (15 件) で、両方とも winner-take-all 排他 = tracker が置換する対象そのもの。harmonic-parent 側の修正候補として実測 partial table が存在 (**ただし C4 未確定・弱クラスタ不安定のため「検証済みの置換手段」とまでは言えない** — 監査指摘 2)
5. #206 のカスケード脆弱性 (segment 一括棄却 + recent-note memory) は tracker の状態ベース explaining-away が構造的に解消する型
6. 位相勾配による partial/実打鍵識別・レベル変動頑健性 (ユーザー仮説 2 件) など、観測モデルの独立な伸び代が複数ある。partial table の主要クラスタは timing perturbation ±40ms + pinned 再構築で安定 (監査対応で検証済み)

## GO に反対する材料 (counter-evidence)

1. **precision は未解決のまま、しかもギャップは当初想定より大きい**: guard 排除後の生 precision は 0.16 (recall 0.97 の代償)。explaining-away 層は机上設計であり、P 0.16 → 実用域への回収は最大の実装リスク。guard を残せば P 0.25-0.29 だが recall 0.77 に戻る — このトレードオフの解消が tracker の存在理由であり、失敗すれば kill
2. **partial table の脆弱性**: 収音チェーン依存が強く magnetic pickup 環境では bleed 項がほぼ消える。C4 未確定・A4 は n=4。「実測+検証済み」の検証は現状 1 楽器 (テスター 17-C) に偏る
3. **共鳴の原理的難所**: 機械結合の impulse 伝達は位相でも区別困難 — tracker でも残る FN/FP 領域がある
4. **recall と precision のトレードオフ解消は未実証**: full-guard 動作点では R=0.77、guard off では R=0.974 だが P=0.159。tracker が「R≈0.97 側の獲物を保ったまま precision を実用域へ回収できる」ことはまだどこにも実証されていない
5. **機会費用**: dual-run 評価は S5-S7 の 3 スプリント枠 (K6)。NO-GO 分岐 (較正系 #172-174) もテスター録音の環境メタデータが揃いつつあり価値が上がっている
6. a9e30986 (R=0.444、最難録音) の FN 45 は多声+速いパッセージ由来が多く、carryover/倍音抑圧の 2 型だけでは説明できない可能性 (tracker の獲物範囲外の FN が相当数残るリスク)

## NO-GO 分岐の計画 (監査指摘 8 対応 — GO 側と同粒度)

NO-GO 時は較正系 #172-174 を前倒しする。M1 (汎化) の実装本体であり、「別の楽器・環境で使えるようにする」route は per-tine tracker と独立に価値がある:

| # | 内容 | 初回実験 | 指標 | 期待改善 | 工数感 |
|---|---|---|---|---|---|
| #172 per-tine fundamental_ratio 較正 | tine 個体の cents ずれ (実測済み: テスター A4 +20.3c) を認識帯域に反映 | きらきら星 GT で per-tine offset を較正 → dual-run | 非飽和 headline (n=9) + fixture 非劣化 | A4 系の miss/誤同定の解消 (70cc6637 FN 30 のうち A4 関与分)。効果は限定的だが確実 | 小 (offset 測定は実測済み、適用機構のみ) |
| #173 per-recording backward_attack_gain 正規化 | 収音チェーン差 (mic 近接 / magnetic / iPhone) で gain 分布が動く問題 | GT 済み 15 本で backward_attack_gain 分布を録音別に実測 → 正規化式を導出 | 同上 + 環境別 FP/FN 分布 | 絶対閾値の環境脆弱性 (#206 で実証済みの型) の緩和。レベル頑健性はユーザー仮説とも接続 | 中 |
| #174 BPM 適応 noise_floor gap | 高速演奏で silent gap 検出が崩れる問題 (47902d34 の flux 崩壊と同根) | 47902d34 (高速テイク GT) で gap 検出の BPM 依存を実測 | 同上 | 高速テイクの R 0.750 改善 | 中 |

- 入力データは既に揃っている (per-tine cents 実測 / partial table の per-chain 差 / テスター録音環境メタデータ / pinned onsets) — S3 成果は NO-GO 側でもそのまま資産になる
- 弱点: 較正系は「単発時点検出の構造的死角 (carryover-mask 38 音、guard トレードオフ)」には効かない。unpinnable 型 FN は残存する見込み — ここが GO/NO-GO の本質的な差
- #206 短期防御 (棄却時の低 confidence 候補保持、非 pass 形) も NO-GO 側の初手に含める

## 判定手順 (残り)

1. 別系モデル (Codex 推奨) による敵対的監査 — 監査パッケージは別途用意
2. ユーザーの GO/NO-GO 判断 → #203 に記録
3. GO 時: S5 = tracker 実装 (research branch + dual-run、kill 判定は kill-criteria doc の S4 基準値で行う)。NO-GO 時: 較正系 #172-174 前倒し or 出力/UX 再配分
