# #149 衝突プローブ結果 (per-tine GO 条件 G1 の判定材料)

- 実施: 2026-07-05 (第 3 期 S0、Fable)。スクリプト: `scripts/audio-analysis/research/tine_partial_collision_probe.py`
- 方法: GT 済み全録音から孤立単独発音イベント (前後 350ms に他イベント無し) を収集し、body 窓 (onset+50ms から 200ms) の FFT で基音の 1.2-4.4 倍帯の partial ピークを tine ごとに実測。実測 partial と全 tine 基音の衝突 (±50 cents) を列挙し、衝突ペアの汚染度 (striker 単独発音時の victim 帯エネルギー / striker 基音エネルギー) を測定

## 結果

### 実測 per-tine partial (17-C、初の実測テーブルの種)

| tine | n | 実測 partial 比 |
|---|---|---|
| C4 | 8 | ×1.474, ×3.023 |
| E4 | 4 | ×1.347, ×2.914 |
| G4 | 19 | ×1.265 |
| C5 | 38 | ×1.263, ×2.911 |
| E5 | 40 | ×2.821 |
| C6 | 6 | ×1.280 |
| A4/B4/D5/F5/G5/A5/B5/E6 | 9-55 | 有意な partial クラスタなし (高域は基音支配) |

非整数性は Chapman (2012) の梁理論と整合 (2nd partial が ×1.26-1.47 で tine ごとに異なる)。**高域 tine は partial が弱く、per-tine table は低中域のみで足りる可能性が高い**。

### 衝突マップ (実測 partial ↔ 他 tine 基音、±50c)

| striker partial | victim | 距離 | 汚染度 (median) |
|---|---|---|---|
| C4 ×1.474 | G4 | -28c | 0.002 |
| C4 ×3.023 | G5 | +15c | 0.002 |
| E4 ×1.347 | A4 | +16c | 0.040 |
| **E4 ×2.914** | **B5** | **-48c** | **0.182** |
| G4 ×1.265 | B4 | +7c | 0.021 |
| C5 ×1.263 | E5 | +4c | 0.018 |
| C6 ×1.280 | E6 | +27c | 0.010 |

### 衝突外の高汚染ペア (裾・広帯域漏れ)

A4→G4 0.205 (n=26、200c 近接の裾)。E6→F5 0.550 / C6→E4 0.450 / E6→D5 0.415 は n=4-6 で要追試 (高域打鍵の広帯域成分の疑い)。

## 判定 (GO 条件 G1 への入力)

**「per-tine partial table が隣接 tine 汚染で自壊する」ことは示されなかった。** 衝突ペアの汚染は 6/7 で ≤4%、最大でも 18% (E4→B5) であり、閾値でなく cross-tine の explaining-away 項 (「この帯域の上昇は striker の実測 partial で説明できるか」) の対象として明確に有限。むしろ実測 partial 比が tine ごとに固有 (×1.26 vs ×1.47) であることは、partial を識別特徴として使える方向の材料でもある。

留意: A4→G4 型の裾汚染 (衝突ではない 0.2 級) は位相追跡プローブの skirt bleed (D4 誤検出) と同根で、tracker の cross-tine 項の必要性を再確認する結果。高域 3 ペアの高汚染は n 不足のため S3 (partial 実測本実施) で追試する。

---

## 追試 (2026-07-05、第 3 期 S3): 楽器グループ分離 + ring-out 減算

S0 実行の 2 つの方法論的欠陥を修正して再実行した (同スクリプト改修):

1. **楽器グループ分離**: S0 は全 GT 録音を単一の 17-C テーブルに pool していたが、実際には 34L-C 2 本 + G-low 1 本 (2026-06-13 GT 化、magnetic pickup) が混入しており、さらに「17-C」の主要イベント源はテスター楽器 (iPhone 録音: 17ea7626/47902d34/70cc6637/a9e30986/d7a82772、ユーザー確認 2026-07-05) だった。tuning × 奏者 (author/tester) の 4 グループに分離。
2. **ring-out 減算 (fresh 汚染度)**: ISOLATION_SEC=0.35 は前打鍵の**数秒級リンギング**を除外できない (17-C|author の C5→B4 raw 2.81 という >1 の異常値が発見の契機)。onset 直前の同長窓のエネルギーを引いた `fresh = max(E_body − E_pre, 0) / E_striker` を追加した。fresh≈0 は「その帯域のエネルギーは打鍵前から存在 = 残響持ち越し」を意味する。

### 結果

- **S0 の高汚染 3 ペア (E6→F5 0.550 / C6→E4 0.441 / E6→D5 0.415) は全て fresh=0.000** — striker partial でも広帯域漏れでもなく、前打鍵の残響持ち越しだった。C6→E4 はそもそも 34L-C 録音由来で 17-C の性質ですらない。**per-tine partial table の自壊要因からは除外** (G1 判定は「自壊を示さず」のまま強化)。
- raw 汚染 0.2-0.7 級のペアは各グループに多数あるが、fresh で生き残るのは実質 2 つだけ:
  - **E4 ×2.913 → B5 (-49c): fresh=0.205** (17-C tester、S0 の 0.182 を fresh 基準で確認)
  - **A4 ×3.003 → E6 (+4c): fresh=0.352** (17-C tester、n=4 — 新発見。テスター楽器の A4 は +20.3c 個体で partial も高め)
- A4 ×2.372 → C6 は raw 0.700 だが fresh 0.000 — **partial 抽出リスト自体にも ring-out ghost が乗りうる**ことを示す。partial table 本実施では fresh 基準 (pre-onset 減算) で partial を抽出すべき (S3 設計入力)。
- (同日追記) fresh 基準の table 本実施 ([`2026-07-per-tine-partial-table.md`](2026-07-per-tine-partial-table.md)) により、S0 テーブルの **×1.26-1.47 クラスタは大部分が「上方長 3 度 tine の残響」のゴースト**と判明。本文書 S0 節の「2nd partial が ×1.26-1.47」という解釈は取り下げ (実在の第 2 モードは ×2.8-2.9)。

### 解釈

- 「単発時点の帯域エネルギー観測は raw 汚染 0.2-0.7 に常時さらされるが、その大半は残響であり、per-tine の減衰状態を保持すれば説明可能」— これは per-tine tracker (状態保持) の動機を S0 より直接的に強化する結果。single-instant 判定が fixture 回帰 2-5 本を出した spike 実測とも整合。
- 実 fresh 衝突は低中域 tine の第 3 partial (×2.9-3.0) が高域 tine 基音に乗る形に限られ、explaining-away の対象は有限 (現データで 2 ペア)。
- **per-tine partial table は楽器ごとに作る必要がある** (#172 較正系と直結)。author 17-C の孤立イベントは 31 個しかなく (敵対的テイクは密度が高く孤立イベントが少ない)、author 楽器のテーブル構築には追加データか非孤立イベントからの抽出法が必要。
- 注意: fresh は GT timeSec の精度に依存する (onset が実際より遅く記録されていると pre 窓に attack が混入し過小評価)。timing-sensitive な本利用は spectral pin 済み録音で検証する (ガードレール 13)。
