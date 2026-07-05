# per-tine partial table 実測 (第 3 期 S3、research line 共通前提部品)

- 実施: 2026-07-05 (S3、#149 プローブ追試の直後)。スクリプト: `scripts/audio-analysis/research/build_partial_table.py`
- 成果物 (機械可読): [`per-tine-partial-table.json`](per-tine-partial-table.json)
- 位置づけ: per-tine tracker / NMF 対抗馬 / 再合成距離の共通前提部品 (sprint-plan-2026-07c S3)。**実測 + 検証記録付き**であり、ガードレール 2 (改訂)「実測+検証済み partial table の使用は可」の要件を満たすことを意図する

## 方法

#149 プローブ追試 (2026-07-05) で確立した **fresh スペクトル** (body 窓 − onset 直前の同長窓、床 0) から partial ピークを抽出する。楽器グループ (tuning × 奏者) ごとに単独発音 GT イベントを集め、比クラスタ (0.05 幅) のうち **support ≥ 0.4** (そのノートのイベントの 40% 以上に出現) のみを採録。

- ring-out 減算により前打鍵の残響は除去されるため、isolation は前方 0.10s / 後方 0.27s に緩和 (S0 の ±0.35s より母数増)
- 検証として strict (±0.35s isolation) でも構築し、relaxed でしか出ないクラスタに `relaxedOnly` を付与
- 各エントリは ratio / medianRelAmp (基音比) / support を持つ

## 主結果

### 1. S0 の「2nd partial ×1.26-1.47」は大部分が残響ゴーストだった (S0 解釈の訂正)

fresh 減算後、S0 テーブルの ×1.26-1.47 クラスタはほぼ全て消失した (17-C tester で生き残るのは G5 ×1.269 sup 0.67 のみ)。**比 1.26 = 長 3 度**であり、消えたペアは C5×1.263→E5 / G4×1.336→C5 / G5×1.333→C6 と全て diatonic の 4 半音上 — メロディ (特にきらきら星の C5/E5 交替) で直前に鳴った**上方 3 度の tine の残響を partial と誤認**していたと解釈できる。S0 doc の「Chapman 梁理論と整合 (×1.26-1.47)」という解釈は取り下げる (梁理論の非整数性自体は ×2.8-2.9 で引き続き整合)。

### 2. 実在する安定 partial は第 2 モード ×2.8-2.9

17-C tester (n=138 イベント): E4 ×2.913 (sup 1.0) / G4 ×2.883 (sup 0.96) / C5 ×2.91 (sup 0.95) / E5 ×2.821 (sup 0.82) / A4 ×3.003 (sup 1.0, n=4) / G5 ×2.652 (sup 1.0)。これらの周波数は 17-C の tine 基音の間に落ちる (例: G4×2.883=1130Hz は C6 と D6 の間) ため残響では説明できず、本物の tine 第 2 モードと判断。比が tine ごとに固有 (×2.65-3.00) である点は per-tine 識別特徴としての利用可能性を維持する。

### 3. 収音チェーン依存が強い (較正系 #172/#173 の設計入力)

- magnetic pickup 系 (34L-C / G-low、author): 安定 partial がほぼ出ない (34L-C は C5 ×2.912 のみ、G-low はゼロ)
- 17-C author (VB-Audio チェーン、敵対的テイク): n=52 で安定 partial ゼロ — テイクが短く密で母数不足の面もある
- **partial table は楽器 × 収音チェーンごとに構築する必要がある**。tracker の観測モデルは「partial が見えない環境」でも成立する設計 (partial 項を optional に) が必須

### 4. explaining-away の実対象 (プローブ追試と整合)

fresh 衝突として確認済みなのは 2 ペアのみ: **E4 ×2.913 → B5 (-49c, 汚染 0.205)** / **A4 ×3.003 → E6 (+4c, 汚染 0.352)**。いずれも第 2 モードが高域 tine 基音に乗る形。

## 既知のギャップ

- **C4 の partial が未確定** (author n=5 / tester 孤立イベント無し)。S2 dogfooding の C4×3→G5 混同実例は C4 ×~3.0 partial の関与を示唆するが、本テーブルでは裏取りできていない。追加データ (1955b5bd/98019f67 の GT 化後の再実行、またはユーザー新録音) で再測定する
- A4 tester は n=4 と少ない (ただし sup 1.0 で fresh 衝突とも整合)
- GT timeSec は近似 (ガードレール 13)。fresh 減算は onset が遅く記録されていると過小評価側に倒れる (偽 partial は作らない)。timing-sensitive 利用の前に spectral pin 済み録音での再確認を推奨

## 再現

```
uv run python scripts/audio-analysis/research/build_partial_table.py
```

GT 追加後に再実行すれば JSON が更新される。グループ分けの奏者判定は `tine_partial_collision_probe.py` の `KNOWN_TESTER` (ユーザー確認 2026-07-05) + client メタデータ (f617f8c 以降)。
