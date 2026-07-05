# per-tine 確率トラッカー + causal onset 統合設計 (第 3 期 S3)

- 作成: 2026-07-05 (S3)。**S4 実装ゲートの判定材料** — 実装本体は S4 の人間 GO 後にのみ着手する (ガードレール 11)
- research line: #141 (umbrella)、kill/GO 条件は [`2026-07-per-tine-kill-criteria.md`](2026-07-per-tine-kill-criteria.md) (S0 固定値)
- 入力: #149 プローブ + 追試 ([`2026-07-149-collision-probe.md`](2026-07-149-collision-probe.md))、実測 partial table ([`2026-07-per-tine-partial-table.md`](2026-07-per-tine-partial-table.md))、位相追跡 ROC ([`2026-07-phase-tracking-roc.md`](2026-07-phase-tracking-roc.md))、spectral pin (#201 v1)、S5 spike 群 (branch claude/s5-agenda4-bg-reattack-rescue)、S2 用途検証・dogfooding の設計入力
- 旧 [`../per-note-onset-detection-design.md`](../per-note-onset-detection-design.md) (mute-dip/pass 構成) は概念的前身。本書は「連続状態推定」へ再構成したもので、mute-dip は観測モデルの 1 特徴に降格する

## 1. なぜ「状態保持」が必須か (実測 3 系統の収束)

1. **spike 実測**: single-instant 統合 (bg rescue / phase-reset rescue) は carryover-mask 2 録音で 0.769→0.933 を達成したが fixture 回帰 2-5 本と引き換えだった — 時点判定では真偽の分離に必要な情報が足りない
2. **プローブ追試 (fresh 汚染度)**: 単発時点の帯域観測は raw 汚染 0.2-0.7 に常時さらされるが、その大半は前打鍵の残響 = **per-tine の減衰状態を保持すれば説明できる成分**
3. **ROC 非単調性**: 閾値を緩めると winner-take-all guard が spurious に真イベントを殺させ recall が下がる。「候補を増やして後段で選ぶ」は単発時点排他では成立せず、全 36 combo で precision ≤ 0.29

## 2. 状態空間 (連続、離散 state machine は採らない)

tine ごとに連続状態を保持する (attack/body/late_decay の離散 3 状態機械は採用しない — 境界閾値が新たな定数群を生むため。reassessment §1 の懸念と整合):

| 状態変数 | 内容 | 更新 |
|---|---|---|
| a_k(t) | 帯域複素包絡の振幅推定 (鳴っているか・どれだけ強く) | heterodyne 復調の causal LPF 出力 |
| φ_k(t) | unwrapped 位相の線形トレンド (周波数オフセット込み) | 直近 100ms の線形フィット |
| d_k(t) | 減衰率推定 (log-envelope の傾き) | 打鍵後の指数減衰フィット、既定は tuning 別 prior |
| t_k^last | 最終打鍵時刻 (自己 onset 履歴) | onset 判定時に更新 |

**予測**: 次 hop の期待値 â = a·exp(d·Δt)、φ̂ = 線形外挿。**観測との残差**が onset 証拠の一次量になる (AR(1) 予測残差案はこの離散化と等価なのでここに吸収)。

## 3. 観測モデル (特徴量の比較と採否)

| 特徴 | 判別対象 | causal | マイク距離揺らぎ頑健性* | 実測状況 | 採否 |
|---|---|---|---|---|---|
| 位相 RMS 予測誤差 | 再打鍵 (自己位相トレンド破壊) | ✓ (60ms 先読みのみ) | **高** (1cm ≈ 0.08rad @440Hz ≪ bar 0.7) | ROC 済み: unpinnable 38 中 30 回収 | **採用 (中核)** |
| envelope jerk (相対正規化) | 新規打鍵の立ち上がり | ✓ | 中 (common-mode は正規化で相殺、速い swell に残余) | ROC 済み | **採用 (中核)** |
| fresh narrow FFT (pre 減算) | 打鍵の新規寄与 vs 残響 | ✓ (窓 250ms 遅延) | 中 | プローブ追試で確立 | **採用 (検証層)** |
| 実測 partial 比 + 期待 bleed 振幅 | cross-tine explaining-away | ✓ | 高 (比は距離に不変) | table 構築済み (×2.8-3.0、bleed 0.002-0.35) | **採用 (explaining-away)** |
| 減衰予測残差 | 残響 vs 新エネルギー | ✓ | 中 | d_k 推定は未実装 | **採用** |
| envelope 形状相関 (S2 判別器) | 同時打鍵 vs bleed (形が同じなら bleed) | ✓ | 高 (形状は振幅スケール不変) | S2 で個別実証 | 採用 (第 2 層) |
| sustain チェック (100-250ms 保持) | フィルタ過渡 vs 実打鍵 | 250ms 遅延 | 高 | プローブで有効 | 採用 (streaming では確定遅延として許容) |
| 位相コヒーレンス声部分離 | 同時和音の分離 | ✓ | 高 | 未実測 | 保留 (位相 RMS と同一物理量、冗長なら削る) |
| integer 倍音 harmonic-parent guard | (現行) partial bleed 抑圧 | ✓ | — | **ROC で系統的 FN の主因と判明** | **不採用 (実測 partial に置換)** |
| winner-take-all dominance 排他 | (現行) skirt bleed 抑圧 | ✓ | — | ROC 非単調性の主因 | **不採用 (状態ベース説明に置換)** |

\* ユーザー仮説 (2026-07-05): マイク距離揺らぎの音量変動は振幅系を騙すが位相系は頑健なはず。物理評価: 距離変化 δd の位相影響は 2πf·δd/c (440Hz で 0.08 rad/cm) で bar に対し余裕大。振幅は 1/r 直撃。**検証**: augmentation 資産で slow AM (0.5-2Hz、±3-6dB) を適用し jerk 系 vs 位相系の検出安定性を比較するメタモルフィックテスト (§6)。

### explaining-away の定式化 (シンプル形)

tine k の帯域で観測された新規エネルギー ΔE_k に対し、「他 tine の打鍵/残響で説明できる量」を実測テーブルから引く:

```
ΔE_k^unexplained = ΔE_k − Σ_j (bleed_amp[j→k] × ΔE_j)   (j: 同時窓内に onset 証拠のある tine)
                        − decay_residual_k                 (自己残響の予測残差)
```

bleed_amp は実測 partial table の medianRelAmp (楽器グループ別)。**partial が見えない収音チェーン (magnetic pickup 系) では bleed 項がほぼゼロになり、位相+jerk のみで動作する** — partial 項は optional でなければならない (table 実測の主教訓)。

## 4. causal 化共通基盤 (A1 統合)

- 全特徴は heterodyne 復調 (混合 + causal LPF + 5ms hop 間引き) の上に載る。butter 3 次 sosfilt は逐次実行可能で、**native (Python) と WASM (Rust) で同一カーネルを共有できる** — kalimba-dsp crate に `tine_demod` を足せば browser 側 (A1) の streaming onset がこの基盤に乗る
- 固定 look-ahead は 60ms (位相外挿窓) + sustain 確認 250ms。streaming では「暫定 onset (60ms 遅延) → 確定 (250ms 遅延)」の 2 段 emit として自然に表現できる。M2 進入条件「causal 化判断」はこの設計で充足見込み
- 計算量: 17 tine × (複素混合 + 3 次 IIR ×2) ≈ 数 MFLOP/s 級。ブラウザ実時間で問題なし

## 5. dual-run 計画 (S5、GO 後)

- **構成**: main recognizer (無変更) vs research branch (tracker 統合版) を同一録音に対し並走。tracker は**まず後段 rescue/reject の判定器として接続** (broadband events を置き換えず、carryover-mask 型の追加 onset 提案 + 共鳴 FP の降格提案) — C2 (本線漏出) を避ける最小結合
- **評価** (kill 条件文書の規定どおり、機械出力を #203 に貼る):
  - K2: carryover-mask 2 録音 (4e1ae5c6/9ce7df83) 合計 recall ≥ 0.875
  - K3: 非飽和 headline micro R +0.015 以上 (CI 併記)
  - K4: 非飽和 pooled FP 悪化 +15% 以内
  - C1: full fixture suite 回帰の有無 (596 tests)
- **新計器**: spectral pin の unpinnable リスト (70cc6637 で 38 音) の回収率。単発時点の上限 (79% @ ROC) を超えるかが tracker の付加価値の直接測定になる
- 評価録音の timing 検証は pin 済み 2 本を使う (ガードレール 13 充足済み)

## 6. 検証バックログ (実装前後)

1. **AM メタモルフィック** (ユーザー仮説検証): 実録音への slow AM で位相系 vs 振幅系の安定性比較。警報 v0 の変換群に追加する形で実装可
2. **C4 partial の実測補完**: table の既知ギャップ。1955b5bd/98019f67 の GT 化後に build_partial_table.py 再実行 (C4×3→G5 混同実例の裏取り)
3. **ebecf0c6 D5/F5 lowpass 脆弱性** (警報 v0 WARN): **root cause 解明済み ([#206](https://github.com/ayokura/kalimba-transcription/issues/206))** — lowpass による flux 相対閾値の低下が偽 onset を作り、segment 再分割 → residual-decay 一括棄却 → recent-note memory カスケードの 4 段連鎖。F5 の位相リセットは lowpass/ゲインに不変であり、位相ベース検出ならこの WARN 自体が発生しない (レベル頑健性仮説の実例)。segment 一括棄却 + recent-note memory は explaining-away で置換すべき機構の筆頭
4. playability 拘束は **GT 側 prior のみ** (FP 信号としては dead — bets 判定 2026-07-05)。tracker には入れない

## 7. リスクと未解決

- **A4/C4 など低 n の partial 項**: n=4-5 での medianRelAmp は誤差が大きい。bleed 減算は保守側 (過大減算しない clip) に倒す
- **同時和音 (真の同時打鍵) と bleed の縮退**: 形状相関 + partial 比の 2 層で分離する設計だが、オクターブ同時 (C4+C5) が最難関のまま残る可能性。ROC の FN 構造から、少なくとも「integer 倍音 guard で無条件に殺す」よりは改善するはず
- **減衰率 d_k の環境依存**: 部屋残響で見かけの減衰が変わる。per-recording 較正 (#173 と同系) の接続点として設計しておく
- 位相コヒーレンスの冗長性判定は実装時の ablation で決める (kill 条件の巡数を消費しない範囲で)
