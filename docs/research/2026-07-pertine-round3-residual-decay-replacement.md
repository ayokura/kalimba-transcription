# 第 3 巡設計: residual-decay 一括棄却連鎖の per-tine 置換 (#206 → merge 条件 3)

- 作成: 2026-07-06 (S5 exit 後、ユーザー承認範囲 = 設計・分析のみ)。**実装は S6 GO 後の第 3 巡** (巡 = dual-run 1 回とその改修サイクル、kill 条件文書の定義)
- 目的: #141 merge 3 条件のうち未実証の **(3) 既存 suppression pass / gate reason の削減** を、#206 の residual-decay 連鎖の置換で実証する
- probe artifact: [`pertine-round3-probe.json`](pertine-round3-probe.json) (生成: `scripts/audio-analysis/research/pertine_round3_probe.py`、全数値は再生成可能)
- 入力: [#206](https://github.com/ayokura/kalimba-transcription/issues/206) root cause (4 段連鎖) / [`2026-07-per-tine-tracker-design.md`](2026-07-per-tine-tracker-design.md) §6.3 (置換すべき機構の筆頭指名) / S5 第 2 巡の統合 tracker ([`pertine-dualrun-round2-integrated.json`](pertine-dualrun-round2-integrated.json))

## 1. 置換対象 (コード所在と役割)

| 機構 | 所在 | 役割 |
|---|---|---|
| residual-decay 一括棄却 | `peaks.py` `_resolve_primary` 内 (recent note + `is_residual_decay` + mute-dip なし → segment 丸ごと drop、reason `residual-decay-no-reattack`) | 前打鍵の共鳴 segment の FP 抑制 |
| residual-forward-scan | 同分岐内 (recent notes を mute-dip で走査 → 代替 primary 昇格、octave-up rescue 込み) | 棄却の巻き添え救済 (**recent-note memory 依存**) |
| 第 2 サイト | `peaks.py` multi-primary branching 側 (同条件、forward-scan なし) | 分岐仮説の同型棄却 |
| 下流の緩和 | `pipeline.py` slot 保持 (#178) + mute-dip sub-onset rescue | 棄却後の部分回収 |

#206 の 4 段連鎖 (偽 onset → segment 再分割 → 一括棄却 → recent-note カスケード) の 3-4 段目がこの機構。**candidate 保持なしの hard drop と、履歴 (recent-note memory) 依存の rescue の結合**が連鎖の増幅器になっている。

## 2. Stakes 実測 (Probe A: 15 GT 録音、tracker OFF 側)

- residual-decay 棄却 = **33 segment / 13 録音**
- うち **15 segment が GT FN 計 21 音と重なる** (機構起因 FN の上界。非飽和 truth 419 に対し R 換算最大 ~+0.05 — S5 第 2 巡の ΔR +0.021 の 2 倍超の headroom)
- 18 segment は clean suppression (機構が守っている precision)
- forward-scan 昇格は **10 回** (70cc6637 ×6 / ebecf0c6 ×2 / 8039a34c ×2) — 置換対象の rescue が現役で使われている量

## 3. 判別力実測 (Probe B: veto 核の分離性能)

**veto は rescue と別物**: rescue (第 2 巡の追加型) は broadband の裏付けなしに event を作るため carryover 制約 (pre-ring) + tier マージンが必要だが、veto は **broadband が既に onset を主張した segment** の中で「どの tine が新鮮な attack か」だけを裁く。よって検出核の生 bars (phase RMS ≥ 0.7 / jerk ≥ 150、carryover 制約なし) を適用した:

| 対象 | 結果 |
|---|---|
| FN 音 21 (棄却 segment 内) | **16 発火** (= veto が救える)、slot 単位では **12/15 回収可能** |
| clean suppression の dropped primary 18 | **発火 1 のみ** (誤 veto 率 5.6%) |

取り残し 5 音の内訳 (置換の限界として記録):
- 同 note 軟再打鍵 2 (47902d34 G4 / a9e30986 D5): 検出核 hit ゼロ。dropped primary 自身が真の再打鍵だった事例で、mute-dip 証拠の領分 (既存 mute-dip rescue が残る理由)
- 境界 3 (ebecf0c6 F5 jerk 22 / 70cc6637 E4 phase 0.59 / C5 phase 0.29): bars 未達。うち phase 0.29 は位相トレンド無傷 = bleed の可能性もあり GT 側の再検証候補

誤 veto 1 (2bf55c75 F#5) は全軸 marginal: phase 0.81 (bar 0.7 直上)、hit 時刻が segment 終端の +0.029s 外、reinject 0.4。**較正手段は複数あり** (窓意味論の厳密化 / phase margin / tier 降格の再利用) が、probe n=1 への過適合を避けるため**選択は実装巡の較正に委ねる** (reinject ≥ 1.0 の一律追加は回収 16 中 5 を失うため不採用と判明済み)。なお 2bf55c75 は corpus floor 1.000 のため、誤 veto の event 化は C1 違反になる — 実装巡の必須解消項目。

**実装巡の較正決定 (2026-07-06 追記、上記の事前判断を fixture 実測で更新)**:

1. **窓意味論の厳密化を採用**: hit は slot の [start, end] 内のみ有効 (pad なし)。誤 veto 1 (+0.029s 外) が閾値追加なしで消える。犠牲は境界 1 発火 (47902d34 F5, +0.017s 外) のみ
2. **条件 (5) reinject を veto にも復活** — 上記「一律追加は不採用」を覆す。根拠: 検出核のみの veto は和音 repeat 系 fixture で破綻する (a4-d4-f4-triad-repeat-01 単体で +16 event、**全て非演奏 tine C4/E4、全件 reinject 0.32–0.86**。同時リンギング tine の相互 beating が phase/jerk bar を通過する regime で、reinject はまさにその beating guard として round 1 で較正済み)。probe 側の犠牲 (発火 16→10、slot 12→6) のうち C5 系は round-2 rescue path が統合パイプラインで既に回収する carryover-mask 級 (probe は tracker OFF で FN を数えるため重複計上) — 限界損失は 2×2 の arm C vs A で実測する
3. pre-ring は設計どおり免除を維持 (#206 級 = 鳴っていなかった masked tine の fresh strike)
4. **既存 event 同時性 guard**: fire が既存 event (任意 note) の ±ATTACKER_WIN 内なら veto 対象外。broadband が event を出した瞬間はフルスコアリング済みで、veto の使命は「全落ちした瞬間」の裁定のみ (17-c bwv147 の D5: 認識済み和音 attack 内で chord selector が棄却済みの音を 3 特徴で再審して誤昇格していた)
5. **veto tier (dominance margin の再利用)**: reinject guard 後も残る誤発火 4 件 (bwv147 G5 envRatio 1.21 / triad-02 G4+C4 相互 / c4-to-e6 D5 / ebecf0c6 C6 reinject 1.27) はすべて「強い同時 strike に対し非優勢な quiet tine の発火 = table が説明しきれない bleed」で、round-2 tier margin (attacker 比 ≥1.5 + reinject ≥1.5、pre-ring 条項のみ免除) がそのまま分離する。非優勢 fire は event でなく candidate slot (`pertine-autopsy-candidate`) に降格
6. **窓境界の同 note event 規則**: dropped 窓の縁 (±EXISTING_TOL) に同 note の既存 event が接している場合、窓内 fire はその attack が segment 分割で早く/遅く見えたものであり昇格すると二重発行になる (c4-to-e6: dropped C5-residual 窓 [4.432, 4.859] の終端 = 既存 D5 event の開始で、窓内 D5@4.535 が二重化)

## 4. 頑健性実測 (Probe C: #206 の lowpass 不変性)

ebecf0c6 baseline vs lowpass 8kHz (metamorphic alarm と同変換、zero-phase):

| 対象 | baseline | lowpass 8kHz |
|---|---|---|
| F5@2.603 (連鎖で drop された真打鍵) | fires / phase 4.05 / jerk 3472 / reinject 8.84 | **完全同値** |
| F5@3.243 (同) | fires / phase 1.74 / jerk 360 / reinject 2.48 | **完全同値** |
| D5 residual (両時刻) | 発火なし (probe floor jerk 10 でも hit ゼロ) | 同 |

物理的必然: heterodyne 復調の LPF 帯域は F5 で ~17 Hz (≪ 8kHz cutoff) であり、tracker の特徴空間は 8kHz lowpass を「見る」ことすらできない。**#206 の WARN は tracker ベースの判定では原理的に発生しない** — ユーザー仮説 (2026-07-05)「レベル/帯域変動に位相は頑健」の定量実例。

## 5. 置換アーキテクチャ (提案)

**形: post-stage segment autopsy** (第 2 巡の最小結合を維持、C2 非成立のまま):

1. pipeline 統合点 (現 pertine hook) で、`residual-decay-no-reattack` の dropped slot を tracker 特徴で adjudicate する
2. slot 窓内に検出核が発火する tine があれば、その note を event 昇格 (昇格の tier 判定は第 2 巡の規則を再利用するか veto 用に較正 — 実装巡で決定)。発火なしなら現状どおり slot 保持
3. **forward-scan ablation dual-run** で置換を実証する: 2×2 (forward-scan on/off × autopsy on/off) の fixture suite + benchmark で「fscan off + autopsy on」≥「fscan on + autopsy off」を全指標で確認
4. 成立したら forward-scan (recent-note 走査 + octave-up rescue) の除去を提案 — **peaks.py の変更なので C2 相談トリガー**。成立データ (2×2 表) を添えて相談する手順とする。除去が承認されれば merge 条件 (3) の実証が完成

注: forward-scan の on/off toggle は現状存在しないため、実装巡で settings flag (ablation switch 節) を 1 つ足す (これは C2 の「本線への新規定数」ではなく既存 ablation 機構の追加スイッチだが、念のため相談に含める)。

## 6. 検証プロトコル (実装巡)

- 2×2 ablation matrix: full fixture suite + 統合 dual-run (K2/K3/K4 継続測定、非飽和 headline + CI)
- metamorphic alarm: ebecf0c6 × lowpass の WARN 解消 (dropped 5 → 0) を green 化の直接判定に使う
- held-out: 1955b5bd / 98019f67 の GT 化後に全数値を再測 (第 2 巡 caveat 1 の解消と同時)
- 誤 veto 較正: §3 の選択肢から 1 つを選び、2bf55c75 floor 1.000 の維持を C1 で確認

## 6.5 実装巡の測定結果と判定 (2026-07-06 追記 — 本設計の帰結)

実装 (755274d、較正 4 段) 後の実測で、本置換設計は**両命題とも不成立**と判定した:

1. **置換命題「fscan off + autopsy on ≥ fscan on + autopsy off」: 不成立** (`pertine-round3-ablation.json`)。B は A 比 R −0.002 / FP +1、per-recording 回帰 2 件 (70cc6637 0.852→0.844、8039a34c 1.000→0.998)。arm 差分の同定により、autopsy の統合後の正味 GT 効果は **70cc6637 の FP +1 のみ** — probe A/B が見込んだ FN 回収は、fixture 較正で必要になった 4 guard に抑止されるか、round-2 rescue / forward-scan が既に回収していた (probe は tracker OFF 側で FN を数えており、統合系での限界寄与を過大評価していた — §2 の headroom ~+0.05 は成立しない)
2. **頑健性命題「metamorphic WARN 解消 (dropped 5→0)」: 不成立**。autopsy ON でも WARN 継続 (dropped 4 + added 2、diff 6 > thr 2)。F5 は lowpass 下で autopsy が救うが hit 時刻が +0.1s ずれ ±50ms 照合を外れる。D5 系は純減のまま
3. **副産物 (正の結果)**: kill 継続判定 (第 2 カウント巡、arm C vs base) は K2 14/16 PASS / K3 ΔR+0.021 CI[0.0048, 0.0502] PASS / K4 FP54≤61 PASS。また arm D (fscan+autopsy 両 off) も K2 14/16 であり、**carryover-mask の獲物は round-2 rescue 単独で回収できている**。fscan の残存寄与は +0.002 R (70cc6637/8039a34c) に縮小 — forward-scan は「除去を正当化できないが、依存も既に小さい」状態と実測された

**処置**: `use_pertine_residual_autopsy` を既定 OFF に戻し (event 昇格の撤退)、コード・mechanism テスト・2×2 スクリプト・本測定を負の結果の資産として research branch に保存。forward-scan 除去の C2 相談は行わない (証拠が支持しない)。merge 条件 (3) は本ルートでは達成されず、#206 の WARN は未解決のまま残る (真の原因は「lowpass による偽 onset → segment 再分割」の上流であり、post-stage 裁定では時刻ずれとして残ることが判明 — 対処するなら segment 形成段が対象)。

## 7. リスク・未解決

- 同 note 軟再打鍵 (mute-dip class) は本置換の範囲外 — mute-dip rescue は残す (置換対象は forward-scan のみ)
- multi-primary branching 側の第 2 サイトは forward-scan を持たないため、autopsy が同様に効くかは実装巡で確認
- #188 (orphan-onset の noise-floor 不変化) は同族の設計課題だが範囲外 (autopsy の基盤が流用できる可能性のみ記録)
- 巡カウント: 本実装は kill 判定の第 2 カウント巡を消費する。K6 期限 2026-08-04
