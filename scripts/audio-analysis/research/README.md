# per-tine tracker research probes (S5 / bets #4 / #141)

S5 アジェンダ 4 の帰結として作成した第 0 次プローブ。ラベル付き 14 時刻
(REAL=GT 実証重ね弾き 5 / FP=bg rescue の fixture 誤爆 4 / RESO=真の共鳴 5)
に対する判別力の実測記録。

## pertine_decay_probe.py — 減衰カーブ予測逸脱

band energy (40ms 窓) の対数線形フィットで onset 時のエネルギーを予測し、
実測/予測比を測る。**結果: 分離不能** (REAL 0.72-1.98 vs FP 0.03-1.20 vs
RESO 0.12-1.21)。スカラー energy 系列では bleed と再打鍵を区別できない。

## pertine_phase_probe.py — 位相リセット + 振幅 jerk

狭帯域 (±2.5%) Hilbert 解析。onset 前 100ms の位相を線形外挿し、onset 後
の RMS 位相誤差 (rad) と正規化振幅急変率 (jerk /s) を測る。

**結果 (2026-07-05): REAL vs RESO は両軸で完全分離** — per-tine tracker
の中核仮説の初実証:

| group | phase_rms (rad) | amp_jerk (/s) |
|---|---|---|
| REAL (再打鍵) | 0.73–5.06 | 117–2159 |
| RESO (共鳴) | 0.13–0.34 | -0.3–6.1 |
| FP (bleed 等) | 0.11–11.28 | 28–837 |

FP 4 件中 3 件は「phase≥0.7 AND jerk≥50」で排除可能。残 1 件
(C6@191.653, bwv163) は 100 cents 下の B5 強打鍵の bleed が ±2.5% 帯域を
汚すケース — 絶対閾値では不可分で、**全 tine の相対比較 (最強のリセットを
示す tine の特定) が次の設計要素**。

## 実装制約

scipy (butter/sosfiltfilt/hilbert) は recognizer 本体に持ち込めない
(pure-numpy 方針、#187/#193)。本実装時は FFT ベースの解析信号
(rfft→負周波数ゼロ化→irfft) と FFT バンドパスで置換する (WASM 適合)。
