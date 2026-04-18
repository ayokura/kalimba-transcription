# Kalimba Physical Layouts

各調律の物理 tine 配置を記録する。key 番号は tuning JSON (`apps/api/app/tunings/*.py` 由来) の key field と一致し、物理的な mount 位置順に並ぶ。配置情報は recognizer の sympathetic / mechanical coupling 判定 (`peaks.py::_apply_inharmonic_partial_gate` など) で使用される。

## 34L-C (34-tine Lingting C Major)

### 構造

34 tines が 2 段 (row) に分かれて mount される:

- **Lower row (K1-17)**: diatonic + 2 octaves、中央 C4 起点の zigzag (左降順 / 右昇順)
- **Upper row (K18-34)**: chromatic + duplicate diatonic、lower row の上に stacked mounting

key index offset = 17 が 2 つの row の物理 pairing。つまり K9 (C4) の物理的真上に K26 (C#4) が mount され、body を通じた mechanical coupling が発生しうる。

### Layout 表

```
Lower row (diatonic)         Upper row (stacked above)
┌─────────────────────┐      ┌─────────────────────┐
│ K1  D6              │      │ K18 D#6             │
│ K2  B5              │      │ K19 C6  (duplicate) │
│ K3  G5              │      │ K20 G#5             │
│ K4  E5              │      │ K21 F5  (duplicate) │
│ K5  C5              │      │ K22 C#5             │
│ K6  A4              │      │ K23 A#4             │
│ K7  F4              │      │ K24 F#4             │
│ K8  D4              │      │ K25 D#4             │
│ K9  C4 ────────────┼──────┼─▶ K26 C#4           │
│ K10 E4 ────────────┼──────┼─▶ K27 F4  (duplicate)│
│ K11 G4 ────────────┼──────┼─▶ K28 G#4           │
│ K12 B4 ────────────┼──────┼─▶ K29 C5  (duplicate)│
│ K13 D5 ────────────┼──────┼─▶ K30 D#5           │
│ K14 F5 ────────────┼──────┼─▶ K31 F#5           │
│ K15 A5 ────────────┼──────┼─▶ K32 A#5           │
│ K16 C6 ────────────┼──────┼─▶ K33 C#6           │
│ K17 E6 ────────────┼──────┼─▶ K34 F6            │
└─────────────────────┘      └─────────────────────┘
                 (上段は下段の真上に mount)
```

### 物理結合の観測

Lower row の tine を叩くと、真上の upper row の tine が body coupling で sympathetic に振動する。観測された例 (fixture `ac1a5c58-d9db-406f-8cef-31a33f3182ac`, evt7 @ 15.56 s):

| 下段 (struck) | 上段 (coupled) | 音程差 | cross-note sustain ratio |
|---|---|---|---|
| K13 D5 (587 Hz) | K30 D#5 (622 Hz) | 半音上 (ratio 1.060) | 0.09 (9 %) |
| K14 F5 (698 Hz) | K31 F#5 (740 Hz) | 半音上 (ratio 1.060) | 0.02 (2 %) |

結合 tine 自身は振動しているが pluck 起源ではないため late-sustain (300-600 ms) が急速に崩壊する。これが `peaks.py::_apply_inharmonic_partial_gate` の `adjacent-semitone-leakage` zone (ratio 1.03-1.12 / 0.89-0.97) + cross-note sustain 閾値 0.15 で検出される FP パターン。

### 重複 tine

Upper row には 3 つの diatonic duplicate tine (K19 C6, K21 F5, K27 F4, K29 C5) がある。同じ音名で frequency も同一だが、position が異なるため奏者の指運びで使い分けられる。recognizer は `key` field で区別し、ranked candidates 上で両方を評価する (`peaks.py` の candidate scoring 参照)。

## 17-C (17-tine C Major diatonic)

Single-row diatonic のみ。chromatic tine が存在しないので半音差の tine pair 自体がなく、adjacent-semitone coupling は発生しない (gate が自然に no-op)。

## 17-G-low (17-tine G Major, low octave)

Single-row diatonic (G major scale)。scale の中に半音間隔が 2 箇所存在:
- B3 (247 Hz) ↔ C4 (262 Hz): ratio 1.059
- F#4 (370 Hz) ↔ G4 (392 Hz): ratio 1.057

これらは physical pairing ではなく diatonic scale の一部なので、mechanical coupling は 34L のような stacked 構造ほど強くない可能性がある (検証未了)。`_SPECTRAL_LEAKAGE_SUSPECT_ZONES` の adjacent-semitone zone は tuning を問わず半音 ratio で発動するため、17-G-low でも genuine な B+C や F#+G の chord が弱い方だけ sustain しない場合は demote されうる。

Fixture `kalimba-17-g-low-bwv147-sequence-163-01` の末尾 2 event では、この gate 導入で F#4 と G4 の event 境界が 1 note 分シフト (合計 note set は保存)。2026-04-18 時点では pending 扱いとして要再確認。

## 研究参考

- Chapman, D. M. F. (2012). *The tones of the kalimba (African thumb piano)*. JASA. — 非整数倍 partials (inharmonic harmonics) の実測。`_INHARMONIC_PARTIAL_ZONES` の 2nd/3rd/4th partial zone 根拠
- 34L の stacked mounting による mechanical coupling は既存文献には未確認 (adjacent-semitone FP は ac1a5c58 での実測から発見、Chapman の inharmonic partial theory とは別機構)
