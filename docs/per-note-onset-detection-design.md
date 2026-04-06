# Per-Note Onset Detection: Design Notes

## 背景

現在の onset 検出は librosa の broadband spectral flux に全面依存している。これは大半の onset を正しく検出するが、以下の構造的限界がある:

1. **Polyphonic masking**: 他の音の残響が大きいとき、単音 attack の spectral flux が埋もれる（E162 B4: onset_strength=0.6 vs background 0.3-0.5）
2. **Gap onset の flux 不足**: genuine attack でも high-band flux が低いと `is_valid_attack=False` で gap collector に昇格されない（E46, E62 — カテゴリA修正済みだが、閾値緩和には限界がある）
3. **Same-note re-attack の残響混同**: 前のイベントと同じノートの再打鍵が residual decay と区別できない（E83 G4）

## 目標

Broadband onset 検出を**置き換えず**、per-note の物理的シグナルで**補完**する追加パスを設計する。

## 核心的な洞察: mute-dip の物理的一意性

カリンバのタインを再打鍵するには、指でタインに触れる（ミュート）→ 弾く の動作が必要。このミュートにより note-band energy が急落（>100x）→ ゼロ近傍 → 新 spike のパターンが生じる。

この mute-dip は:
- **Per-note でしか観測できない**: broadband spectral flux では他の音のエネルギーに埋もれる
- **False positive が物理的にほぼ起きない**: タインに触れないと dip しない。sympathetic excitation では dip_ratio=0.7+ で棄却可能
- **既に実装済み**: `_has_mute_dip_reattack()` (peaks.py) が MUTE_DIP_REATTACK_MAX_DIP_RATIO=0.1 で判定

現在この検出は peaks 層（segment 内部）でのみ使われている。これを onset/segment 層に持ち上げることで、broadband が見逃した re-attack を補完できる。

## アーキテクチャ

### 3つの独立した per-note パス + peaks 層の既存処理

初期設計では4パスを検討したが、レビューの結果 **in-segment mute-dip hint（旧 Pass 2）は peaks 層の既存処理として維持**が適切と判断した。理由:

- 全 fixture を通して peaks 層の mute-dip save は 1 件のみ — segments 層に持ち上げる実益がない
- `residual-decay-no-reattack` 棄却の大半（60+ 件）は正しい棄却（本当に残響）
- 「この segment の primary は残響か genuine か」は peaks 層の責務として自然

```
[既存] broadband onset → segments
  ↓ 順に適用
[Pass 1] gap mute-dip rescue      — gap 内に新規 segment を生成
[Pass 2] onset splitting           — 既存 segment を per-note attack 時刻差で分割
[Pass 3] post-merge                — 誤分割の吸収（既存 merge 層と統合）

[既存・変更なし] peaks 層の mute-dip による residual-decay 免除
```

### Pass 1: gap mute-dip rescue

**対象問題**: broadband onset が完全に見逃す same-note re-attack（gap 内に segment がない）

**判別基準**: mute-dip（物理的保証あり）

**peaks 層への指示**: `confirmed_primary` — primary がほぼ確定。`_resolve_primary` スキップ可能。residual-decay 棄却を免除。

**処理フロー**:
```
for each gap between segments:
    for note in scan_candidates:
        scan gap region for mute-dip pattern:
            pre_energy = note_band_energy(gap_start - margin)
            scan for energy minimum in gap (5ms step)
            if dip_ratio < 0.1 and recovery_ratio >= 0.9:
                create new segment at recovery point
                attach confirmed_primary = note
```

**閾値の安全性**: `MUTE_DIP_REATTACK_MAX_DIP_RATIO=0.1` は既存の peaks 層で検証済み。100x の energy drop は genuine re-attack にしか起きない。false positive リスクが低い。

**スコープ制限**: first attack（その note が初めて鳴る場合）は mute-dip がないため対象外。broadband onset 側に任せる。

**scan_candidates の選定**: 「未解決の問い」セクション参照。

### Pass 2: onset splitting

**対象問題**: 複数の note の attack が近接して1つの segment に合流（E162 型）

**判別基準**: per-note energy gradient（推定ベース、mute-dip ほどの物理的保証なし）

**peaks 層への指示**: `hint_primary` — primary の可能性が高いが確定ではない。`_resolve_primary` はスキップせず、ranked candidates の優先順序に影響を与える程度。

**E162 の実態**: B4 は直前に強く鳴っていない（energy 4-9K、残響レベル）。mute-dip の定義（100x drop → recovery）に当てはまらない。**mute-dip ではなく first-attack に近い** → energy gradient でしか検出できない。

**処理フロー**:
```
for each segment:
    for each tuning note with significant energy in segment:
        attack_time = find_note_attack_time(note_frequency)
    if max(attack_times) - min(attack_times) > SPLIT_THRESHOLD:
        split segment at intermediate attack times
        attach hint_primary to earlier sub-segment
```

**SPLIT_THRESHOLD**: 20-30ms 程度。STFT 窓の実効時間分解能より大きく、slide の note-to-note 間隔より小さい値。

**注意**: slide/gliss では各 note が数ms-20ms 間隔で連続するため、threshold 設定で除外できる可能性はあるが、完全ではない。Pass 4 での post-merge が安全ネットになる。

### Pass 3: post-merge

**対象問題**: Pass 2 で生じた誤分割の吸収

**既存 merge 層との関係**:
- `merge_adjacent_events()` (events.py): 同一 notes かつ隣接する events を結合
- `merge_short_chord_clusters()` (events.py): 短い chord cluster を結合

Pass 2 の誤分割は「短い segment が slide の途中に挿入される」形になるため、既存の merge で吸収できる可能性がある。ただし:
- 既存 merge は events 層（segment_peaks 通過後）で動作
- 誤分割 segment が segment_peaks で unexpected な note を検出すると merge 条件が成立しない可能性

→ **既存 merge の挙動を棚卸し**してから Pass 3 の設計要否を判断すべき。

## Segment への provenance 追加

### 問題

現在の `Segment(start, end, sources)` は「どの collector が生成したか」のみ。per-note パスで生成/修飾した segment は **「どの note の情報を根拠にしたか」** と **「その確信度」** が必要。

### 理由

peaks 層がこの情報を知らないと:
1. 全 tuning note をスキャンして primary を決定
2. 残響が強い別の note が primary になる
3. residual-decay-no-reattack で棄却 → per-note パスの意味が消失

### 2段階の確信度

| フィールド | 設定元 | 確信度 | peaks 層での扱い |
|---|---|---|---|
| `confirmed_primary` | Pass 1 (mute-dip) | 高（物理的保証） | `_resolve_primary` スキップ、residual-decay 免除 |
| `hint_primary` | Pass 2 (energy gradient) | 中（推定） | `_resolve_primary` は実行するが、ranked candidates の優先に影響 |

### 設計

```python
@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    sources: frozenset[str]
    confirmed_primary: str | None = None  # Pass 1: mute-dip 由来、物理的に確定
    hint_primary: str | None = None       # Pass 2: energy gradient 由来、推定
```

**`confirmed_primary` がある場合の peaks 処理:**
- confirmed_primary を primary として採用
- ranked candidates で confirm（FFT energy が存在するか検証）
- 存在しない場合は通常パスにフォールバック
- residual-decay 棄却を免除（mute-dip が物理的に re-attack を証明済み）

**`hint_primary` がある場合の peaks 処理:**
- `_resolve_primary` は通常通り実行
- ranked candidates 内に hint_primary と一致する note があれば、スコアにボーナスを加える or tie-break で優先
- 一致しない場合は無視（energy gradient の推定が外れた）

### peaks 層の mute-dip 処理との関係

peaks 層の `_has_mute_dip_reattack()` による residual-decay 免除は**現状維持**。理由:

- 全 fixture を通して peaks 層の mute-dip save は 1 件のみ — segments 層に移す実益がない
- residual-decay 棄却の 60+ 件中、大半は正しい棄却（本当に残響）
- peaks 層の責務（「この primary は genuine か residual か」の判定）として自然

Pass 1 の `confirmed_primary` 付き segment については peaks 層の mute-dip チェックは冗長になるが、`confirmed_primary` がない通常 segment では従来通り peaks 層でチェックが必要。短期は共存で問題ない。

### E83 型の考慮事項

E83 G4 は「segment 存在 + mute-dip なし + 残響 D5 が primary → 棄却」のパターン。energy trace で G4 に mute-dip が確認されないため、Pass 1/2 いずれも対象外。

耳確認の結果次第で:
- **re-attack なし** → alignment_override（recognizer は正しい）
- **re-attack あり** → mute-dip 以外の per-note 判別基準が必要（例: energy gradient の微小変化、backtrack timing のズレ等）。現時点では設計材料が不足しているため、耳確認後に再検討。

## 因果性と Streaming 互換性

| 部品 | 因果的? | 備考 |
|------|---------|------|
| `_note_band_energy()` | ○ | center_time 周辺のみ |
| `onset_energy_gain()` | △ | onset 後 80ms を参照 |
| `onset_backward_attack_gain()` | ○ | 過去データのみ |
| `_has_mute_dip_reattack()` | △ | onset 後 50ms スキャン |
| gap mute-dip rescue (Pass 1) | △ | gap の end が既知なら因果的 |
| onset splitting (Pass 2) | △ | segment の end が既知なら因果的 |

Streaming では segment の end を「次の onset 検出時」に確定できるため、小さなバッファ（~100ms）で運用可能。

## WebAssembly 互換性

全ての核心処理は:
- FFT (`np.fft.rfft`) → WebAssembly FFT ライブラリで代替可能
- Band energy 抽出 (`peak_energy_near`) → 配列のスライス + max
- Energy gradient → 差分 + argmax

**librosa 依存は broadband onset（1段目）のみ**。per-note 部分は完全に portable。

## 実装優先順

1. **Segment provenance** — `confirmed_primary` / `hint_primary` フィールド追加。他の全パスの前提
2. **Pass 1 (gap mute-dip rescue)** — 最も確実で false positive リスクが低い。既存部品でほぼ組める
3. **Pass 2 (onset splitting)** — E162 型の解決。判別基準が mute-dip と異なるため設計判断が多い
4. **Pass 3 (post-merge)** — 既存 merge 層の棚卸し後に設計。Pass 2 と同時またはその後

Note dataclass の導入は Pass 1 と同時またはそれ以前。

## 未解決の問い

- **scan_candidates の選定**: 全 tuning notes? recent_notes? — mute-dip の pre_energy 条件 (>3.0) が自然なフィルタになるため、全 tuning notes をスキャンしても false positive は出にくい。計算コストのみの問題。ただし gap の数 × 17-34 notes × 5ms step scan のコスト感は要評価
- **Pass 1 の segment end time**: recovery point から次の既存 segment start まで? 固定 duration (0.24s)? 次の onset まで?
- **Pass 2 の SPLIT_THRESHOLD**: slide/gliss の note-to-note 間隔との兼ね合い。既存 fixture の slide timing を計測して決定すべき
- **Pass 2 の「significant energy」の定義**: 全 tuning notes をスキャンするのはコストが高い。閾値が必要
- **Pass 3**: 既存 merge で十分か、新規 merge ルールが要るか。Stage 7 棚卸しが前提
- **`confirmed_primary` が FFT で confirm できなかった場合のフォールバック戦略**: 通常パスに戻すのが安全だが、mute-dip が物理的に保証しているのに FFT に現れないケースがどれだけあるか
- **E83 型 (mute-dip なし re-attack)**: 耳確認結果次第。re-attack ありと判明した場合、mute-dip 以外の per-note 判別基準が必要になる
- **パス間の順序依存**: Pass 1 が gap に新規 segment を生成すると gap 構造が変わる。Pass 2/3 は Pass 1 の出力を含む segment リストに適用される。実質的に Pass 1 → 2 → 3 の順序が必要

---

## Appendix: Note dataclass の導入

### 現状の問題

Note 関連の情報が複数の型とユーティリティ関数に散在している:

| 現在の型/関数 | 持つ情報 | 用途 |
|---|---|---|
| `TuningNote` (Pydantic) | key, note_name, frequency, layer | API スキーマ |
| `NoteCandidate` (dataclass) | key, note_name, frequency, pitch_class, octave, score | peaks 層の候補 |
| `parse_note_name()` | 文字列 → (pitch_class, octave) | 都度パース |
| `note_name_to_frequency()` | 文字列 → frequency | 都度計算 |
| note 間の関係 | ad-hoc 計算 (peaks.py 各所) | octave 判定、隣接半音等 |

`parse_note_name()` は peaks.py だけで複数箇所から呼ばれ、pitch_class + octave の組を毎回取り出している。note 間の関係（octave、semitone distance）も peaks.py の各関数内で個別に計算。

### 提案

```python
@dataclass(frozen=True)
class Note:
    name: str           # "G4"
    pitch_class: str    # "G"
    octave: int         # 4
    frequency: float    # 391.99
    midi: int           # 67

    @staticmethod
    def from_name(name: str) -> Note: ...

    def semitone_distance(self, other: Note) -> int:
        return abs(self.midi - other.midi)

    def is_octave_of(self, other: Note) -> bool:
        return self.pitch_class == other.pitch_class and self.octave != other.octave

    def octave_above(self) -> Note: ...
    def octave_below(self) -> Note: ...
```

### 効果

- **NoteCandidate**: `Note` を内包（`note: Note` + `key: int` + `score: float`）。pitch_class / octave / frequency の重複フィールドが解消
- **TuningNote → Note 変換**: `build_tuning()` で一度だけ生成。以降は `Note` として流通
- **confirmed_primary**: `str` ではなく `Note` 型。frequency 情報が自動的に付随するため、per-note energy 計算で再ルックアップ不要
- **note 間の関係**: `is_octave_of()`, `semitone_distance()` 等が型のメソッドに閉じる。peaks.py の ad-hoc 計算が整理される
- **楽器汎化 (#33)**: note の物理特性がクラスに集約されるため、異なるチューニングへの対応が容易

### タイミング

per-note onset detection と同時または先行して導入。per-note onset の `confirmed_primary` や gap scan の `recent_notes` が `Note` 型であれば、frequency ルックアップなしで `_note_band_energy()` 等に直接渡せる。遅くとも per-note onset detection と同時に導入する。
