# 自動採譜 Product / UX サーベイ補遺 (2026-06-26)

## 目的

free-performance 転写では、recognizer が一発で完全正解を出すことよりも、
**ユーザーが短時間で正しい譜面へ直せること**が重要になる。

本補遺は、UX/product 担当サーベイの結果を主ソースとして、既存 audio-to-MIDI / automatic transcription
製品の公式 docs・機能説明・公開レビューから、Review UI と `Correction Burden` の設計に使える知見を整理する。
口コミは定量評価ではなく、不満カテゴリの抽出に限定する。

---

## 1. 製品 / UX パターン

### 1.1 Basic Pitch demo: 完成譜ではなく editable MIDI draft

見る対象:

- Basic Pitch demo / About
- Spotify `basic-pitch` GitHub

観測:

- Basic Pitch は単一楽器録音を中心に audio-to-MIDI を返す。
- 公式 About でも、MIDI にして DAW で tweak / fine tune する導線を明示している。
- GitHub 版は MIDI だけでなく raw model output や note events CSV も保存できる。

示唆:

- 自動採譜 UX は「正しい譜面を即完成」ではなく、**編集可能なイベント列を返す**パターンが実用的。
- kalimba-transcription の初期 Review UI も、五線譜エディタより `ScoreEvent` / `candidateSlots` /
  `alternateGroupings` を中心にしたイベント列レビューが合う。
- MusicXML 出力は後段でよい。

### 1.2 AnthemScore: spectrogram + note grid + 修正が製品中核

見る対象:

- AnthemScore getting started / docs

観測:

- 典型ワークフローは audio file を開く → instrument parts → measures/beats → notes → PDF/MIDI/MusicXML export。
- spectrogram 上の note grid を使う。
- note add/remove/move、duration adjust、AI score による note slider、measure/downbeat editing、slow playback が説明されている。

示唆:

- 自動採譜の UX では「結果表示」だけでなく、なぜその候補なのかを音響的に確認できる evidence が重要。
- kalimba MVP で full spectrogram を作り込む必要はないが、イベント単位の audio loop、onset marker、narrow-band energy mini trace、candidate score は早めに効く。

### 1.3 ScoreCloud: Time & Rhythm → Pitch → Export の段階的修正

見る対象:

- ScoreCloud Songwriter quick-start / tutorials

観測:

- まず playback / display で問題箇所を把握する。
- 次に Time & Rhythm を先に確認する。
- 個別 note rhythm / pitch は後で直す。
- barline drag、pickup、time signature、octave move、pitch drag、enharmonic shift、note timing / duration、triplet subdivision、add/remove/split note、PDF/MIDI/MusicXML export を持つ。

示唆:

- 音響イベント認識と記譜リズム整形を混ぜると修正負担が膨らむ。
- kalimba-transcription では、まず absolute onset 秒ベースの event / note set を正しくし、拍子・小節・量子化は `NotationLayerBurden` として別に扱う。

### 1.4 Klangio / Melody Scanner: mobile capture と desktop edit の分離、score grammar 制約

見る対象:

- Klangio Edit Mode / FAQ / quality guide
- Melody Scanner App Store 説明・レビュー

観測:

- Klangio は Edit Mode を持つが、mobile browser では画面が小さすぎるため Edit Mode を隠している。
- 音符削除は拍子整合性を壊すため、削除ではなく rest になることがある。
- PDF の高度なカスタマイズは MusicXML を外部ソフトに渡す方針。
- Melody Scanner は、複数楽器分離なし、live recognition なし、100% note detection なし、wrong detection あり、と制約を明示している。

示唆:

- カリンバ録音はスマホで行われる可能性が高いが、細かい correction は desktop-first でよい。
- mobile では「候補を採用」「候補に置換」「余分な note を外す」など、tap で完結する candidate-assisted 操作に絞る。
- score grammar の都合で修正が連鎖する領域は、音響 event correction とは分ける。

### 1.5 Setup / re-transcribe も UX の一部

見る対象:

- Klangio quality guide

観測:

- clean recording
- reverb / delay / distortion の悪影響
- constant tempo
- tight playing
- key / time signature / BPM range / quantization / mode などの追加情報
- retranscribe

が品質改善に重要とされる。

示唆:

- Review UI に入ってから 1音ずつ直すより、録音品質・tuning・profile・tempo 設定を直して再解析した方が速いケースがある。
- `Correction Burden` には **Re-run Burden** も含める。
- 「この録音は直すより再録が早い」を UI が言えると実ユーザーの負担が下がる。

---

## 2. よくある修正作業

### 2.1 False Positive / extra note 削除

観測:

- 弾いていない音。
- 和音内の余分な音。
- 片手では不自然な同時音。
- wrong notes / wrong octaves。

kalimba での原因:

- sympathetic resonance
- late decay
- 非整数倍音
- 隣接 tine の共鳴
- octave / partial alias

操作:

- `delete_event`
- `remove_note_from_event`

### 2.2 False Negative / missing note の追加

観測:

- missing parts
- missing bars
- incomplete transcription

kalimba での原因:

- weak attack
- 低域
- 連続音
- residual suppression
- onset gate / segmenter で落ちた event

操作:

- 候補がある場合: `enable_candidate_slot`
- 候補がない場合: `insert_event_manual`

重要:

- 候補にもない missing event は、ユーザーが再生して探し、時刻を決め、音を選ぶ必要があり重い。
- hard drop ではなく `candidateSlot` として残す価値が高い。

### 2.3 Wrong pitch / octave / spelling

観測:

- pitch drag
- semitone move
- octave shortcut
- enharmonic shift
- similar pitch selection

kalimba での扱い:

- 自由な pitch drag より、tuning に存在する tine の候補から選ぶ UI がよい。
- `replace_note_candidate`
- `octave_shift`
- `replace_note_manual`

を別操作として扱う。

### 2.4 Timing / rhythm / meter 修正

観測:

- barline drag
- time signature
- pickup
- double / halve note values
- note timing / duration
- AI transcription 由来 MusicXML の time signature collapse

kalimba での扱い:

- まず absolute onset 秒で event を直す。
- 拍子・小節・duration quantization は別レイヤに分ける。
- free-performance では meter まで同時評価すると、recognizer の音高誤りと notation cleanup 誤りが混ざる。

### 2.5 Split / merge / gesture 修正

観測:

- note split
- triplet
- tie
- duration
- measure / beat editing
- note deletion が rest 化される score grammar 制約

kalimba での扱い:

- `strict_chord`
- `slide_chord`
- `arpeggio`
- `separated_notes`

の分類が重要。
音高 F1 では測れない「演奏 gesture の修正」を Correction Burden に入れる。

### 2.6 Setup / re-run / export cleanup

観測:

- recording quality
- constant tempo
- additional information
- retranscribe
- desktop 版で広い編集機能

kalimba での扱い:

- tuning mismatch
- clipping / noise
- source profile
- tempo stability
- mic distance

を Review 前に出し、必要なら再録・再解析を促す。

---

## 3. Correction Burden の設計

基本定義:

> predicted event sequence から ground truth event sequence へ到達するための semantic edit cost。

クリック数ではなく、意味的な修正単位で数える。

### 3.1 分けて測るべき3種類

#### Edit Burden

実際に event / note / gesture を変える負担。

例:

- note add/remove
- event delete
- candidate enable
- merge/split
- gesture relabel

#### Review Burden

誤りを探すための負担。

例:

- audition count
- seek count
- candidate panel open
- event inspect count
- review time

#### Re-run Burden

設定・録音条件を変えて再解析する負担。

例:

- tuning/profile変更
- quantization設定変更
- rerecord
- retranscribe

### 3.2 初期コスト案

| 操作単位 | コスト案 | 意味 |
|---|---:|---|
| `accept_event` | 0 | 正しいので編集不要。確認再生は Review Burden 側で記録。 |
| `delete_event` | 1 | resonance / ghost event の削除。 |
| `restore_event` | 1 | 削除済み event の復元。 |
| `remove_note_from_event` | 1 | 和音内の extra note を外す。 |
| `add_note_from_candidate` | 1 | event 内に候補 note を追加。 |
| `replace_note_candidate` | 1 | alternate candidate へ置換。add+remove より低コスト。 |
| `enable_candidate_slot` | 1 | dropped candidate を採用。Candidate Recall が効く最重要操作。 |
| `select_alternate_grouping` | 1 | merge/split/別 note set 候補を選ぶ。 |
| `gesture_relabel` | 1 | strict chord / slide / arpeggio / separated を変更。 |
| `octave_shift` | 1 | octave だけの修正。 |
| `timing_nudge_small` | 1 | 小さい onset 補正。例: ±50–100ms。 |
| `replace_note_manual` | 2 | 候補外の tine を手動選択。 |
| `split_event` | 2 | 1 event を複数 event に分割。候補があれば 1 に下げる。 |
| `merge_adjacent_events` | 2 | 近接 event を chord / slide / arpeggio として統合。候補があれば 1。 |
| `move_event_time_large` | 2 | 大きい時刻移動。seek + drag / 入力が必要。 |
| `insert_event_manual` | 3 | 候補なしの missing event。場所を探す + 時刻を決める + 音を選ぶ。 |
| `set_tuning_or_profile_and_rerun` | 3+ | 設定変更と再解析。再確認が必要。 |
| `set_meter_tempo_quantization` | 2–3 | 記譜レイヤー導入後に計測。 |
| `manual_reentry_region` | 高 | その区間は直すより打ち直しが早い失敗ケース。 |

### 3.3 指標化の要点

- 同じ F1 miss でも、正解が `candidateSlot` に残っていれば `enable_candidate_slot = 1`。
- 正解が完全に消えていれば `insert_event_manual = 3`。
- Candidate Recall は候補乱発で上げられるため、必ず Correction Burden / Review Burden と対で見る。
- Score grammar による rest / tie / meter の連鎖修正は、初期の acoustic event Correction Burden とは分けて `NotationLayerBurden` にする。

---

## 4. Review UI に入れるべき項目

### 4.1 Event-first Review UI

まず五線譜エディタではなく、event card / timeline / score preview の構成にする。

各 event card に出す項目:

- event time
- selected note set
- origin: `recognizer` / `edited` / `inserted-slot` / `inserted-manual`
- confidence
- `needsReview`
- `dropReason` / `rescueReason`
- nearby `candidateSlots`
- `alternateGroupings`
- short audition button
- undo / reset / save

### 4.2 Candidate-assisted correction

最重要操作:

- dropped candidate をワンクリック採用
- alternate note にワンクリック置換
- event 内 note add/remove
- event delete / restore
- manual insert
- merge/split suggestion
- gesture suggestion
- candidate rank / confidence / reason code 表示

特に `candidateSlots` と `alternateGroupings` は debug 出力ではなく、UI と評価指標の一次データにする。

### 4.3 Acoustic evidence mini panel

AnthemScore のような full spectrogram は強力だが、MVP では重い。
まず event card に以下を出す。

- event 前後 150–300ms の loop 再生
- broadband onset marker
- narrow-band energy mini trace
- primary / secondary candidate score
- residual / no-reattack / orphan-onset などの reason
- original audio と synthesized candidate の A/B playback

### 4.4 Triage-first workflow

全 event を順番に確認させるのではなく、怪しい箇所を先に出す。

優先表示する条件:

- low confidence
- 1st/2nd candidate が僅差
- dropped candidate が近くにある
- onset はあるが segment がない
- residual / late decay 判定
- tuning 外または kalimba range 外
- 異常に dense な cluster
- impossible fast repeat
- merge/split ambiguity
- gesture ambiguity

### 4.5 Setup / re-run panel

Review 前に以下を出す。

- selected tuning
- tuning mismatch warning
- recording level / clipping / noise
- source profile: acoustic real / app synth
- estimated tempo stability
- resonance risk
- microphone / room guidance
- re-run with different tuning/profile

「この録音は直すより再録が早い」を UI が判断できると、実ユーザーの負担が大きく下がる。

---

## 5. 評価に入れるべき項目

### 5.1 `CorrectionBurden.total`

ground truth event 列へ到達する最小操作コスト。
合計だけでなく内訳を必ず出す。

内訳例:

- `event_delete_cost`
- `event_insert_manual_cost`
- `candidate_enable_cost`
- `note_add_cost`
- `note_remove_cost`
- `note_replace_cost`
- `timing_nudge_cost`
- `merge_split_cost`
- `gesture_relabel_cost`
- `rerun_cost`

### 5.2 `CandidateAssistedFixRate`

修正のうち、候補から低コストで直せた割合。

```text
candidate_assisted_fixes / all_fixes
```

1-best が外れても候補に残っていれば、ユーザーには優しい。
通常の note F1 だけでは見えない価値。

### 5.3 `HardMissRate`

ground truth の event / note が 1-best にも candidate にも存在しない率。

```text
hard_miss_events / ground_truth_events
```

これは `insert_event_manual` を強制するので、Correction Burden 上もっとも重い。

### 5.4 `ReviewBurden`

実 UI ログで測る。

- `audition_count`
- `audition_total_sec`
- `seek_count`
- `opened_event_count`
- `opened_candidate_panel_count`
- `time_to_first_correction`
- `total_review_time`
- `manual_reentry_region_count`

最初は CI 指標ではなく、manual study / local analysis 用でもよい。

### 5.5 `ConfidenceCalibration`

`needsReview` が本当に誤りを含んでいるかを見る。

- flagged event precision
- missed-error rate
- high-confidence wrong events
- low-confidence correct events

目標は「全部見てください」ではなく、「ここだけ見れば大半が直ります」に近づけること。

### 5.6 `NotationLayerBurden`

将来の sheet music 化で別に測る。

- meter correction
- barline correction
- duration quantization correction
- pickup correction
- tie / rest / slur correction
- layout / export cleanup

これは recognizer の音響認識評価とは分離する。

---

## 6. やらない方がよいこと

- 最初から汎用 notation editor を作る。
- spectrogram UI を先に作り込みすぎる。
- correction 操作をログ化しない。
- teacher model と user correction を provenance なしで混ぜる。
- gesture classification をユーザーに見せないまま内部だけで確定する。
- 音響 event correction と notation cleanup を同じ指標に混ぜる。

---

## 7. 結論

既存製品の共通点は、**自動採譜は完成品ではなく、修正可能な draft を作る UX** だという点。
ユーザー不満の中心は「間違うこと」だけではなく、**どこを直すべきか分からない、候補がない、記譜構造が崩れて手入力の方が早い**ことにある。

kalimba-transcription では、最初から汎用 notation editor を作るより、`candidateSlots` / `alternateGroupings` /
reason code / audio evidence を Review UI に出し、Correction Burden を **candidate-aware な編集距離**として測るのが最も効果的。

## 履歴

- 2026-06-26: 新規作成。audio-to-MIDI / 自動採譜 product の correction workflow 観点を整理。
- 2026-06-27: UX/product 担当サーベイ結果を主ソースとして全文再稿。Basic Pitch demo, AnthemScore, ScoreCloud, Klangio, Melody Scanner, product complaints, Edit/Review/Re-run Burden を反映。
