# Recognition Roadmap

## Current State (2026-04-16)

### Fixture カバレッジ

44 fixtures total:
- **31 completed** — strict regression target
- **8 pending** — recognizer 改善待ち
- **2 reference_only** — 参照用（regression 対象外）
- **1 review_needed** — メタデータ要確認
- **1 rerecord** — 再録音優先

### Gesture families (completed)
- single notes（C4, D4, D5 等の繰り返し）
- octave dyads（C4+C5, D4+D5）
- triads（C4+E4+G4, A4+D4+F4, E4+G4+B4）
- four-note strict / rolled / gliss chords（E4+G4+B4+D5）
- ascending sequences（C4→E6 17音、C6→E6 13/15音）
- descending sequences（E6→C4 17/51音、E6→G4 6音、D6→E6 10音）
- mixed phrases（混合シーケンス）
- BWV147 scoped phrases（6 sub-fixtures: late-upper-tail, lower-context-roll, lower-mixed-roll, upper-mixed-cluster, restart-prefix, restart-tail 等）

### BWV147 practical coverage
- **17-key 164-event sequence**: **completed** (164/164 100%, gap-rise rescue (#186) で E148 C6 復活 → 163→164 events に拡張、`6222b90` で completed 復帰)
- **34-key 163-event sequence**: pending (162/163 99% as of #153 Phase B `36cb3de`, 残り 1 件は R1 E83 G4 → D5+G4 carryover で #153 scope 外)
- 6 scoped BWV sub-fixtures: 4 completed, 2 pending

### 現時点で acoustic regression target ではないもの
- legacy four-note fixture（broken metadata, `reference_only`）
- smartphone app reference video/audio

## Current Bottleneck

31 completed fixtures が安定した regression baseline を形成している。主な残課題:

- **34-key R1 E83 carryover** ([#153](https://github.com/ayokura/kalimba-transcription/issues/153) scope 外): L6 E82 `<D5,B4,G4>` の D5 carryover が threshold 突破して E83 (expected `G4`) に extra D5 として追加される。`recent-carryover-candidate` / `weak-secondary-onset` 等の既存 gate 領域。次セッションで energy trace 確認 → 別 issue 起票 or override 判断
- **heuristic constants の audit** ([#162](https://github.com/ayokura/kalimba-transcription/issues/162)): #153 Phase A + B + #154 で導入した 27 個の新定数の inventory、環境依存性の特定、data-driven 化候補の抽出。設計 audit のみ
- **peaks redesign / chord selector** ([#111](https://github.com/ayokura/kalimba-transcription/issues/111)): `_evidence_rescue_gate` の複雑化、sequential accept loop の構造的制約。Phase A/B 完了後の再評価マイルストーン待ち
- **`arpeggio` modeling** ([#6](https://github.com/ayokura/kalimba-transcription/issues/6)): `slide_chord` との分離。Phase 1 設計は [arpeggio-design.md](arpeggio-design.md) に記載

### 解決済 (#153 Phase A + B / #154 / #186 で完了)
- **17-key BWV147 sequence-163**: 162/163 → 163/163 (#153) → **164/164** (#186 gap-rise rescue で E148 C6 復活、completed 復帰 `6222b90`)
- **17-key R1 E97 G4** (early-window primary、broadband onset 168.0827s が gap 内): Phase B `recover_pre_segment_attack_via_narrow_fft` で解決
- **17-key R5 E148 C6** (octave-coincident chord, C6 vs C5 2倍音): Phase A.2 で一度解決 → Rust 整数化 (#185) で drift 依存が判明し pending に後退 → **#186 gap-rise rescue で物理的に正当な機構で復活** (Rust `detect_gap_rise_attack` + Python two-snapshot dominance check)
- **34-key R1 E97 / R4 E133 D5+F5 (gliss splitting + masked re-attack)**: Phase A.3 + A.4 で解決
- **34-key R2 E100 C4** (early-window primary): Phase B で解決
- **34-key R3 E121 / E127 prefix splitting**: Phase A.3 で解決
- **#125 ranked candidate 不在問題 (E148 C6)**: Phase A.2 で解決済 (要 close 確認)
- **noise floor calibration** ([#154](https://github.com/ayokura/kalimba-transcription/issues/154)): per-recording per-band で実装、narrow-FFT 系 pass の閾値基盤として使用

## Active Fixture Policy

### ステータス分布

- `completed` (31): strict regression target。`test_manual_capture_completed.py` が自動検証
- `pending` (8): recognizer 改善待ち。smoke probe のみ実行
- `reference_only` (2): 参照用。regression 対象外
- `review_needed` (1): メタデータ要確認
- `rerecord` (1): 再録音優先

詳細なステータス定義は [testing.md](testing.md) を参照。

### Historical context
- strict four-note reference (`four-note-strict-repeat-02/03/04`): 初期の認識精度検証に使用。現在は 30 completed fixtures のうちの一部
- legacy four-note reference (`four-note-repeat-01`): `reference_only` — broken scenario metadata
- BWV147 fixture-specific ルールの管理は [recognizer-local-rules.md](recognizer-local-rules.md) を参照

## ユーザー向け Gesture Families

現在の canonical なファミリー:
- `strict_chord`
- `slide_chord`
- `separated_notes`
- `ambiguous`

カリンバのセマンティクスでは、`rolled_chord` と `gliss` は `slide_chord` に統一されている。

## 将来の境界: Arpeggio

`arpeggio` は `slide_chord` に統合**すべきではない**。

理由:
- `slide_chord` は依然として 1 つの chord gesture family に解決される
- `arpeggio` は独自の時間構造を持つ ordered broken-chord パターン
- 将来の記譜・編集ではノート順序と方向が必要（harmonic set だけでは不十分）

計画方針:
- `arpeggio` を `slide_chord` と分離して維持
- main harmonic event に attach するか、明示的にリンクする
- 1 つの音楽的アイデアを無関係な複数の top-level chord event として二重計上しない

詳細設計: [arpeggio-design.md](arpeggio-design.md)
Tracking issue: [#6](https://github.com/ayokura/kalimba-transcription/issues/6)

## Local Rule Debt

fixture 固有または非常に局所的な recognizer ルール（将来の Free Performance 転写で debt になりうるもの）は [recognizer-local-rules.md](recognizer-local-rules.md) を参照。
Strategy B の gap-candidate 設計と candidate/promotion プロトタイプは [strategy-b-gap-candidates.md](strategy-b-gap-candidates.md) を参照。

## Immediate Next Engineering Tasks

1. 30 completed fixtures の regression baseline を維持する
2. peaks redesign (#111): chord selector による sequential accept loop の構造改善
3. ranked candidate 不在問題 (#125): onset-focused FFT window 等の spectral acquisition 改善
4. BWV147 full-sequence の pending 解消（onset detection 層 + post-processing debt）
5. `arpeggio` Phase 1 の vocabulary 導入（[arpeggio-design.md](arpeggio-design.md) 参照）

## 将来のサンプル収集マトリクス

収集再開時は、同じ pitch set に対する paired sample を優先:

### 同一ノートセットで
- `strict_chord`
- `slide_chord`
- `arpeggio`

### 推奨ノートセット
- `C4 + E4 + G4`
- `A4 + C5 + E5`
- `E4 + G4 + B4 + D5`

pitch content を変えずに family 境界をテスト可能にする。

長期的には、intent は recording-level メタデータから event-level メタデータに移行すべき。現在の one-intent-per-capture モデルは、fixture が意図的に one-gesture-per-take で収集されている間のみ許容される。
## 次の録音優先度

現在の acoustic recognizer には repeated-pattern redesign の強い blocker はない。次に有用なデータは方向・音域に敏感な practical material。

### Priority 1: 下降 separated-note run

目的:
- ascending-only の local carryover cleanup が汎化するか検証
- descending run に専用ロジックが必要か、データ追加だけで十分かを測定

推奨キャプチャ:
- E6 -> D6 -> C6 -> B5 -> A5 -> G5 -> F5 -> E5 -> D5 -> C5 -> B4 -> A4 -> G4 -> F4 -> E4 -> D4 -> C4
- single pass
- 同一フレーズ 3 回繰り返し（ノート間に意図的な無音なし）

### Priority 2: 高音域 short-tine coverage

目的:
- D6/E6 にノート固有の処理が必要か、fixture サポートの拡充だけで十分かを測定
- ユーザーが指摘した楽器上端付近の短い sustain / 異なる timbre をキャプチャ

推奨キャプチャ:
- D6 / E6 交互の単音、5 サイクル
- C6 / D6 / E6 昇順・降順、各 5 サイクル
- A5 / B5 / C6 / D6 / E6 などの短いフレーズ末尾

### Priority 3: 高音域 mixed phrase endings

目的:
- より密な mid-register 素材に続く D6/E6 付近のフレーズ末尾が robust かテスト

推奨キャプチャ:
- C4 -> ... -> B5 -> C6 -> D6 -> E6
- E6 -> C4 restart（短い gap の後）
- 孤立した unit pattern だけでなく、realistic なフレーズ末尾を 1-2 パターン

収集ルール:
- real-device fixture を優先
- 目的が特にノート間の無音を要求しない限り、自然なアーティキュレーションを維持
- フレーズ内で technique が混在する場合、現在のスキーマが recording-level でもメモに intent を記載
## 将来の入力ソースプロファイル

現在の acoustic recognizer と local app/synth 参照音声の詳細な比較は [app-synth-audio-gap-analysis.md](app-synth-audio-gap-analysis.md) を参照。

長期的に、recognizer は単一の acoustic 環境を前提とすべきではない。異なる入力ソースを明示的に表現する必要がある。

初期プロファイル分割:
- `acoustic_real`: マイクで録音された実カリンバ
- `app_synth`: スマートフォンまたはソフトウェアカリンバアプリの音声/映像

`acoustic_real` 内の将来的な次元:
- close mic vs room mic
- 静かな部屋 vs 騒がしい部屋
- 異なる電話 / ラップトップマイクの周波数応答
- 異なるカリンバモデルと resonance 特性

ポリシー:
- primary regression は `acoustic_real` で維持
- アプリ由来の素材は別プロファイルが存在するまで `reference_only`
- source profile は fixture status, evaluation policy, 将来の feature normalization に影響すべき

重要な理由:
- phone app audio はパターン発見やシンボリック参照には依然有用
- ただし real-device regression pool に直接混在させると recognizer チューニングの判断が曖昧になる

## スマートフォンアプリ参照映像

ローカルパス:
- `C:\src\calimba-score\.codex-media\source-videos\ScreenRecording_03-23-2026_13-09-56_1.mov`

用途（許可）:
- visual vocabulary 参照
- 将来の `reference_only` UI 素材の候補

用途（禁止）:
- acoustic regression 入力
- real-device performance の ground truth

理由:
- アプリのレンダリングと phone capture の挙動を反映しており、実カリンバの acoustics や手の technique ではない

## App-Video Arpeggio Candidate

スマートフォンアプリ参照映像の後半は `slide_chord` よりも `arpeggio candidate` / broken-chord 参照として理解するのが適切。

`.codex-media/derived-analysis/kira-kira-expected-performance.json` からの根拠:
- 最有力候補ブロック: 約 `15.41s-22.83s`
- 最も明確な arpeggio 風サブブロック: 約 `19.25s-20.89s`
- そのサブブロックの推定シーケンス: `F4+F5 / A4 / C5 / A5 / A4 / F4`

解釈:
- 将来の `arpeggio` セマンティクスとサンプル計画の参照として有用
- 現在の acoustic regression target ではない
- `slide_chord` セマンティクスに統合すべきではない

短期的用途:
- ソースメディアは `.codex-media/` 配下にローカル保持
- derived event sequence は `reference_only` の設計根拠としてのみ使用
- recognizer 作業の開始前に同等の real-device sample を収集

## 将来の Real-Device Sample Families

app-video 分析から、以下の real-device family が将来の収集候補:
- wide-register dyad: `G5 + C4`, `C5 + G4`, `F4 + F5`
- wide-register triad: `G5 + E4 + G4`
- chord-to-single continuation: `B4 + D5 -> D5`
- broken-chord / `arpeggio` パターン: `F4+F5 / A4 / C5 / A5 / A4 / F4`

これらは app video から直接導出せず、real-device fixture として録音すべき。

## 録音リクエストテンプレート

新しいマニュアルデータを依頼する際は以下の構造を使う:

- `goal`: 目的
- `gesture`: 演奏スタイル
- `notes`: ノート
- `repetitions`: 繰り返し回数
- `spacing`: 間隔
- `success criteria`: 成功基準

strict four-note rerecord の例:
- goal: クリーンな同時 four-note reference の再構築
- gesture: `strict_chord`
- notes: `E4 + G4 + B4 + D5`
- repetitions: `5`
- spacing: take 間に約 `1s` の無音
- success criteria: `5 events`、各 `E4+G4+B4+D5`

