# Recognizer Local Rules Ledger

## 目的

以下に該当する recognizer ルールを追跡する:
- 事実上 fixture 固有のもの
- 将来の Free Performance 転写（expected metadata なし）で debt になりうるほど狭いもの

これらのルールを即座に削除することが目的ではない。以下の条件が整ったときに再検討できるよう可視化しておくことが目的:
- BWV147 の practical coverage が拡大した
- scoped-fixture debt が削減された
- Free Performance 認識が first-class target になった

## 分類

### Keep For Now（当面維持）

局所的なルールだが、複数の practical fixture や広範な onset 品質を保護している。

| ルール / ヘルパー | コードパス | 現在の役割 | 保護している fixture / テスト | 維持理由 | 将来の削除条件 |
| --- | --- | --- | --- | --- | --- |
| `collapse_active_range_head_onsets` | `apps/api/app/transcription/segments.py` | active range 内の密集 head-onset cluster を畳み、フレーズ冒頭の over-segmentation を低減 | `#30` practical recovery set。parametrized regression via `test_manual_capture_completed.py` で検証 | 単一 BWV ルールではなく、cross-fixture の onset-shape 修正 | より原理的な onset-boundary モデルが head-cluster collapse を置き換えた時点 |
| `refine_onset_times_by_attack_profile` | `apps/api/app/transcription/profiles.py` | 近傍の弱い / invalid onset をより強い valid-attack onset で置換 | `#30` practical recovery set および manual-capture regressions | 局所的だが attack-profile ベースで汎用性がある | onset detection 自体が close invalid-valid ペアを確実に解決できるようになった時点 |

### Candidate For Later Removal（将来削除候補）

現時点で有用だが、特定の practical failure shape に強く結びついている。debt として定期的に再評価すべき。

| ルール / ヘルパー | コードパス | 処理概要 | 主目的 | 保護している fixture / テスト | debt シグナル | 望ましい長期的代替 |
| --- | --- | --- | --- | --- | --- | --- |
| ~~`collect_two_onset_terminal_tail_segments`~~ | **削除済み** (#122) — AVC trailing collector で代替。詳細は [deferred-ideas.md](deferred-ideas.md) の「Terminal collector 5種」を参照 | — | — | — | — | — |
| `recent-upper-octave-alias-primary` promotion | `apps/api/app/transcription/peaks.py` (`maybe_promote_recent_upper_octave_alias_primary`) | current primary が直近の lower alias に見える場合、upper octave candidate を primary に昇格 | restart tail での正しい高音域ノートの復元 | `kalimba-17-c-bwv147-restart-tail-01` (pending); parametrized regression via `test_manual_capture_completed.py` | restart フレーズ tail の高音域 alias failure に強く結びつく | candidate ranking 時点でのより良い alias 解消（事後的な primary 置換ではなく） |
| `collapse_restart_tail_subset_into_following_chord` | `apps/api/app/transcription/events.py` | 後続の chord が明らかに吸収する transient subset event を除去 | restart-tail の最終 chord 手前の subset blip を抑制 | `kalimba-17-c-bwv147-restart-tail-01` practical regression path | 1つの restart-tail failure mode に特化した post-merge cleanup | restart tail 周辺でのより良い event bundling（subset blip が生成されないようにする） |
| `suppress_recent_upper_echo_mixed_clusters` | `apps/api/app/transcription/patterns.py` | 短い upper echo を除去し、後続 mixed cluster から繰り返し upper carryover を除去 | mixed upper BWV フレーズでの extra upper-note echo / over-bundling を防止 | `kalimba-17-c-bwv147-upper-mixed-cluster-01` (completed); parametrized regression via `test_manual_capture_completed.py` | 1つの mixed upper echo パターンに強く特化 | post-merge cleanup 前のより良い carryover / resonance モデリング |
| `lower-mixed-roll-extension` | `apps/api/app/transcription/peaks.py` (`segment_peaks` 内 inline) | 残余 candidate から onset / score 制約を満たす lower/mixed ノートを 1つ追加 | mixed lower-roll フレーズ内の `G4` クラスの欠落ノートを復元 | `kalimba-17-c-bwv147-lower-mixed-roll-01` (completed); `kalimba-17-c-bwv147-lower-context-roll-01` (completed); parametrized regression via `test_manual_capture_completed.py` | BWV lower-roll の語彙と onset shape に狭くチューニングされている | 専用 extension ルールなしでの lower/mixed フレーズ向けのより良い multi-note selection |
| ~~`collect_post_sparse_gap_run_segments`~~ | **削除済み** — `collect_multi_onset_gap_segments` 内の candidate fallback (`_promote_gap_candidates_by_structure`) で代替 | — | — | — | — | — |

## Fixture Coverage Map

### local rules で保護されている BWV147 fixtures

| Fixture | Status | 実質的に保護しているルール | 備考 |
| --- | --- | --- | --- |
| `kalimba-17-c-bwv147-restart-tail-01` | `pending` | `recent-upper-octave-alias-primary`, `collapse_restart_tail_subset_into_following_chord` | 3-note chord の検出が未完。restart-tail-specific cleanup debt の典型例 |
| `kalimba-17-c-bwv147-lower-context-roll-01` | `completed` | `lower-mixed-roll-extension`, `_promote_gap_candidates_by_structure` による candidate promotion | context を保持した lower-roll phrase |
| `kalimba-17-c-bwv147-lower-mixed-roll-01` | `completed` | `lower-mixed-roll-extension`, `_promote_gap_candidates_by_structure` による candidate promotion | scoped-evaluation 解消により completed に昇格 |

### local onset logic が影響する非 BWV practical coverage

| ルール / 領域 | 保護している practical scope |
| --- | --- |
| `collapse_active_range_head_onsets` | `#30` practical fixture recovery set および completed/manual-capture regressions |
| `refine_onset_times_by_attack_profile` | `#30` practical fixture recovery set および completed/manual-capture regressions |

## レビューガイダンス

新しい local rule を追加する際は、以下をすべて記録すること:
- 正確な helper / promotion / suppression 名
- "Keep For Now" か "Candidate For Later Removal" か
- それを必要とする fixture
- パッチしている具体的な failure shape
- 削除可能にする長期的な代替

Free Performance 認識が active goal になった時点で、"Candidate For Later Removal" テーブルを最初にレビューすること。
