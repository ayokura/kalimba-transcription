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

**2026-04-09 時点: 現時点で候補なし。**  2026-04-09 の G ablation 第 1 + 2 弾 (計 7 関数削除) により、過去に debt として記録されていた fixture-specific ルールはすべて削除済み。今後新たに狭い fixture 固有ルールを追加した場合のみ本テーブルに掲載する。

過去に本テーブルに掲載されていた削除済みエントリ:

| ルール | 削除 commit | 削除日 |
| --- | --- | --- |
| `collect_two_onset_terminal_tail_segments` | #122 | — |
| `collect_post_sparse_gap_run_segments` | — | — |
| `suppress_recent_upper_echo_mixed_clusters` | 9af12c3 | 2026-04-09 |
| `lower-mixed-roll-extension` (`_extend_lower_mixed_roll`) | 3fde670 | 2026-04-09 |
| `recent-upper-octave-alias-primary` promotion | c92b040 | 2026-04-09 |
| `collapse_restart_tail_subset_into_following_chord` | c92b040 | 2026-04-09 |
| `suppress_descending_restart_residual_cluster` | 2c901f4 | 2026-04-09 |
| `suppress_descending_terminal_residual_cluster` | a47f13e | 2026-04-09 |
| `suppress_resonant_carryover` | 677feaf | 2026-04-09 |

## Fixture Coverage Map

### local rules で保護されている BWV147 fixtures

**2026-04-09 時点: 空。** 過去に local rules で保護されていた BWV147 child fixture はすべて完了 (sequence-163 に昇華) または削除済み。

| Fixture | Status | 実質的に保護しているルール | 備考 |
| --- | --- | --- | --- |

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
