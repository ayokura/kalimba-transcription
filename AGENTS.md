# AGENTS

## Scope

- This file is the shared source of truth for all agents working in this repository.
- Keep `main` runnable. Do not leave `main` in a knowingly broken state.
- Shared rules go here. Agent-specific rules go in the agent-specific Notes sections within this document, or in files referenced from those sections.

## Research References

- AMT (Automatic Music Transcription) の研究サーベイと現行パイプラインへの適用分析が [`docs/research/`](/docs/research/) にある。読み順は [`index.md`](/docs/research/index.md)。
- **新規の設計判断やアルゴリズム選択では、まず [`20260626-unbiased-amt-reassessment.md`](/docs/research/20260626-unbiased-amt-reassessment.md) を確認すること** (実コード確認済みの実装事実テーブル + NOW/NEXT/LATER 方針)。旧 `20260406-*` サーベイ群は LLM 由来バイアスありの deprecated 資料であり、設計判断の根拠に使わない。
- 中期の作業計画とその優先順位は [`docs/sprint-plan-2026-07b.md`](/docs/sprint-plan-2026-07b.md) (第 2 期、2026-07-04〜) を参照。第 1 期 [`docs/sprint-plan-2026-07.md`](/docs/sprint-plan-2026-07.md) は superseded (実績記録として凍結)。
- 期をまたぐ戦略判断の経緯は [`docs/decision-log.md`](/docs/decision-log.md) に日付入りで記録されている。大きな方向転換をした時はここに追記する (追記のみ、書き換え不可)。
- 特に以下の点は設計上の前提として意識する (reassessment §1 の実装事実に基づく):
  - カリンバの倍音は非整数比（梁振動由来）— 整数倍 harmonic comb の限界を認識する
  - onset gate は実装済み・既定有効 (broadband / per-note / backward-attack の弱 AND)。共鳴 FP 抑制の中核だが「安全に落とす」より「低 confidence 候補として保持する」方向が現方針
  - attack / body / late_decay の note-state machine は**未実装**であり、導入は research spike / ablation 限定 (本線は broadband + patch、下記「Broadband patch vs per-note onset detection」参照)
  - per-tine partial scoring は実装済みだが既定無効。信念による既定化はしない (#149)

## Product Vision and Technical Direction

- The end goal is transcription of free-form kalimba performance into sheet music, without any Expected Performance or prior knowledge of what will be played.
- Current fixtures with Expected Performance are stepping stones for building and validating the recognizer; the final product must work without them.
- Design decisions should favor approaches that work without Expected Performance over those that depend on it.
- Future UX direction includes near-real-time (streaming) transcription. Prefer causal/streaming-friendly algorithms over batch-only approaches where quality is comparable.
- A future milestone is browser-side-only implementation (no server round-trip) using WebAudio API and/or WebAssembly. Keep recognizer logic portable:
  - Avoid deep coupling to Python-specific libraries (librosa, numpy) in core algorithm design.
  - Prefer simple, well-defined numerical operations over complex library-specific abstractions.
  - This does not mean avoiding these libraries now, but the algorithmic logic should be expressible without them.

## Python / Test Environment

- Primary API test environment: uv (managed via `pyproject.toml` + `uv.lock`)
- API test command (run from the repository root):
  - `uv run pytest apps/api/tests -q`
- `pytest.ini` sets `pythonpath = . apps/api`, so no `PYTHONPATH` environment variable is needed.

## Host-Local Runtime & Deploy

- Production serving, systemd units, ports, the Cloudflare tunnel, and the deploy procedure are **host-specific** and live outside git in `.runtime-local/` (gitignored; agent-neutral, not tied to any one agent).
- **Before any production deploy or runtime operation (restart, log inspection, port assumptions), read `.runtime-local/deploy.md` on this host.** It is non-portable and may differ or be absent on other hosts/checkouts — never assume it exists or that its values apply elsewhere.
- Keep secrets out of `.runtime-local/`; record only their locations (e.g. cloudflared credentials live under `~/.cloudflared/`).
- `docs/deploy-cloudflare-tunnel.md` (checked in) is the portable procedure template; `.runtime-local/deploy.md` holds this host's concrete values. Keep host-specific facts in the latter, not in committed docs.

## Fixture Policy

- Prefer practical musical fixtures over synthetic microbenchmarks when the two conflict.
- If a change improves a practical fixture and regresses only synthetic or single-event-repeat fixtures, the practical fixture may take precedence.
- If a change improves one practical fixture but regresses another practical fixture, do not merge it directly. Keep it as a spike until the tradeoff is resolved.
- If a fixture contains a locally invalid take or fragment, prefer `ignoredRanges` or fixture reclassification over distorting recognizer logic to force a pass.
- Use statuses deliberately:
  - `completed`: stable regression target
  - `pending`: valuable target, recognizer still needs work
  - `rerecord`: data quality or capture intent is not good enough
  - `review_needed`: metadata or interpretation still unclear
  - `reference_only`: retain for reference, not active regression

## Corpus Management / Rights Review

- Free-performance corpus management is documented in [`docs/corpus-management.md`](docs/corpus-management.md).
- Repository-managed free-performance recordings live under
  `apps/api/tests/fixtures/free-performance-corpus/<transaction-id>/`.
- Audio / teacher data may be committed there only after a **human**
  copyright/rights review. `metadata.json.rightsReview.status` must be
  `approved_for_repository`, and `copyright.status` must be cleared
  (`original_performance`, `public_domain`, or equivalent).
- Useful but copyright-unknown or non-clearable recordings should remain
  local-only in `data/transactions/` and/or ignored transaction-capture
  fixtures. Do not commit their audio or teacher data.
- When capture hardware is known, record it in corpus metadata (e.g.
  `recording.device`, `recording.microphone`) because it may become useful for
  future recording-profile calibration.
- Review-UI-derived `ground_truth.json` (`method: "user_corrected"`) is reliable
  primarily for note identity / order / grouping, but onset `timeSec` is
  approximate. Recognizer event starts can deviate from perceptual/spectral
  onsets, `inserted-slot` uses dropped-segment boundaries, and `inserted-manual`
  is hand-placed. Use wider `toleranceSec` and require spectral/human
  re-verification before timing-sensitive training or calibration.

## Fixture Alignment / Overrides

- `score_structure.json` は楽譜の意図を表す。録音固有の演奏差分に合わせて変更しない。
- `alignment_overrides.json` は録音固有の差分（楽譜 → 録音の変換規則）を記録する。
- `alignment_overrides.json` と `ignoredRanges` は、**ユーザーからの明示的な許可または指示がある場合に限り**追加・変更できる。エージェントが独自判断で追加してはならない。
- 各 override には `reason` フィールドで根拠（耳確認、スペクトル分析等）を記録すること。
- schema と `score_structure.json` / `expected.json` との関係は [`docs/fixture-alignment.md`](docs/fixture-alignment.md) を参照。

## Spike / Rollback Policy

- Main-agent-only rule: for promising but not-yet-mergeable recognizer changes, use a dedicated `codex/...` branch.
- Do not keep speculative or knowingly regressive spikes on `main`.
- Preserve reusable failed experiments only when all of the following are true:
  - the target practical fixture clearly improves
  - the approach may be useful later
  - the regression risk is understandable and documented
- Discard low-value experiments that only add noise.
- If the primary agent tries a change on `main` and later decides not to keep it, archive that experiment by:
  - recreating it on a dedicated `codex/...` branch
  - committing it there with a detailed commit body
  - returning `main` to the clean accepted state
- If the archived experiment is important enough to track, the primary agent may add a short indexed issue comment using the spike tags below.

## How To Record Spike History

- Put the detailed rationale in the commit body.
- For primary-agent spike archives, add a short issue comment as an index, not as the full writeup, when the spike is important enough to keep discoverable.
- Use the following tags in the issue comment so spike history can be filtered:
  - `[spike-archive]`
  - `[fixture: <fixture-id>]`
  - `[regressed: <fixture-id-or-none>]`
  - `[branch: <branch-name>]`
  - `[commit: <sha>]`
- The issue comment should say only:
  - what improved
  - what regressed
  - why it was not merged
  - that the detailed explanation is in the commit body
- Subagents should not create separate spike-archive branches of their own for this purpose. Because subagents already work on isolated branches/worktrees, they should preserve undoable experiment history with normal commits plus explicit revert commits when needed.

## Common Agent Workflow

- When asking subagents to inspect an issue, do not pass only an issue number.
- Always include either:
  - the issue title and summary, or
  - the relevant local problem statement directly
- Close subagents after their result is integrated.
- Use subagents for independent, high-value parallel analysis when the active runtime and higher-priority instructions permit it; keep write scopes explicit when delegating implementation.
- Explorer subagents should treat file edits as exceptional, not normal.
- Explorer subagents must not edit the main worktree directly.
- If an explorer concludes that a file edit is necessary for the investigation, it must:
  - stop before editing
  - report that need back to the primary agent
  - let the primary agent either apply the change itself or move the task into a dedicated editing worktree
- Any subagent that may edit files must use a dedicated worktree rather than the main worktree.
- For editing subagents:
  - use a dedicated branch for that subagent
  - do not let the subagent edit the main worktree directly
- Dedicated worktree setup, integration, and cleanup should be handled by the primary agent unless the active toolset provides equivalent isolation automatically.
- If a subagent is only doing inspection or debugging, prefer `explorer` plus no file edits over creating a worktree.

## Issue Labeling

- GitHub Issues use a three-layer label taxonomy:
  - `area:*` for broad product surface
  - `type:*` for the nature of the work
  - optional `component:*` for a concrete code or ownership slice
- New issues should normally get one `area:*` label and one `type:*` label.
- Add `component:*` only when the implementation target is already clear enough to be useful for routing or filtering.
- When package-boundary or cross-module work does not fit a narrower component cleanly, prefer a broader component label rather than forcing a misleading one.
- See [`docs/task-management.md`](/docs/task-management.md) for the current label set and examples.

## Test Architecture

詳細は [`docs/testing.md`](docs/testing.md) を参照。AGENTS.md では以下のみ常時ルールとして維持する。

- テストは **Mechanism / Fixture regression / Ablation-variant / Corpus benchmark regression** の4層モデルに従う。
- Fixture regression の権威は `test_manual_capture_completed.py` の parameterized test と `expected.json`。
- Corpus benchmark regression の権威は `free-performance-corpus/benchmark_baseline.json` の per-recording floor (`test_free_performance_corpus.py` が検証)。baseline の更新は改善方向のみ (`--write-baseline`)。低下を伴う更新は fixture policy 相当の明示的 tradeoff 判断が必要。
- Fixture test で `payload["debug"]` の内部構造を exact-match しない。
- Mechanism test は構築入力または marshal した中間データを使い、フルパイプライン実行に依存しない。
- `ground_truth.json` は人間確認済み onset 時刻を絶対秒で記録する optional timing assertion。
- Fixture 調査で full audio を見るのはよいが、最終 validation は必ず regression test と同じ eval_scope で行う。

## Recognizer Strategy Notes

- Treat repeated-pattern normalizers as suspicious until proven necessary. Favor local/causal explanations over corpus-wide dominant-pattern rewrites.
- Before large recognizer redesigns, add ablation controls and provenance first.
- **過適合ゲート: 閾値調整を伴う recognizer 改修は、GT レビュー済みの非飽和録音 (F1 < 1.0) が 2 件以上あることを条件とする。** 非飽和録音 1 件を相手に閾値を調整すると、tuning-set 飽和 (F1=1.000 で指標が何も教えなくなった状態) の再演になる。構造的欠陥の修正 (onset は検出済みなのに segment が形成されない #197 型) はこの条件の対象外。F1=1.000 は成功指標ではなく飽和のサインとして扱う。
- **Verify the physical premise before implementing rescue/suppression logic.** Before adding a new pass or tuning a gate, confirm with energy trace + narrow FFT probe + broadband onset times (`gapValidatedOnsetTimes`) that the proposed mechanism matches what the audio actually shows. The originally-stated cause is often wrong in subtle ways: e.g., #153 Phase B's E97 G4 was first thought to need a noise-floor multiplier change, but the actual cause was a broadband-detected onset that the segmenter discarded — a different mechanism entirely. Investigation-first prevents whole-day rabbit holes on the wrong rescue path.
- **Discriminator design beats constant tuning.** When no threshold cleanly separates true positives from false positives, consider whether the candidate iteration order itself is wrong. #153 Phase B replaced narrow-FFT-score-ordered iteration with backward-attack-gain-ordered iteration, which changed the problem from "tighten the constants" to "evaluate the strongest fresh-attack signal first" — and several constants became unnecessary. Ordering by a single physical signal is often cleaner than ordering by a composite score and then patching exceptions.
- **Heuristic constants live in `apps/api/app/transcription/constants.py`.** The original inventory audit [#162](https://github.com/ayokura/kalimba-transcription/issues/162) is **closed (completed)**; its data-driven-replacement work was split into open successors: [#131](https://github.com/ayokura/kalimba-transcription/issues/131) (migrate tunable thresholds to `RecognizerSettings`) and [#172](https://github.com/ayokura/kalimba-transcription/issues/172) / [#173](https://github.com/ayokura/kalimba-transcription/issues/173) / [#174](https://github.com/ayokura/kalimba-transcription/issues/174) (per-tine / per-recording / BPM-adaptive calibration). When adding a new constant, include its calibration data in the inline comment, and route any data-driven-replacement candidate to the relevant open successor (#131 or #172–#174). Do **not** append to the closed #162 audit body.

### Broadband patch vs per-note onset detection

現在の recognizer は broadband onset detection（pure-numpy 化された spectral flux ベース。librosa からの移植コードだが、recognizer 自体は #187 / #193 で librosa-free）をベースに、個別の rescue/gate patch を積み上げて精度を上げている。一方 [#141](https://github.com/ayokura/kalimba-transcription/issues/141) では per-note onset detection という根本的な architecture 変更が提案されている。

**既定方針 (2026-07-04 改訂)**: broadband ベースは維持するが、**events.py への新規 suppression pass の追加は禁止** — トリガー 4 (下記) が限界域に達したため (pass 32 / gate reason ~40 vs fixture 35、2026-07-04 監査)。新規 pass に相当する変更は #141 research spike (research branch + dual-run) 経由でのみ試す。**非 pass 形の改修 (候補保持 / 降格 / provenance / 既存 pass の除去・簡素化) は従来どおり可**。per-note への全面移行は以下のトリガーのいずれかが発生した時点で判断する (トリガーの数値判定は ablation observatory — 第 2 期 S4 — で自動化予定):

1. **Patch が衝突する** — ある patch が別の patch の前提を壊し、全体として整合的な物理モデルにならなくなったとき
2. **broadband で物理的に検出不能な音が出る** — weak attack で spectral flux が閾値に届かないケース。broadband detection が通っているケース (今日の 10.939s D5 など) は patch で拾える
3. **リアルタイム要求 (streaming transcription)** — batch 前提の broadband 解析では間に合わなくなったとき。per-note state machine (`OFF → ATTACK → BODY → LATE_DECAY`) への移行が必要
4. **Patch 数が fixture 数に近づく** — 一般化できないローカル解決が蓄積したとき

**streaming / WASM 適合性は直交**: broadband patch も per-note も FFT / band energy ベースで WASM 化できる。recognizer は既に librosa-free (#187 / #193 で pure-numpy 化済み) なので、ライブラリ独立は per-note を選ぶ理由にはならない。

**並行路線を推奨**: main line は patch で完成度を上げ、research line (別 branch) で per-note を実験的に検証する。patch で解けないケースを per-note 側で解く、が明確になった時点で merge を判断する。

## Claude Code-Specific Notes

- Claude Code reads this file via `@AGENTS.md` in CLAUDE.md.
- Detailed agent-specific rules for Claude Code are maintained in CLAUDE.md; this section only summarizes cross-cutting conventions and defaults.
- Explorer subagents: use `subagent_type: "Explore"` — this type has no Edit/Write access by design, satisfying the "explorer must not edit" rule automatically.
- Editing subagents: use `isolation: "worktree"` on the Agent tool — equivalent isolation is provided automatically, no manual worktree setup needed.
- Branch prefix for Claude Code-initiated spikes: `claude/` (mirrors Codex's `codex/` convention).

## Codex-Specific Notes

- In Codex, editing subagents must use dedicated worktrees under `.codex-worktrees/<agent-name>/` unless the active toolset provides equivalent automatic isolation.
- In Codex, editing subagent branches should use the `codex/` prefix.
- In Codex subagent coordination, do not pass only an issue number; include the issue title/summary or the local problem statement.
- `.codex-*` paths are local-only and must remain ignored.
- Runtime-specific guidance for Codex should be applied explicitly by shell/runtime:
  - WSL/Linux:
    - Prefer the repo-standard `uv` workflow directly.
    - Standard API test command remains `uv run pytest apps/api/tests -q`.
    - If the standard pytest run fails with a temp/capture `FileNotFoundError` (seen on Windows-mounted `/mnt/...` worktrees), rerun with `TMPDIR=/tmp uv run pytest apps/api/tests -q`.
    - `gh` auth, when present, is read from a local `.codex-gh/gh.env` (gitignored, may not exist in every checkout). If that file sets `GH_CONFIG_DIR` to a Windows path, normalize it to the `.codex-gh` directory inside the current repo root before running `gh`.
  - Windows PowerShell:
    - The repo has shifted toward WSL/Linux as the primary runtime; call out the environment mismatch explicitly before assuming parity with the user's shell.
    - `gh` auth, when configured, is read from the local `.codex-gh/gh.env` (gitignored; may be absent).
