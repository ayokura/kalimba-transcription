# AGENTS

## Scope

- This file is the shared source of truth for all agents working in this repository.
- Keep `main` runnable. Do not leave `main` in a knowingly broken state.
- Shared rules go here. Agent-specific rules go in the agent-specific Notes sections within this document, or in files referenced from those sections.

## Research References

- AMT (Automatic Music Transcription) の研究サーベイと現行パイプラインへの適用分析が [`docs/research/`](/docs/research/) にある。
- 設計判断やアルゴリズム選択の際は [`20260406-research-to-implementation-mapping.md`](/docs/research/20260406-research-to-implementation-mapping.md) を参照し、研究知見との整合性を確認すること。
- 特に以下の点は設計上の前提として意識する:
  - カリンバの倍音は非整数比（梁振動由来）— 整数倍 harmonic comb の限界を認識する
  - onset の有無を note-on の gate として使う設計が共鳴 FP 抑制に最も効果的
  - attack / body / late_decay の状態遷移モデルが sympathetic resonance との区別に有効

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

- テストは **Mechanism / Fixture regression / Ablation-variant** の3層モデルに従う。
- Fixture regression の権威は `test_manual_capture_completed.py` の parameterized test と `expected.json`。
- Fixture test で `payload["debug"]` の内部構造を exact-match しない。
- Mechanism test は構築入力または marshal した中間データを使い、フルパイプライン実行に依存しない。
- `ground_truth.json` は人間確認済み onset 時刻を絶対秒で記録する optional timing assertion。
- Fixture 調査で full audio を見るのはよいが、最終 validation は必ず regression test と同じ eval_scope で行う。

## Recognizer Strategy Notes

- Treat repeated-pattern normalizers as suspicious until proven necessary. Favor local/causal explanations over corpus-wide dominant-pattern rewrites.
- Before large recognizer redesigns, add ablation controls and provenance first.
- **Verify the physical premise before implementing rescue/suppression logic.** Before adding a new pass or tuning a gate, confirm with energy trace + narrow FFT probe + broadband onset times (`gapValidatedOnsetTimes`) that the proposed mechanism matches what the audio actually shows. The originally-stated cause is often wrong in subtle ways: e.g., #153 Phase B's E97 G4 was first thought to need a noise-floor multiplier change, but the actual cause was a broadband-detected onset that the segmenter discarded — a different mechanism entirely. Investigation-first prevents whole-day rabbit holes on the wrong rescue path.
- **Discriminator design beats constant tuning.** When no threshold cleanly separates true positives from false positives, consider whether the candidate iteration order itself is wrong. #153 Phase B replaced narrow-FFT-score-ordered iteration with backward-attack-gain-ordered iteration, which changed the problem from "tighten the constants" to "evaluate the strongest fresh-attack signal first" — and several constants became unnecessary. Ordering by a single physical signal is often cleaner than ordering by a composite score and then patching exceptions.
- **Heuristic constants live in `apps/api/app/transcription/constants.py`.** The original inventory audit [#162](https://github.com/ayokura/kalimba-transcription/issues/162) is **closed (completed)**; its data-driven-replacement work was split into open successors: [#131](https://github.com/ayokura/kalimba-transcription/issues/131) (migrate tunable thresholds to `RecognizerSettings`) and [#172](https://github.com/ayokura/kalimba-transcription/issues/172) / [#173](https://github.com/ayokura/kalimba-transcription/issues/173) / [#174](https://github.com/ayokura/kalimba-transcription/issues/174) (per-tine / per-recording / BPM-adaptive calibration). When adding a new constant, include its calibration data in the inline comment, and route any data-driven-replacement candidate to the relevant open successor (#131 or #172–#174). Do **not** append to the closed #162 audit body.

### Broadband patch vs per-note onset detection

現在の recognizer は broadband onset detection（pure-numpy 化された spectral flux ベース。librosa からの移植コードだが、recognizer 自体は #187 / #193 で librosa-free）をベースに、個別の rescue/gate patch を積み上げて精度を上げている。一方 [#141](https://github.com/ayokura/kalimba-transcription/issues/141) では per-note onset detection という根本的な architecture 変更が提案されている。

**既定方針**: 既存の broadband + patch で対処できるケースは patch で進める。per-note への全面移行は以下のトリガーのいずれかが発生した時点で判断する:

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
