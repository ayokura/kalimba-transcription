# AGENTS

## Scope

- This file is the shared source of truth for all agents working in this repository.
- Keep `main` runnable. Do not leave `main` in a knowingly broken state.
- Shared rules go here. Agent-specific rules go in the agent-specific Notes sections within this document, or in files referenced from those sections.

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

## Alignment Overrides

- `alignment_overrides.json` は、score_structure が楽譜として正しいが録音上の演奏が楽譜と異なる場合に、特定イベントの「この録音での正解」をパッチするための仕組みである。
- score_structure.json は楽譜の意図を表すものであり、変更しない。override は録音固有の事実を記録する。
- `ignoredRanges` と同様、**ユーザーからの明示的な許可または指示がある場合に限り**追加・変更できる。エージェントが独自判断で追加してはならない。
- 各 override には `reason` フィールドで根拠（耳確認、スペクトル分析等）を記録すること。

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
- Use subagents aggressively for parallel analysis, but keep write scopes explicit when delegating implementation.
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
- See [`docs/task-management.md`](/mnt/c/src/calimba-score/docs/task-management.md) for the current label set and examples.

## Test Architecture

テストは3層モデルに従う:

| Tier | 目的 | 入力 | アサーション対象 |
|------|------|------|----------------|
| **Mechanism** | 個別の recognizer 関数の動作確認 | 構築した RawEvent/NoteCandidate、または marshal した中間データ | 関数の戻り値 |
| **Fixture regression** | 実音声に対する転写結果の回帰検出 | WAV ファイル (HTTP 経由) | expected.json のアサーション (イベント数、ノートセット、順序) |
| **Ablation/variant** | フィーチャーフラグ変更が既存 fixture を壊さないか | WAV ファイル + monkeypatch | expected.json のアサーション |

### ルール

1. **Fixture テストで debug 構造をアサートしない。** `payload["debug"]` のサブフィールド（`segmentCandidates`, `multiOnsetGapSegments`, `secondaryDecisionTrail`, `mergedEvents` 等）を exact-match でピン留めしない。recognizer リファクタリングで内部出力が変わっても転写結果が正しければテストは壊れてはならない。
2. **Mechanism テストは構築入力を使う。** `RawEvent`/`NoteCandidate` を直接構築して関数を呼ぶ。合成音声で `segment_peaks` を直接呼ぶのも可。
3. **実データが必要な mechanism テストは marshal する。** パイプラインの中間状態を JSON にダンプし、`apps/api/tests/fixtures/mechanism-snapshots/` に保存して読み込む。フルパイプライン実行で中間状態を取得してはならない。
4. **Fixture 回帰の権威は parameterized テスト。** `test_manual_capture_completed.py` が全 completed fixture を自動的に検証する。個別の fixture テストを `test_api.py` に追加する前に、`expected.json` のアサーションで表現できないか検討する。
5. **個別 fixture テストは parameterized で表現不可能な場合のみ。** pending/review_needed fixture の暫定チェック、またはイベント間の関係性など `expected.json` で表現できないアサーションに限る。
6. **ground_truth.json でタイミング情報を管理する。** 人間が耳・スペクトログラムで確認した onset 時刻を `ground_truth.json` に記録する。librosa の onset 検出に依存しない絶対秒で記録し、自動テストでの timing 検証に使用する。

### ground_truth.json スキーマ

```json
{
  "version": 1,
  "toleranceSec": 0.05,
  "onsets": [
    {"timeSec": 1.05, "notes": ["C4"], "method": "ear_verified"},
    {"timeSec": 2.03, "notes": ["D4"], "toleranceSec": 0.08, "method": "spectrogram_verified", "comment": "soft attack"}
  ]
}
```

- `timeSec`: audio.wav 先頭からの絶対秒
- `toleranceSec`: デフォルト50ms、onset ごとにオーバーライド可能
- `method`: `ear_verified`, `spectrogram_verified`, `aubio_cross_checked`
- ファイルはオプショナル（存在する fixture のみ timing チェック実施）

## Recognizer Strategy Notes

- Treat repeated-pattern normalizers as suspicious until proven necessary. Favor local/causal explanations over corpus-wide dominant-pattern rewrites.
- Before large recognizer redesigns, add ablation controls and provenance first.

## Claude Code-Specific Notes

- Claude Code reads this file via `@AGENTS.md` in CLAUDE.md.
- Detailed agent-specific rules for Claude Code are maintained in CLAUDE.md; this section only summarizes cross-cutting conventions and defaults.
- Explorer subagents: use `subagent_type: "Explore"` — this type has no Edit/Write access by design, satisfying the "explorer must not edit" rule automatically.
- Editing subagents: use `isolation: "worktree"` on the Agent tool — equivalent isolation is provided automatically, no manual worktree setup needed.
- Branch prefix for Claude Code-initiated spikes: `claude/` (mirrors Codex's `codex/` convention).

## Codex-Specific Notes

- In Codex, editing subagents must use dedicated worktrees under `.codex-worktrees/<agent-name>/` because the toolset does not provide equivalent automatic isolation.
- In Codex, editing subagent branches should use the `codex/` prefix.
- In Codex subagent coordination, do not pass only an issue number; include the issue title/summary or the local problem statement.
- In Codex/GPT-5.4-era reasoning controls, enter `xhigh` only when starting actual large-scale redesign, not for preparatory audits or narrow local fixes. Re-evaluate this guidance if the toolset or reasoning-tier definitions change.
- `.codex-*` paths are local-only and must remain ignored.
- Runtime-specific guidance for Codex should be applied explicitly by shell/runtime:
  - WSL/Linux:
    - Prefer the repo-standard `uv` workflow directly.
    - Standard API test command remains `uv run pytest apps/api/tests -q`.
    - In this WSL `/mnt/c/...` worktree, if the standard pytest run fails with a temp/capture `FileNotFoundError`, rerun with `TMPDIR=/tmp uv run pytest apps/api/tests -q`.
    - In this workspace, `gh` auth is expected from `.codex-gh/gh.env`; if that file sets `GH_CONFIG_DIR` to a Windows path, normalize or override it to `/mnt/c/src/calimba-score/.codex-gh` before running `gh`.
  - Windows PowerShell:
    - The repo has shifted toward WSL/Linux as the primary runtime; call out the environment mismatch explicitly before assuming parity with the user's shell.
    - In this workspace, `gh` auth is expected from `.codex-gh/gh.env`.
