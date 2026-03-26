# AGENTS

## Scope

- This file captures repo-specific collaboration rules for Codex/agents working in `C:\src\calimba-score`.
- Keep `main` runnable. Do not leave `main` in a knowingly broken state.

## Python / Test Environment

- Primary API test environment:
  - `C:\src\calimba-score\.venv313`
- API test command:
  - `$env:PYTHONPATH='C:\src\calimba-score\apps\api'; .\.venv313\Scripts\python -m pytest apps/api/tests -q`

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

## Subagent Coordination

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
- Any subagent that may edit files must work in its own git worktree under `.codex-worktrees/<agent-name>/`.
- For editing subagents:
  - create a dedicated branch for that subagent
  - keep the branch name under the `codex/` prefix
  - do not let the subagent edit the main worktree directly
- The primary agent is responsible for:
  - creating the worktree and branch before delegation
  - integrating or cherry-picking the result
  - cleaning up stale `.codex-worktrees/...` directories after the work is resolved
- If a subagent is only doing inspection or debugging, prefer `explorer` plus no file edits over creating a worktree.

## Recognizer Strategy Notes

- Treat repeated-pattern normalizers as suspicious until proven necessary. Favor local/causal explanations over corpus-wide dominant-pattern rewrites.
- Before large recognizer redesigns, add ablation controls and provenance first.
- Enter `xhigh` reasoning only when starting actual large-scale redesign, not for preparatory audits or narrow local fixes.

## Local Paths

- `.codex-*` paths are local-only and must remain ignored.
