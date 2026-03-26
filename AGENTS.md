# AGENTS

## Scope

- This file is the shared source of truth for all agents working in this repository.
- Keep `main` runnable. Do not leave `main` in a knowingly broken state.
- Shared rules go here. Agent-specific rules go in the agent-specific Notes sections within this document, or in files referenced from those sections.

## Python / Test Environment

- Primary API test environment:
  - `.venv313`
- API test command (run from the repository root so that `PYTHONPATH=apps/api` resolves correctly):
  - PowerShell: `$env:PYTHONPATH='apps/api'; .\.venv313\Scripts\python -m pytest apps/api/tests -q`
  - Bash (Git Bash on Windows, Windows-created venv): `PYTHONPATH=apps/api .venv313/Scripts/python -m pytest apps/api/tests -q`
  - Other environments (Linux/macOS): venv layout and paths differ; test commands need separate consideration.
- Note: `PYTHONPATH` is required because `apps/api` is not installed as a package. See issue #24 for a potential fix via `pyproject.toml`.

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
