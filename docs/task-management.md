# Task Management

## Recommended Setup

Use a mixed approach.

- GitHub Issues and a small GitHub Project are the source of truth for prioritized work
- Repo-local docs and fixture folders are the source of truth for recording evidence, fixture state, and regression details

This project already has a local-first loop:

1. record in the debug UI
2. export a capture pack
3. import it into fixtures
4. promote it to regression only when it is stable

That loop does not map cleanly to "one GitHub issue per recording".

## Put in GitHub

Create Issues for work that should survive a single debug session.

Examples:

- DSP accuracy themes
- gliss / rolled-chord classification
- debug UI improvements
- editor workflow changes
- test infrastructure work
- performance and deployment milestones

Recommended labels:

- `dsp`
- `fixture`
- `rerecord`
- `debug-ui`
- `editor`
- `test-infra`
- `research`

Recommended Project columns:

- `Inbox`
- `Active`
- `Waiting on Recording/Audit`
- `Done`

## Keep in Repo

Keep short-lived evidence and operational truth in the repository.

- `apps/api/tests/fixtures/manual-captures/...`
- `docs/testing.md`
- fixture `notes.md` and `expected.json`

This is where recording intent, rerecord guidance, fixture status, and audit notes belong.

## Practical Rule

- If several captures point to the same underlying recognition problem, open or update one GitHub Issue
- If a capture is only evidence for an existing problem, keep it local and link it from the issue if needed
- Do not create GitHub Issues for every ad-hoc recording

## Current Medium-Term Themes

1. gesture classification
   - strict chord
   - rolled chord
   - gliss / sweep
   - ambiguous
2. debug review UX
   - expected vs detected diff
   - rerecord / review-needed states
   - fragment warnings
3. editor UX
   - merge as chord
   - bundle as gliss
4. broader recognition robustness
   - contact noise
   - room noise
   - mic position variance

