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
- `slide_chord` classification
- debug UI improvements
- editor workflow changes
- test infrastructure work
- performance and deployment milestones

Issue labels now use the shared taxonomy below instead of older ad-hoc labels.

## Current GitHub Label Taxonomy

Use labels in three layers.

- `area:*`
  - broad product surface
  - choose exactly one when possible
- `type:*`
  - the kind of work being tracked
  - choose exactly one when possible
- `component:*`
  - concrete code or ownership slice
  - use only when the target is clear; it is optional

Current `area:*` labels:

- `area:recognizer`
- `area:web`
- `area:fixture`
- `area:docs`
- `area:infra`

Current `type:*` labels:

- `type:analysis`
- `type:design`
- `type:implementation`
- `type:refactor`
- `type:research`
- `type:docs`
- `type:perf`

Current `component:*` labels:

- `component:transcription-package`
- `component:transcription-segments`
- `component:transcription-peaks`
- `component:transcription-patterns`
- `component:fixture-metadata`
- `component:web-review-workspace`
- `component:dependency-management`

Practical rules:

- `area:*` and `type:*` are the default minimum set for every issue, including closed issues kept for history.
- Add `component:*` only when it helps route work or filter related issues. Leave it unset if the issue spans multiple layers or the target is still ambiguous.
- Prefer the narrowest component that is still stable across refactors. For example, use `component:transcription-peaks` for `segment_peaks` work, but use `component:transcription-package` for package-boundary work such as module splits or `_legacy.py` cleanup.
- When an issue clearly spans two concrete components, multiple `component:*` labels are acceptable, but keep that exceptional.

Examples:

- residual-decay rejection in `segment_peaks`
  - `area:recognizer`, `type:implementation`, `component:transcription-peaks`
- fixture audit of `expected.json`
  - `area:fixture`, `type:analysis`, `component:fixture-metadata`
- review workspace playback UI
  - `area:web`, `type:implementation`, `component:web-review-workspace`
- dependency registry / lockfile work
  - `area:infra`, `type:implementation`, `component:dependency-management`

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
   - `slide_chord`
   - `slide_chord` / sweep
   - ambiguous
2. debug review UX
   - expected vs detected diff
   - rerecord / review-needed states
   - fragment warnings
3. editor UX
   - merge as chord
   - bundle as `slide_chord`
4. broader recognition robustness
   - contact noise
   - room noise
   - mic position variance

