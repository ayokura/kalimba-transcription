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

## Alignment Overrides

- `alignment_overrides.json` は、score_structure が楽譜として正しいが録音上の演奏が楽譜と異なる場合に、特定イベントの「この録音での正解」をパッチするための仕組みである。
- score_structure.json は楽譜の意図を表すものであり、変更しない。override は録音固有の事実を記録する。
- `ignoredRanges` と同様、**ユーザーからの明示的な許可または指示がある場合に限り**追加・変更できる。エージェントが独自判断で追加してはならない。
- 各 override には `reason` フィールドで根拠（耳確認、スペクトル分析等）を記録すること。

### Schema

v1 (現行互換) と v2 (op 拡張) をサポート。`op` を省略すると v1 と同じ `replace` 動作。

```json
{
  "version": 2,
  "overrides": [
    {"op": "replace", "eventIndex": 64, "expectedNotes": ["C5","E4"], "reason": "..."},
    {"op": "insert", "afterEventIndex": 115, "expectedNotes": ["E5"],
     "reason": "R3 playing error: E116 D5 missed, E5 played instead; restarted from D5"},
    {"op": "skip", "eventIndex": 170, "reason": "performer skipped this note"}
  ]
}
```

- **replace**: 既存イベント `eventIndex` の `expectedNotes` を上書き。v1 互換 (op 省略可)。
- **insert**: スコアにない余分な演奏 event を `afterEventIndex` の直後に追加。label は `E{afterEventIndex}{suffix}` (例: `E115a`)。同じ `afterEventIndex` に複数 insert がある場合は自動で `a`, `b`, `c`... が付く。明示したい場合は `suffix` フィールドで指定。
- **skip**: スコア上の `eventIndex` が録音では弾かれていない場合に欠番化。

### score_structure との関係

- `score_structure.json` は楽譜の真実 (不変)
- `alignment_overrides.json` は録音固有の差分 (楽譜 → 録音の変換規則)
- `expected.json:expectedEventNoteSetsOrdered` は録音の実順 (テスト assertion 用、alignment_overrides 適用後の最終形と整合するよう手書き)
- diagnosis tool (`score_alignment_diagnosis.py`) は score_structure に alignment_overrides を適用して recognizer 出力と突合する

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
- See [`docs/task-management.md`](/docs/task-management.md) for the current label set and examples.

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
7. **Fixture investigation: eval_scope vs full audio.** `test_manual_capture_completed.py` 系のテストは `transcribe_manual_capture_fixture(...)` 経由でデフォルト `use_evaluation_scope=True` (`evaluationWindows` / `ignoredRanges` でトリミングされた audio)。物理現象を綺麗に観察したい調査時は `transcribe_manual_capture_fixture_full_audio(...)` を使うとクリーンな silent region 分布 + splice 影響なしの状態で確認できるが、**最終的な validation は必ず eval_scope で行う** (テストはそれで動いているため)。両モードで挙動が変わるパスは特に警戒する (#154 noise floor は両モードで `noise_floor[G4]` が ~1.5x ずれた実例)。

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
- **Verify the physical premise before implementing rescue/suppression logic.** Before adding a new pass or tuning a gate, confirm with energy trace + narrow FFT probe + broadband onset times (`gapValidatedOnsetTimes`) that the proposed mechanism matches what the audio actually shows. The originally-stated cause is often wrong in subtle ways: e.g., #153 Phase B's E97 G4 was first thought to need a noise-floor multiplier change, but the actual cause was a broadband-detected onset that the segmenter discarded — a different mechanism entirely. Investigation-first prevents whole-day rabbit holes on the wrong rescue path.
- **Discriminator design beats constant tuning.** When no threshold cleanly separates true positives from false positives, consider whether the candidate iteration order itself is wrong. #153 Phase B replaced narrow-FFT-score-ordered iteration with backward-attack-gain-ordered iteration, which changed the problem from "tighten the constants" to "evaluate the strongest fresh-attack signal first" — and several constants became unnecessary. Ordering by a single physical signal is often cleaner than ordering by a composite score and then patching exceptions.
- **Heuristic constants live in `apps/api/app/transcription/constants.py` and are tracked in [#162](https://github.com/ayokura/kalimba-transcription/issues/162).** When adding a new constant, include its calibration data in the inline comment and append it to the #162 audit body so the inventory and the data-driven-replacement candidate list stay current.

### Broadband patch vs per-note onset detection

現在の recognizer は broadband onset detection（librosa spectral flux 系）をベースに、個別の rescue/gate patch を積み上げて精度を上げている。一方 [#141](https://github.com/ayokura/kalimba-transcription/issues/141) では per-note onset detection という根本的な architecture 変更が提案されている。

**既定方針**: 既存の broadband + patch で対処できるケースは patch で進める。per-note への全面移行は以下のトリガーのいずれかが発生した時点で判断する:

1. **Patch が衝突する** — ある patch が別の patch の前提を壊し、全体として整合的な物理モデルにならなくなったとき
2. **broadband で物理的に検出不能な音が出る** — weak attack で spectral flux が閾値に届かないケース。broadband detection が通っているケース (今日の 10.939s D5 など) は patch で拾える
3. **リアルタイム要求 (streaming transcription)** — batch 前提の broadband 解析では間に合わなくなったとき。per-note state machine (`OFF → ATTACK → BODY → LATE_DECAY`) への移行が必要
4. **Patch 数が fixture 数に近づく** — 一般化できないローカル解決が蓄積したとき

**streaming / WASM 適合性は直交**: broadband patch も per-note も FFT / band energy ベースで WASM 化できる。librosa からの独立は両者で共通の作業量であり、per-note を選ぶ理由にはならない。

**並行路線を推奨**: main line は patch で完成度を上げ、research line (別 branch) で per-note を実験的に検証する。patch で解けないケースを per-note 側で解く、が明確になった時点で merge を判断する。

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
