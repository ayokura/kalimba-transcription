#!/usr/bin/env python3
"""Metamorphic alarm v0: GT-free output-consistency check (S2, sprint-plan-2026-07c.md).

A metamorphic test does not need ground truth: it applies a transform that is
*known* to preserve onset time and pitch (a "benign" DSP transform) and checks
that the recognizer's own output does not change. Where a benign transform
does change the output, that is evidence of a fragile decision boundary
between suppression/rescue passes (third-term guardrail "トリガー 1: patch が
衝突する" in AGENTS.md's Broadband patch vs per-note onset detection section) —
not a labeled quality regression, since there is no ground truth involved here
at all.

REPORT-ONLY / NON-BLOCKING (AGENTS.md guardrail 7, docs/sprint-plan-2026-07c.md
S2 instrumentation lane): this script never gates CI, is not wired into
pytest, and has no baseline file. Exit code is always 0 unless ``--strict`` is
passed. WARN output is a lead for investigation, not a pass/fail verdict.

Reused assets (not reimplemented):
- ``augmentation_robustness.py`` supplies the transform implementations
  (``transform_none`` / ``transform_gain`` / ``transform_lowpass``), the
  ``Condition`` dataclass, audio loading (``load_audio_mono``), WAV encoding
  (``encode_wav_bytes``), and the TestClient transcribe helper
  (``transcribe_bytes``) — imported and called directly, not copied.
- ``note_f1_benchmark.py`` supplies ``collect_one_best`` (event/notes ->
  flat note-time pairs) and, critically, ``match_pairs`` — the same
  nearest-unused-same-note-within-tolerance matching algorithm used for GT
  comparison is reused here for baseline-output-vs-transformed-output
  comparison, by treating the baseline recognizer output as a pseudo ground
  truth. ``match_pairs``'s "false negatives" become "dropped events" (present
  before the transform, missing after) and its "false positives" become
  "added events" (new after the transform, absent before).

Transform-set selection rationale (v0, 3 conditions) — read from
``reports/augmentation_robustness.md`` (2026-07-03 run, 2 recordings:
17ea7626-... and bbd6797f-...), re-derived per-recording (not just the
corpus-mean, since the corpus mean can hide a large per-recording swing
behind an already-saturated, insensitive second recording):

  family     level    dF1 (17ea7626)  dPred (17ea7626)  dF1 (bbd6797f)  dPred (bbd6797f)
  gain       +6dB     +0.0028         +2                +0.0000         +0
  lowpass    8000hz   -0.0090         +1                +0.0000         +0
  reverb     light    -0.1159         -4                +0.0000         +0
  gain       -10dB    -0.0842         +0 (but TP -4: note-identity swap)  -0.2000  +0
  noise_pink snr30    -0.1665         -6                +0.0000         +0
  noise_white snr30   -0.2221         -17               -0.2000         +0

Only ``gain +6dB`` and ``lowpass 8000hz`` show a negligible F1/predicted-count
delta on *both* recordings, including the non-saturated one where a real
change would actually show up (bbd6797f is F1=1.000 at baseline, so it cannot
reveal degradation for most families). Every reverb/noise family's mildest
tested level already moves F1 by 0.08-0.22 on the non-saturated recording —
not "unchanged", so those families are excluded from v0 even though the
report's cross-recording *mean* delta looks small for reverb light/medium
(the mean is diluted by the saturated recording contributing exactly 0). A
future v1 could recalibrate much lighter reverb/noise levels specifically for
alarm use (distinct from the F1-robustness-map calibration, which optimizes
for a realistic degradation *slope*, not for zero-delta invariance).

The third v0 condition, ``identity/control`` (``transform_none`` applied and
re-transcribed independently of the one-time baseline transcription), is not
a DSP perturbation at all — it is a determinism/reproducibility canary. If
byte-identical audio ever produces a different transcription across two
TestClient calls, that is a more fundamental problem than patch-boundary
fragility and this control condition exists to catch it in the same run.

Alarm rule v0: per (recording, condition), let ``diff`` = number of added
events + number of dropped events (symmetric difference over note-level
onset+name matches at the tolerance below). WARN when
``diff > max(min_abs_notes, min_pct * baseline_note_count)``
(defaults: 2 notes / 5%).

Cadence: on-demand only for v0. Nightly/scheduled execution is a human
decision not made yet (docs/sprint-plan-2026-07c.md open question #11) — do
not wire this into any cron/CI job without that decision.

Usage:
  uv run python scripts/audio-analysis/metamorphic_alarm.py
  uv run python scripts/audio-analysis/metamorphic_alarm.py --json
  uv run python scripts/audio-analysis/metamorphic_alarm.py --tx 17ea7626-3c5d-450d-ae74-0116dea6e881
  uv run python scripts/audio-analysis/metamorphic_alarm.py --markdown-out scripts/audio-analysis/reports/metamorphic_alarm.md
  uv run python scripts/audio-analysis/metamorphic_alarm.py --strict   # exit 1 if any WARN
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import augmentation_robustness as aug  # noqa: E402
import note_f1_benchmark as nfb  # noqa: E402

from apps.api.app.fingerprints import (  # noqa: E402
    kalimba_dsp_fingerprint,
    recognizer_fingerprint,
)

DEFAULT_TOLERANCE_SEC = nfb.DEFAULT_TOLERANCE_SEC  # 0.05s, same as GT matching
DEFAULT_MIN_ABS_NOTES = 2
DEFAULT_MIN_PCT = 0.05
DEFAULT_JSON_OUT = SCRIPT_DIR / "reports" / "metamorphic_alarm.json"
DEFAULT_MARKDOWN_OUT = SCRIPT_DIR / "reports" / "metamorphic_alarm.md"

# v0 transform set. See module docstring for the selection rationale (only
# conditions with a negligible F1/predicted-count delta on BOTH recordings in
# the augmentation_robustness.py report, plus one determinism control).
CONDITIONS: list[aug.Condition] = [
    aug.Condition("identity", "control", aug.transform_none, {}),
    aug.Condition("gain", "+6dB", aug.transform_gain, {"db": 6.0}),
    aug.Condition("lowpass", "8000hz", aug.transform_lowpass, {"cutoff_hz": 8000.0}),
]


def corpus_tx_ids() -> list[str]:
    """Repo-managed free-performance corpus recordings (GT not required).

    Unlike augmentation_robustness.py / note_f1_benchmark.py, this alarm is
    ground-truth-free by design, so recordings without a reviewed
    ground_truth.json are still in scope (broader than the F1 benchmarks).
    Restricted to the repo-managed corpus (not local-only transaction
    captures) so the alarm is reproducible from a fresh checkout.
    """
    ids = []
    for d in sorted(nfb.FREE_PERFORMANCE_CORPUS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / "audio.wav").is_file() and (d / "request.json").is_file():
            ids.append(d.name)
    return ids


def run_condition(
    client,
    tx_id: str,
    audio,
    sample_rate: int,
    baseline_pseudo_truth: list[dict],
    baseline_note_count: int,
    cond: "aug.Condition",
    *,
    min_abs_notes: int,
    min_pct: float,
) -> dict:
    kwargs = dict(cond.params)
    transformed, meta = cond.fn(audio, sample_rate, **kwargs)
    audio_bytes = aug.encode_wav_bytes(transformed, sample_rate)
    response = aug.transcribe_bytes(client, tx_id, audio_bytes, debug=False)
    if response.status_code != 200:
        return {
            "txId": tx_id,
            "family": cond.family,
            "level": cond.level,
            "params": meta,
            "error": f"HTTP {response.status_code}: {response.text[:300]}",
        }
    payload = response.json()
    transformed_notes = nfb.collect_one_best(payload)
    # Reuse note_f1_benchmark's match_pairs for output-to-output comparison:
    # the baseline transcription's notes stand in for "ground truth" here.
    # match["falseNegatives"] = baseline notes with no matching transformed
    # note ("dropped"); match["falsePositives"] = transformed notes with no
    # matching baseline note ("added").
    match = nfb.match_pairs(baseline_pseudo_truth, transformed_notes)
    dropped = match["falseNegatives"]
    added = match["falsePositives"]
    diff = len(dropped) + len(added)
    threshold = max(min_abs_notes, min_pct * baseline_note_count)
    return {
        "txId": tx_id,
        "family": cond.family,
        "level": cond.level,
        "params": meta,
        "baselineNotes": baseline_note_count,
        "transformedNotes": len(transformed_notes),
        "tp": match["tp"],
        "addedCount": len(added),
        "droppedCount": len(dropped),
        "diff": diff,
        "threshold": threshold,
        "warn": diff > threshold,
        "added": [{"time": round(a["time"], 3), "note": a["note"]} for a in added],
        "dropped": [{"time": round(d_["time"], 3), "note": d_["note"]} for d_ in dropped],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metamorphic alarm v0 (GT-free output-consistency check, non-blocking)"
    )
    parser.add_argument("--tx", action="append", dest="tx_ids", help="restrict to these tx ids (repeatable)")
    parser.add_argument("--json", action="store_true", help="print JSON to stdout instead of a table")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="write full JSON results here")
    parser.add_argument(
        "--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT, help="write the markdown summary here"
    )
    parser.add_argument("--no-write", action="store_true", help="do not write report files, stdout only")
    parser.add_argument(
        "--tolerance-sec", type=float, default=DEFAULT_TOLERANCE_SEC,
        help="onset-time tolerance for note matching (default: %(default)s)",
    )
    parser.add_argument(
        "--min-abs-notes", type=int, default=DEFAULT_MIN_ABS_NOTES,
        help="absolute WARN floor on added+dropped note count (default: %(default)s)",
    )
    parser.add_argument(
        "--min-pct", type=float, default=DEFAULT_MIN_PCT,
        help="WARN threshold as a fraction of baseline note count (default: %(default)s)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 if any WARN fired (still not wired into CI/pytest by default)",
    )
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from apps.api.app.main import app

    tx_ids = args.tx_ids or corpus_tx_ids()
    if not tx_ids:
        print("No repo-managed free-performance corpus recordings found.", file=sys.stderr)
        return 1

    client = TestClient(app)
    results: list[dict] = []
    any_warn = False

    for tx_id in tx_ids:
        audio, sample_rate = aug.load_audio_mono(tx_id)
        baseline_transformed, _ = aug.transform_none(audio, sample_rate)
        baseline_bytes = aug.encode_wav_bytes(baseline_transformed, sample_rate)
        baseline_response = aug.transcribe_bytes(client, tx_id, baseline_bytes, debug=False)
        baseline_response.raise_for_status()
        baseline_notes = nfb.collect_one_best(baseline_response.json())
        baseline_pseudo_truth = [
            {"time": n["time"], "note": n["note"], "tol": args.tolerance_sec} for n in baseline_notes
        ]
        baseline_count = len(baseline_notes)

        for cond in CONDITIONS:
            outcome = run_condition(
                client,
                tx_id,
                audio,
                sample_rate,
                baseline_pseudo_truth,
                baseline_count,
                cond,
                min_abs_notes=args.min_abs_notes,
                min_pct=args.min_pct,
            )
            results.append(outcome)
            if outcome.get("warn"):
                any_warn = True

    summary = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "recognizerFingerprint": recognizer_fingerprint(),
        "kalimbaDspFingerprint": kalimba_dsp_fingerprint(),
        "txIds": tx_ids,
        "toleranceSec": args.tolerance_sec,
        "minAbsNotes": args.min_abs_notes,
        "minPct": args.min_pct,
        "reportOnly": True,
        "nonBlocking": True,
        "anyWarn": any_warn,
        "note": (
            "non-blocking metamorphic alarm v0 (docs/sprint-plan-2026-07c.md S2 /"
            " AGENTS.md guardrail 7): not a regression gate, no CI wiring, no baseline file."
        ),
    }
    output = {"summary": summary, "results": results}

    if not args.no_write:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(output), encoding="utf-8")

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print_table(output)

    if args.strict and any_warn:
        return 1
    return 0


def print_table(output: dict) -> None:
    summary = output["summary"]
    print(
        f"recognizer {summary['recognizerFingerprint']} dsp {summary['kalimbaDspFingerprint']}"
        f"  tolerance={summary['toleranceSec']}s"
        f"  threshold=max({summary['minAbsNotes']}, {summary['minPct']:.0%} of baseline notes)"
    )
    print()
    print(f"{'txId':38} {'family':10} {'level':10} {'base':>5} {'xform':>5} {'add':>4} {'drop':>5} {'thr':>5}  WARN")
    warns = []
    for r in output["results"]:
        if "error" in r:
            print(f"{r['txId'][:36]:38} {r['family']:10} {r['level']:10} ERROR: {r['error']}")
            continue
        mark = "*** WARN ***" if r["warn"] else ""
        if r["warn"]:
            warns.append(r)
        print(
            f"{r['txId'][:36]:38} {r['family']:10} {r['level']:10}"
            f" {r['baselineNotes']:>5} {r['transformedNotes']:>5}"
            f" {r['addedCount']:>4} {r['droppedCount']:>5} {r['threshold']:>5.2f}  {mark}"
        )
    print()
    if warns:
        print(f"WARN summary ({len(warns)} of {len(output['results'])} rows):")
        for r in warns:
            print(f"  {r['txId']} / {r['family']}-{r['level']}: diff={r['diff']} > threshold={r['threshold']:.2f}")
            for a in r["added"]:
                print(f"      + added   {a['time']:8.3f}s {a['note']}")
            for d_ in r["dropped"]:
                print(f"      - dropped {d_['time']:8.3f}s {d_['note']}")
    else:
        print("No WARNs.")


def render_markdown(output: dict) -> str:
    summary = output["summary"]
    results = output["results"]
    lines: list[str] = []
    lines.append("# Metamorphic Alarm v0 (S2 instrumentation lane)")
    lines.append("")
    lines.append(
        "**REPORT-ONLY / NON-BLOCKING.** Not a regression gate, no CI wiring, no baseline"
        " file (AGENTS.md guardrail 7, docs/sprint-plan-2026-07c.md S2). Ground-truth-free:"
        " compares each recording's own transcription before/after a benign transform."
    )
    lines.append("")
    lines.append(f"- Generated: {summary['generatedAt']}")
    lines.append(f"- Recognizer fingerprint: `{summary['recognizerFingerprint']}`")
    lines.append(f"- kalimba_dsp fingerprint: `{summary['kalimbaDspFingerprint']}`")
    lines.append(f"- Tolerance: {summary['toleranceSec']}s")
    lines.append(
        f"- WARN threshold: diff > max({summary['minAbsNotes']} notes,"
        f" {summary['minPct']:.0%} of baseline note count)"
    )
    lines.append(f"- Recordings: {', '.join(summary['txIds'])}")
    lines.append(f"- Any WARN this run: {'**yes**' if summary['anyWarn'] else 'no'}")
    lines.append("")

    lines.append("## Matrix (recording x condition)")
    lines.append("")
    lines.append("| txId | family | level | baseline notes | transformed notes | added | dropped | threshold | WARN |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['txId']} | {r['family']} | {r['level']} | - | - | - | - | - | ERROR |")
            continue
        warn_mark = "**WARN**" if r["warn"] else "-"
        lines.append(
            f"| {r['txId']} | {r['family']} | {r['level']} | {r['baselineNotes']} |"
            f" {r['transformedNotes']} | {r['addedCount']} | {r['droppedCount']} |"
            f" {r['threshold']:.2f} | {warn_mark} |"
        )
    lines.append("")

    warns = [r for r in results if r.get("warn")]
    lines.append("## WARN detail")
    lines.append("")
    if not warns:
        lines.append("No WARNs on this run.")
    else:
        for r in warns:
            lines.append(f"### {r['txId']} / {r['family']}-{r['level']}")
            lines.append("")
            lines.append(f"diff={r['diff']} > threshold={r['threshold']:.2f}")
            lines.append("")
            if r["added"]:
                lines.append("Added (present after transform, absent at baseline):")
                for a in r["added"]:
                    lines.append(f"- {a['time']:.3f}s {a['note']}")
            if r["dropped"]:
                lines.append("Dropped (present at baseline, missing after transform):")
                for d_ in r["dropped"]:
                    lines.append(f"- {d_['time']:.3f}s {d_['note']}")
            lines.append("")
    lines.append(render_analysis_section())
    return "\n".join(lines) + "\n"


def render_analysis_section() -> str:
    return """## Notes

- Transform set (v0, see script docstring for full rationale): ``identity/control``
  (determinism canary — re-transcribes byte-identical audio and should never
  WARN), ``gain +6dB`` (headroom), ``lowpass 8000hz`` (mic-distance/muffling
  proxy). All three showed a negligible F1/predicted-count delta on both
  recordings in the 2026-07-03 ``augmentation_robustness.py`` run; reverb and
  additive-noise families are excluded from v0 because their mildest tested
  levels already move F1 on the non-saturated reference recording — using
  them here would conflate "known DSP degradation" with "patch-boundary
  fragility".
- A WARN here is a *lead*, not a verdict: it means a benign transform changed
  the recognizer's own output, which is consistent with (but not proof of)
  AGENTS.md's guardrail-11 trigger 1 ("patch が衝突する"). Follow up with the
  usual audio-diagnose / energy-trace tools on the specific added/dropped
  onsets before concluding a pass conflict.
"""


if __name__ == "__main__":
    raise SystemExit(main())
