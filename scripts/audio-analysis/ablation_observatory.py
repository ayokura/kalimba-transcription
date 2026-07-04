#!/usr/bin/env python3
"""ablation observatory 第 1 巡 (第 2 期 S4、bets #2)。

既存トグル (feature flag 9 + ablate 5 + disabled_gates 38) を 1 つずつ倒し、
(a) 実 fixture スイート (test_manual_capture_completed + test_free_performance_corpus)
(b) 非飽和 GT 6 録音の note F1 benchmark
の両方で単独 ablation の影響を測る。トグルの伝搬は KALIMBA_SETTINGS_OVERRIDES
env フック (settings.py) — ad-hoc 比較ではなく実テストスイートを使う
(feedback_fixture_evaluation の教訓)。

Usage:
  uv run python scripts/audio-analysis/ablation_observatory.py [--out DIR] [--limit N]

出力: <out>/report.jsonl (1 トグル 1 行、resume 対応) + summary.md
判定: dead = fixture 影響なし & |ΔmicroF1| < 0.002 & |ΔcR@3| < 0.002
      protective = 無効化で悪化 / harmful = 無効化で改善 (要精査)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.transcription.peaks import GATE_CATEGORIES  # noqa: E402

FIXTURE_TESTS = [
    "apps/api/tests/test_manual_capture_completed.py",
    "apps/api/tests/test_free_performance_corpus.py",
]
NON_SATURATED_TX = [
    "4e1ae5c6-df9a-4876-917d-b7e47699c8e5",
    "9ce7df83-33a0-455d-bf86-c9392ce6f777",
    "ebecf0c6-7e41-430b-bd60-8111a495185e",
    "ea7edd71-e815-4638-a248-a47fe21e5061",
    "a9e30986-5300-4401-8b69-152cba821042",
    "d7a82772-f77f-4820-9798-00133ae45f4e",
]

FEATURE_FLAGS = {
    # 既定 True → False で ablation
    "use_attack_validated_gap_collector": False,
    "filter_gap_onsets_by_attack_profile": False,
    "use_iterative_harmonic_suppression": False,
    "use_evidence_gate_rescue": False,
    "use_multi_primary_branching": False,
    "use_onset_gate": False,
    "use_alternate_groupings": False,
    "use_soft_candidate_alternates": False,
    "use_phase_c_octave_dyad_rescue": False,
    # 既定 False → True (逆向き: 有効化の影響)
    "use_per_tine_partial_scoring": True,
}
ABLATE_SWITCHES = [
    "ablate_sparse_gap_tail",
    "ablate_multi_onset_gap",
    "ablate_trailing_chord_cluster",
    "ablate_collapse_active_range_head",
    "ablate_snap_range_start_to_onset",
]

PYTEST_SUMMARY_RE = re.compile(r"(\d+) passed|(\d+) failed")


def run_pytest(env: dict) -> dict:
    proc = subprocess.run(
        ["uv", "run", "pytest", *FIXTURE_TESTS, "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )
    passed = failed = 0
    for m in PYTEST_SUMMARY_RE.finditer(proc.stdout):
        if m.group(1):
            passed = int(m.group(1))
        if m.group(2):
            failed = int(m.group(2))
    failed_ids = re.findall(r"FAILED \S+::(\S+)", proc.stdout)
    return {"passed": passed, "failed": failed, "failedTests": failed_ids[:10]}


def run_benchmark(env: dict) -> dict | None:
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/audio-analysis/note_f1_benchmark.py",
         "--json", *NON_SATURATED_TX],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    summary = doc.get("summary", {})
    per_rec = {
        r["txId"][:8]: round(r.get("oneBest", {}).get("onsetF1", 0), 4)
        for r in doc.get("results", [])
        if isinstance(r, dict) and r.get("txId")
    }
    return {
        "microF1": round(summary.get("microF1", 0), 4),
        "recallAt3": round(summary.get("candidates", {}).get("recallAt3", 0), 4),
        "hardMisses": summary.get("candidates", {}).get("hardMisses"),
        "perRecording": per_rec,
    }


def measure(name: str, overrides: dict, base_env: dict) -> dict:
    env = dict(base_env)
    if overrides:
        env["KALIMBA_SETTINGS_OVERRIDES"] = json.dumps(overrides)
    return {
        "toggle": name,
        "overrides": overrides,
        "fixtures": run_pytest(env),
        "benchmark": run_benchmark(env),
        "measuredAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def classify(row: dict, baseline: dict) -> str:
    fx = row["fixtures"]
    bl_fx = baseline["fixtures"]
    bench = row.get("benchmark") or {}
    bl_bench = baseline.get("benchmark") or {}
    d_f1 = (bench.get("microF1") or 0) - (bl_bench.get("microF1") or 0)
    d_cr3 = (bench.get("recallAt3") or 0) - (bl_bench.get("recallAt3") or 0)
    fixture_broken = fx["failed"] > bl_fx["failed"]
    if fixture_broken or d_f1 < -0.002:
        return "protective"
    if d_f1 > 0.002 or d_cr3 > 0.002:
        return "harmful-when-enabled"  # 無効化で改善 = 現行実装が足を引っ張る疑い
    if abs(d_f1) <= 0.002 and abs(d_cr3) <= 0.002 and not fixture_broken:
        return "dead-in-current-data"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: 先頭 N トグルのみ")
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "ablation_observatory" / datetime.now(timezone.utc).strftime("%Y%m%d")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.jsonl"

    done: dict[str, dict] = {}
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["toggle"]] = row
            except json.JSONDecodeError:
                pass

    base_env = {k: v for k, v in os.environ.items() if k != "KALIMBA_SETTINGS_OVERRIDES"}

    plan: list[tuple[str, dict]] = [("baseline", {})]
    plan += [(f"flag:{k}={v}", {k: v}) for k, v in FEATURE_FLAGS.items()]
    plan += [(f"ablate:{k}", {k: True}) for k in ABLATE_SWITCHES]
    plan += [
        (f"gate:{g}", {"disabled_gates": [g]})
        for g in sorted(GATE_CATEGORIES)
    ]
    if args.limit:
        plan = plan[: args.limit]

    with report_path.open("a", encoding="utf-8") as fh:
        for i, (name, overrides) in enumerate(plan):
            if name in done:
                print(f"[{i+1}/{len(plan)}] skip (done): {name}", flush=True)
                continue
            print(f"[{i+1}/{len(plan)}] measuring: {name}", flush=True)
            row = measure(name, overrides, base_env)
            done[name] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    baseline = done.get("baseline")
    if baseline is None:
        print("ERROR: baseline missing", file=sys.stderr)
        return 1

    lines = [
        "# Ablation Observatory 第 1 巡",
        "",
        f"- 測定日: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- baseline: fixtures {baseline['fixtures']['passed']} passed / "
        f"microF1 {baseline['benchmark']['microF1']} / cR@3 {baseline['benchmark']['recallAt3']}"
        f" / hardMiss {baseline['benchmark']['hardMisses']} (非飽和 6 録音)",
        "",
        "| toggle | category | fixtures Δfail | ΔmicroF1 | ΔcR@3 | ΔhardMiss | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    counts: dict[str, int] = {}
    for name, row in done.items():
        if name == "baseline":
            continue
        verdict = classify(row, baseline)
        counts[verdict] = counts.get(verdict, 0) + 1
        bench = row.get("benchmark") or {}
        bl_bench = baseline["benchmark"]
        gate = name.split(":", 1)[1] if name.startswith("gate:") else ""
        category = GATE_CATEGORIES.get(gate, "-")
        lines.append(
            f"| {name} | {category} | "
            f"{row['fixtures']['failed'] - baseline['fixtures']['failed']:+d} | "
            f"{(bench.get('microF1') or 0) - bl_bench['microF1']:+.4f} | "
            f"{(bench.get('recallAt3') or 0) - bl_bench['recallAt3']:+.4f} | "
            f"{(bench.get('hardMisses') or 0) - bl_bench['hardMisses']:+d} | "
            f"{verdict} |"
        )
    lines += ["", f"判定集計: {json.dumps(counts, ensure_ascii=False)}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path} and {out_dir / 'summary.md'}")
    print("判定集計:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
