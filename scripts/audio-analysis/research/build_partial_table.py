"""Build the measured per-tine partial table (third-term S3, research line).

Constructs, per instrument group (tuning x performer), a machine-readable
table of measured partial ratios from single-note GT events — the shared
prerequisite for the per-tine tracker / NMF / resynthesis lines (bets S3).

Method (learned from the #149 collision-probe rerun, 2026-07-05):
- Partials are extracted from the *fresh* spectrum S_body - S_pre (floored
  at 0), where S_pre is a same-length window ending just before the onset.
  This cancels the seconds-long ring-out of earlier notes, which the rerun
  showed both inflates contamination and injects ghost partial peaks.
- Because the subtraction handles earlier notes, isolation is only required
  *forward*: no other GT event inside the body window. A small backward gap
  keeps the previous attack out of the pre window (subtraction then
  over-subtracts, which only biases toward missing a partial, never toward
  inventing one).
- Per (group, note), ratio clusters need >= MIN_SUPPORT of events to enter
  the table; each entry records its support fraction and median relative
  amplitude so downstream users can weigh evidence.

Validation baked in: the table is built twice — strict (S0-style +-0.35s
isolation) and relaxed (forward-only) — and clusters that appear only in
the relaxed build are marked "relaxedOnly" for scrutiny.

Output: docs/research/data/per-tine-partial-table.json (+ stdout summary).

Usage: uv run python scripts/audio-analysis/research/build_partial_table.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tine_partial_collision_probe import (  # noqa: E402
    BODY_OFFSET,
    BODY_WINDOW,
    DATA_DIR,
    ISOLATION_SEC,
    MIN_PEAK_REL,
    PARTIAL_BAND,
    REPO,
    audio_for,
    collect_gt,
    load_audio,
    peak_near,
    resolve_source,
    window_spectrum,
)

PREV_GAP_MIN = 0.10          # keep the previous attack out of the pre window
NEXT_GAP_MIN = BODY_OFFSET + BODY_WINDOW + 0.02
MIN_EVENTS = 3               # per (group, note)
MIN_SUPPORT = 0.4            # cluster must appear in >=40% of the note's events
CLUSTER_GAP = 0.05           # ratio clustering bin, matches the probe
# NOTE: not under a "data/" subdir — the repo-wide "data/" gitignore pattern
# would swallow it, and this table is a committed research asset.
OUT_PATH = REPO / "docs" / "research" / "per-tine-partial-table.json"


def fresh_spectrum(audio: np.ndarray, sr: int, t: float):
    body = window_spectrum(audio, sr, t + BODY_OFFSET)
    pre = window_spectrum(audio, sr, t - 0.01 - BODY_WINDOW)
    if body is None:
        return None
    fr, sb = body
    if pre is not None:
        sb = np.maximum(sb - pre[1], 0.0)
    return fr, sb


def collect_events(strict: bool):
    """{group: (freqs_map, [(tx, t, note), ...])} passing the gap rule."""
    groups: dict[str, tuple[dict[str, float], list]] = {}
    for tx, onsets in collect_gt().items():
        src = resolve_source(tx)
        if src is None or audio_for(tx) is None:
            continue
        _tuning, group, freqs_map = src
        bucket = groups.setdefault(group, (freqs_map, []))[1]
        times = [t for t, _ in onsets]
        for i, (t, notes) in enumerate(onsets):
            if len(notes) != 1 or notes[0] not in freqs_map:
                continue
            prev_gap = t - times[i - 1] if i > 0 else 99
            next_gap = times[i + 1] - t if i + 1 < len(times) else 99
            if strict:
                if prev_gap < ISOLATION_SEC or next_gap < ISOLATION_SEC:
                    continue
            else:
                if prev_gap < PREV_GAP_MIN or next_gap < NEXT_GAP_MIN:
                    continue
            bucket.append((tx, t, notes[0]))
    return groups


def measure(group_events, freqs_map):
    """note -> list of per-event [(rel_amp, ratio), ...] from fresh spectra."""
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    per_note: dict[str, list[list[tuple[float, float]]]] = {}
    for tx, t, note in group_events:
        if tx not in audio_cache:
            audio_cache[tx] = load_audio(audio_for(tx))
        audio, sr = audio_cache[tx]
        fs = fresh_spectrum(audio, sr, t)
        if fs is None:
            continue
        fr, spec = fs
        f0 = freqs_map[note]
        e0 = peak_near(fr, spec, f0)
        if e0 <= 0:
            continue
        lo, hi = f0 * PARTIAL_BAND[0], f0 * PARTIAL_BAND[1]
        m = (fr >= lo) & (fr <= hi)
        band_f, band_s = fr[m], spec[m]
        is_pk = np.zeros(len(band_s), bool)
        is_pk[1:-1] = (band_s[1:-1] > band_s[:-2]) & (band_s[1:-1] >= band_s[2:])
        strong = sorted(
            [(float(bs / e0), float(bf / f0)) for bf, bs in zip(band_f[is_pk], band_s[is_pk]) if bs >= e0 * MIN_PEAK_REL],
            reverse=True,
        )[:4]
        per_note.setdefault(note, []).append(strong)
    return per_note


def cluster_table(per_note, freqs_map):
    """note -> [{ratio, medianRelAmp, support, n}] using CLUSTER_GAP bins."""
    table = {}
    for note in sorted(per_note, key=lambda n: freqs_map[n]):
        events = per_note[note]
        if len(events) < MIN_EVENTS:
            continue
        flat = sorted([(r, a) for ev in events for a, r in ev])
        if not flat:
            table[note] = {"n": len(events), "partials": []}
            continue
        clusters: list[list[tuple[float, float]]] = [[flat[0]]]
        for r, a in flat[1:]:
            if r - clusters[-1][-1][0] <= CLUSTER_GAP:
                clusters[-1].append((r, a))
            else:
                clusters.append([(r, a)])
        partials = []
        for c in clusters:
            # support: fraction of this note's events contributing to the cluster
            ratios = [r for r, _ in c]
            amps = [a for _, a in c]
            lo, hi = min(ratios) - 1e-9, max(ratios) + 1e-9
            hits = sum(1 for ev in events if any(lo <= r <= hi for _, r in ev))
            support = hits / len(events)
            if support >= MIN_SUPPORT:
                partials.append({
                    "ratio": round(float(np.median(ratios)), 3),
                    "medianRelAmp": round(float(np.median(amps)), 4),
                    "support": round(support, 2),
                })
        table[note] = {"n": len(events), "partials": partials}
    return table


def main() -> int:
    strict_groups = collect_events(strict=True)
    relaxed_groups = collect_events(strict=False)
    out = {
        "method": "fresh-spectrum (body minus pre-onset window) peak clustering on single-note GT events",
        "generated": str(date.today()),
        "params": {
            "bodyWindowSec": BODY_WINDOW, "bodyOffsetSec": BODY_OFFSET,
            "prevGapMinSec": PREV_GAP_MIN, "nextGapMinSec": round(NEXT_GAP_MIN, 3),
            "minPeakRel": MIN_PEAK_REL, "minSupport": MIN_SUPPORT,
            "clusterGap": CLUSTER_GAP, "partialBand": list(PARTIAL_BAND),
        },
        "caveats": [
            "GT timeSec is approximate; timing-sensitive use requires spectral-pinned onsets (guardrail 13)",
            "relaxedOnly clusters lack strict-isolation corroboration — verify before load-bearing use",
            "ratios are medians per instrument (tuning x performer); do not pool across instruments (#172)",
        ],
        "groups": {},
    }
    for group in sorted(relaxed_groups, key=lambda g: -len(relaxed_groups[g][1])):
        freqs_map, events = relaxed_groups[group]
        relaxed_pn = measure(events, freqs_map)
        relaxed_tbl = cluster_table(relaxed_pn, freqs_map)
        strict_tbl = {}
        if group in strict_groups:
            s_freqs, s_events = strict_groups[group]
            strict_tbl = cluster_table(measure(s_events, s_freqs), s_freqs)
        recordings = sorted({tx[:8] for tx, _, _ in events})
        print(f"\n[{group}] events: relaxed={len(events)} strict={len(strict_groups.get(group, (None, []))[1])} recordings={','.join(recordings)}")
        gout = {"recordings": recordings, "notes": {}}
        for note, entry in relaxed_tbl.items():
            strict_ratios = [p["ratio"] for p in strict_tbl.get(note, {}).get("partials", [])]
            for p in entry["partials"]:
                corroborated = any(abs(p["ratio"] - sr_) <= CLUSTER_GAP for sr_ in strict_ratios)
                if not corroborated:
                    p["relaxedOnly"] = True
            gout["notes"][note] = {"f0": freqs_map[note], **entry}
            desc = ", ".join(
                f"x{p['ratio']}(amp {p['medianRelAmp']}, sup {p['support']}{', relaxedOnly' if p.get('relaxedOnly') else ''})"
                for p in entry["partials"]) or "(no stable partial)"
            print(f"  {note:4s} n={entry['n']:3d}: {desc}")
        out["groups"][group] = gout
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
