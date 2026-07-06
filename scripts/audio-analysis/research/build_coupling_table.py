"""Measured per-pair coupling table (S5 round 1.8, research line).

The dual-run leak analysis showed the remaining tracker FPs concentrate on
the magnetic-pickup instruments (G-low / 34L-C BWV sequences, 43/48 fixture
additions) — exactly the chains where the partial table is empty, so the
measured-partial explaining-away term has nothing to say there.

But those recordings have complete GT: every strike is labelled. That lets
us measure, per instrument group, how much *fresh* energy a strike of tine j
injects at tine k's own fundamental (mechanical coupling through the mount,
skirt leak, unlabelled partials — the cause does not matter, the transfer
ratio does). This is the same fresh (pre-onset-subtracted) quantity the #149
collision-probe rerun measures, aggregated as a (striker -> victim) table.

Output: docs/research/per-tine-coupling-table.json
  {group: {striker: {victim: {"median": r, "p90": r, "n": n}}}}

Usage: uv run python scripts/audio-analysis/research/build_coupling_table.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_partial_table import collect_events, fresh_spectrum  # noqa: E402
from tine_partial_collision_probe import (  # noqa: E402
    REPO,
    audio_for,
    load_audio,
    peak_near,
    window_spectrum,
)

ATTACK_WIN = 0.08


def attack_fresh_spectrum(audio, sr, t):
    """Fresh spectrum over the ATTACK window [t, t+0.08] (pre = same length
    ending just before t). The tracker's candidates fire on the attack
    transient, where cross-tine transfer (mechanical kick + broadband click)
    is far stronger than in the body window the partial probe uses —
    round-1.8 leak analysis: body-window transfer ratios under-bound the
    attack-instant leak and the coupling term never fired."""
    import numpy as _np
    a = int(t * sr); b = a + int(ATTACK_WIN * sr)
    pa = int((t - 0.01 - ATTACK_WIN) * sr); pb = pa + int(ATTACK_WIN * sr)
    if a < 0 or b > len(audio) or pa < 0:
        return None
    win = _np.hanning(b - a)
    spec = _np.abs(_np.fft.rfft(audio[a:b] * win, n=1 << 16))
    pre = _np.abs(_np.fft.rfft(audio[pa:pb] * win, n=1 << 16))
    freqs = _np.fft.rfftfreq(1 << 16, 1.0 / sr)
    return freqs, _np.maximum(spec - pre, 0.0)

MIN_N = 3
MIN_MEDIAN = 0.005   # ignore pairs with negligible transfer
OUT = REPO / "docs" / "research" / "per-tine-coupling-table.json"


def main() -> int:
    groups = collect_events(strict=False)
    out = {"generated": str(date.today()),
           "method": "fresh (pre-onset-subtracted) energy at victim fundamentals during "
                     "single-note GT strikes; relaxed isolation (fwd 0.27s / back 0.10s)",
           "groups": {}}
    for group, (freqs_map, events) in groups.items():
        audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        acc: dict[tuple[str, str], list[float]] = {}          # body-window transfer
        acc_atk: dict[tuple[str, str], list[float]] = {}      # attack-window transfer
        for tx, t, note in events:
            if tx not in audio_cache:
                audio_cache[tx] = load_audio(audio_for(tx))
            audio, sr = audio_cache[tx]
            for spec_fn, dest in ((fresh_spectrum, acc), (attack_fresh_spectrum, acc_atk)):
                fs = spec_fn(audio, sr, t)
                if fs is None:
                    continue
                fr, spec = fs
                e0 = peak_near(fr, spec, freqs_map[note])
                if e0 <= 0:
                    continue
                for other, fo in freqs_map.items():
                    if other == note:
                        continue
                    dest.setdefault((note, other), []).append(peak_near(fr, spec, fo) / e0)
        g: dict[str, dict] = {}
        n_pairs = 0
        for (j, k), vals in acc.items():
            if len(vals) < MIN_N:
                continue
            med = float(np.median(vals))
            atk = acc_atk.get((j, k), [])
            atk_p90 = float(np.percentile(atk, 90)) if len(atk) >= MIN_N else None
            if med < MIN_MEDIAN and (atk_p90 is None or atk_p90 < MIN_MEDIAN):
                continue
            g.setdefault(j, {})[k] = {"median": round(med, 4),
                                      "p90": round(float(np.percentile(vals, 90)), 4),
                                      "attackP90": round(atk_p90, 4) if atk_p90 is not None else None,
                                      "n": len(vals)}
            n_pairs += 1
        out["groups"][group] = g
        print(f"{group}: {n_pairs} pairs (median >= {MIN_MEDIAN}, n >= {MIN_N})")
        top = sorted(((j, k, v["median"]) for j, d in g.items() for k, v in d.items()),
                     key=lambda x: -x[2])[:6]
        for j, k, m in top:
            print(f"  {j}->{k}: {m}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
