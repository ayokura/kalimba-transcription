"""Round-4: mute-dip marginal contribution over the in-stage oracle (#206).

NOTE (retirement, user-approved 2026-07-06): the mute-dip OR-backup and its
provenance marker were physically removed after the held-out measurement
(zero marginal, 0/17 recordings). On current code this script always
reports 0 — to re-audit the retirement against a NEW recording set, run it
on a checkout of 38ad720 (the last marker-instrumented commit).

Counts, per GT recording (branch defaults: oracle ON, OR wiring), the
provenance marker `residual-fresh-mute-dip-only` — segments the residual-
decay rejection would have dropped under the oracle alone but mute-dip
saved. Also reports the oracle-regime rejection volume
(residual-decay-no-fresh-attack slots) for context. If the marker count is
~zero corpus-wide, the mute-dip term can retire on that evidence
(merge condition 3 route; user-flagged population 2026-07-06).

Usage: uv run python scripts/audio-analysis/research/pertine_round4_mutedip_margin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import note_f1_benchmark as nfb  # noqa: E402
from pertine_dualrun import gt_tx_ids  # noqa: E402

OUT = REPO / "docs" / "research" / "pertine-round4-mutedip-margin.json"
MARKER = "residual-fresh-mute-dip-only"


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    rows, total_marker, total_reject = [], 0, 0
    for tx in gt_tx_ids():
        payload = nfb.transcribe_payload(client, tx, debug=True)
        debug_blob = json.dumps(payload.get("debug") or {})
        marker = debug_blob.count(MARKER)
        rejects = sum(1 for s in payload.get("candidateSlots") or []
                      if s.get("dropReason") == "residual-decay-no-fresh-attack")
        rows.append({"tx": tx[:8], "muteDipOnlySaves": marker,
                     "oracleRejections": rejects})
        total_marker += marker
        total_reject += rejects
        print(f"{tx[:8]} muteDipOnlySaves={marker:2d} oracleRejections={rejects:2d}")

    print(f"\nTOTAL muteDipOnlySaves={total_marker} oracleRejections={total_reject}")
    print("mute-dip retirement evidence:" ,
          "SUPPORTED (~zero marginal)" if total_marker == 0 else
          f"NOT supported ({total_marker} saves depend on mute-dip)")
    OUT.write_text(json.dumps({
        "marker": MARKER, "rows": rows,
        "totals": {"muteDipOnlySaves": total_marker,
                   "oracleRejections": total_reject},
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
