"""Run Spotify basic-pitch on a wav and emit note events as JSON (stdout).

Runs in an ISOLATED Python 3.11 environment (basic-pitch pins tensorflow
2.15, no cp312+/cp314 wheels).  Invoke via:

    uv run --python 3.11 --no-project --with basic-pitch \
        python scripts/audio-analysis/research/basic_pitch_infer.py <audio.wav>

Part of the S6 bets #3 spike (consensus-teacher disagreement map, #199).
Output: [{"start": s, "end": s, "midi": int, "amplitude": f}, ...]
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: basic_pitch_infer.py <audio.wav>", file=sys.stderr)
        return 2
    # Import after argv check: tensorflow import takes seconds.
    import contextlib
    import io

    from basic_pitch.inference import predict

    # predict() prints progress lines to stdout; keep stdout JSON-clean.
    with contextlib.redirect_stdout(io.StringIO()):
        _model_output, _midi, note_events = predict(sys.argv[1])
    rows = [
        {
            "start": round(float(start), 4),
            "end": round(float(end), 4),
            "midi": int(pitch),
            "amplitude": round(float(amp), 4),
        }
        for start, end, pitch, amp, *_ in note_events
    ]
    rows.sort(key=lambda r: r["start"])
    json.dump(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
