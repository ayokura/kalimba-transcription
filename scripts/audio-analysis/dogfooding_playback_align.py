#!/usr/bin/env python3
"""dogfooding day-2/3 event-sequence alignment harness (reproducible rebuild).

昨夜送りになった playback の照合 harness の再構築。Needleman-Wunsch で 2 つの
event 列を note-set Jaccard 距離で整列し、exact/SUB/INS/DEL と event 一致率を出す。

入力は tx (response.json の events) または played-GT (ground_truth.json)。
- 描画忠実度: playback-recognition vs day1-recognition   (confidence C, recognizer bias 共有)
- played-GT 基準: playback played-GT を hyp に差し替えれば同じ harness で再計算 (turnkey)

usage:
  python dogfooding_align.py <ref_tx_or_file> <hyp_tx_or_file> [--label NAME]
"""
import json, sys, argparse
from pathlib import Path

DATA = Path("data/transactions")

def load_events(spec):
    """tx prefix / path を受け取り、順序付き note-set 列 [(t, frozenset{'C5',...})] を返す。"""
    p = Path(spec)
    if not p.exists():
        # tx prefix として解決
        cands = sorted(DATA.glob(f"{spec}*"))
        if not cands:
            raise SystemExit(f"not found: {spec}")
        # response.json を優先、なければ ground_truth.json
        d = cands[0]
        p = d / "response.json"
        if not p.exists():
            p = d / "ground_truth.json"
    obj = json.loads(p.read_text())
    events = obj.get("events")
    if events is None:  # ground_truth.json 形式の fallback
        events = obj.get("notes") or obj.get("groundTruthEvents") or []
    out = []
    for e in events:
        notes = e.get("notes", [])
        ns = frozenset(f"{n['pitchClass']}{n['octave']}" for n in notes)
        t = e.get("startTimeSec", e.get("timeSec", 0.0))
        if ns:
            out.append((t, ns))
    return out

def jaccard_dist(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)

def align(ref, hyp, sub_gap=1.0):
    """Needleman-Wunsch。gap cost=sub_gap。substitution cost=jaccard 距離。
    戻り: 整列トレース [(ri, hi, op)] op in exact/sub/ins/del"""
    n, m = len(ref), len(hyp)
    INF = float("inf")
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i*sub_gap
    for j in range(1, m+1): dp[0][j] = j*sub_gap
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = jaccard_dist(ref[i-1][1], hyp[j-1][1])
            dp[i][j] = min(dp[i-1][j-1]+c, dp[i-1][j]+sub_gap, dp[i][j-1]+sub_gap)
    # backtrace
    i, j, trace = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = jaccard_dist(ref[i-1][1], hyp[j-1][1])
            if abs(dp[i][j] - (dp[i-1][j-1]+c)) < 1e-9:
                op = "exact" if c == 0.0 else "sub"
                trace.append((i-1, j-1, op)); i -= 1; j -= 1; continue
        if i > 0 and abs(dp[i][j] - (dp[i-1][j]+sub_gap)) < 1e-9:
            trace.append((i-1, None, "del")); i -= 1; continue
        trace.append((None, j-1, "ins")); j -= 1
    trace.reverse()
    return trace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("hyp")
    ap.add_argument("--label", default="")
    ap.add_argument("--show-loci", action="store_true", help="per-locus 詳細 (bias 注意: played-GT 確定前は surface しない)")
    a = ap.parse_args()
    ref, hyp = load_events(a.ref), load_events(a.hyp)
    tr = align(ref, hyp)
    ex = sum(1 for _,_,o in tr if o=="exact")
    sub = sum(1 for _,_,o in tr if o=="sub")
    ins = sum(1 for _,_,o in tr if o=="ins")
    dl = sum(1 for _,_,o in tr if o=="del")
    print(f"[{a.label}] ref(day1-recog)={len(ref)}  hyp(playback)={len(hyp)}")
    print(f"  exact={ex}  SUB(pitch誤り)={sub}  INS(playback余分)={ins}  DEL(playback欠落)={dl}")
    print(f"  event一致率 = {ex}/{len(ref)} = {100*ex/len(ref):.1f}%")
    if a.show_loci:
        for ri,hi,o in tr:
            if o == "exact": continue
            rt = f"{ref[ri][0]:.2f}s {sorted(ref[ri][1])}" if ri is not None else "-"
            ht = f"{hyp[hi][0]:.2f}s {sorted(hyp[hi][1])}" if hi is not None else "-"
            print(f"    {o.upper():4} ref[{rt}]  hyp[{ht}]")

if __name__ == "__main__":
    main()
