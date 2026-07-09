#!/usr/bin/env python3
"""dogfooding day-2/3 event-sequence alignment harness (reproducible rebuild).

Needleman-Wunsch で 2 つの event 列を note-set Jaccard 距離で整列し、
exact/SUB/INS/DEL と event 一致率を出す。

ref/hyp は tx (response.json の events) / played-GT (ground_truth.json) / 原譜 の
いずれか:
- 描画忠実度: playback-recognition vs day1-recognition   (confidence C, recognizer bias 共有)
- played-GT 基準: playback played-GT を hyp に差し替えれば同じ harness で再計算 (turnkey)
- end-to-end: ref を原譜 (SCORE) にすれば「弾き戻し vs 原曲」を測れる (#204 B0 day-3)

usage:
  # 描画忠実度 / played-GT 基準
  python dogfooding_playback_align.py <ref_tx_or_file> <hyp_tx_or_file> [--label NAME]
  # end-to-end (原譜 vs playback)
  python dogfooding_playback_align.py SCORE <hyp_tx_or_file> \\
      --score-md docs/dogfooding/dogfooding-scores-2026-07.md --song 曲1 [--label NAME]
  # 原譜パースの確認だけ (note列を出す)
  python dogfooding_playback_align.py SCORE --score-md <md> --song 曲1 --dump-score
"""
import json, re, argparse
from pathlib import Path

DATA = Path("data/transactions")
_NOTE = re.compile(r"[A-G]#?\d")


def load_events(spec):
    """tx prefix / path を受け取り、順序付き note-set 列 [(t, frozenset{'C5',...})] を返す。"""
    p = Path(spec)
    if not p.exists():
        cands = sorted(DATA.glob(f"{spec}*"))
        if not cands:
            raise SystemExit(f"not found: {spec}")
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


def load_score_notes(md_path, song):
    """原譜 md から、指定曲の打鍵イベント列 (A→B 各 1 回) を順序付き frozenset 列で返す。

    初回テイク規律は「A→B を 1 回ずつ」(dogfooding-scores の演奏メモ)。リズム記号
    (♪♩𝅗𝅥. 等)・タイ (~)・休符 ((休)/(♩休))・繰り返し (:‖) は打鍵イベントを生まないので
    無視し、音高集合のみ抽出する。dyad [X/Y] は 1 イベント {X,Y}。
    戻り値は load_events と同形の [(index, frozenset), ...]。
    """
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    hdr = next(
        (i for i, ln in enumerate(lines)
         if ln.startswith(f"## {song}:") or ln.startswith(f"## {song}：")),
        None,
    )
    if hdr is None:
        raise SystemExit(f"song section not found: {song}")
    end = next((i for i in range(hdr + 1, len(lines)) if lines[i].startswith("## ")), len(lines))

    events = []
    in_block = False
    for ln in lines[hdr:end]:
        if ln.strip().startswith("```"):
            in_block = not in_block
            continue
        if not in_block:
            continue
        # コメント (← 以降) と繰り返し記号を落とす
        body = re.split(r"←|:‖|:｜", ln)[0]
        # 行頭ラベル (小節番号: / 拾い:) を落とす
        body = re.sub(r"^\s*\d+\s*[:：]", "", body)
        body = re.sub(r"^\s*拾い\s*[:：]", "", body)
        for tok in re.split(r"[\s　]+", body):
            if not tok or tok.startswith("~"):  # 空 / タイ
                continue
            ns = frozenset(_NOTE.findall(tok))
            if ns:
                events.append((len(events), ns))
    return events


def jaccard_dist(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def align(ref, hyp, sub_gap=1.0):
    """Needleman-Wunsch。gap cost=sub_gap、substitution cost=jaccard 距離。
    戻り: 整列トレース [(ri, hi, op)] op in exact/sub/ins/del"""
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * sub_gap
    for j in range(1, m + 1):
        dp[0][j] = j * sub_gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = jaccard_dist(ref[i - 1][1], hyp[j - 1][1])
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + sub_gap, dp[i][j - 1] + sub_gap)
    i, j, trace = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = jaccard_dist(ref[i - 1][1], hyp[j - 1][1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + c)) < 1e-9:
                trace.append((i - 1, j - 1, "exact" if c == 0.0 else "sub"))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + sub_gap)) < 1e-9:
            trace.append((i - 1, None, "del"))
            i -= 1
            continue
        trace.append((None, j - 1, "ins"))
        j -= 1
    trace.reverse()
    return trace


def _fmt(seq, k):
    return "-" if k is None else f"{seq[k][0]}{'' if isinstance(seq[k][0], int) else 's'} {sorted(seq[k][1])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="tx/file、または end-to-end 用の 'SCORE'")
    ap.add_argument("hyp", nargs="?", help="hyp の tx/file (--dump-score 時は不要)")
    ap.add_argument("--label", default="")
    ap.add_argument("--score-md", help="原譜 md (ref=SCORE 時)")
    ap.add_argument("--song", help="曲1 / 曲2 (ref=SCORE 時)")
    ap.add_argument("--dump-score", action="store_true", help="原譜 note列を出して終了")
    ap.add_argument("--show-loci", action="store_true",
                    help="per-locus 詳細 (bias 注意: played-GT 確定前は surface しない)")
    a = ap.parse_args()

    if a.ref.upper() == "SCORE":
        if not a.score_md or not a.song:
            raise SystemExit("ref=SCORE には --score-md と --song が必要")
        ref = load_score_notes(a.score_md, a.song)
        ref_label = f"score:{a.song}"
    else:
        ref = load_events(a.ref)
        ref_label = "ref"

    if a.dump_score:
        print(f"[{ref_label}] events={len(ref)}")
        print("  " + "  ".join(f"{i}:{'+'.join(sorted(ns))}" for i, ns in ref))
        return

    if not a.hyp:
        raise SystemExit("hyp が必要")
    hyp = load_events(a.hyp)
    tr = align(ref, hyp)
    ex = sum(1 for _, _, o in tr if o == "exact")
    sub = sum(1 for _, _, o in tr if o == "sub")
    ins = sum(1 for _, _, o in tr if o == "ins")
    dl = sum(1 for _, _, o in tr if o == "del")
    print(f"[{a.label}] {ref_label}={len(ref)}  hyp={len(hyp)}")
    print(f"  exact={ex}  SUB(pitch誤り)={sub}  INS(hyp余分)={ins}  DEL(hyp欠落)={dl}")
    print(f"  event一致率 = {ex}/{len(ref)} = {100 * ex / len(ref):.1f}%")
    if a.show_loci:
        for ri, hi, o in tr:
            if o == "exact":
                continue
            print(f"    {o.upper():4} ref[{_fmt(ref, ri)}]  hyp[{_fmt(hyp, hi)}]")


if __name__ == "__main__":
    main()
