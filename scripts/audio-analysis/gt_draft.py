"""Generate a ground-truth *draft* for a local transaction recording.

第 2 期 S2 (docs/sprint-plan-2026-07b.md) の GT 化フロー前半:
エージェントがドラフトを作り、ユーザーは試聴して差分確認だけで済む形にする。

Per transaction this script runs the *raw* front half of the pipeline directly
(no segmenter, no gates — so collapsed recordings where the recognizer emitted
events=1/slots=0 are still fully analyzable):

  1. raw onsets   — kalimba_dsp.onset_strength -> onset_detect(backtrack)
                    (recognizer defaults: hop=256, n_fft=2048, n_mels=128)
  2. pitch top-3  — per-onset window (onset -> next onset, capped 0.25 s),
                    adaptive_n_fft(min_bins=2) + chunk_spectrum +
                    rank_tuning_candidates — the wasm-demo / recognizer
                    integer-harmonic-comb default path
  3. alignment    — raw onsets x recognized events x candidateSlots x expected
                    sequence (request.json:expectedPerformance, or --menu の
                    破壊メニュー期待列, or none for free performance)

Outputs (data/gt_drafts/, gitignored):
  <tx8>.md                  — 試聴確認用の整列表 (ユーザー向け)
  <tx8>.rows.json           — /debug/gt-review ページ用の行データ (機械可読)
  <tx8>.ground_truth.json   — draft GT (schema = transaction-captures GT;
                              method: agent_draft)。ユーザー確認後に method を
                              ear_verified へ書き換えて
                              apps/api/tests/fixtures/transaction-captures/<tx>/
                              ground_truth.json に配置する。

Usage:
  uv run python scripts/audio-analysis/gt_draft.py 4e1ae5c6 --menu carryover-mask
  uv run python scripts/audio-analysis/gt_draft.py 2cc06261            # free perf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import kalimba_dsp as K  # noqa: E402

from apps.api.app.transcription.audio import condition_input_audio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "transactions"
OUT_DIR_DEFAULT = REPO_ROOT / "data" / "gt_drafts"

# Recognizer onset defaults — must match apps/api/app/transcription/constants.py
# and apps/web/src/lib/wasm/onset.ts.
HOP_LENGTH = 256
N_FFT = 2048
N_MELS = 128

# Pitch-ID window heuristics — mirror apps/web/src/lib/wasm/pitch.ts.
MAX_WINDOW_SEC = 0.25
MIN_CHUNK_SAMPLES = 256
MIN_BINS = 2

try:
    from apps.api.app.transcription.peaks import HARMONIC_BAND_CENTS
except Exception:  # pragma: no cover - fallback matches constants.py
    HARMONIC_BAND_CENTS = 40.0

# 破壊メニューの期待列。正は apps/web/src/lib/adversarialMenu.ts (events フィールド)。
# 1caec83 以前の録音は request.json に expectedPerformance が無いため、この
# ローカルコピーで手動整列する。TS 側を変えたらここも同期すること。
ADVERSARIAL_MENU_EVENTS: dict[str, list[list[str]]] = {
    "carryover-mask": [["B4"], ["C5"], ["B4"], ["C5"], ["B4"], ["C5"], ["B4"], ["C5"]],
    "dense-cluster": [
        ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["E6"],
        ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["E6"],
    ],
    "weak-attack": [["C4"], ["E4"], ["G4"], ["C5"], ["C4"], ["E4"], ["G4"], ["C5"]],
    "adjacent-tine": [
        ["C4", "E4"], ["C4", "E4"], ["C4", "E4"], ["C4", "E4"],
        ["C6", "E6"], ["C6", "E6"], ["C6", "E6"], ["C6", "E6"],
    ],
    "mute-reattack": [["C4"]] * 8,
    "dynamics-contrast": [["C5"], ["E5"], ["G5"], ["C5"], ["E5"], ["G5"]],
    "tremolo": [["C5"]] * 8,
}

ALIGN_TOLERANCE_SEC = 0.12  # raw onset <-> recognized event / slot の対応付け幅


def resolve_tx(prefix: str) -> Path:
    matches = sorted(p for p in DATA_DIR.iterdir() if p.name.startswith(prefix))
    if len(matches) != 1:
        raise SystemExit(f"tx prefix '{prefix}' matched {len(matches)} dirs: {[p.name for p in matches]}")
    return matches[0]


def load_audio(tx_dir: Path) -> tuple[np.ndarray, int, dict]:
    audio, sr = sf.read(tx_dir / "audio.wav", dtype="float32")
    # server parity: 優勢チャンネル選択 + 増幅のみ peak 正規化 (2026-07-05)
    audio, conditioning, _peak = condition_input_audio(np.asarray(audio))
    return audio, int(sr), conditioning


def detect_raw_onsets(audio: np.ndarray, sr: int) -> list[float]:
    env = np.asarray(K.onset_strength(audio, sr, HOP_LENGTH, N_FFT, N_MELS), dtype=np.float32)
    frames = K.onset_detect(env, sr, HOP_LENGTH, True)
    return [float(f) * HOP_LENGTH / sr for f in frames]


def identify_pitch(
    audio: np.ndarray,
    sr: int,
    onsets: list[float],
    note_names: list[str],
    note_freqs: np.ndarray,
) -> list[dict]:
    """Per-onset top-3 note candidates (wasm-demo identifyNotesInBrowser parity)."""
    min_freq = float(note_freqs.min())
    max_window = round(MAX_WINDOW_SEC * sr)
    results = []
    for i, t in enumerate(onsets):
        start = max(0, round(t * sr))
        next_start = round(onsets[i + 1] * sr) if i + 1 < len(onsets) else len(audio)
        end = min(start + max_window, next_start, len(audio))
        chunk_len = end - start
        entry: dict = {"timeSec": round(t, 4), "top": []}
        if chunk_len >= MIN_CHUNK_SAMPLES:
            chunk = np.ascontiguousarray(audio[start:end], dtype=np.float32)
            n_fft = K.adaptive_n_fft(sr, min_freq, chunk_len, MIN_BINS, HARMONIC_BAND_CENTS)
            spectrum = np.asarray(K.chunk_spectrum(chunk, sr, n_fft), dtype=np.float64)
            freqs = np.arange(len(spectrum), dtype=np.float64) * (sr / n_fft)
            scores = np.asarray(
                K.rank_tuning_candidates(freqs, spectrum, note_freqs, HARMONIC_BAND_CENTS),
                dtype=np.float64,
            )
            # 34L-C 等の 2-row stacked tuning は同名 tine を持つ — noteName で
            # dedupe (最大スコア採用)。share は正スコアのみで正規化 (負スコアの
            # 無音/ノイズ窓で share が発散しないように)。
            best_by_name: dict[str, float] = {}
            for name, s in zip(note_names, scores):
                if name not in best_by_name or float(s) > best_by_name[name]:
                    best_by_name[name] = float(s)
            ranked = sorted(best_by_name.items(), key=lambda kv: kv[1], reverse=True)[:3]
            pos_total = sum(max(s, 0.0) for s in best_by_name.values()) or 1.0
            entry["top"] = [
                {"note": n, "score": s, "share": max(s, 0.0) / pos_total} for n, s in ranked
            ]
            if ranked and ranked[0][1] <= 0.0:
                entry["lowEvidence"] = True
        else:
            entry["skipped"] = f"window {chunk_len} samples < {MIN_CHUNK_SAMPLES}"
        results.append(entry)
    return results


def load_expected(tx_dir: Path, menu_id: str | None) -> tuple[list[list[str]] | None, str]:
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    perf = request.get("expectedPerformance")
    if isinstance(perf, dict) and isinstance(perf.get("events"), list):
        events = [[k["noteName"] for k in ev.get("keys", [])] for ev in perf["events"]]
        return events, f"request.json:expectedPerformance ({perf.get('summary', '')})"
    if menu_id:
        if menu_id not in ADVERSARIAL_MENU_EVENTS:
            raise SystemExit(f"unknown menu id '{menu_id}' (known: {sorted(ADVERSARIAL_MENU_EVENTS)})")
        return ADVERSARIAL_MENU_EVENTS[menu_id], f"adversarial menu '{menu_id}' (manual alignment)"
    return None, "none (free performance — ear-verify all rows)"


def nearest_within(t: float, items: list[tuple[float, str]], tol: float) -> str | None:
    best = None
    best_dt = tol
    for it, label in items:
        dt = abs(it - t)
        if dt <= best_dt:
            best_dt = dt
            best = f"{label} (Δ{(it - t) * 1000:+.0f}ms)"
    return best


# DP 整列のスコア。top-1 一致を最優先しつつ、carryover で top-1 が直前音に
# 支配されたケース (masked C5 が top-2 に share 0.3-0.5 で残る等) を拾うため
# top-2/3 一致も低スコアで許す。無理な位置に expected を割り当てるより
# MISS として残す方が安全なので、期待 gap のペナルティは軽め。
MATCH_SCORE_BY_RANK = (2.0, 1.4, 1.1)
MATCH_MIN_SHARE_RANK2 = 0.12
GAP_EXPECTED = -0.5  # expected を未配置 (MISS) にするコスト
GAP_ONSET = 0.0  # onset を未対応 (EXTRA) のままにするコスト (余分 onset は普通にある)
# 位置事前分布: 期待列 j 番目は録音の (j+0.5)/m 付近に来るはず、という弱い仮定。
# メニューの演奏指示は全項目とも録音全体を使うので、順序一致だけだと発生する
# 「末尾の接触ノイズ帯 (top-2 に直前音の残響が並ぶ) へ列ごと吸い寄せられる」誤整列を防ぐ。
POSITION_PRIOR_WEIGHT = 3.0


def match_score(pitch: dict, exp_notes: list[str]) -> tuple[float, int] | None:
    """(score, matched_rank) — expected の構成音が top-3 に居れば一致とみなす。"""
    members = set(exp_notes)
    for rank, cand in enumerate(pitch.get("top", [])):
        if cand["note"] in members and (rank == 0 or cand["share"] >= MATCH_MIN_SHARE_RANK2):
            return MATCH_SCORE_BY_RANK[rank], rank
    return None


def align_expected(
    pitches: list[dict], expected: list[list[str]], duration_sec: float
) -> list[tuple[int, int] | None]:
    """Order-preserving DP alignment (Needleman-Wunsch) with a positional prior.

    Returns per-onset (expected_index, matched_rank) or None.
    """
    n, m = len(pitches), len(expected)

    def pair_score(i: int, j: int) -> tuple[float, int] | None:
        ms = match_score(pitches[i], expected[j])
        if ms is None:
            return None
        pos_onset = pitches[i]["timeSec"] / duration_sec if duration_sec > 0 else 0.0
        pos_expected = (j + 0.5) / m
        return ms[0] - POSITION_PRIOR_WEIGHT * abs(pos_onset - pos_expected), ms[1]

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_EXPECTED
        back[0][j] = "e"
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_ONSET
        back[i][0] = "o"
        for j in range(1, m + 1):
            best, bb = score[i - 1][j] + GAP_ONSET, "o"
            alt = score[i][j - 1] + GAP_EXPECTED
            if alt > best:
                best, bb = alt, "e"
            ps = pair_score(i - 1, j - 1)
            if ps is not None:
                alt = score[i - 1][j - 1] + ps[0]
                if alt >= best:
                    best, bb = alt, "m"
            score[i][j], back[i][j] = best, bb
    mapping: list[tuple[int, int] | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == "m":
            rank = match_score(pitches[i - 1], expected[j - 1])[1]  # type: ignore[index]
            mapping[i - 1] = (j - 1, rank)
            i -= 1
            j -= 1
        elif b == "e":
            j -= 1
        else:
            i -= 1
    return mapping


def build_draft(tx_dir: Path, menu_id: str | None, out_dir: Path) -> Path:
    tx_id = tx_dir.name
    tx8 = tx_id[:8]
    audio, sr, input_conditioning = load_audio(tx_dir)
    duration = len(audio) / sr

    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    tuning = request["tuning"]
    notes_sorted = sorted(tuning["notes"], key=lambda n: n["frequency"])
    note_names = [n["noteName"] for n in notes_sorted]
    note_freqs = np.asarray([n["frequency"] for n in notes_sorted], dtype=np.float64)

    onsets = detect_raw_onsets(audio, sr)
    pitches = identify_pitch(audio, sr, onsets, note_names, note_freqs)

    response = json.loads((tx_dir / "response.json").read_text(encoding="utf-8"))
    rec_events = [
        (float(ev["startTimeSec"]), "+".join(f"{n['pitchClass']}{n['octave']}" for n in ev["notes"]))
        for ev in response.get("events", [])
    ]
    slots = [
        (float(s["startTime"]), f"{s['primaryNote']['pitchClass']}{s['primaryNote']['octave']}")
        for s in response.get("candidateSlots", [])
    ]

    expected, expected_src = load_expected(tx_dir, menu_id)

    mapping = align_expected(pitches, expected, duration) if expected else [None] * len(pitches)
    mapped = {m[0] for m in mapping if m is not None}
    unplaced = (
        [{"index": j, "notes": expected[j]} for j in range(len(expected)) if j not in mapped]
        if expected
        else []
    )

    memo = None
    memo_path = tx_dir / "memo.txt"
    if memo_path.is_file():
        memo = memo_path.read_text(encoding="utf-8").strip()

    # --- 整列表 (markdown) ---
    lines = [
        f"# GT draft — {tx_id}",
        "",
        f"- 音声: `data/transactions/{tx_id}/audio.wav` ({duration:.2f}s @ {sr}Hz)",
        f"- 試聴: https://score.ayokura.net/score/{tx_id}/review (または /debug/triage)",
        f"- 期待列ソース: {expected_src}",
        f"- raw onsets: {len(onsets)} / recognized events: {len(rec_events)} / candidateSlots: {len(slots)}",
    ]
    if memo:
        lines.append(f"- memo: {memo}")
    lines += [
        "",
        "flag の読み方: ✅ = top-1 と期待が一致 / ⚠PITCH = 期待音は top-2/3 に居るが top-1 は別音"
        " (carryover 誤同定疑い、耳確認) / ⚠CHORD = 和音の構成音のみ確認 (他構成音を耳確認)"
        " / ⚠EXTRA = 期待に無い onset (演奏外ノイズ or 弾き直し、ignore するか判断)",
        "",
        "| # | time | pitch top-1 (share) | top-2/3 | recognized | slot | expected | flag |",
        "|---|------|--------------------|---------|------------|------|----------|------|",
    ]
    draft_onsets = []
    ui_rows: list[dict] = []
    for i, p in enumerate(pitches):
        t = p["timeSec"]
        tp = p["top"]
        if not tp:
            top1_str = f"? ({p.get('skipped', '')})"
        elif p.get("lowEvidence"):
            top1_str = f"{tp[0]['note']} (score≤0 — 無音/ノイズ窓?)"
        else:
            top1_str = f"{tp[0]['note']} ({tp[0]['share']:.2f})"
        alt = " / ".join(f"{c['note']} {c['share']:.2f}" for c in tp[1:]) if len(tp) > 1 else ""
        rec = nearest_within(t, rec_events, ALIGN_TOLERANCE_SEC) or ""
        slot = nearest_within(t, slots, ALIGN_TOLERANCE_SEC) or ""
        m = mapping[i]
        exp_str = ""
        if m is not None:
            exp_j, rank = m
            exp_label = "+".join(expected[exp_j])
            exp_str = f"[{exp_j + 1}] {exp_label}"
            gt_notes = expected[exp_j]
            if rank == 0 and len(expected[exp_j]) == 1:
                flag = "✅"
                comment = f"top1={tp[0]['note']} share={tp[0]['share']:.2f}"
            elif rank == 0:
                # 和音の構成音が top-1: 単音 top-1 では他の構成音を確認できない
                flag = "⚠CHORD"
                comment = (
                    f"NEEDS EAR CHECK: top1={tp[0]['note']} は和音 {exp_label} の構成音。"
                    "他構成音が鳴っているか耳確認"
                )
            else:
                flag = "⚠PITCH"
                comment = (
                    f"NEEDS EAR CHECK: top1={tp[0]['note'] if tp else '?'} だが expected "
                    f"{exp_label} が top-{rank + 1} (share {tp[rank]['share']:.2f}) — "
                    "carryover 誤同定の疑い。期待音を耳確認"
                )
        else:
            gt_notes = [tp[0]["note"]] if tp else []
            flag = "⚠EXTRA" if expected else "耳確認"
            comment = "NEEDS EAR CHECK: " + (
                "期待列に無い onset — ignore するか判断" if expected else f"free performance, top1 draft"
            )
        lines.append(f"| {i + 1} | {t:.3f} | {top1_str} | {alt} | {rec} | {slot} | {exp_str} | {flag} |")
        if gt_notes:
            draft_onsets.append(
                {"timeSec": t, "notes": gt_notes, "method": "agent_draft", "comment": comment}
            )
        ui_rows.append(
            {
                "index": i + 1,
                "timeSec": t,
                "top": tp,
                "lowEvidence": bool(p.get("lowEvidence")),
                "recognized": rec,
                "slot": slot,
                "expectedIndex": (m[0] + 1) if m is not None else None,
                "expectedNotes": expected[m[0]] if m is not None else None,
                "flag": {"✅": "ok", "⚠PITCH": "pitch", "⚠CHORD": "chord", "⚠EXTRA": "extra"}.get(
                    flag, "ear"
                ),
                "draftNotes": gt_notes,
                "comment": comment,
            }
        )

    if not expected and pitches:
        # 自由演奏: 全行耳確認は重いので、フレーズ単位で俯瞰できる推定メロディを
        # 併記する。演奏者は曲を知っているため、行単位で「らしいか」を照合し、
        # 怪しい行だけ表に戻れる。低確信 (share<0.2) は「?」を付す。
        lines += [
            "",
            "## 推定メロディ (top-1 連結 / 1.0s 超の無音で改行 / share<0.20 は ? 付き)",
            "",
        ]
        phrase: list[str] = []
        phrase_start = pitches[0]["timeSec"]
        prev_t = phrase_start

        def flush() -> None:
            if phrase:
                lines.append(f"- `{phrase_start:6.2f}s` {' '.join(phrase)}")

        for p in pitches:
            if p["timeSec"] - prev_t > 1.0:
                flush()
                phrase = []
                phrase_start = p["timeSec"]
            tp = p["top"]
            if not tp or p.get("lowEvidence"):
                phrase.append("×")
            else:
                phrase.append(tp[0]["note"] + ("?" if tp[0]["share"] < 0.2 else ""))
            prev_t = p["timeSec"]
        flush()

    if unplaced:
        lines += [
            "",
            "## ⚠MISS — raw onset に対応が見つからなかった期待イベント",
            "",
            "耳で該当時刻を特定するか、「実際は弾いていない」と判断して破棄してください。",
            "",
        ]
        for u in unplaced:
            lines.append(f"- 期待 [{u['index'] + 1}] {'+'.join(u['notes'])}")

    lines += [
        "",
        "## 確認後の手順",
        "",
        "1. ⚠ 行を試聴で裁定 (PITCH → 正しい音名に修正 / EXTRA → 行削除 or 採用 / MISS → 時刻特定 or 破棄)",
        f"2. `{tx8}.ground_truth.json` の method を `ear_verified` に置換",
        f"3. `apps/api/tests/fixtures/transaction-captures/{tx_id}/ground_truth.json` へ配置",
        "4. `uv run python scripts/audio-analysis/note_f1_benchmark.py " + tx8 + "` で F1 を確認",
    ]

    draft_gt = {
        "version": 1,
        "toleranceSec": 0.08,
        "source": {
            "type": "gt-draft",
            "transactionId": tx_id,
            "generator": "scripts/audio-analysis/gt_draft.py",
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "expectedSource": expected_src,
            "comment": "raw onset (backtrack) + pitch top-1/expected 整列のドラフト。"
            "onset timing は spectral backtrack 由来で perceptual onset より早め得る (tol 0.08)。"
            "ユーザー試聴確認後 method を ear_verified へ。",
        },
        "onsets": draft_onsets,
        "unplacedExpected": unplaced,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{tx8}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{tx8}.ground_truth.json").write_text(
        json.dumps(draft_gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # /debug/gt-review (temporary dev ページ) 用の機械可読な行データ。
    # 同名 tine (34L-C) を dedupe した音名リストを修正 UI の note picker に使う。
    seen_names: set[str] = set()
    tuning_note_names = []
    for n in notes_sorted:
        if n["noteName"] not in seen_names:
            seen_names.add(n["noteName"])
            tuning_note_names.append(n["noteName"])
    ui_doc = {
        "txId": tx_id,
        "tx8": tx8,
        "durationSec": round(duration, 3),
        "sampleRate": sr,
        "memo": memo,
        "menuId": menu_id,
        "expectedSource": expected_src,
        "expectedCount": len(expected) if expected else None,
        "generatedAt": draft_gt["source"]["generatedAt"],
        # 元録音の peak (正規化前)。/debug/gt-review の試聴ブースト量の根拠
        "inputPeakDbfs": input_conditioning.get("inputPeakDbfs"),
        "tuningNotes": tuning_note_names,
        "rows": ui_rows,
        "unplacedExpected": unplaced,
    }
    (out_dir / f"{tx8}.rows.json").write_text(
        json.dumps(ui_doc, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_check = sum(1 for o in draft_onsets if "NEEDS EAR CHECK" in o["comment"])
    print(
        f"{tx8}: onsets={len(onsets)} draft_rows={len(draft_onsets)} "
        f"ear_check={n_check} miss={len(unplaced)} -> {md_path}"
    )
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tx_prefixes", nargs="+", help="transaction id prefix(es)")
    parser.add_argument("--menu", default=None, help="adversarial menu id (期待列の手動指定)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    for prefix in args.tx_prefixes:
        build_draft(resolve_tx(prefix), args.menu, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
