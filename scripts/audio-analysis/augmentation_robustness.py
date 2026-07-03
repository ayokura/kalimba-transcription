#!/usr/bin/env python3
"""DSP augmentation robustness map for the free-performance corpus (S2 bets #1).

Applies known, onset/pitch-preserving DSP transforms (gain, additive noise,
synthetic reverb, low-pass filtering) to the repo-managed free-performance
corpus recordings and re-scores each transformed take against the *original*
``ground_truth.json`` using the exact same note-level matching logic as
``note_f1_benchmark.py`` (imported, not re-implemented). This produces a
surface of "which transform, at which strength, breaks note-level F1".

REPORT-ONLY (docs/sprint-plan-2026-07b.md S2 bets #1 / AGENTS.md guardrail 7):
this script's results must NOT be used as a regression gate, and augmented
recordings do NOT count toward the "non-saturated held-out" n used by the
overfitting gate or the S5 branch condition. That n is reserved for real
recordings with human-reviewed ground truth.

Design constraints (see task brief for the full rationale):
- Only pitch/onset-time-preserving transforms are used. Resampling-based
  speed change / time-stretch is explicitly EXCLUDED because it would
  invalidate the reused ground-truth onset times and note identities.
- Ground truth is reused verbatim from the source recording's
  ``ground_truth.json`` because every transform here preserves onset time and
  pitch (gain/noise/reverb/low-pass do not shift note fundamentals or attack
  sample-alignment; the reverb IR's direct-path tap is placed at lag 0 and the
  low-pass filter is zero-phase via ``scipy.signal.filtfilt``).
- Recognizer source code is never modified by this script; it only exercises
  the existing pipeline through the same FastAPI TestClient path
  ``note_f1_benchmark.py`` uses.
- Transformed audio is synthesized in-memory and never written under a
  tracked path; only the resulting JSON/Markdown report is written to disk.

Usage:
  uv run python scripts/audio-analysis/augmentation_robustness.py
  uv run python scripts/audio-analysis/augmentation_robustness.py --json
  uv run python scripts/audio-analysis/augmentation_robustness.py --tx 17ea7626-3c5d-450d-ae74-0116dea6e881
  uv run python scripts/audio-analysis/augmentation_robustness.py --markdown-out scripts/audio-analysis/reports/augmentation_robustness.md
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.signal as sig
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import note_f1_benchmark as nfb  # noqa: E402

from apps.api.app.fingerprints import (  # noqa: E402
    kalimba_dsp_fingerprint,
    recognizer_fingerprint,
)

# Fixed base seed so every run of this script (same code + same audio) reproduces
# byte-identical synthetic noise/reverb, hence identical F1 numbers.
BASE_SEED = 20260704
DEFAULT_JSON_OUT = SCRIPT_DIR / "reports" / "augmentation_robustness.json"
DEFAULT_MARKDOWN_OUT = SCRIPT_DIR / "reports" / "augmentation_robustness.md"


def _seed_for(*parts: Any) -> int:
    """Stable, process-independent seed derived from a string key.

    Avoids relying on Python's (hash-randomized) built-in ``hash()`` so seeds
    are reproducible across interpreter invocations.
    """
    key = "|".join(str(p) for p in parts)
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Transforms. Each returns (transformed_audio_float32, meta_dict). None of
# these change len(audio) or introduce a global time shift, so the original
# ground_truth.json onset times remain valid without adjustment.
# ---------------------------------------------------------------------------


def transform_none(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    return audio.copy(), {}


def transform_gain(audio: np.ndarray, sample_rate: int, *, db: float) -> tuple[np.ndarray, dict]:
    factor = 10.0 ** (db / 20.0)
    out = audio.astype(np.float64) * factor
    clipped_fraction = float(np.mean(np.abs(out) > 1.0))
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32), {"gainDb": db, "clippedFraction": clipped_fraction}


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs = freqs.copy()
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid /0 at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    std = float(np.std(pink))
    if std > 1e-12:
        pink = pink / std
    return pink


def transform_noise(
    audio: np.ndarray,
    sample_rate: int,
    *,
    snr_db: float,
    color: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n = len(audio)
    if color == "white":
        noise = rng.standard_normal(n)
    elif color == "pink":
        noise = _pink_noise(n, rng)
    else:
        raise ValueError(f"unknown noise color: {color}")
    noise_std = float(np.std(noise))
    if noise_std > 1e-12:
        noise = noise / noise_std
    signal_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * target_noise_rms
    out = audio.astype(np.float64) + noise
    clipped_fraction = float(np.mean(np.abs(out) > 1.0))
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32), {
        "snrDb": snr_db,
        "color": color,
        "clippedFraction": clipped_fraction,
    }


def _synthetic_ir(sample_rate: int, rt60_sec: float, seed: int) -> np.ndarray:
    """Exponentially-decaying white-noise impulse response.

    ``ir[0]`` is forced to 1.0 so the direct path stays at lag 0 (no group
    delay / no onset-time shift when convolved). Energy is normalized to 1 so
    "wet" mixing amount below has a stable, comparable meaning across RT60s.
    """
    rng = np.random.default_rng(seed)
    length = max(int(sample_rate * rt60_sec * 1.5), 8)
    t = np.arange(length) / sample_rate
    # -60 dB (1e-3 amplitude) reached at t = rt60_sec.
    tau = rt60_sec / np.log(1000.0)
    envelope = np.exp(-t / tau)
    ir = rng.standard_normal(length) * envelope
    ir[0] = 1.0
    energy = float(np.sqrt(np.sum(ir.astype(np.float64) ** 2)))
    if energy > 1e-12:
        ir = ir / energy
    return ir.astype(np.float64)


def transform_reverb(
    audio: np.ndarray,
    sample_rate: int,
    *,
    rt60_sec: float,
    wet: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    ir = _synthetic_ir(sample_rate, rt60_sec, seed)
    wet_signal = sig.fftconvolve(audio.astype(np.float64), ir, mode="full")[: len(audio)]
    dry_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    wet_rms = float(np.sqrt(np.mean(wet_signal ** 2)) + 1e-12)
    wet_signal = wet_signal * (dry_rms / wet_rms)
    out = (1.0 - wet) * audio.astype(np.float64) + wet * wet_signal
    clipped_fraction = float(np.mean(np.abs(out) > 1.0))
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32), {
        "rt60Sec": rt60_sec,
        "wet": wet,
        "clippedFraction": clipped_fraction,
    }


def transform_lowpass(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float,
    order: int = 4,
) -> tuple[np.ndarray, dict]:
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = sig.butter(order, normalized_cutoff, btype="low")
    # filtfilt = zero-phase (forward-backward) filtering: no group delay, so
    # onset sample alignment is preserved exactly.
    out = sig.filtfilt(b, a, audio.astype(np.float64))
    clipped_fraction = float(np.mean(np.abs(out) > 1.0))
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32), {
        "cutoffHz": cutoff_hz,
        "order": order,
        "clippedFraction": clipped_fraction,
    }


@dataclass(frozen=True)
class Condition:
    family: str
    level: str
    fn: Callable[..., tuple[np.ndarray, dict]]
    params: dict = field(default_factory=dict)
    needs_seed: bool = False


CONDITIONS: list[Condition] = [
    Condition("none", "baseline", transform_none, {}),
    # 1) Gain scaling. +6dB is expected to clip on recordings already close to
    # full scale (report-only condition: clipping distortion is itself a
    # realistic degradation mode per the task brief, not a bug in the script).
    Condition("gain", "-30dB", transform_gain, {"db": -30.0}),
    Condition("gain", "-20dB", transform_gain, {"db": -20.0}),
    Condition("gain", "-10dB", transform_gain, {"db": -10.0}),
    Condition("gain", "0dB", transform_gain, {"db": 0.0}),
    Condition("gain", "+6dB", transform_gain, {"db": 6.0}),
    # 2) Additive noise, white and pink, at a range of SNRs.
    Condition("noise_white", "snr30", transform_noise, {"snr_db": 30.0, "color": "white"}, True),
    Condition("noise_white", "snr20", transform_noise, {"snr_db": 20.0, "color": "white"}, True),
    Condition("noise_white", "snr10", transform_noise, {"snr_db": 10.0, "color": "white"}, True),
    Condition("noise_white", "snr5", transform_noise, {"snr_db": 5.0, "color": "white"}, True),
    Condition("noise_white", "snr0", transform_noise, {"snr_db": 0.0, "color": "white"}, True),
    Condition("noise_pink", "snr30", transform_noise, {"snr_db": 30.0, "color": "pink"}, True),
    Condition("noise_pink", "snr20", transform_noise, {"snr_db": 20.0, "color": "pink"}, True),
    Condition("noise_pink", "snr10", transform_noise, {"snr_db": 10.0, "color": "pink"}, True),
    Condition("noise_pink", "snr5", transform_noise, {"snr_db": 5.0, "color": "pink"}, True),
    Condition("noise_pink", "snr0", transform_noise, {"snr_db": 0.0, "color": "pink"}, True),
    # 3) Synthetic reverb (exponential-decay white-noise IR), RT60/wet pairs
    # from light room tail to unrealistically extreme (surface endpoints).
    Condition("reverb", "light", transform_reverb, {"rt60_sec": 0.3, "wet": 0.2}, True),
    Condition("reverb", "medium", transform_reverb, {"rt60_sec": 0.6, "wet": 0.3}, True),
    Condition("reverb", "heavy", transform_reverb, {"rt60_sec": 1.2, "wet": 0.4}, True),
    Condition("reverb", "extreme", transform_reverb, {"rt60_sec": 2.0, "wet": 0.5}, True),
    # 4) Low-pass (mic distance / muffling proxy), zero-phase Butterworth.
    Condition("lowpass", "8000hz", transform_lowpass, {"cutoff_hz": 8000.0}),
    Condition("lowpass", "4000hz", transform_lowpass, {"cutoff_hz": 4000.0}),
    Condition("lowpass", "2000hz", transform_lowpass, {"cutoff_hz": 2000.0}),
    Condition("lowpass", "1000hz", transform_lowpass, {"cutoff_hz": 1000.0}),
]


def corpus_tx_ids() -> list[str]:
    """Repo-managed free-performance corpus recordings with ground_truth.json.

    Restricted to the repo-managed corpus (not local-only transaction-captures)
    so this script is reproducible from a fresh checkout with no local data/.
    """
    ids = []
    for d in sorted(nfb.FREE_PERFORMANCE_CORPUS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (
            (d / "ground_truth.json").is_file()
            and (d / "audio.wav").is_file()
            and (d / "request.json").is_file()
        ):
            ids.append(d.name)
    return ids


def load_audio_mono(tx_id: str) -> tuple[np.ndarray, int]:
    tx_dir = nfb.FREE_PERFORMANCE_CORPUS_DIR / tx_id
    audio, sample_rate = sf.read(tx_dir / "audio.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sample_rate)


def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe_bytes(client, tx_id: str, audio_bytes: bytes, *, debug: bool = False) -> dict:
    """Same POST shape as note_f1_benchmark.transcribe_payload, but with
    caller-supplied (transformed) audio bytes instead of the on-disk file."""
    tx_dir = nfb.transaction_dir_for(tx_id)
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "true" if debug else "false",
            "dryRun": "true",
            "force": "true",
        },
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    return response


def run_condition(client, tx_id: str, audio: np.ndarray, sample_rate: int, truth: list[dict], cond: Condition) -> dict:
    kwargs = dict(cond.params)
    if cond.needs_seed:
        kwargs["seed"] = _seed_for(cond.family, cond.level, tx_id, BASE_SEED)
    transformed, meta = cond.fn(audio, sample_rate, **kwargs)
    audio_bytes = encode_wav_bytes(transformed, sample_rate)
    response = transcribe_bytes(client, tx_id, audio_bytes, debug=False)
    if response.status_code != 200:
        return {
            "txId": tx_id,
            "family": cond.family,
            "level": cond.level,
            "params": meta,
            "error": f"HTTP {response.status_code}: {response.text[:300]}",
        }
    payload = response.json()
    primary = nfb.collect_one_best(payload)
    match = nfb.match_pairs(truth, primary)
    return {
        "txId": tx_id,
        "family": cond.family,
        "level": cond.level,
        "params": meta,
        "truthNotes": match["truthNotes"],
        "predictedNotes": match["predictedNotes"],
        "tp": match["tp"],
        "precision": match["precision"],
        "recall": match["recall"],
        "f1": match["f1"],
        "warnings": payload.get("warnings", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DSP augmentation robustness map (report-only)")
    parser.add_argument("--tx", action="append", dest="tx_ids", help="restrict to these tx ids (repeatable)")
    parser.add_argument("--json", action="store_true", help="print JSON to stdout instead of a table")
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_JSON_OUT, help="write full JSON results here"
    )
    parser.add_argument(
        "--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT, help="write the markdown surface report here"
    )
    parser.add_argument("--no-write", action="store_true", help="do not write report files, stdout only")
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from apps.api.app.main import app

    tx_ids = args.tx_ids or corpus_tx_ids()
    if not tx_ids:
        print("No repo-managed free-performance corpus recordings with ground_truth.json found.", file=sys.stderr)
        return 1

    client = TestClient(app)
    baselines: dict[str, dict] = {}
    results: list[dict] = []

    for tx_id in tx_ids:
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx_id))
        audio, sample_rate = load_audio_mono(tx_id)
        for cond in CONDITIONS:
            outcome = run_condition(client, tx_id, audio, sample_rate, truth, cond)
            results.append(outcome)
            if cond.family == "none":
                baselines[tx_id] = outcome

    # Cross-check against note_f1_benchmark.py's own (unmodified-audio) F1,
    # computed via the exact same code path (transcribe_payload + match_pairs),
    # to catch any accidental divergence in the matching-logic reuse.
    reference_check = []
    for tx_id in tx_ids:
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx_id))
        payload = nfb.transcribe_payload(client, tx_id, debug=False)
        primary = nfb.collect_one_best(payload)
        match = nfb.match_pairs(truth, primary)
        aug_f1 = baselines.get(tx_id, {}).get("f1")
        reference_check.append(
            {
                "txId": tx_id,
                "noteF1BenchmarkF1": match["f1"],
                "augmentationBaselineF1": aug_f1,
                "matches": aug_f1 is not None and abs(match["f1"] - aug_f1) < 1e-9,
            }
        )

    summary = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "baseSeed": BASE_SEED,
        "recognizerFingerprint": recognizer_fingerprint(),
        "kalimbaDspFingerprint": kalimba_dsp_fingerprint(),
        "txIds": tx_ids,
        "reportOnly": True,
        "note": (
            "report-only per docs/sprint-plan-2026-07b.md S2 bets #1 / AGENTS.md guardrail 7:"
            " not a regression gate, not counted toward overfitting-gate or S5-branch"
            " non-saturated-recording n."
        ),
        "referenceCheck": reference_check,
    }

    output = {"summary": summary, "results": results}

    if not args.no_write:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(output), encoding="utf-8")

    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    print_table(output)
    return 0


def print_table(output: dict) -> None:
    summary = output["summary"]
    print(f"recognizer {summary['recognizerFingerprint']} dsp {summary['kalimbaDspFingerprint']}")
    for check in summary["referenceCheck"]:
        status = "OK" if check["matches"] else "MISMATCH"
        print(
            f"[{status}] {check['txId']}: note_f1_benchmark F1={check['noteF1BenchmarkF1']:.4f}"
            f" vs augmentation-script baseline F1={check['augmentationBaselineF1']}"
        )
    print()
    print(f"{'txId':38} {'family':13} {'level':10} {'GT':>4} {'pred':>5} {'TP':>4} {'F1':>7}")
    for r in output["results"]:
        if "error" in r:
            print(f"{r['txId'][:36]:38} {r['family']:13} {r['level']:10} ERROR: {r['error']}")
            continue
        print(
            f"{r['txId'][:36]:38} {r['family']:13} {r['level']:10}"
            f" {r['truthNotes']:>4} {r['predictedNotes']:>5} {r['tp']:>4} {r['f1']:7.4f}"
        )


def render_markdown(output: dict) -> str:
    summary = output["summary"]
    results = output["results"]
    lines: list[str] = []
    lines.append("# DSP Augmentation Robustness Map (S2 bets #1)")
    lines.append("")
    lines.append(
        "**REPORT-ONLY.** Not a regression gate. Augmented recordings do not count toward"
        " the overfitting-gate or S5-branch non-saturated-recording n (AGENTS.md guardrail 7,"
        " docs/sprint-plan-2026-07b.md S2 bets #1)."
    )
    lines.append("")
    lines.append(f"- Generated: {summary['generatedAt']}")
    lines.append(f"- Recognizer fingerprint: `{summary['recognizerFingerprint']}`")
    lines.append(f"- kalimba_dsp fingerprint: `{summary['kalimbaDspFingerprint']}`")
    lines.append(f"- Base seed: `{summary['baseSeed']}`")
    lines.append(f"- Recordings: {', '.join(summary['txIds'])}")
    lines.append("")
    lines.append("## Baseline reproduction check")
    lines.append("")
    lines.append("| txId | note_f1_benchmark.py F1 | augmentation-script (no-op) F1 | match |")
    lines.append("|---|---|---|---|")
    for check in summary["referenceCheck"]:
        mark = "yes" if check["matches"] else "**NO — investigate**"
        lines.append(
            f"| {check['txId']} | {check['noteF1BenchmarkF1']:.4f} |"
            f" {check['augmentationBaselineF1']:.4f} | {mark} |"
        )
    lines.append("")

    by_tx: dict[str, list[dict]] = {}
    for r in results:
        by_tx.setdefault(r["txId"], []).append(r)

    families = list(dict.fromkeys(r["family"] for r in results))
    for tx_id, rows in by_tx.items():
        lines.append(f"## {tx_id}")
        lines.append("")
        lines.append("| family | level | params | GT | pred | TP | F1 | clipped |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['family']} | {r['level']} | {r['params']} | - | - | - | ERROR | - |")
                continue
            clipped = r["params"].get("clippedFraction")
            clipped_text = f"{clipped:.1%}" if clipped else "-"
            params_text = ", ".join(f"{k}={v}" for k, v in r["params"].items() if k != "clippedFraction")
            lines.append(
                f"| {r['family']} | {r['level']} | {params_text or '-'} | {r['truthNotes']} |"
                f" {r['predictedNotes']} | {r['tp']} | {r['f1']:.4f} | {clipped_text} |"
            )
        lines.append("")

    lines.append("## Mean F1 by family/level (across recordings)")
    lines.append("")
    lines.append("| family | level | mean F1 | recordings |")
    lines.append("|---|---|---|---|")
    seen = set()
    for r in results:
        key = (r["family"], r["level"])
        if key in seen or "error" in r:
            continue
        seen.add(key)
        vals = [x["f1"] for x in results if x["family"] == r["family"] and x["level"] == r["level"] and "error" not in x]
        if vals:
            lines.append(f"| {r['family']} | {r['level']} | {sum(vals) / len(vals):.4f} | {len(vals)} |")
    lines.append("")
    lines.append(render_analysis_section())
    return "\n".join(lines) + "\n"


def render_analysis_section() -> str:
    """Static interpretation written against the 2026-07-04 baseline run.

    Re-validate this prose (not just the tables above) after major recognizer
    or corpus changes -- it describes *why* the surface looks the way it does,
    which can go stale even when the numeric tables stay auto-generated.
    """
    return """## Interpretation: predicted recognizer weak points (2026-07-04 baseline run)

No condition thinning was needed: the full grid (24 conditions x 2 recordings
= 48 recognizer calls) completed in well under a minute, far inside the
30-minute budget in the task brief.

1. **Gain scaling is a cliff, not a slope.** Mean F1 is essentially flat
   across -10dB/0dB/+6dB (~0.79-0.93) then collapses to ~0.39 at -20dB and to
   0.00 at -30dB on both recordings. This is not the pipeline's silence guard
   (`read_audio` rejects only when peak amplitude < 1e-4; at -30dB the
   quieter recording's peak is still ~30x that floor). Both recordings also
   surface the pipeline's own `"Only a small number of note events were
   detected."` warning exactly at the two collapsing levels. Because most
   calibrated onset constants in `constants.py` are expressed as *ratios*
   (`ONSET_GATE_MIN_ONSET_GAIN`, backward-attack-gain thresholds, etc.), a
   clean ratio-based design should degrade smoothly with gain; the observed
   cliff instead points to some absolute-magnitude-dependent stage in the
   broadband onset/spectral-flux pipeline. This corroborates the existing
   project note `feedback_gain_vs_attack_profile.md` ("gain絶対量はマイク距離
   変化に弱い") — quiet or mic-distant captures are a predicted practical
   failure mode, not just louder/quieter versions of the same transcription.

2. **Additive noise degrades fast even at nominal high SNR.** Mean F1 drops
   from ~0.93 (clean) to ~0.72-0.85 already at SNR 30dB (white/pink) and
   continues to ~0.20-0.30 by SNR 0dB, close to monotonic for both colors.
   SNR here is defined against whole-recording RMS; a supplementary check
   confirmed the RMS in +/-100ms windows around actual onsets is within ~2dB
   of the whole-file RMS for both recordings, so this is not a
   silence-diluted-SNR artifact — the effective SNR near real note attacks is
   close to nominal. Predicted-note *counts* drop under noise rather than
   balloon (e.g. 17ea7626: 43 predicted at baseline vs 13-26 under noise),
   meaning noise mostly suppresses true attacks rather than only adding false
   positives. This predicts broadband spectral-flux onset detection is more
   noise-floor-sensitive than gain-sensitive: room hiss, HVAC noise, or
   electrical hum during real capture is a meaningfully larger practical risk
   than the current (comparatively controlled) fixture/corpus recordings
   would suggest.

3. **Reverb is the gentlest-sloped family — a likely surface/kill-condition
   mismatch.** Mean F1 barely moves at "light"/"medium" RT60 (0.87, at or
   above the additive-noise floor) and only reaches ~0.70 at "extreme"
   (RT60=2.0s, wet=0.5 — an unrealistically large hall for solo kalimba).
   This is weaker than the task brief's working hypothesis ("reverb wet 高で
   F1 が崩れる → carryover 判別が弱い"). The likely reason is the IR family
   used here (exponential-decay *white noise*) has no coherent per-note
   spectral structure, unlike real kalimba sympathetic-tine carryover/decay,
   which retains the ringing note's own partials. **This is the sharpest kill
   condition to watch**: if a real adversarial "carryover" recording (Mech2,
   per the S1/S2 敵対的セルフ録音 menu) shows large F1 collapse that this
   synthetic reverb sweep did not predict, that is direct evidence this
   augmentation family should be downgraded to invariance-testing use only
   (per the task brief's explicit kill condition) rather than used to predict
   real reverberant-carryover weaknesses.

4. **Low-pass filtering degrades close to linearly with cutoff** — the most
   "expected", least surprising family. Mean F1 ~0.93 at an 8kHz cutoff
   (near baseline; fundamentals and first partials of essentially all kalimba
   notes survive) down to ~0.63 at 1kHz (cuts into the fundamental/first
   partial region for mid/high notes). Predicted-note counts stay closer to
   truth-note counts than in the noise family, suggesting this failure mode
   is more about note-identity substitution (per-tine partial matching /
   narrow-FFT confusion) than dropped onsets — a natural follow-up would be a
   false-positive/false-negative breakdown, out of scope for this
   report-only pass.

**Overall predicted risk ranking** (steepest to gentlest F1 collapse in this
surface): gain (cliff at quiet levels) > additive noise > low-pass/muffling >
reverb. This ranking is a testable prediction against the S1/S2 adversarial
recording menu once those takes have human-reviewed ground truth: if the
ranking or the reverb/carryover mismatch above does not hold on real
recordings, downgrade this bet to an invariance-testing tool only, per the
task brief's kill condition (docs/sprint-plan-2026-07b.md 頑健性マップの
「予測される弱点機構」が実新録音の弱点と一致しなければ不変性テスト用途に
格下げ).

Transforms were applied as independent single-family sweeps (no pairwise/grid
combinations) to keep the surface interpretable and the runtime small;
combined conditions (e.g. noise + reverb) are a natural extension if this bet
graduates past the kill condition above.
"""


if __name__ == "__main__":
    raise SystemExit(main())
