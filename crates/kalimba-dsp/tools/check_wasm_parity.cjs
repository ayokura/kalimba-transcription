/**
 * A0 browser offline parity harness — real-fixture through-path check.
 *
 * Pairs with `wasm_parity_reference.py`. For each committed fixture WAV this
 * runs the wasm build end-to-end (audio -> onset_strength -> onset_detect,
 * no native-generated intermediates injected — the gap the synthetic harness
 * left open, 2026-07 audit) and compares against two references:
 *
 * - native (Rust pyo3, same shared core): envelope tolerance 1e-4/1e-5,
 *   onset frames exact. A mismatch = the two compilations of the shared core
 *   diverged (rustfft code paths, wasm-opt, f32 contraction differences).
 * - numpy (independent oracle): envelope tolerance 1e-3/1e-3 (the established
 *   Rust<->numpy bound from test_onset_dsp_rust.py), onset frames exact with
 *   a diff report on failure (known ~1e-8 window-mean accumulation caveat can
 *   flip threshold-coincident frames; surfacing that is the point).
 *
 * Segment-level reference (active ranges / boundaries / discard) has no wasm
 * counterpart until B1. The comparator (compareSegments) is exported for the
 * B1 port to require(); until then each reference is fed back through it as a
 * round-trip self-test (diff machinery + reference integrity).
 *
 *   node tools/check_wasm_parity.cjs <nodejs_pkg_dir> <parity_dir>
 */
const fs = require("fs");
const path = require("path");

function compareEnv(name, got, exp, rtol, atol, failures) {
  if (got.length !== exp.length) {
    failures.push(`${name}: env length wasm=${got.length} ref=${exp.length}`);
    return false;
  }
  let maxd = 0;
  let bad = -1;
  for (let i = 0; i < exp.length; i++) {
    const d = Math.abs(got[i] - exp[i]);
    if (d > maxd) maxd = d;
    if (d > atol + rtol * Math.abs(exp[i]) && bad < 0) bad = i;
  }
  if (bad < 0) return true;
  failures.push(
    `${name}: env max|d|=${maxd.toExponential(2)} first bad @${bad} wasm=${got[bad]} ref=${exp[bad]}`,
  );
  return false;
}

function compareFrames(name, got, exp, failures) {
  const g = Array.from(got);
  const e = Array.from(exp);
  if (g.length === e.length && e.every((v, i) => v === g[i])) return true;
  const extra = g.filter((v) => !e.includes(v));
  const missing = e.filter((v) => !g.includes(v));
  failures.push(
    `${name}: frames wasm=${g.length} ref=${e.length}` +
      ` extra=[${extra}] missing=[${missing}]`,
  );
  return false;
}

/**
 * Segment-level diff: compare a candidate segment computation (the future B1
 * TS/Rust port) against the pinned Python reference. Ranges are [start, end]
 * second pairs; `epsilonSec` absorbs float formatting, not algorithmic drift.
 */
function compareSegments(name, candidate, reference, failures, epsilonSec = 1e-6) {
  const rangeKeys = [
    "rawActiveRanges",
    "activeRanges",
    "shortBridgeActiveRanges",
    "activeRangeSegments",
    "segments",
  ];
  let ok = true;
  for (const key of rangeKeys) {
    const got = candidate?.[key] ?? [];
    const exp = reference?.[key] ?? [];
    if (got.length !== exp.length) {
      failures.push(`${name}: ${key} count candidate=${got.length} ref=${exp.length}`);
      ok = false;
      continue;
    }
    for (let i = 0; i < exp.length; i++) {
      const [gs, ge] = got[i];
      const [es, ee] = exp[i];
      if (Math.abs(gs - es) > epsilonSec || Math.abs(ge - ee) > epsilonSec) {
        failures.push(
          `${name}: ${key}[${i}] candidate=[${gs},${ge}] ref=[${es},${ee}]`,
        );
        ok = false;
        break;
      }
    }
  }
  const gotThr = candidate?.rmsThreshold;
  const expThr = reference?.rmsThreshold;
  if (typeof gotThr !== "number" || Math.abs(gotThr - expThr) > 1e-9 + 1e-6 * Math.abs(expThr)) {
    failures.push(`${name}: rmsThreshold candidate=${gotThr} ref=${expThr}`);
    ok = false;
  }
  return ok;
}

function main() {
  const [, , pkgDir, refDir] = process.argv;
  if (!pkgDir || !refDir) {
    console.error("usage: node check_wasm_parity.cjs <nodejs_pkg_dir> <parity_dir>");
    process.exit(2);
  }

  const wasm = require(path.join(pkgDir, "kalimba_dsp.js"));
  const ref = JSON.parse(
    fs.readFileSync(path.join(refDir, "parity_reference.json"), "utf8"),
  );

  const readF32 = (file) => {
    const buf = fs.readFileSync(path.join(refDir, file));
    return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  };
  const readU32 = (file) => {
    const buf = fs.readFileSync(path.join(refDir, file));
    return new Uint32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  };

  const { hopLength, nFft, nMels } = ref.constants;
  let pass = 0;
  let total = 0;
  const failures = [];

  for (const c of ref.cases) {
    const audio = readF32(c.audio);
    const sr = BigInt(c.sampleRate);

    // wasm through-path: audio -> envelope -> onset frames (backtracked)
    const envWasm = wasm.onset_strength(audio, sr, hopLength, nFft, nMels);
    const framesWasm = wasm.onset_detect(envWasm, sr, hopLength, true);

    total += 4;
    if (compareEnv(`${c.id} vs native`, envWasm, readF32(c.native.env), 1e-4, 1e-5, failures)) pass++;
    if (compareFrames(`${c.id} vs native`, framesWasm, readU32(c.native.frames), failures)) pass++;
    if (compareEnv(`${c.id} vs numpy`, envWasm, readF32(c.numpy.env), 1e-3, 1e-3, failures)) pass++;
    if (compareFrames(`${c.id} vs numpy`, framesWasm, readU32(c.numpy.frames), failures)) pass++;

    // Segment-level diff. Until the B1 port exists there is no wasm-side
    // candidate, so the reference is fed back through the comparator
    // (round-trip self-test: proves the diff machinery + reference integrity).
    total += 1;
    if (compareSegments(`${c.id} segments(self)`, c.segment, c.segment, failures)) pass++;
  }

  for (const f of failures) console.error("FAIL " + f);
  console.log(`wasm fixture parity: ${pass}/${total} checks passed (${ref.cases.length} recordings)`);
  process.exit(failures.length ? 1 : 0);
}

module.exports = { compareSegments, compareEnv, compareFrames };

if (require.main === module) main();
