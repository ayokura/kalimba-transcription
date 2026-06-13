/**
 * Replay native-Rust reference cases through the wasm build and assert equality.
 *
 * Pair of `wasm_reference.py` (which produced the reference dir using the native
 * pyo3 extension). Both bindings share the pure-Rust core, so any mismatch is a
 * binding-glue regression (Float32Array marshalling, i64->BigInt / u32 ABI), not
 * an algorithm change.
 *
 *   node tools/check_wasm.cjs <nodejs_pkg_dir> <reference_dir>
 *
 * Scalar-output functions only for now. When R3 exposes array-returning
 * primitives (mel_filterbank, onset_strength envelope, peak_pick indices),
 * extend the dispatch + comparison below to handle Float32Array / index outputs.
 */
const fs = require("fs");
const path = require("path");

const [, , pkgDir, refDir] = process.argv;
if (!pkgDir || !refDir) {
  console.error("usage: node check_wasm.cjs <nodejs_pkg_dir> <reference_dir>");
  process.exit(2);
}

const wasm = require(path.join(pkgDir, "kalimba_dsp.js"));
const ref = JSON.parse(fs.readFileSync(path.join(refDir, "reference.json"), "utf8"));

const audioCache = {};
function loadAudio(name) {
  if (!(name in audioCache)) {
    const meta = ref.audioFiles[name];
    const buf = fs.readFileSync(path.join(refDir, meta.file));
    audioCache[name] = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  }
  return audioCache[name];
}

let pass = 0;
const failures = [];
for (const c of ref.cases) {
  const fn = wasm[c.fn];
  if (typeof fn !== "function") {
    failures.push(`${c.name}: wasm export '${c.fn}' missing`);
    continue;
  }
  // i64 sample_rate maps to a JS BigInt; everything else is a plain number.
  const args = c.audio
    ? [loadAudio(c.audio), BigInt(c.sampleRate), ...c.scalars]
    : [BigInt(c.sampleRate), ...c.scalars];
  const got = fn(...args);

  let ok;
  if (c.exact) {
    ok = got === c.expected;
  } else {
    const atol = c.atol ?? 0;
    const rtol = c.rtol ?? 1e-6;
    ok = Math.abs(got - c.expected) <= atol + rtol * Math.abs(c.expected);
  }
  if (ok) pass++;
  else failures.push(`${c.name}: wasm=${got} native=${c.expected}`);
}

for (const f of failures) console.error("FAIL " + f);
console.log(`wasm equivalence: ${pass}/${ref.cases.length} cases passed`);
process.exit(failures.length ? 1 : 0);
