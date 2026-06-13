/**
 * Replay native-Rust reference cases through the wasm build and assert equality.
 *
 * Pairs with `wasm_reference.py` (which produced the reference dir using the
 * native pyo3 extension). Both bindings share the pure-Rust core, so any
 * mismatch is a binding-glue regression (Float32Array / Uint32Array marshalling,
 * i64->BigInt / u32 ABI), not an algorithm change.
 *
 *   node tools/check_wasm.cjs <nodejs_pkg_dir> <reference_dir>
 *
 * Handles scalar, float-array, and uint32-index outputs. See wasm_reference.py
 * for the case schema.
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

function readF32(file) {
  const buf = fs.readFileSync(path.join(refDir, file));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}
function readU32(file) {
  const buf = fs.readFileSync(path.join(refDir, file));
  return new Uint32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}

function buildArg(spec) {
  if ("f32arr" in spec) return readF32(spec.f32arr);
  if ("u32arr" in spec) return readU32(spec.u32arr);
  if ("i64" in spec) return BigInt(spec.i64);
  if ("f64" in spec) return spec.f64;
  if ("u32" in spec) return spec.u32;
  if ("bool" in spec) return spec.bool;
  throw new Error("unknown arg spec: " + JSON.stringify(spec));
}

let pass = 0;
const failures = [];
for (const c of ref.cases) {
  const fn = wasm[c.fn];
  if (typeof fn !== "function") {
    failures.push(`${c.name}: wasm export '${c.fn}' missing`);
    continue;
  }
  const got = fn(...c.args.map(buildArg));

  if ("expected" in c) {
    const ok = c.exact
      ? got === c.expected
      : Math.abs(got - c.expected) <= (c.atol ?? 0) + (c.rtol ?? 1e-6) * Math.abs(c.expected);
    if (ok) pass++;
    else failures.push(`${c.name}: wasm=${got} native=${c.expected}`);
  } else if ("expectedArray" in c) {
    const exp = readF32(c.expectedArray);
    let maxd = 0, bad = -1;
    if (got.length !== exp.length) {
      failures.push(`${c.name}: length wasm=${got.length} native=${exp.length}`);
      continue;
    }
    const atol = c.atol ?? 0, rtol = c.rtol ?? 1e-6;
    for (let i = 0; i < exp.length; i++) {
      const d = Math.abs(got[i] - exp[i]);
      if (d > maxd) maxd = d;
      if (d > atol + rtol * Math.abs(exp[i]) && bad < 0) bad = i;
    }
    if (bad < 0) pass++;
    else failures.push(`${c.name}: max|d|=${maxd.toExponential(2)} first bad @${bad} wasm=${got[bad]} native=${exp[bad]}`);
  } else if ("expectedIndices" in c) {
    const exp = readU32(c.expectedIndices);
    const ok = got.length === exp.length && exp.every((v, i) => v === got[i]);
    if (ok) pass++;
    else failures.push(`${c.name}: indices wasm=[${Array.from(got)}] native=[${Array.from(exp)}]`);
  } else {
    failures.push(`${c.name}: no output spec`);
  }
}

for (const f of failures) console.error("FAIL " + f);
console.log(`wasm equivalence: ${pass}/${ref.cases.length} cases passed`);
process.exit(failures.length ? 1 : 0);
