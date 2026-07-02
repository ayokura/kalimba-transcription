#!/usr/bin/env bash
# Build the wasm package and verify it is numerically equivalent to the native
# pyo3 extension across a battery of inputs (Float32Array marshalling + ABI).
#
# Requires: wasm-pack, node, and the uv-managed Python env (for the native
# kalimba_dsp extension that produces the reference values). The wasm32 target
# must be installed once: `rustup target add wasm32-unknown-unknown`.
#
#   crates/kalimba-dsp/check_wasm.sh
#
# Exit non-zero on any wasm-vs-native mismatch. Safe to wire into CI once the
# toolchain is provisioned.
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CRATE_DIR/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
NODE_PKG="$WORK/pkg-node"
REF_DIR="$WORK/ref"
mkdir -p "$REF_DIR"

PARITY_DIR="$WORK/parity"
mkdir -p "$PARITY_DIR"

echo "[1/5] building nodejs-target wasm"
( cd "$CRATE_DIR" && wasm-pack build --target nodejs --out-dir "$NODE_PKG" -- \
    --no-default-features --features wasm ) >/dev/null

echo "[2/5] generating native reference values (synthetic)"
( cd "$REPO_ROOT" && PYTHONPATH=apps/api uv run python \
    "$CRATE_DIR/tools/wasm_reference.py" "$REF_DIR" )

echo "[3/5] checking wasm outputs vs native (synthetic)"
node "$CRATE_DIR/tools/check_wasm.cjs" "$NODE_PKG" "$REF_DIR"

echo "[4/5] generating fixture parity references (real WAVs, native + numpy + segments)"
( cd "$REPO_ROOT" && PYTHONPATH=apps/api uv run python \
    "$CRATE_DIR/tools/wasm_parity_reference.py" "$PARITY_DIR" )

echo "[5/5] checking wasm through-path (audio->onset_strength->onset_detect) vs references"
node "$CRATE_DIR/tools/check_wasm_parity.cjs" "$NODE_PKG" "$PARITY_DIR"
