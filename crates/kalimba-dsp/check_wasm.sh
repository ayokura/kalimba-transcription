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

echo "[1/3] building nodejs-target wasm"
( cd "$CRATE_DIR" && wasm-pack build --target nodejs --out-dir "$NODE_PKG" -- \
    --no-default-features --features wasm ) >/dev/null

echo "[2/3] generating native reference values"
( cd "$REPO_ROOT" && PYTHONPATH=apps/api uv run python \
    "$CRATE_DIR/tools/wasm_reference.py" "$REF_DIR" )

echo "[3/3] checking wasm outputs vs native"
node "$CRATE_DIR/tools/check_wasm.cjs" "$NODE_PKG" "$REF_DIR"
