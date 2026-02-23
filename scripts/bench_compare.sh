#!/bin/bash
# Peregrine vs ML Frameworks wall-clock benchmark orchestrator.
#
# Builds Peregrine in release mode, then runs each framework benchmark
# sequentially with nice -n 10 to keep resource usage under 80%.
#
# Usage: ./scripts/bench_compare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OUT_DIR="target/bench_compare"
mkdir -p "$OUT_DIR"

echo "=== Peregrine vs ML Frameworks — Wall-Clock Benchmark ==="
echo ""

# --- Build Peregrine in release mode ---
echo "[1/7] Building Peregrine (release)..."
cargo build --release --bench wallclock 2>&1 | tail -1
echo ""

# --- Run PyTorch benchmark ---
echo "[2/7] Running PyTorch benchmark..."
nice -n 10 .venv/bin/python scripts/bench_pytorch.py
echo ""

# --- Run MLX benchmark ---
echo "[3/7] Running MLX benchmark..."
nice -n 10 .venv/bin/python scripts/bench_mlx.py
echo ""

# --- Run TensorFlow benchmark ---
echo "[4/7] Running TensorFlow benchmark..."
nice -n 10 .venv/bin/python scripts/bench_tensorflow.py
echo ""

# --- Run tinygrad benchmark ---
echo "[5/7] Running tinygrad benchmark..."
nice -n 10 .venv/bin/python scripts/bench_tinygrad.py
echo ""

# --- Run Peregrine benchmark ---
echo "[6/7] Running Peregrine benchmark..."
nice -n 10 cargo bench --bench wallclock 2>&1
echo ""

# --- Compare results ---
echo "[7/7] Comparing results..."
echo ""
.venv/bin/python scripts/compare_bench.py
echo ""
echo "=== Done ==="
